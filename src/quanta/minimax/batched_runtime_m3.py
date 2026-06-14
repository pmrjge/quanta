"""Batched (B>1) RAM-resident MiniMax-M3-VL runtime — multi-stream serving over shared weights.

Wraps :class:`quanta.minimax.runtime_m3.MiniMaxM3ResidentModel` to amortize the routed-expert read
across many concurrent decode streams without touching the proven single-stream forward (the
parity-gated M1/M2 reference lives there). Single-stream M3 decode is memory-bandwidth bound on the
routed-experts read (per token, per MoE layer: top-4 expert slots × {gate_up, down} + the always-on
shared expert, all ``gather_qmm``-fed); batching ``B`` streams reads the **same** expert weight tile
once and dispatches all ``B`` tokens through one MoE call (``mx.gather_qmm`` is B-aware via the
existing ``[B,1,h] → [N=B,h]`` reshape in :meth:`quanta.minimax.model_m3.MiniMaxM3MoE.__call__`).
Target operating point: B≈32 (the fleet cohort point), ~order-of-magnitude aggregate throughput.

**Design A** — per-stream caches, stack-for-MoE (the parity-trivial path; mirrors
:mod:`quanta.qwen35.batched_runtime` but on M3's uniform all-GQA backbone — no GDN hybrid, no YaRN):

* Each stream keeps its OWN per-layer GQA :class:`quanta.minimax.model_m3.KVCache` list (one per
  decoder layer). Per-stream KV is unavoidable: each stream's k/v rows are its own positions.
* Per-layer decode step (one token per stream, ``B`` streams):
    1. **per-stream attention step** — iterate streams, run the same ``MiniMaxM3Block`` attention
       half (``x + attn(in_norm(x))``) through each stream's per-layer cache exactly as the
       single-stream runtime does. This loop is a bounded IO loop over the small batch (rule 3 —
       NOT a hot per-token loop), and the GQA work inside each call is fully vectorized over
       heads/dims; each call is M=1 so the packed q/k/v/o ``mx.quantized_matmul`` is bit-exact vs
       the single-stream decode. Each stream's cache mutates in place.
    2. **stack** the ``B`` post-attention residuals ``[1,1,hidden]`` → ``[B,1,hidden]``.
    3. **batched MoE / dense-FFN sub-block** — feed ``[B,1,hidden]`` through the **same** block FFN.
       For MoE layers (3–59) the existing ``MiniMaxM3MoE`` routes ``[B,1,h] → [N=B,h]``: top-4
       routing runs on ``B`` rows in one shot; ``gather_qmm`` over the packed int6 stacks reads each
       touched expert's codes ONCE for every token that routes to it; the shared expert runs once
       over ``[B,h]``. This is the bandwidth win. Dense layers (0–2) run the FFN once over ``[B,h]``.
    4. **split** ``[B,1,hidden]`` back into per-stream ``[1,1,hidden]`` residuals.
* Every token completes ALL layers (advancing its per-stream KV) before the next begins, so
  per-stream output is causally identical to a single-stream decode at the same offset — gated
  model-free in ``parity/minimax_m3_batched_test.py`` (the routed ``gather_qmm`` slots are the
  per-(token,slot) M=1 matvec, batch-invariant; only the batched router/shared GEMM can ULP-reorder
  ⇒ greedy-token-equivalent, top-1 exact).

Out of scope here (later M3 sub-milestones): the GQA loop-kill (ONE batched attention across streams
— needs ragged-offset padded SDPA; M3-3), paged-KV + prefix caching (int8 KV), and chunked prefill.
Prefill is single-stream (:meth:`prefill` consumes one prompt into one stream's cache via the proven
cached forward); the win is in DECODE, where multi-stream serving spends ~all of its time.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx

from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.model_m3 import KVCache, MiniMaxM3Block
from quanta.minimax.runtime_m3 import MiniMaxM3ResidentModel


def _attn_step(blk: MiniMaxM3Block, kv: KVCache, x_t: mx.array, *, use_fast: bool = True) -> mx.array:
    """One stream's attention half through its per-layer KV cache: ``x + attn(in_norm(x))``.

    Mirrors the first half of :meth:`MiniMaxM3Block.__call__` (rule 4 — the exact reference attention,
    the cache mutates in place). ``x_t`` ``[1,1,hidden]`` → post-attention residual ``[1,1,hidden]``.
    M3 carries no length-dependent RoPE factor (native 1M, no YaRN), so the attention reads its
    absolute position from ``kv.offset`` with nothing else to thread."""
    y = blk.self_attn(blk.input_layernorm(x_t), cache=kv, use_fast=use_fast)
    return x_t + y


def batched_decode_step(
    layers: Sequence[MiniMaxM3Block],
    embed_w: mx.array,
    norm_w: mx.array,
    lm_head_w: mx.array,
    cfg: MiniMaxM3Config,
    stream_token_ids: Sequence[int],
    stream_caches: Sequence[Sequence[KVCache]],
    offsets: Sequence[int],
    *,
    use_fast: bool = True,
) -> list[mx.array]:
    """One batched decode step over ``B = len(stream_token_ids)`` streams. Returns per-stream logits
    ``[1,1,vocab]`` — one per stream, in input order, so the caller can sample each independently.

    Pure function — no model state outside the per-stream caches (which mutate in place) and the
    passed weights / layers. The Design-A per-stream attention step iterates streams; the FFN
    sub-block runs ONCE on the stacked ``[B,1,hidden]`` (the routed-expert read amortizes across all
    ``B``). Per-stream output is greedy-token-equivalent (top-1 exact) to feeding the same token
    through :class:`quanta.minimax.runtime_m3.MiniMaxM3ResidentModel` at the same offset against the
    same cache (the per-stream attention + per-slot ``gather_qmm`` are M=1 ⇒ bit-exact; only the
    batched router/shared GEMM ULP-reorders). Drives the resident runtime AND the model-free gate.
    """
    b = len(stream_token_ids)
    if b < 1:
        raise ValueError("step needs >= 1 stream")
    if len(stream_caches) != b or len(offsets) != b:
        raise ValueError(f"len(stream_token_ids)={b}, len(stream_caches)={len(stream_caches)}, "
                         f"len(offsets)={len(offsets)} — must match")
    n_layers = len(layers)
    # per-stream offset sanity: each stream's per-layer caches must already sit at its declared offset
    # (fail loud — rule 6: refuse to silently advance a desynced orchestrator). All per-layer caches in
    # a stream grow in lockstep, so caches[0].offset is the stream offset.
    for s, (caches_s, off) in enumerate(zip(stream_caches, offsets, strict=True)):
        if len(caches_s) != n_layers:
            raise ValueError(f"stream {s}: len(caches)={len(caches_s)} != num_layers={n_layers} "
                             f"(one KV cache per layer; refusing to mis-thread state — rule 6)")
        if caches_s[0].offset != off:
            raise ValueError(f"stream {s}: cache.offset={caches_s[0].offset} != declared offset={off} "
                             f"(orchestrator desynced; refusing to silently advance)")

    # per-stream embed (one gather across all B token ids in ONE indexing op)
    ids_b = mx.array([int(t) for t in stream_token_ids], dtype=mx.int32)            # [B]
    h_b = embed_w[ids_b][:, None].astype(mx.bfloat16)                               # [B,1,hidden]
    hs: list[mx.array] = [h_b[i:i + 1] for i in range(b)]                           # B × [1,1,hidden]

    for layer_i, blk in enumerate(layers):
        # 1) per-stream attention step (proven M=1 path; bounded IO loop over the small batch, rule 3).
        after_attn = [_attn_step(blk, stream_caches[s][layer_i], hs[s], use_fast=use_fast)
                      for s in range(b)]
        # 2) stack the B post-attention residuals -> [B,1,hidden] for the FFN sub-block
        stacked = mx.concatenate(after_attn, axis=0) if b > 1 else after_attn[0]    # [B,1,hidden]
        # 3) ONE batched FFN over all B tokens — MoE gather_qmm reads each touched routed-expert tile
        # once for all B rows that route to it (the bandwidth win); the shared expert + dense FFN run
        # once over [B,h]. The module's [B,S,h] -> [N=B,h] reshape is B-aware.
        h_post = blk.post_attention_layernorm(stacked)
        y = blk.mlp(h_post, sparse=True) if blk.is_moe else blk.mlp(h_post)         # [B,1,hidden]
        stacked = stacked + y
        # 4) split back to per-stream views for the next layer
        hs = [stacked[i:i + 1] for i in range(b)]
        mx.eval(stacked)                                                            # bound the per-layer graph

    # final Gemma (1+w) RMSNorm + lm_head over the stacked [B,1,hidden]
    final = mx.concatenate(hs, axis=0) if b > 1 else hs[0]                          # [B,1,hidden]
    final = mx.fast.rms_norm(final, norm_w.astype(final.dtype), cfg.norm_eps)
    logits_b = final @ lm_head_w.T.astype(final.dtype)                              # [B,1,vocab]
    return [logits_b[i:i + 1] for i in range(b)]


class MiniMaxM3BatchedResidentModel:
    """Batched (B>1) MiniMax-M3 decode over shared resident weights — Design A (per-stream caches).

    Loads the artifact ONCE via :class:`MiniMaxM3ResidentModel` (one-layer-resident, rule 8) and wraps
    it; weights are shared across streams. Each stream keeps its own per-layer
    :class:`quanta.minimax.model_m3.KVCache` list. The FFN sub-block is fed ``[B,1,hidden]`` so the
    routed-expert / shared-expert reads amortize across all ``B`` streams (the existing B-aware MoE).

    ``packed=True`` (default — the serving config) holds the int8 mixer (GQA q/k/v/o + dense-FFN)
    packed as ``nn.QuantizedLinear`` (``mx.quantized_matmul``); ``packed_experts=True`` (default)
    holds the routed experts packed int6 (``gather_qmm``). Both are greedy-exact on the SAME codes the
    bf16 reference dequantizes (gated). The shared expert + router gate/bias stay bf16/F32.

    Surface the orchestrator drives:

    * ``step_batch(stream_token_ids, stream_caches, offsets) -> list[mx.array]`` — one decode step for
      each stream; returns per-stream logits ``[1,1,vocab]``.
    * ``prefill(prompt_ids, state) -> mx.array`` — single-stream prefill into one stream's cache.
    * ``make_caches()`` / ``make_batch_caches(B)`` — per-stream KV cache factories.
    * ``.cfg``, ``.num_layers``, ``.embed_w``, ``.norm_w``, ``.lm_head_w``, ``.layers`` — same handles
      as :class:`MiniMaxM3ResidentModel`.
    """

    def __init__(self, art_dir: str | Path, *, max_batch: int = 32,
                 n_layers: int | None = None, packed: bool = True,
                 packed_experts: bool = True) -> None:
        if max_batch < 1:
            raise ValueError(f"max_batch must be >= 1, got {max_batch}")
        self._inner = MiniMaxM3ResidentModel(art_dir, n_layers=n_layers, packed=packed,
                                             packed_experts=packed_experts)
        self.max_batch = int(max_batch)
        self.cfg: MiniMaxM3Config = self._inner.cfg
        self.num_layers: int = self._inner.num_layers
        self.embed_w: mx.array = self._inner.embed_w
        self.norm_w: mx.array = self._inner.norm_w
        self.lm_head_w: mx.array = self._inner.lm_head_w
        self.layers: list[MiniMaxM3Block] = self._inner.layers
        self.packed = bool(packed)                       # int8 mixer held packed (mx.quantized_matmul)
        self.packed_experts = bool(packed_experts)       # routed experts packed int6 (gather_qmm)

    @classmethod
    def from_inner(cls, layers: list[MiniMaxM3Block], embed_w: mx.array, norm_w: mx.array,
                   lm_head_w: mx.array, cfg: MiniMaxM3Config, *,
                   max_batch: int = 32) -> MiniMaxM3BatchedResidentModel:
        """Construct from pre-built layers / final-form weights (bypasses artifact load) so the
        model-free parity gate can drive ``step_batch`` / ``prefill`` against a tiny synthetic model
        without a checkpoint — same step machinery, model-free. ``norm_w`` is the already-``(1+w)``-
        folded final norm. ``packed`` / ``packed_experts`` are detected from the passed layers."""
        self = cls.__new__(cls)
        self._inner = None
        self.max_batch = int(max_batch)
        self.cfg = cfg
        self.num_layers = len(layers)
        self.embed_w = embed_w
        self.norm_w = norm_w
        self.lm_head_w = lm_head_w
        self.layers = list(layers)
        moe = [b for b in layers if b.is_moe]
        self.packed_experts = bool(moe) and isinstance(moe[0].mlp.experts_gate_up, dict)
        import mlx.nn as nn
        self.packed = bool(layers) and isinstance(layers[0].self_attn.q_proj, nn.QuantizedLinear)
        return self

    # --- per-stream cache factory ---------------------------------------------
    def make_caches(self) -> list[KVCache]:
        """One stream's fresh per-layer GQA KV cache list (bf16; int8/paged KV is a later lever)."""
        return [KVCache() for _ in range(self.num_layers)]

    def make_batch_caches(self, batch: int) -> list[list[KVCache]]:
        """A list of ``batch`` independent per-stream cache lists (one per concurrent stream)."""
        if batch < 1:
            raise ValueError(f"batch must be >= 1, got {batch}")
        if batch > self.max_batch:
            raise ValueError(f"batch {batch} exceeds max_batch {self.max_batch} "
                             f"(raise max_batch on construction)")
        return [self.make_caches() for _ in range(batch)]

    # --- prefill (single-stream; consumes the prompt into ONE stream's cache) --
    def prefill(self, prompt_ids, state: Sequence[KVCache], *, use_fast: bool = True,
                sparse: bool = True) -> mx.array:
        """Consume ``prompt_ids`` into ``state`` (one stream's per-layer cache list); return the last
        position's logits ``[1,1,vocab]`` (the next-token distribution).

        Single-stream by design (Design A): full-attention prefill needs a common offset / causal mask
        across the consumed window, which cannot be shared with other streams' independent state. The
        orchestrator calls this once per new stream's prompt; decoding then proceeds via
        :meth:`step_batch`. Runs the proven cached forward over the whole ``[1,T,hidden]`` window (the
        ``T``-token cached-over-fresh-caches path the M3-1 gate proves bit-identical to the
        ``caches=None`` prefill), so the seeded state is bit-identical to a fresh single-stream run."""
        ids = mx.array([int(t) for t in prompt_ids], dtype=mx.int32)
        if int(ids.shape[0]) < 1:
            raise ValueError("prompt_ids is empty (prefill needs >= 1 token)")
        if len(state) != self.num_layers:
            raise ValueError(f"len(state)={len(state)} != num_layers={self.num_layers} "
                             f"(one KV cache per layer; refusing to mis-thread state — rule 6)")
        h = self.embed_w[ids][None].astype(mx.bfloat16)                            # [1,T,hidden]
        for blk, cache in zip(self.layers, state, strict=True):
            h = blk(h, cache=cache, use_fast=use_fast, sparse=sparse)
        mx.eval(h)
        hh = mx.fast.rms_norm(h[:, -1:], self.norm_w.astype(h.dtype), self.cfg.norm_eps)
        return hh @ self.lm_head_w.T.astype(hh.dtype)                              # [1,1,vocab]

    # --- one-step batched decode ----------------------------------------------
    def step_batch(self, stream_token_ids: Sequence[int],
                   stream_caches: Sequence[Sequence[KVCache]],
                   offsets: Sequence[int], *, use_fast: bool = True) -> list[mx.array]:
        """One decode step for ``B = len(stream_token_ids)`` concurrent streams.

        ``stream_token_ids[b]`` is the int token id feeding stream ``b``; ``stream_caches[b]`` is that
        stream's per-layer cache list (mutated in place); ``offsets[b]`` is the absolute position of
        the new token in stream ``b`` (== ``stream_caches[b][0].offset`` before the step). Returns the
        per-stream logits as a list of ``[1,1,vocab]`` arrays. Per-stream output is greedy-token-
        equivalent (top-1 exact) to the single-stream runtime at the same offset against the same
        cache — gated in ``parity/minimax_m3_batched_test.py``."""
        if len(stream_token_ids) > self.max_batch:
            raise ValueError(f"batch {len(stream_token_ids)} exceeds max_batch {self.max_batch}")
        return batched_decode_step(self.layers, self.embed_w, self.norm_w, self.lm_head_w,
                                   self.cfg, stream_token_ids, stream_caches, offsets,
                                   use_fast=use_fast)
