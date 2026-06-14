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
    1. **attention step** — TWO output-equivalent paths (rule-4 flag ``loopkill``):

       * *per-stream loop* (``loopkill=False`` — the M3-2 reference): iterate streams, run the same
         ``MiniMaxM3Block`` attention half (``x + attn(in_norm(x))``) through each stream's per-layer
         cache exactly as the single-stream runtime does. Bounded IO loop over the small batch (rule
         3); each call is M=1 so the packed q/k/v/o ``mx.quantized_matmul`` is bit-exact vs the
         single-stream decode.
       * *GQA loop-kill* (``loopkill=True`` — M3-3, the graduated serving default): ONE batched
         attention across all ``B`` streams via
         :meth:`quanta.minimax.model_m3.MiniMaxM3Attention.decode_step_batched` — batched q/k/v/o
         projections (each weight read ``⌈B/chunk⌉×`` instead of ``B×``: the mixer-read bandwidth win,
         mirroring the MoE expert-read amortization), a per-stream RoPE *kernel* loop (only the
         absolute offset differs — M3 has no YaRN), and the shared fused padded SDPA across all ``B``
         (:func:`quanta.modeling.batched_attention.batched_decode_attention_kv`). The projections are
         applied in ``<=chunk`` row-slices so each ``mx.quantized_matmul`` stays in the M=1-equivalent
         gemv regime (#153 option B) ⇒ bit-exact projections; only the fused padded SDPA softmax
         reorders ⇒ greedy-token-equivalent (top-1 exact). Requires ``packed`` (a dense-bf16 projection
         reorders across batch-M — enforced, rule 4/6).

       Each stream's cache mutates in place.
    2. **stack** the ``B`` post-attention residuals ``[1,1,hidden]`` → ``[B,1,hidden]``.
    3. **batched MoE / dense-FFN sub-block** — feed ``[B,1,hidden]`` through the **same** block FFN.
       For MoE layers (3–59) the existing ``MiniMaxM3MoE`` routes ``[B,1,h] → [N=B,h]``: top-4
       routing runs on ``B`` rows in one shot; ``gather_qmm`` over the packed int6 stacks reads each
       touched expert's codes ONCE for every token that routes to it; the shared expert runs once
       over ``[B,h]``. This is the bandwidth win. Dense layers (0–2) run the FFN once over ``[B,h]``.
    4. **split** ``[B,1,hidden]`` back into per-stream ``[1,1,hidden]`` residuals.
* Every token completes ALL layers (advancing its per-stream KV) before the next begins, so
  per-stream output is causally identical to a single-stream decode at the same offset — gated
  model-free in ``parity/minimax_m3_batched_test.py`` (per-stream loop, bit-exact dispatch) and
  ``parity/minimax_m3_loopkill_test.py`` (loop-kill == per-stream loop, greedy-token-equivalent).

**M3-4 — paged-KV + prefix caching (int8 KV).** M3 is the clean dense-GQA paged case (all 60 layers
attention, NO recurrent state — like InternLM2.5), so this model also exposes the #152 paged contract
(``paged_kv_spec`` / ``make_paged_state`` / ``prefill_paged`` / ``has_recurrent_state=False``) the shared
:class:`quanta.shim.omlx._BaseBatchedSession` drives: per-stream state becomes one
:class:`quanta.paged.PagedKVCacheView` per layer over a shared :class:`quanta.paged.PagedKVCacheManager`,
so concurrent / multi-turn requests that share a prompt prefix store its KV **once** (ref-counted blocks).
The KV is **int8 g64** (``kv_quantized=True``; GQA 4 kv heads ⇒ the KV is already cheap, int8 halves it
again) — quant groups on ``head_dim`` are orthogonal to the seq-axis blocks, so a paged gather is
bit-identical to the discrete :class:`KVCache` (gated). The M3-3 loop-kill's KV step takes the paged
loop-kill (``paged_batched`` → ONE ``write_batched`` + ONE ``gather_batched``) when serving paged views.

Out of scope here (later M3 sub-milestones): chunked prefill (the per-token :meth:`prefill_paged` is the
admit path for chat-length suffixes), the trained block-sparse long-context lever, and the oMLX shim
(the ``_MiniMaxM3BatchedSession`` engine route that consumes this paged contract). The win is in DECODE,
where multi-stream serving spends ~all of its time.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx

from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.model_m3 import KVCache, MiniMaxM3Block
from quanta.minimax.runtime_m3 import MiniMaxM3ResidentModel
from quanta.paged import MINIMAX_M3_PAGED_KV_BATCHED_DEFAULT, PagedKVCacheView

# --- M3-3 GQA loop-kill default (the graduated serving path) ---------------------------------------
# When ON, the serving decode step (:func:`batched_decode_step`) replaces the per-stream attention loop
# with ONE batched attention across all B streams (M3 is all-GQA, so the whole mixer): batched q/k/v/o
# projections (each mixer weight read ONCE for all B — the bandwidth win, mirroring the MoE expert-read
# amortization) + per-stream RoPE kernel loop + the shared fused padded SDPA. **GRADUATED ON (M3-3)**
# after the real-model B=8 re-gate (``parity/minimax_m3_loopkill_real.py`` on the 397B int6-g64 bake):
# with the option-B packed+chunked projections it is greedy-token-equivalent to the per-stream loop at
# every B AND a decode win on top of M3-2's batched MoE. The loop-kill REQUIRES packed (a dense-bf16
# projection reorders across batch-M), enforced by ``_check_loopkill_requires_packed``. Gated vs the
# per-stream loop in ``parity/minimax_m3_loopkill_test.py``.
MINIMAX_M3_BATCHED_LOOPKILL_DEFAULT = True

# --- M3-3 option-B loop-kill chunk size (the batch-M bit-exact regime) -----------------------------
# The loop-kill batches the mixer projections across all B streams. Under the packed runtime those are
# ``mx.quantized_matmul``, which is batch-M BIT-EXACT only for M<=~10 (a per-row gemv kernel); at M>=12
# it switches to a tiled GEMM that REORDERS the K-reduction (bf16). So the batched projections are
# applied in row-chunks of <= this, keeping every matmul in the bit-exact regime — equalling the
# per-stream M=1 loop BIT-FOR-BIT at any B (only the fused padded SDPA spans all B). Proven model-free
# in ``parity/minimax_m3_loopkill_test.py`` §M0 (== the qwen35 #153 option-B finding); if a future MLX
# drops the gemv→GEMM threshold below 8, §M0 fails loudly (the signal to lower this).
MINIMAX_M3_LOOPKILL_CHUNK = 8


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
    loopkill: bool = False,
    chunk: int = MINIMAX_M3_LOOPKILL_CHUNK,
    paged_batched: bool = False,
) -> list[mx.array]:
    """One batched decode step over ``B = len(stream_token_ids)`` streams. Returns per-stream logits
    ``[1,1,vocab]`` — one per stream, in input order, so the caller can sample each independently.

    Pure function — no model state outside the per-stream caches (which mutate in place) and the
    passed weights / layers. The FFN sub-block always runs ONCE on the stacked ``[B,1,hidden]`` (the
    routed-expert read amortizes across all ``B``). The attention half has two output-equivalent paths:

    * ``loopkill=False`` (the M3-2 reference, the free-function default — rule 4): a per-stream
      attention loop (each call M=1 ⇒ bit-exact vs the single-stream decode).
    * ``loopkill=True`` (M3-3): ONE batched attention across all ``B`` streams (batched chunked
      projections + per-stream RoPE + fused padded SDPA), greedy-token-equivalent to the loop. The
      caller MUST hold the mixer packed (enforced upstream — a dense-bf16 projection reorders across
      batch-M); ``chunk`` keeps each ``mx.quantized_matmul`` in the bit-exact gemv regime.

    ``paged_batched`` (M3-4): when True AND ``stream_caches`` holds :class:`quanta.paged.PagedKVCacheView`
    lists, the loop-kill's KV step does ONE ``write_batched`` + ONE ``gather_batched`` over the shared
    manager instead of the per-stream ``.update()`` loop — bit-identical (both end in the same fused SDPA;
    no-op for discrete caches or the per-stream-loop path). Threaded into :meth:`MiniMaxM3Attention.
    decode_step_batched`.

    Per-stream output is greedy-token-equivalent (top-1 exact) to feeding the same token through
    :class:`quanta.minimax.runtime_m3.MiniMaxM3ResidentModel` at the same offset against the same
    cache. Drives the resident runtime AND the model-free gates.
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
        # 1) attention step -> stacked [B,1,hidden] post-attention residuals.
        if loopkill:
            # M3-3 GQA loop-kill: ONE batched attention across all B streams (chunked projections +
            # per-stream RoPE + fused padded SDPA), reading each mixer weight ⌈B/chunk⌉× for all B (the
            # bandwidth win). Batched input-norm (RMSNorm is row-wise → per-row identical to the
            # per-stream norm), then the batched mixer, then the batched residual — greedy-token-
            # equivalent to the per-stream loop (rule-4 flag; gated in minimax_m3_loopkill_test.py).
            x_stacked = mx.concatenate(hs, axis=0) if b > 1 else hs[0]              # [B,1,hidden]
            h_norm = blk.input_layernorm(x_stacked)
            kv_for_layer = [stream_caches[s][layer_i] for s in range(b)]
            y = blk.self_attn.decode_step_batched(h_norm, kv_for_layer=kv_for_layer,
                                                  offsets=list(offsets), chunk=chunk,
                                                  paged_batched=paged_batched)  # [B,1,hidden]
            stacked = x_stacked + y                                                 # [B,1,hidden]
        else:
            # per-stream attention step (proven M=1 path; bounded IO loop over the small batch, rule 3).
            after_attn = [_attn_step(blk, stream_caches[s][layer_i], hs[s], use_fast=use_fast)
                          for s in range(b)]
            stacked = mx.concatenate(after_attn, axis=0) if b > 1 else after_attn[0]  # [B,1,hidden]
        # 2) ONE batched FFN over all B tokens — MoE gather_qmm reads each touched routed-expert tile
        # once for all B rows that route to it (the bandwidth win); the shared expert + dense FFN run
        # once over [B,h]. The module's [B,S,h] -> [N=B,h] reshape is B-aware.
        h_post = blk.post_attention_layernorm(stacked)
        y = blk.mlp(h_post, sparse=True) if blk.is_moe else blk.mlp(h_post)         # [B,1,hidden]
        stacked = stacked + y
        # 3) split back to per-stream views for the next layer
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
      each stream; returns per-stream logits ``[1,1,vocab]``. Accepts either per-stream discrete
      :class:`KVCache` lists (unpaged) or :class:`quanta.paged.PagedKVCacheView` lists (paged).
    * ``prefill(prompt_ids, state) -> mx.array`` — single-stream prefill into one stream's cache.
    * ``make_caches()`` / ``make_batch_caches(B)`` — per-stream KV cache factories (the configured mode).
    * **#152 paged contract** (M3-4; consumed by :class:`quanta.shim.omlx._BaseBatchedSession`):
      ``has_recurrent_state=False``, ``paged_kv_spec``, ``make_paged_state(mgr, seq)``,
      ``prefill_paged(suffix_ids, state, *, offset, recurrent_in, block_size)``.
    * ``.cfg``, ``.num_layers``, ``.embed_w``, ``.norm_w``, ``.lm_head_w``, ``.layers`` — same handles
      as :class:`MiniMaxM3ResidentModel`.
    """

    # #152 paged contract: M3 is PURE dense GQA attention — every layer's k/v pair is paged and there
    # is NO recurrent/derived state to snapshot at block boundaries (unlike Nemotron's Mamba / DSV4's
    # compressor). So has_recurrent_state is False and prefill_paged returns no boundary payloads; the
    # base session's _admit_paged short-circuits the recurrent branch (recurrent_in is always None).
    has_recurrent_state = False

    def __init__(self, art_dir: str | Path, *, max_batch: int = 32,
                 n_layers: int | None = None, packed: bool = True,
                 packed_experts: bool = True, loopkill: bool | None = None,
                 kv_quantized: bool = True, kv_group_size: int = 64, kv_bits: int = 8) -> None:
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
        # KV storage mode for the per-stream / paged caches — int8 g64 the M3-4 serving default (GQA 4 kv
        # heads ⇒ cheap KV; int8 halves it). Threaded to the paged spec so a PagedKVCacheManager is built
        # bit-identical (rule 6, never hardcoded).
        self._kv_quantized = bool(kv_quantized)
        self._kv_group_size = int(kv_group_size)
        self._kv_bits = int(kv_bits)
        self._loopkill = (bool(MINIMAX_M3_BATCHED_LOOPKILL_DEFAULT) if loopkill is None
                          else bool(loopkill))           # M3-3 GQA loop-kill (graduated ON; rule-4 flag)
        self._paged_kv_batched = bool(MINIMAX_M3_PAGED_KV_BATCHED_DEFAULT)  # M3-4 paged KV loop-kill
        self._check_loopkill_requires_packed()

    @classmethod
    def from_inner(cls, layers: list[MiniMaxM3Block], embed_w: mx.array, norm_w: mx.array,
                   lm_head_w: mx.array, cfg: MiniMaxM3Config, *,
                   max_batch: int = 32, loopkill: bool | None = None,
                   kv_quantized: bool = False, kv_group_size: int = 64,
                   kv_bits: int = 8) -> MiniMaxM3BatchedResidentModel:
        """Construct from pre-built layers / final-form weights (bypasses artifact load) so the
        model-free parity gate can drive ``step_batch`` / ``prefill`` against a tiny synthetic model
        without a checkpoint — same step machinery, model-free. ``norm_w`` is the already-``(1+w)``-
        folded final norm. ``packed`` / ``packed_experts`` are detected from the passed layers.

        ``loopkill`` overrides the per-instance loop-kill flag: ``None`` (default) inherits the
        graduated :data:`MINIMAX_M3_BATCHED_LOOPKILL_DEFAULT` (so the serving ``from_inner`` gets the
        loop-kill, paired with the layers' ``packed`` state); the M3-2 Design-A gate passes
        ``loopkill=False`` to pin the bit-exact per-stream path it gates.

        KV flags default **bf16** (unlike ``__init__``'s int8 serving default) so a tiny synthetic
        config (``head_dim < 32`` cannot meet ``mx.quantize``'s min group_size) stays on the bf16 path;
        the M3-4 paged gate passes ``kv_quantized=True`` with a ``head_dim>=32`` config to exercise int8."""
        self = cls.__new__(cls)
        self._inner = None
        self.max_batch = int(max_batch)
        self.cfg = cfg
        self.num_layers = len(layers)
        self.embed_w = embed_w
        self.norm_w = norm_w
        self.lm_head_w = lm_head_w
        self.layers = list(layers)
        self._kv_quantized = bool(kv_quantized)
        self._kv_group_size = int(kv_group_size)
        self._kv_bits = int(kv_bits)
        moe = [b for b in layers if b.is_moe]
        self.packed_experts = bool(moe) and isinstance(moe[0].mlp.experts_gate_up, dict)
        import mlx.nn as nn
        self.packed = bool(layers) and isinstance(layers[0].self_attn.q_proj, nn.QuantizedLinear)
        self._loopkill = (bool(MINIMAX_M3_BATCHED_LOOPKILL_DEFAULT) if loopkill is None
                          else bool(loopkill))
        self._paged_kv_batched = bool(MINIMAX_M3_PAGED_KV_BATCHED_DEFAULT)  # M3-4 paged KV loop-kill
        self._check_loopkill_requires_packed()
        return self

    # --- loop-kill ⇒ packed enforcement (rule 4/6) ----------------------------
    def _check_loopkill_requires_packed(self) -> None:
        """Fail loud if the M3-3 GQA loop-kill is enabled on a non-packed (dense-bf16) runtime.

        The loop-kill batches the mixer projections across all ``B`` streams; a dense-bf16 GEMM
        reorders its accumulation across batch-M (the ``feedback_batched_rope_bf16`` finding). Only the
        packed runtime (``mx.quantized_matmul``, chunked ``<=`` :data:`MINIMAX_M3_LOOPKILL_CHUNK`) is
        batch-M bit-exact, so the loop-kill REQUIRES packed. Checked at construction AND on every
        :meth:`step_batch` (so a runtime ``self._loopkill`` toggle cannot bypass it — rule 6: refuse to
        silently emit batch-M-reordered logits)."""
        if self._loopkill and not self.packed:
            raise ValueError(
                "M3-3 GQA loop-kill requires the packed runtime (packed=True): dense-bf16 mixer "
                "projections reorder their accumulation across batch-M. Construct the batched runtime "
                "with packed=True, or disable the loop-kill (MINIMAX_M3_BATCHED_LOOPKILL_DEFAULT / "
                "loopkill=False).")

    # --- per-stream cache factory ---------------------------------------------
    def make_caches(self) -> list[KVCache]:
        """One stream's fresh per-layer GQA KV cache list in the configured KV mode (``_kv_quantized`` /
        ``_kv_group_size`` / ``_kv_bits`` — int8 g64 the serving default, bf16 for the model-free gates)."""
        return [KVCache(quantized=self._kv_quantized, group_size=self._kv_group_size,
                        bits=self._kv_bits) for _ in range(self.num_layers)]

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
        cache — gated in ``parity/minimax_m3_batched_test.py`` (per-stream loop) and
        ``parity/minimax_m3_loopkill_test.py`` (loop-kill). The attention path is the GQA loop-kill
        when ``self._loopkill`` (the graduated serving default), else the per-stream loop.

        ``stream_caches`` may hold per-stream discrete :class:`KVCache` lists (unpaged) or per-stream
        :class:`quanta.paged.PagedKVCacheView` lists (paged, the omlx session path). For paged views the
        loop-kill's KV step takes the paged loop-kill (``write_batched`` + ``gather_batched``) when
        ``self._paged_kv_batched`` (M3-4); the per-stream loop path reads paged views directly via
        ``view.update`` (no flag needed)."""
        if len(stream_token_ids) > self.max_batch:
            raise ValueError(f"batch {len(stream_token_ids)} exceeds max_batch {self.max_batch}")
        self._check_loopkill_requires_packed()   # rule 6: a runtime toggle cannot bypass loopkill⇒packed
        # paged KV loop-kill engages only when (a) it's enabled, (b) the loop-kill attention is on (the
        # per-stream loop reads paged views directly), and (c) the caches are actually paged views.
        paged = (self._paged_kv_batched and self._loopkill and bool(stream_caches)
                 and isinstance(stream_caches[0][0], PagedKVCacheView))
        return batched_decode_step(self.layers, self.embed_w, self.norm_w, self.lm_head_w,
                                   self.cfg, stream_token_ids, stream_caches, offsets,
                                   use_fast=use_fast, loopkill=self._loopkill,
                                   chunk=MINIMAX_M3_LOOPKILL_CHUNK, paged_batched=paged)

    # --- #152 paged contract (M3-4) -------------------------------------------
    @property
    def paged_kv_spec(self) -> dict:
        """Shape/codec the shared :class:`quanta.paged.PagedKVCacheManager` must use to be bit-exact with
        the discrete :class:`KVCache` — threaded from this runtime's own KV flags, never hardcoded (rule
        6: a wrong width silently mis-decodes). All 60 layers are paged (uniform dense GQA); k/v pair (no
        ``single_stream``); no recurrent cache (``has_recurrent_state=False``)."""
        return {"n_layers": self.num_layers,
                "group_size": self._kv_group_size,
                "bits": self._kv_bits,
                "quantized": self._kv_quantized}

    def make_paged_state(self, manager, seq) -> list:
        """Per-stream paged state = one :class:`quanta.paged.PagedKVCacheView` per layer over the shared
        ``manager`` (so the prefix blocks dedup across requests). The plain list is exactly what
        :meth:`step_batch` / :meth:`prefill_paged` thread per layer (each view ducks the discrete
        :class:`KVCache` ``offset`` / ``update`` surface, so the unchanged forward writes/reads KV
        through it)."""
        return [manager.view(seq, i) for i in range(self.num_layers)]

    def prefill_paged(self, suffix_ids, state: list, *, offset: int,
                      recurrent_in=None, block_size: int = 0) -> tuple[mx.array, list]:
        """Prefill ONLY the uncached suffix into ``state`` (per-layer paged views over the resident prefix
        blocks), with the suffix tokens at absolute positions ``[offset, offset + T)`` for RoPE — the
        :meth:`quanta.shim.omlx._BaseBatchedSession._admit_paged` hook.

        M3 is pure dense GQA ⇒ NO recurrent state to restore (``recurrent_in`` MUST be ``None`` — rule 6)
        and NO boundary snapshot to produce (returns ``[]``); the engine's paged admit short-circuits the
        recurrent branch (``has_recurrent_state=False``). Each layer's attention writes the suffix k/v
        through its view and reads back the full (prefix + suffix) stream for the bottom-right-causal SDPA,
        which is **bit-identical** to a discrete continue-from-prefix prefill of the same split (the paged
        gather == discrete :class:`KVCache` foundation — gated in ``parity/minimax_m3_paged_test.py``).
        ``block_size`` is unused (no boundary work)."""
        del block_size  # no recurrent/derived boundary state for a dense model
        if recurrent_in is not None:
            raise ValueError("MiniMaxM3 prefill_paged: recurrent_in must be None (dense GQA has no "
                             "recurrent state — rule 6)")
        if len(state) != self.num_layers:
            raise ValueError(f"len(state)={len(state)} != num_layers={self.num_layers} "
                             f"(one paged view per layer; refusing to mis-thread state — rule 6)")
        if int(state[0].offset) != int(offset):
            raise ValueError(f"prefill_paged: view offset {int(state[0].offset)} != declared offset "
                             f"{int(offset)} (the manager must have advanced the prefix first — rule 6)")
        logits = self.prefill(suffix_ids, state)   # the unchanged cached forward over the paged views
        return logits, []
