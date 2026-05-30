"""Batched (B>1) RAM-resident Qwen3.5 runtime — agentic-loop serving over shared weights.

Wraps :class:`quanta.qwen35.runtime.Qwen35ResidentModel` to amortize the read of the routed-expert
stacks across many concurrent decode streams without touching the proven single-stream code (the
parity-gated reference lives there). Single-stream Qwen3.5 decode is memory-bandwidth bound on the
routed-experts read (per token: 10 expert slots × ~3 projections + shared expert × ~3 projections,
all gather_qmm-fed); batching ``B`` streams reads the **same** expert/shared weights once and
dispatches all ``B`` tokens through one MoE call (``mx.gather_qmm`` is B-aware via the existing
``[N=B,dim] → [N,dim]`` reshape in :func:`quanta.qwen35.moe.qwen35_moe`). Target: B=32 → ~10×
aggregate throughput vs B=1.

**Design A** — per-stream caches, stack-for-MoE (the parity-trivial path):

* Each stream keeps its OWN :class:`quanta.qwen35.decode.Qwen35Cache` (per-layer GDN recurrent state
  OR per-layer GQA KV cache, by ``cfg.is_linear_attention(i)``). Per-stream state is unavoidable for
  both mixers — GDN's ``[Hv,Dk,Dv]`` recurrent state is non-mergeable across streams; GQA's KV cache
  is non-mergeable across streams (each stream's k/v rows are its own positions).
* Per-layer decode step (one token per stream, ``B`` streams):
    1. **per-stream mixer step** — iterate streams, call the same ``Qwen35Block`` mixer through
       each stream's cache exactly as :class:`Qwen35ResidentModel._decode_block` does (this loop is
       a bounded IO loop over the small batch — not a hot per-token loop, rule 3 — and the GDN /
       GQA work is fully vectorized inside each call). Each stream's cache mutates in place.
    2. **stack** the ``B`` post-mixer residuals ``[1,1,hidden]`` → ``[B,1,hidden]``.
    3. **batched MoE sub-block** — feed ``[B,1,hidden]`` through the **same** ``Qwen35MoEModule``
       call. The existing :func:`quanta.qwen35.moe.qwen35_moe` already routes ``[B,S,h] → [N,h]``
       with ``N=B*S=B``: top-10 routing runs on ``B`` rows in one shot; ``gather_qmm`` over
       ``[E,2*inter,h] / [E,h,inter]`` reads each touched expert's weights ONCE for all the tokens
       that route to it; the shared expert runs once over ``[B,h]``. This is the bandwidth win.
    4. **split** ``[B,1,hidden]`` back into per-stream ``[1,1,hidden]`` residuals.
* Every token completes ALL layers (advancing its per-stream state) before the next begins, so
  per-stream output is causally identical to a single-stream decode at the same offset — gated
  model-free in ``parity/qwen35_batched_test.py``.

Out of scope: ragged per-stream offsets in **prefill** (full attention's RoPE / SDPA mask need a
common offset). Prefill is therefore single-stream (``prefill(prompt_ids, state)`` runs the
single-stream :class:`Qwen35ResidentModel` forward once per stream). The win is in DECODE — where
the agentic loop spends ~all of its time.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx

from quanta.qwen35.attention import KVCache, Qwen35Attention
from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.decode import Qwen35Cache, _GDNLayerState
from quanta.qwen35.gated_deltanet import GatedDeltaNet

# --- #153 hybrid loop-kill default (Qwen3.5-scoped; UNPAGED — not the shared paged flag) -----------
# When ON, the serving decode step (:func:`batched_decode_step`) replaces the per-stream mixer loop with
# ONE batched mixer across all B streams, for BOTH halves of the 3:1 hybrid:
#   * full-attention (GQA) layers — batched q/k/v/o projections + per-stream RoPE kernel loop + the
#     shared ``batched_decode_attention_kv`` fused padded SDPA (M1);
#   * linear-attention (Gated DeltaNet) layers — gather the B streams' (conv,recurrent) state, ONE
#     batched recurrence, scatter+commit back per-stream via :func:`_gdn_step_batched` (M2, the bigger
#     lever: 45 of 60 layers are GDN).
# This is the #18-style loop-kill brought to the Qwen prod path, but UNPAGED (Qwen serving forces
# ``paged_kv=False``) so it cannot reuse the shared ``PAGED_KV_BATCHED_DEFAULT``. Defaults OFF (rule 4:
# naive per-stream path until parity-proven); M1+M2 land it behind this flag, M3 flips it ON after the
# real-model B-sweep. Gated vs the per-stream loop in ``parity/qwen35_batched_loopkill_test.py``: GDN is
# bit-exact at B=1 / fp-tolerance at B>1 (the projection-GEMM batch-M reorder), GQA is greedy-exact (it
# also reorders the padded SDPA). ``batch_step`` (tree-spec verify) is untouched (per-replica rollback).
QWEN35_BATCHED_LOOPKILL_DEFAULT = False


def _gdn_step_through_cache(blk, lc: _GDNLayerState, x_t: mx.array) -> mx.array:
    """One linear-layer decode step for ONE stream through its recurrent layer-state.

    Mirrors :meth:`quanta.qwen35.runtime.Qwen35ResidentModel._decode_block` for the linear branch
    (rule-4: same forward as the single-stream parity reference; the per-stream cache mutates in
    place). Returns the post-mixer residual ``[1,1,hidden]``.
    """
    m = blk.mixer
    assert isinstance(m, GatedDeltaNet)
    state = lc.recurrent_state
    conv = lc.conv_state
    if conv is None:
        # fresh stream: seed zero recurrent + zero conv window so the O(1) recurrence engages
        # from absolute offset 0 (== single-stream runtime, gated in qwen35_decode_attn_test).
        state = mx.zeros((1, m.hv, m.dk, m.dv), dtype=mx.float32)
        conv = mx.zeros((1, m.k - 1, m.conv_dim), dtype=x_t.dtype)
    # the block's mixer call returns (out, recurrent_state, conv_state) — thread through.
    h = blk.input_layernorm(x_t)
    y, rec_out, conv_out = m(h, state=state, conv_state=conv)
    lc.commit(conv_out, rec_out)
    return x_t + y


def _gqa_step_through_cache(blk, kv: KVCache, x_t: mx.array, seq_hint: int) -> mx.array:
    """One full-layer decode step for ONE stream through its KV cache (YaRN factor pinned)."""
    m = blk.mixer
    assert isinstance(m, Qwen35Attention)
    h = blk.input_layernorm(x_t)
    y = m(h, cache=kv, use_fast=True, seq_hint=seq_hint)
    return x_t + y


def _gdn_step_batched(m: GatedDeltaNet, lcs: Sequence[_GDNLayerState], h_norm: mx.array) -> mx.array:
    """Batched Gated-DeltaNet decode step across ``B`` streams — the #153 M2 loop-kill for the
    serving decode path (the linear-attention half of the Qwen3.5 hybrid). Returns the mixer output
    ``y`` ``[B,1,hidden]`` (the caller adds the residual ``x_stacked + y``).

    ``h_norm`` ``[B,1,hidden]`` are the ``B`` streams' post-input-norm residuals (stacked);
    ``lcs[s]`` is stream ``s``'s :class:`_GDNLayerState` for THIS layer (mutated in place). Gathers
    each stream's recurrent ``(conv_state, recurrent_state)`` into a ``[B,...]`` batch (seeding zeros
    where a stream is fresh — bit-identical to :func:`_gdn_step_through_cache`'s per-stream seed),
    runs ONE :meth:`GatedDeltaNet.__call__` decode recurrence over all ``B`` (so ``in_proj_qkv`` /
    ``out_proj`` / the conv weights are read ONCE for all ``B`` — the memory-bandwidth win, mirroring
    the MoE expert-read amortization), then scatters the new ``(conv_out, rec_out)`` back into each
    stream's state via :meth:`_GDNLayerState.commit` (advancing its offset + snapshot ring exactly as
    the per-stream path does).

    Per-row equivalent to the per-stream loop: the depthwise-conv window, the gated-delta recurrence,
    and both gated RMSNorms have NO cross-row op, and GDN carries no positional term at all (position
    lives in the recurrent state), so ragged per-stream offsets are irrelevant to its math. The ONLY
    batch-size sensitivity is the projection GEMM's accumulation order — so row ``s`` is **bit-exact** to
    the standalone ``B=1`` call when ``B==1`` (the ``b==1`` path feeds identical ``[1,1,hidden]`` shapes,
    a strict passthrough) and matches at **fp tolerance** for ``B>1`` (the ~1e-7 fp32 ``[B,...]@W``
    reorder, with NO padded-SDPA softmax reorder — tighter than the GQA half, greedy-token-stable).
    Gated in ``parity/qwen35_batched_loopkill_test.py`` (§GDN).

    Memory: the committed ``conv_out[s:s+1]`` / ``rec_out[s:s+1]`` are MLX slices that share the batched
    output buffer (MLX slices are views — confirmed: holding one row retains the whole ``[B,...]``
    parent). This does NOT blow up the snapshot ring: its depth is bounded (``snapshot_depth``) and
    continuous-batching streams commit in lockstep, so at any step at most ``snapshot_depth`` recent
    batched buffers are referenced (== the per-stream path's total snapshot footprint), and a buffer
    ages out of every active stream's ring within ``snapshot_depth`` steps. An active stream is never
    paused mid-decode, so rings stay time-aligned and no released stream's rows pin a stale buffer.
    """
    b = len(lcs)
    conv_rows: list[mx.array] = []
    rec_rows: list[mx.array] = []
    for lc in lcs:
        if lc.conv_state is None:
            # fresh stream: seed zero conv window + zero recurrent state so the O(1) recurrence
            # engages from absolute offset 0 (identical to _gdn_step_through_cache's per-stream seed).
            conv_rows.append(mx.zeros((1, m.k - 1, m.conv_dim), dtype=h_norm.dtype))
            rec_rows.append(mx.zeros((1, m.hv, m.dk, m.dv), dtype=mx.float32))
        else:
            conv_rows.append(lc.conv_state)
            rec_rows.append(lc.recurrent_state)
    conv = mx.concatenate(conv_rows, axis=0) if b > 1 else conv_rows[0]    # [B,K-1,conv_dim]
    rec = mx.concatenate(rec_rows, axis=0) if b > 1 else rec_rows[0]       # [B,Hv,Dk,Dv]
    y, rec_out, conv_out = m(h_norm, state=rec, conv_state=conv)           # ONE batched recurrence
    for s, lc in enumerate(lcs):
        lc.commit(conv_out[s:s + 1], rec_out[s:s + 1])                     # per-stream offset + snapshot
    return y


def batched_decode_step(
    layers: Sequence,
    embed_w: mx.array,
    norm_w: mx.array,
    lm_head_w: mx.array,
    cfg: Qwen35Config,
    stream_token_ids: Sequence[int],
    stream_caches: Sequence[Qwen35Cache],
    offsets: Sequence[int],
    *,
    loopkill: bool = QWEN35_BATCHED_LOOPKILL_DEFAULT,
) -> list[mx.array]:
    """One batched decode step over ``B = len(stream_token_ids)`` streams. Returns per-stream
    logits ``[1,1,vocab]``.

    Pure function — no model state outside the per-stream caches (which mutate in place) and the
    passed weights / layers. The Design-A per-stream mixer step iterates streams; the MoE
    sub-block runs ONCE on the stacked ``[B,1,hidden]``. Per-stream output is bit-identical to
    feeding the same token through :class:`quanta.qwen35.runtime.Qwen35ResidentModel` at the same
    offset against the same cache.

    ``loopkill`` (#153 — :data:`QWEN35_BATCHED_LOOPKILL_DEFAULT`): when True EVERY layer runs ONE
    batched mixer across all ``B`` streams instead of the per-stream loop — full-attention (GQA) layers
    via :meth:`quanta.qwen35.attention.Qwen35Attention.decode_step_batched` (batched projections +
    per-stream RoPE + shared fused SDPA, M1, greedy-token-equivalent) and linear-attention (Gated
    DeltaNet) layers via :func:`_gdn_step_batched` (gather state + ONE recurrence + scatter, M2; B=1
    bit-exact, B>1 fp-tolerance). Gated in ``parity/qwen35_batched_loopkill_test.py``. Off ⇒ the proven
    per-stream path (rule 4).

    Drives both the resident runtime (post-artifact-load) AND the model-free parity test (via the
    tiny :class:`quanta.qwen35.model.Qwen35Model` set up with random weights), without duplicating
    the step machinery.
    """
    b = len(stream_token_ids)
    if b < 1:
        raise ValueError("step needs >= 1 stream")
    if len(stream_caches) != b or len(offsets) != b:
        raise ValueError(f"len(stream_token_ids)={b}, len(stream_caches)={len(stream_caches)}, "
                         f"len(offsets)={len(offsets)} — must match")
    # per-stream offset sanity: each cache must already be at its declared offset (fail loud — rule 6).
    for i, (cache, off) in enumerate(zip(stream_caches, offsets, strict=True)):
        if cache.offset != off:
            raise ValueError(f"stream {i}: cache.offset={cache.offset} != declared offset={off} "
                             f"(orchestrator desynced; refusing to silently advance)")

    # --- per-stream embed (one gather across all B token ids in ONE indexing op) -------------
    ids_b = mx.array([int(t) for t in stream_token_ids], dtype=mx.int32)            # [B]
    h_b = embed_w[ids_b][:, None].astype(mx.bfloat16)                               # [B,1,hidden]
    hs: list[mx.array] = [h_b[i:i + 1] for i in range(b)]                           # B × [1,1,hidden]

    # --- per layer: mixer (per-stream loop OR batched loop-kill: GQA + GDN), then ONE batched MoE ----
    for layer_i, blk in enumerate(layers):
        if loopkill:
            # #153 loop-kill: ONE batched mixer across all B streams (M1 GQA + M2 GDN), reading each
            # mixer weight ONCE for all B (the bandwidth win, mirroring the MoE expert-read amortization).
            # Batched input-norm (RMSNorm is row-wise → per-row identical to the per-stream norm), then
            # the type-specific batched mixer, then the batched residual — equivalent to the per-stream
            # loop (rule-4 flag; gated in parity/qwen35_batched_loopkill_test.py). GDN is bit-exact at
            # B=1 / fp-tolerance at B>1 (projection-GEMM batch-M reorder; no cross-row op); GQA is
            # greedy-exact (it also reorders the padded SDPA).
            x_stacked = mx.concatenate(hs, axis=0) if b > 1 else hs[0]                # [B,1,hidden]
            h_norm = blk.input_layernorm(x_stacked)
            if blk.is_linear:
                # M2: batched Gated-DeltaNet recurrence — gather the B streams' (conv,recurrent) state,
                # ONE recurrence, scatter+commit back per-stream (offset + snapshot ring preserved).
                lcs = [stream_caches[s][layer_i] for s in range(b)]
                y = _gdn_step_batched(blk.mixer, lcs, h_norm)                         # [B,1,hidden]
            else:
                # M1: batched GQA — batched q/k/v/o proj + per-stream RoPE + shared fused padded SDPA.
                kv_for_layer = [stream_caches[s][layer_i] for s in range(b)]
                seq_hints = [stream_caches[s].yarn_seq(offsets[s] + 1, cfg) for s in range(b)]
                y = blk.mixer.decode_step_batched(h_norm, kv_for_layer=kv_for_layer,
                                                  offsets=list(offsets), seq_hints=seq_hints)  # [B,1,hidden]
            stacked = x_stacked + y                                                   # [B,1,hidden]
        else:
            # 1) per-stream mixer step (proven path; rule-4 default). Bounded IO loop over the small
            # batch (rule 3 allows IO loops); the GDN / GQA work inside each call is fully vectorized
            # over heads/dims. Hit only when the loop-kill flag is off (then BOTH mixer types loop).
            after_mixer: list[mx.array] = []
            for s in range(b):
                lc = stream_caches[s][layer_i]
                # YaRN factor resolved per stream so it matches the single-stream runtime at the same
                # absolute position; >native requires a pinned factor (rule 6 — see Qwen35Cache.yarn_seq).
                seq_hint_s = stream_caches[s].yarn_seq(offsets[s] + 1, cfg)
                if blk.is_linear:
                    out_s = _gdn_step_through_cache(blk, lc, hs[s])
                else:
                    out_s = _gqa_step_through_cache(blk, lc, hs[s], seq_hint_s)
                after_mixer.append(out_s)

            # 2) stack the B post-mixer residuals -> [B,1,hidden] for the MoE call
            stacked = mx.concatenate(after_mixer, axis=0) if b > 1 else after_mixer[0]   # [B,1,hidden]

        # 3) ONE batched MoE sub-block over all B tokens — gather_qmm reads each touched
        # routed-expert weight tile once for all B rows that route to it; shared-expert once
        # over [B,hidden]. The existing module's [B,S,h] -> [N,h] reshape (N=B) is B-aware.
        h_post_norm = blk.post_attention_layernorm(stacked)
        moe_out = blk.mlp(h_post_norm, sparse=True)                                  # [B,1,hidden]
        stacked = stacked + moe_out                                                  # [B,1,hidden]

        # 4) split back to per-stream views for the next layer
        hs = [stacked[i:i + 1] for i in range(b)]

        # eval per-layer to keep the per-layer activation graph bounded.
        mx.eval(stacked)

    # --- final norm + lm_head over the stacked [B,1,hidden] --------------------------------
    final = mx.concatenate(hs, axis=0) if b > 1 else hs[0]                           # [B,1,hidden]
    final = mx.fast.rms_norm(final, norm_w.astype(final.dtype), cfg.norm_eps)
    logits_b = final @ lm_head_w.T.astype(final.dtype)                               # [B,1,vocab]
    return [logits_b[i:i + 1] for i in range(b)]


class Qwen35BatchedResidentModel:
    """Batched (B>1) Qwen3.5 decode over shared resident weights — Design A (per-stream caches).

    Loads the artifact ONCE via :class:`quanta.qwen35.runtime.Qwen35ResidentModel` (one-layer-resident,
    rule 8) and wraps it; weights are shared across streams. Each stream keeps its own
    :class:`Qwen35Cache`. The MoE sub-block is fed ``[B,1,hidden]`` so the routed-expert /
    shared-expert weight reads amortize across all ``B`` streams via the existing
    :func:`quanta.qwen35.moe.qwen35_moe` (already B-aware on its ``[B,S,h] → [N,h]`` input).

    Surface the orchestrator drives:

    * ``step_batch(stream_token_ids, stream_caches, offsets) -> list[mx.array]`` — one decode step
      for each stream; returns per-stream logits ``[1,1,vocab]``.
    * ``prefill(prompt_ids, state) -> mx.array`` — single-stream prefill into one stream's cache.
    * ``.cfg``, ``.num_layers``, ``.embed_w``, ``.lm_head_w``, ``.norm_w`` — same handles as
      :class:`Qwen35ResidentModel`.
    """

    def __init__(self, art_dir: str | Path, *, max_batch: int = 32,
                 n_layers: int | None = None, kv_quantized: bool = True,
                 kv_group_size: int = 64) -> None:
        if max_batch < 1:
            raise ValueError(f"max_batch must be >= 1, got {max_batch}")
        # late import: the real runtime touches the artifact loader; tests that bypass artifact load
        # construct the batched model via ``from_inner`` (see below) without hitting this path.
        from quanta.qwen35.runtime import Qwen35ResidentModel
        self._inner = Qwen35ResidentModel(art_dir, n_layers=n_layers)
        self.max_batch = int(max_batch)
        self.cfg: Qwen35Config = self._inner.cfg
        self.num_layers: int = self._inner.num_layers
        self.embed_w: mx.array = self._inner.embed_w
        self.norm_w: mx.array = self._inner.norm_w
        self.lm_head_w: mx.array = self._inner.lm_head_w
        self.layers = self._inner.layers
        self._kv_quantized = bool(kv_quantized)
        self._kv_group_size = int(kv_group_size)
        self._loopkill = bool(QWEN35_BATCHED_LOOPKILL_DEFAULT)   # #153 M1 GQA loop-kill (rule-4 flag)

    @classmethod
    def from_inner(cls, layers, embed_w: mx.array, norm_w: mx.array, lm_head_w: mx.array,
                   cfg: Qwen35Config, *, max_batch: int = 32, kv_quantized: bool = False,
                   kv_group_size: int = 64) -> Qwen35BatchedResidentModel:
        """Construct from pre-built layers / weights (bypasses artifact load).

        Lets the parity test drive ``step_batch`` against a tiny :class:`quanta.qwen35.model.Qwen35Model`
        without loading a real checkpoint — same step machinery, model-free. ``kv_quantized`` defaults
        to ``False`` so a tiny config (head_dim < kv_group_size) does not hit the int8 KV-quant grid
        constraint; production callers (via the ``__init__`` path) get int8 KV by default."""
        self = cls.__new__(cls)
        self._inner = None
        self.max_batch = int(max_batch)
        self.cfg = cfg
        self.num_layers = len(layers)
        self.embed_w = embed_w
        self.norm_w = norm_w
        self.lm_head_w = lm_head_w
        self.layers = layers
        self._kv_quantized = bool(kv_quantized)
        self._kv_group_size = int(kv_group_size)
        self._loopkill = bool(QWEN35_BATCHED_LOOPKILL_DEFAULT)   # #153 M1 GQA loop-kill (rule-4 flag)
        return self

    # --- per-stream cache factory ---------------------------------------------
    def make_caches(self) -> Qwen35Cache:
        """One stream's fresh decode cache — int8 KV on full-attn layers by default; recurrent
        state on linear-attn layers (same as :meth:`Qwen35ResidentModel.make_caches`). The
        ``kv_quantized`` / ``kv_group_size`` set at construction control the KV-cache storage mode."""
        return Qwen35Cache(self.num_layers, self.cfg, quantized=self._kv_quantized,
                           group_size=self._kv_group_size)

    def make_batch_caches(self, batch: int) -> list[Qwen35Cache]:
        """A list of ``batch`` independent per-stream caches (one per concurrent stream)."""
        if batch < 1:
            raise ValueError(f"batch must be >= 1, got {batch}")
        if batch > self.max_batch:
            raise ValueError(f"batch {batch} exceeds max_batch {self.max_batch} "
                             f"(raise max_batch on construction)")
        return [self.make_caches() for _ in range(batch)]

    # --- prefill (single-stream; consumes the prompt into ONE stream's cache) -
    def prefill(self, prompt_ids, state: Qwen35Cache) -> mx.array:
        """Consume ``prompt_ids`` into ``state`` (one stream's cache); return logits ``[1,1,vocab]``.

        Single-stream by design (Design A): full-attention prefill needs a common offset / mask
        across the consumed window, which cannot be shared with other streams' independent state.
        The orchestrator calls this once per new stream's prompt; decoding then proceeds via
        :meth:`step_batch`. Seeds the cache the same way :func:`quanta.qwen35.generate.generate`
        does — stepping one token at a time so the per-stream state is bit-identical to a fresh
        single-stream run."""
        ids = list(prompt_ids)
        if not ids:
            raise ValueError("prompt_ids is empty (prefill needs >= 1 token)")
        logits = None
        # use the same one-token-at-a-time machinery batched_decode_step uses (so a B=1 prefill is
        # bit-identical to a step-by-step decode at offsets 0..len-1).
        for pos, tid in enumerate(ids):
            logits = batched_decode_step(self.layers, self.embed_w, self.norm_w, self.lm_head_w,
                                         self.cfg, [int(tid)], [state], [pos], loopkill=False)[0]
            mx.eval(logits)
        return logits  # [1,1,vocab] at the last consumed position

    # --- single-stream callable contract (drop-in for Qwen35ResidentModel.__call__) ----------
    def __call__(self, token_ids, *, caches: Qwen35Cache, offset: int = 0,
                 capture_layers=None) -> mx.array:
        """Single-token decode-step contract — the same signature single-stream consumers expect
        (e.g. :func:`quanta.qwen35.generate.generate`). For ``T>1`` token-id input, steps each
        token in sequence; the cache mutates in place, output is logits ``[1,T,vocab]``.

        ``capture_layers`` is not supported in this batched-runtime wrapper (the single-stream
        runtime owns that feature for MTP/spec); pass ``None`` (the default). The orchestrator
        drives :meth:`step_batch` directly when batching; this method exists so generators / spec
        wrappers built for the single-stream contract can use the batched runtime unchanged.
        """
        if capture_layers:
            raise NotImplementedError(
                "Qwen35BatchedResidentModel.__call__ does not support capture_layers; use the "
                "single-stream Qwen35ResidentModel for MTP/spec capture, or step_batch() for B>=1.")
        ids = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids)
        ids = ids.reshape(-1)                                       # [T]
        t_total = int(ids.shape[0])
        if t_total < 1:
            raise ValueError("__call__ needs >= 1 token id")
        # step each token through the B=1 batched path; per-token logits stack to [1,T,vocab].
        per_t: list[mx.array] = []
        for k in range(t_total):
            tid = int(ids[k].item())
            lg = batched_decode_step(self.layers, self.embed_w, self.norm_w, self.lm_head_w,
                                     self.cfg, [tid], [caches], [offset + k],
                                     loopkill=False)[0]    # [1,1,vocab] (single-stream contract)
            per_t.append(lg)
        return per_t[0] if t_total == 1 else mx.concatenate(per_t, axis=1)           # [1,T,vocab]

    # --- one-step batched decode -----------------------------------------------
    def step_batch(self, stream_token_ids: Sequence[int],
                   stream_caches: Sequence[Qwen35Cache],
                   offsets: Sequence[int]) -> list[mx.array]:
        """One decode step for ``B = len(stream_token_ids)`` concurrent streams.

        ``stream_token_ids[b]`` is the int token id feeding stream ``b``; ``stream_caches[b]`` is
        that stream's :class:`Qwen35Cache` (mutated in place); ``offsets[b]`` is the absolute
        position of the new token in stream ``b`` (== ``stream_caches[b].offset`` before the step).
        Returns the per-stream logits as a list of ``[1,1,vocab]`` arrays — one per stream, in
        input order, so the caller can sample each stream independently.

        Per-stream output is bit-identical to feeding the same token through the single-stream
        :class:`Qwen35ResidentModel` at the same offset against the same cache — gated in
        ``parity/qwen35_batched_test.py``. With the #153 GQA loop-kill enabled (``self._loopkill`` ←
        :data:`QWEN35_BATCHED_LOOPKILL_DEFAULT`) the full-attention layers run ONE batched attention
        across streams instead of the per-stream loop — greedy-token-equivalent (gated in
        ``parity/qwen35_batched_loopkill_test.py``).
        """
        if len(stream_token_ids) > self.max_batch:
            raise ValueError(f"batch {len(stream_token_ids)} exceeds max_batch {self.max_batch}")
        return batched_decode_step(self.layers, self.embed_w, self.norm_w, self.lm_head_w,
                                   self.cfg, stream_token_ids, stream_caches, offsets,
                                   loopkill=self._loopkill)

    # --- shared-offset batched step for tree-spec batched verify ---------------
    def batch_step(
        self,
        tokens: Sequence[int],
        *,
        caches: Sequence[Qwen35Cache],
        offset: int,
        capture_layer: int | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """One batched decode step with a SHARED offset across all ``B = len(tokens)`` streams —
        the verify shape :func:`quanta.qwen35.spec.batch_verify` drives for batched tree-spec
        (docs/batched_tree_verify.md). Returns ``(logits [B,1,vocab], captured [B,1,hidden] or None)``.

        Mirrors :meth:`quanta.dsv4.batched_runtime.DSV4BatchedResidentModel.batch_step` but on
        Qwen3.5's 2-D hidden ``[B, 1, hidden]`` (vs DSV4's HC 4-D ``[B, 1, hc, dim]``) and over
        the hybrid mixer schedule (GDN linear-attn layers + GQA full-attn layers, by
        ``cfg.is_linear_attention(i)``). Each replica's cache mutates independently — the linear-
        layer recurrent state and the full-layer KV grow per-replica under MLX immutability so
        sibling replicas never see each other's writes.

        Differences from :meth:`step_batch`:

        * single position (``T == 1``) per call — the spec's verify loops :math:`depth+1` calls;
        * every stream shares ``offset`` (replicas built from the same prefix advance lock-step);
        * optionally captures one layer's residual ``[B, 1, hidden]`` for the MTP feature.

        Reuses the same per-stream mixer (``_gdn_step_through_cache`` / ``_gqa_step_through_cache``)
        + batched MoE pattern as :func:`batched_decode_step`; no new attention / MoE code.
        Numerically equivalent to running each stream through the single-stream runtime at the same
        offset with its own cache — gated model-free in ``parity/qwen35_batched_tree_verify_test.py``.
        """
        b = len(tokens)
        if b < 1:
            raise ValueError("batch_step needs >= 1 stream")
        if b > self.max_batch:
            raise ValueError(f"batch_step: B={b} exceeds max_batch={self.max_batch}")
        if len(caches) != b:
            raise ValueError(
                f"batch_step: len(caches)={len(caches)} != len(tokens)={b}"
            )
        # Each replica must already sit at the shared offset (rule 6 — no silent drift): the verify
        # loop only makes sense when every stream advances the same absolute position.
        for s, cache in enumerate(caches):
            if cache.offset != offset:
                raise ValueError(
                    f"batch_step: caches[{s}].offset={cache.offset} != offset={offset} "
                    f"(all B replicas must sit at the shared verify offset)"
                )
        if capture_layer is not None and not 0 <= capture_layer < self.num_layers:
            raise ValueError(
                f"batch_step: capture_layer={capture_layer} not in [0, {self.num_layers})"
            )

        # Per-stream embed in ONE indexing op (same as batched_decode_step).
        ids_b = mx.array([int(t) for t in tokens], dtype=mx.int32)              # [B]
        h_b = self.embed_w[ids_b][:, None].astype(mx.bfloat16)                  # [B, 1, hidden]
        hs: list[mx.array] = [h_b[i:i + 1] for i in range(b)]                   # B × [1,1,hidden]

        captured: mx.array | None = None
        seq_hint = caches[0].yarn_seq(offset + 1, self.cfg)                     # YaRN pin (shared replicas)

        for layer_i, blk in enumerate(self.layers):
            # 1) per-stream mixer step — bounded B-loop, rule-3 OK. GDN / GQA work is vectorized
            # inside each call; each cache mutates per-replica (immutability isolates siblings).
            after_mixer: list[mx.array] = []
            for s in range(b):
                lc = caches[s][layer_i]
                if blk.is_linear:
                    out_s = _gdn_step_through_cache(blk, lc, hs[s])
                else:
                    out_s = _gqa_step_through_cache(blk, lc, hs[s], seq_hint)
                after_mixer.append(out_s)

            # 2) stack → batched MoE → split (the win: ONE qwen35_moe call over [B,1,hidden]).
            stacked = mx.concatenate(after_mixer, axis=0) if b > 1 else after_mixer[0]   # [B,1,hidden]
            h_post_norm = blk.post_attention_layernorm(stacked)
            moe_out = blk.mlp(h_post_norm, sparse=True)
            stacked = stacked + moe_out
            if capture_layer is not None and layer_i == capture_layer:
                captured = stacked                                                       # [B,1,hidden]
            hs = [stacked[i:i + 1] for i in range(b)]

            mx.eval(stacked)                                                             # per-layer eval

        # Final norm + lm_head over the stacked [B,1,hidden] — one matmul.
        final = mx.concatenate(hs, axis=0) if b > 1 else hs[0]                           # [B,1,hidden]
        final = mx.fast.rms_norm(final, self.norm_w.astype(final.dtype), self.cfg.norm_eps)
        logits = final @ self.lm_head_w.T.astype(final.dtype)                            # [B,1,vocab]
        return logits, captured
