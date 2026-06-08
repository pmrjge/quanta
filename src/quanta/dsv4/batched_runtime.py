"""Batched (B > 1) DSV4-Flash decode runtime — Design A: per-stream caches, batched MoE.

This is the multi-stream sibling of :class:`quanta.dsv4.runtime.DSV4ResidentModel` for concurrent
agentic-loop serving. The single-stream model is the **parity reference**; this runtime is a
*wrapper* that reuses its load path and per-layer parameters unchanged, then replicates the
single-stream decode-block at the right granularity so the bandwidth-bound MoE is amortized across
``B`` simultaneous streams.

Why a wrapper: rule-8 (one layer resident at a time at load) is satisfied by the existing
``DSV4ResidentModel.__init__``; the inner :class:`DSV4ResidentModel` instance is the resident
copy. Prefill is delegated to its ``__call__`` (parity-correct by construction). Decode is split
per layer into two sub-blocks (Design A):

  1. **Per-stream attention sub-block** (loop over the B streams) — ``hc_pre → attn_norm →
     decode_step_dense / decode_step_compressed(cache=stream b, offset=offset[b]) → hc_post``.
     The per-stream KV caches are unavoidable (they hold each stream's growing latent stream /
     compressed stream / raw ring), so this loop is correctness-essential. ``B`` is bounded by
     ``max_batch`` (typically 32), so 43 layers × B = ≤ ~1400 attention calls per token is the
     allowed kind of bounded coarse loop (rule-3 forbids per-token/per-expert/per-hidden loops on
     hot paths; per-stream is acceptable — same shape, same code, just a small set of caches).

  2. **Batched FFN/MoE sub-block** — stack the post-attention HC residuals to ``[B,1,hc,dim]``,
     then ``hc_pre → ffn_norm → dsv4_moe`` ONCE on the stacked tensor. ``dsv4_moe`` reshapes
     ``[B,1,dim] → [B,dim]`` and routes ``B*topk`` slots through ``mx.gather_mm`` in one call —
     **shared expert weight reads are amortized across all B**, and routed expert weight reads
     amortize naturally as more slots land on the same expert (the bandwidth win the B-loop
     unlocks). Output is split back to per-stream ``[1,1,hc,dim]`` for the next layer.

Per-stream sampling / eos / admission live in :mod:`quanta.dsv4.batched_generate`; this module
exposes only the multi-stream forward.

Numerical equivalence to the single-stream runtime is gated MODEL-FREE in
``parity/dsv4_batched_test.py``: B identical prompts/streams must produce bit-identical logits to
the single-stream ``DSV4ResidentModel`` (Design A is output-equivalent to single-stream by
construction — the only batched op is the MoE FFN, which is itself batched-aware already).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from quanta.cache_quant import BITS
from quanta.dsv4.attention import _rms_w
from quanta.dsv4.decode import (
    DSV4Cache,
    _ArenaCacheHandle,
    _KVArenaSet,
    _PagedKVArena,
    _PagedLayerCache,
    decode_step_compressed,
    decode_step_compressed_batched,
    decode_step_dense,
    decode_step_dense_batched,
)
from quanta.dsv4.hyper import hc_expand, hc_post, hc_pre
from quanta.dsv4.moe import dsv4_moe
from quanta.dsv4.runtime import DSV4ResidentModel


def _latent_quant(head_dim: int, *, group_size: int = 128, quantized: bool = True) -> tuple[int, bool]:
    """The ``(group_size, quantized)`` the DSV4 latent KV settles on for ``head_dim`` — mirrors
    :meth:`quanta.dsv4.decode._LayerCache._resolve_quant` so the shared paged manager is bit-identical
    to the discrete cache (rule 6: a width/grouping mismatch would silently mis-decode the prefix).
    Real DSV4 head_dim=128 ⇒ ``(128, True)``; tiny test head_dims fall back exactly as the cache does."""
    if not quantized:
        return group_size, False
    if group_size in (32, 64, 128) and head_dim % group_size == 0:
        return group_size, True
    for g in (128, 64, 32):
        if g <= head_dim and head_dim % g == 0:
            return g, True
    return group_size, False               # head_dim too small / odd -> latent stays bf16


class DSV4BatchedResidentModel:
    """Multi-stream RAM-resident DSV4-Flash runtime — batched MoE amortized over B concurrent streams.

    The model is loaded ONCE (by the inner :class:`DSV4ResidentModel`) and shared across all
    streams. Each stream owns its own :class:`DSV4Cache` and its own decode position (``offset``);
    the inner model's per-layer params, embeddings, final HC head, norms and LM head are read-only
    and shared. ``max_batch`` is a soft ceiling — every batched call must satisfy ``B ≤ max_batch``
    or it fails loudly (rule-6).

    ``packed_experts`` and ``load_mtp`` are forward-compatible flags. Today the inner
    :class:`DSV4ResidentModel` uses dequantized bf16 expert stacks via ``mx.gather_mm`` (DSV4 source
    weights are fp4 → dequantized at load to bf16 stacks); ``packed_experts=True`` is accepted for
    callers that intend to swap in the int4-packed gather_qmm path once it lands — the surface does
    not change. ``load_mtp=True`` materializes the native MTP block params on
    ``self.mtp_params`` so callers can drive a spec-decode loop alongside the batched main forward;
    the batched runtime itself does not consume the MTP head.
    """

    # #152/#175 paged contract: DSV4 is pure attention (no global recurrence), but its compressed-KV /
    # indexer-KV / raw-hidden ring ARE derived boundary state that the compressor pools from RAW hidden
    # (not the latent KV), so they can't be recomputed from the shared latent — they're content-
    # addressed at block boundaries via RecurrentPrefixCache. ``has_recurrent_state=True`` tells the
    # session (quanta.shim.omlx._BaseBatchedSession) to lookup_at/store_at these boundary snapshots.
    has_recurrent_state = True

    def __init__(
        self,
        art_dir: str | Path,
        *,
        max_batch: int = 32,
        packed_experts: bool = True,
        load_mtp: bool = False,
        kv_arena: bool = True,
        n_layers: int | None = None,
    ) -> None:
        if max_batch < 1:
            raise ValueError(f"max_batch must be >= 1, got {max_batch}")
        # Single-stream inner is the parity reference (and the only resident-load path).
        inner = DSV4ResidentModel(art_dir, n_layers=n_layers)
        mtp_params: dict | None = None
        if load_mtp:
            # The MTP block params live in the artifact; load and cast / eval to match the rest of
            # the resident state. The batched runtime exposes them for spec-decode callers; the
            # main batched forward does not invoke the MTP head (which would require per-stream
            # captures and is the orchestrator's job).
            mtp_params = inner.art.mtp(0)
            mx.eval(list(mtp_params.values()))
            inner.art.release()
            mx.clear_cache()
        self._init_from_inner(inner, max_batch=max_batch, packed_experts=packed_experts,
                              mtp_params=mtp_params, kv_arena=kv_arena)

    @classmethod
    def from_inner(cls, inner, *, max_batch: int = 32, packed_experts: bool = True,
                   mtp_params: dict | None = None,
                   kv_arena: bool = True) -> "DSV4BatchedResidentModel":
        """Build a batched runtime around an EXISTING single-stream-shaped inner (parity tests).

        The inner must expose the :class:`DSV4ResidentModel` surface this runtime consumes:
        ``.cfg`` / ``.num_layers`` / ``.embed_w`` / ``.lm_head_w`` / ``.layers`` / ``._head`` /
        ``._rope`` / ``__call__(token_ids, caches=, offset=)``. The model-free parity gate uses a
        tiny fake inner over random params so it can prove batched == single-stream WITHOUT loading
        a real artifact (rule-6: never silently couple test correctness to checkpoint I/O)."""
        obj = cls.__new__(cls)
        obj._init_from_inner(inner, max_batch=max_batch, packed_experts=packed_experts,
                             mtp_params=mtp_params, kv_arena=kv_arena)
        return obj

    def _init_from_inner(self, inner, *, max_batch: int, packed_experts: bool,
                         mtp_params: dict | None, kv_arena: bool = True) -> None:
        """Common init: bind the inner + mirror its surface so callers can drop a batched model in
        wherever a single-stream one is expected (same ``.cfg``/``.num_layers``/``.embed_w``/...)."""
        if max_batch < 1:
            raise ValueError(f"max_batch must be >= 1, got {max_batch}")
        self._inner = inner
        self.max_batch = int(max_batch)
        self.packed_experts = bool(packed_experts)
        self.cfg = inner.cfg
        self.num_layers = inner.num_layers
        self.embed_w = inner.embed_w
        self.lm_head_w = inner.lm_head_w
        self.layers = inner.layers
        self.mtp_params = mtp_params
        # Batched-attention decode default: for the common all-T==1 step, fuse the per-stream attention
        # loop into ONE projection + ONE windowed-sink SDPA per layer across all B streams (the
        # decode_step_*_batched siblings). Gated model-free in parity/dsv4_batched_attention_test.py;
        # the per-stream Design-A loop stays the reference + the multi-token-tail fallback.
        self._fused = True
        # #18 batched KV arena: replace the B ragged per-stream caches' per-stream KV-update loop with a
        # persistent max_batch-sized arena set (one scatter write + one gather read per layer). Default
        # ON since M4: ``make_cache`` leases a row and returns an _ArenaCacheHandle; prefill seeds the
        # latent arena (via the handle's views) + migrates the derived ckv/ikv/ring into the _CompArena
        # set; ``_decode_batched_single`` dispatches to the arena steppers when the cache is a handle.
        # The proven per-stream _LayerCache loop is retained as the flagged reference (kv_arena=False)
        # and is still what a discrete DSV4Cache takes (dispatch keys off the cache type, not the flag).
        self._kv_arena = bool(kv_arena)
        self._arena_set: _KVArenaSet | None = None
        self._arena_quant: tuple[int, bool] = (0, False)
        if self._kv_arena:
            gs, q = _latent_quant(self.cfg.head_dim)
            self._arena_quant = (gs, q)
            comp_specs = [
                ({"ratio": self.cfg.compress_ratio(i), "overlap": self.cfg.overlap(i),
                  "has_indexer": self.cfg.has_indexer(i)}
                 if self.cfg.compress_ratio(i) > 0 else None)
                for i in range(self.num_layers)
            ]
            self._arena_set = _KVArenaSet(self.num_layers, self.max_batch, group_size=gs,
                                          quantized=q, comp_specs=comp_specs)
        # #153 batched-paged KV: when serving paged caches (a paged DSV4Cache, latent in a
        # PagedLatentCacheView), route the latent update through _PagedKVArena (ONE block-table scatter +
        # ONE gather, the paged sibling of the #18 arena loop-kill) instead of the per-stream lcs loop —
        # on DENSE layers (M1) AND COMPRESSED layers (M2, latent-only batched; the derived ckv/ikv/ring
        # stay per-stream so the paged boundary-snapshot lifecycle is unchanged). GRADUATED ON by default
        # (#153 M3: batched-paged == per-stream paged loop is bit-exact model-free through the real
        # session decode, parity/dsv4_paged_latent_test.py §C; rule 4 — the real-model B-sweep is the
        # deferred solo-GPU M4, not a correctness blocker). Read from the module flag at construction; a
        # per-session set still overrides it (the gate flips ON vs OFF that way).
        from quanta.paged import PAGED_KV_BATCHED_DEFAULT
        self._paged_kv_batched = bool(PAGED_KV_BATCHED_DEFAULT)

    # --- convenience pass-throughs --------------------------------------------
    def make_cache(self):
        """A fresh per-stream decode cache. With ``kv_arena`` on (#18, default) an
        :class:`~quanta.dsv4.decode._ArenaCacheHandle` leasing one row of the shared arena set
        (latent + compressed extras), presented as a :class:`DSV4Cache`-shaped object that prefill /
        batched decode drive unchanged; otherwise a discrete :class:`DSV4Cache` (one ``_LayerCache``
        per attention block). Fails loud (rule 6) when all ``max_batch`` arena rows are leased."""
        if self._kv_arena:
            gs, q = self._arena_quant
            row = self._arena_set.alloc()
            return _ArenaCacheHandle(self._arena_set, row, quantized=q, group_size=gs)
        return DSV4Cache(self.num_layers)

    def free_cache(self, cache) -> None:
        """Return an arena-backed cache's leased row to the free-list (serving release, #18). No-op for
        a discrete :class:`DSV4Cache` or when the arena is off — keyed off the cache having a ``row``."""
        row = getattr(cache, "row", None)
        if self._kv_arena and self._arena_set is not None and row is not None:
            self._arena_set.free(row)

    # --- prefill --------------------------------------------------------------
    def prefill(self, prompt_ids: mx.array, cache) -> mx.array:
        """Seed ``cache`` with ``prompt_ids`` (single-stream prefill via the parity-correct
        single-stream forward) and return the final-position logits ``[1, T, vocab]``.

        The DSV4 decode-stepper has no batched cache-fill, so the prompt is consumed by stepping
        each token through the single-stream decode path (the exact pattern
        :func:`quanta.dsv4.generate.generate` uses). This routes through the inner model's
        ``__call__`` which is the parity reference — the same code the single-stream gate
        validates — so prefill is byte-for-byte single-stream-equivalent.

        With an :class:`~quanta.dsv4.decode._ArenaCacheHandle` (#18 arena path) the prefill above
        wrote the latent KV straight into the arena row (through each layer view's ``append_kv``) but
        left the derived ckv/ikv/ring per-object on the views; :meth:`_ArenaCacheHandle.seed_comp`
        migrates them into the shared ``_CompArena`` set (bit-exact code-copy) so batched decode reads
        all state from the arenas."""
        ids = prompt_ids if isinstance(prompt_ids, mx.array) else mx.array(prompt_ids)
        ids = ids.reshape(-1)
        if ids.shape[0] == 0:
            raise ValueError("prefill: empty prompt_ids (need >= 1 token to seed the cache)")
        logits = self._inner(ids, caches=cache, offset=0)
        if hasattr(cache, "seed_comp"):
            cache.seed_comp()
        return logits

    # --- #152/#175 paged contract (driven by quanta.shim.omlx._BaseBatchedSession paged mode) -------
    @property
    def paged_kv_spec(self) -> dict:
        """Shape/codec the shared :class:`~quanta.paged.PagedKVCacheManager` must use to be bit-exact
        with the discrete latent KV stream — threaded from the config, never hardcoded (rule 6). Every
        DSV4 layer is attention (one latent KV head each), so ``n_layers == num_layers``;
        ``single_stream=True`` selects the one-latent-per-token codec (DSV4's MQA latent, not a k/v
        pair)."""
        gs, q = _latent_quant(self.cfg.head_dim)
        return {"n_layers": self.num_layers, "group_size": gs, "bits": BITS,
                "quantized": q, "single_stream": True}

    def make_paged_state(self, manager, seq, *, max_rollback: int = 1):
        """A per-stream paged :class:`~quanta.dsv4.decode.PagedDSV4Cache`: every layer's latent KV is a
        :class:`~quanta.paged.PagedLatentCacheView` into the shared ``manager`` (prefix blocks dedup),
        while the derived ckv/ikv/ring stay per-stream (restored from a boundary snapshot in
        :meth:`prefill_paged`).

        ``max_rollback`` sizes the per-stream derived ring for speculative rollback — leave it 1 for
        plain decode; the tree-spec-over-paged hook passes ``depth + 1`` so a rejected draft suffix
        rolls back losslessly (the same sizing the discrete :func:`quanta.dsv4.spec._make_caches`
        uses). The returned cache also satisfies the batched tree-spec ``replicate(B)`` contract
        (sequence-level COW fork) — see :class:`~quanta.dsv4.decode.PagedDSV4Cache`."""
        from quanta.dsv4.decode import paged_cache
        gs, q = _latent_quant(self.cfg.head_dim)
        return paged_cache(manager, seq, self.num_layers, quantized=q, group_size=gs,
                           max_rollback=max_rollback)

    def prefill_paged(self, suffix_ids, state, *, offset: int, recurrent_in,
                      block_size: int) -> tuple[mx.array, list[tuple[int, list]]]:
        """Prefill ONLY the uncached suffix into ``state`` (a paged :class:`DSV4Cache` whose latent
        slots are paged views over the resident prefix blocks), after restoring the derived
        compressed-KV / indexer-KV / raw-hidden ring from ``recurrent_in`` (the boundary snapshot after
        the shared prefix; ``None`` ⇒ fresh / ``offset == 0``).

        Split in two at the deepest full-block boundary so the derived state THERE can be snapshotted
        for future reuse: part 1 ``[offset, deepest)`` then the partial tail ``[deepest, end)`` — each
        run is the parity-correct single-stream decode loop (:meth:`DSV4ResidentModel.__call__` with
        ``caches=``), which appends latent to the paged views and pools the derived streams seeded by
        the restored ring (the boundary-straddling window reads the prefix tail the ring carries).
        Returns ``(last-position logits, [(deepest, derived_snapshot)])`` — the boundary list is empty
        when the suffix adds no new full block. Bit-identical to a from-scratch full prefill + the KV
        reuse is exact — gated model-free in ``parity/dsv4_paged_latent_test.py``."""
        from quanta.dsv4.decode import restore_derived, snapshot_derived
        if recurrent_in is not None:
            restore_derived(state, recurrent_in)        # seed ckv/ikv/ring (rule 4: == continuous)
        ids = suffix_ids if isinstance(suffix_ids, mx.array) else mx.array(suffix_ids)
        ids = ids.reshape(-1)
        total = int(ids.shape[0])
        if total == 0:
            raise ValueError("prefill_paged: empty suffix (admit must leave >=1 token to recompute)")
        end = offset + total
        deepest = (end // block_size) * block_size      # deepest full-block boundary <= end
        new_boundaries: list[tuple[int, list]] = []
        if deepest > offset:
            n1 = deepest - offset
            logits = self._inner(ids[:n1], caches=state, offset=offset)
            new_boundaries.append((deepest, snapshot_derived(state)))   # derived AFTER prefix+full blocks
            if end > deepest:
                logits = self._inner(ids[n1:], caches=state, offset=deepest)
        else:
            logits = self._inner(ids, caches=state, offset=offset)
        return logits, new_boundaries

    def get_recurrent_state(self, state) -> list:
        """Snapshot the live derived (compressed-KV / indexer-KV / ring) state for a decode-crossed
        block boundary — the per-layer list :class:`~quanta.paged.RecurrentPrefixCache` stores."""
        from quanta.dsv4.decode import snapshot_derived
        return snapshot_derived(state)

    # --- per-layer batched decode block ---------------------------------------
    def _attn_per_stream(
        self,
        h_stream: mx.array,
        p: dict,
        layer_id: int,
        cache: DSV4Cache,
        offset: int,
    ) -> mx.array:
        """One stream's attention sub-block: ``hc_pre → attn_norm → decode_step → hc_post``.

        Mirrors the attention half of :meth:`DSV4ResidentModel._decode_block` exactly. The stepper
        is chosen by ``cfg.compress_ratio(layer_id)`` (dense for ratio==0, compressed otherwise).
        ``h_stream`` is ``[1,1,hc,dim]`` — one stream, one token.
        """
        cfg = self.cfg
        eps, hc, iters, heps = cfg.norm_eps, cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps
        step = decode_step_dense if cfg.compress_ratio(layer_id) == 0 else decode_step_compressed

        # RoPE cos/sin for absolute positions [0, offset+1) — single-stream's cache, reused since
        # RoPE depends only on position, not on the stream's KV.
        cos, sin = self._inner._rope(layer_id, offset + 1)

        res = h_stream
        x, post, comb = hc_pre(
            h_stream, p["hc_attn_fn"], p["hc_attn_scale"], p["hc_attn_base"],
            hc, iters, eps, heps,
        )
        x = _rms_w(x, p["attn_norm"], eps)
        x = step(x, p["attn"], cfg, layer_id, cache, cos, sin, offset)
        return hc_post(x, res, post, comb)

    def _ffn_batched(
        self,
        h_stack: mx.array,
        ids_stack: mx.array,
        p: dict,
        layer_id: int,
    ) -> mx.array:
        """Batched FFN/MoE sub-block on the stacked post-attention residuals.

        ``h_stack``: ``[B,1,hc,dim]`` — all streams' post-attention HC residuals stacked.
        ``ids_stack``: ``[B,1]`` — the streams' input token ids (hash-routing key for the early
        layers; ignored by score-routing layers). One ``dsv4_moe`` call processes all B streams —
        ``B*topk`` slots in a single ``mx.gather_mm`` over the (shared) expert stacks, and ONE
        shared-expert matmul over the stacked input. Output ``[B,1,hc,dim]`` shape-equivalent to
        running each stream individually and stacking the results.
        """
        cfg = self.cfg
        eps, hc, iters, heps = cfg.norm_eps, cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps

        res = h_stack
        x, post, comb = hc_pre(
            h_stack, p["hc_ffn_fn"], p["hc_ffn_scale"], p["hc_ffn_base"],
            hc, iters, eps, heps,
        )
        x = _rms_w(x, p["ffn_norm"], eps)
        x = dsv4_moe(x, p["router"], p["experts"], p["shared"], cfg, layer_id, ids_stack)
        return hc_post(x, res, post, comb)

    # --- batched decode step --------------------------------------------------
    def step_batch(
        self,
        stream_token_ids: list[mx.array],
        caches: list[DSV4Cache],
        offsets: list[int],
    ) -> list[mx.array]:
        """Advance ``B`` streams by one decode step each — Design A: per-stream attention + batched MoE.

        ``stream_token_ids[b]``: the next token id(s) for stream ``b`` (``[1]`` or ``[T_b]`` if a
        stream is replaying a tail). ``caches[b]``: stream ``b``'s :class:`DSV4Cache`.
        ``offsets[b]``: the absolute position of ``stream_token_ids[b][0]`` in stream ``b``.

        Returns ``[1, T_b, vocab]`` logits per stream. Each layer is processed as:

          1. Per-stream attention (loop b): grow ``caches[b]`` and emit ``[1,1,hc,dim]`` per stream.
             Multi-token inputs (``T_b > 1``) are stepped position-by-position through the same
             stepper the single-stream runtime uses, so the cache stays in lock-step with the
             single-stream path.
          2. Stack to ``[B,1,hc,dim]``, batched MoE, split.

        Numerically equivalent to running each stream through the single-stream model in isolation:
        per-stream attention is the same code on the same cache, batched MoE is the same code that
        ``dsv4_moe(x [B,1,dim])`` runs in the single-stream prefill path. Validated model-free in
        ``parity/dsv4_batched_test.py``.
        """
        b = len(stream_token_ids)
        if b == 0:
            raise ValueError("step_batch: empty stream batch (need >= 1 stream)")
        if b > self.max_batch:
            raise ValueError(f"step_batch: B={b} exceeds max_batch={self.max_batch}")
        if len(caches) != b:
            raise ValueError(
                f"step_batch: len(caches)={len(caches)} != len(stream_token_ids)={b}"
            )
        if len(offsets) != b:
            raise ValueError(
                f"step_batch: len(offsets)={len(offsets)} != len(stream_token_ids)={b}"
            )
        # Per-stream cache offsets must match the declared offsets (rule-6 — never silently let a
        # stream drift): each stream's cache offset is the absolute position of its NEXT token.
        for s, (cache, off) in enumerate(zip(caches, offsets, strict=True)):
            if cache.offset != off:
                raise ValueError(
                    f"step_batch: stream {s} cache.offset={cache.offset} != declared offset={off}"
                )

        # Normalize each stream's token ids to a [1, T_b] mx.array; track each T_b for the unstack.
        stream_ids: list[mx.array] = []
        seq_lens: list[int] = []
        for s, ids in enumerate(stream_token_ids):
            arr = ids if isinstance(ids, mx.array) else mx.array(ids)
            arr = arr.reshape(1, -1)
            if arr.shape[1] == 0:
                raise ValueError(f"step_batch: stream {s} has empty token_ids")
            stream_ids.append(arr)
            seq_lens.append(int(arr.shape[1]))

        # #18 arena handles carry their KV in the shared arena set and MUST take the fused single-token
        # path (decode_step_*_batched on the arena). The per-stream Design-A loop / multi-token tail
        # below writes the derived ckv/ikv/ring per-object on the view and would silently diverge from
        # the arena decode reads — so fail loud (rule 6) rather than corrupt a stream.
        if any(hasattr(c, "row") for c in caches) and not (self._fused and all(t == 1 for t in seq_lens)):
            raise ValueError(
                "step_batch: arena cache handles require the fused single-token decode path (T==1 per "
                "stream); multi-token / non-fused decode must use a discrete DSV4Cache (#18)")

        # Fast path — every stream is a plain single-token decode (T_b == 1): fuse the per-stream
        # attention loop into ONE batched pass (decode_step_*_batched). Multi-token tails (T_b > 1,
        # spec replay) keep the position-by-position Design-A loop below — it is also the reference
        # the batched steppers are gated against (parity/dsv4_batched_attention_test.py).
        if self._fused and all(t == 1 for t in seq_lens):
            return self._decode_batched_single(mx.concatenate(stream_ids, axis=0), caches, offsets)

        # Per-stream HC residual stream (initial embed expansion) — [1, T_b, hc, dim] per stream.
        stream_h: list[mx.array] = [
            hc_expand(self.embed_w[arr].astype(mx.bfloat16), self.cfg.hc_mult)
            for arr in stream_ids
        ]

        # Per-stream output collectors (one [hc, dim] per position, stacked at the end).
        stream_outs: list[list[mx.array]] = [[] for _ in range(b)]

        # Step position-by-position across all streams. T_max is the longest stream's input length;
        # streams with shorter inputs simply stop participating once they've exhausted their tokens.
        # This is the bounded outer loop the single-stream runtime also uses (one per input token).
        t_max = max(seq_lens)
        for t in range(t_max):
            # Streams active at position t (those whose input is at least t+1 tokens long).
            active = [s for s in range(b) if t < seq_lens[s]]
            # Per-stream input slice + absolute position for the active set.
            slice_ids = [stream_ids[s][:, t:t + 1] for s in active]               # each [1,1]
            slice_off = [offsets[s] + t for s in active]
            slice_h = [stream_h[s][:, t:t + 1] for s in active]                   # each [1,1,hc,dim]

            for i, p in enumerate(self.layers):
                # (1) Per-stream attention — grows each stream's cache by exactly one position.
                slice_h = [
                    self._attn_per_stream(slice_h[k], p, i, caches[active[k]], slice_off[k])
                    for k in range(len(active))
                ]
                # (2) Stack -> batched MoE -> split. The stack/split is the seam: shared-expert
                # weight bytes amortize across all active streams in this one MoE call.
                h_stack = mx.concatenate(slice_h, axis=0)                          # [B_act,1,hc,dim]
                ids_stack = mx.concatenate(slice_ids, axis=0)                      # [B_act,1]
                h_stack = self._ffn_batched(h_stack, ids_stack, p, i)
                # split back per-stream (shape [1,1,hc,dim] each).
                slice_h = [h_stack[k:k + 1] for k in range(len(active))]

            # Materialize this position's grown caches + residuals (one eval per token, all streams).
            mx.eval([*slice_h])

            # Accumulate this token's [hc, dim] residual on the right stream's output list.
            for k, s in enumerate(active):
                stream_outs[s].append(slice_h[k][0, 0])

        # Reduce per-stream residual stacks to logits — final HC head + norm + lm_head, applied per
        # stream so the head matmul stays at ``[T_b, vocab]`` (no padding waste).
        out_logits: list[mx.array] = []
        for s in range(b):
            h_s = mx.stack(stream_outs[s], axis=0)[None]                          # [1, T_b, hc, dim]
            out_logits.append(self._inner._head(h_s))
        return out_logits

    def _decode_batched_single(self, ids_b: mx.array, caches: list[DSV4Cache],
                               offsets: list[int]) -> list[mx.array]:
        """One batched decode step across ``B`` streams with the attention **fused across streams**
        (the ``self._fused`` all-T==1 path). ``ids_b``: ``[B,1]`` token ids; ``caches[b]``/``offsets[b]``:
        stream ``b``'s :class:`DSV4Cache` / absolute position. Per layer: ONE batched attention sub-block
        (``decode_step_*_batched`` — batched projection + per-stream cache/pool IO + one windowed-sink
        SDPA) wrapped in the batched HC mix, then the already-batched :meth:`_ffn_batched`. Returns the
        per-stream ``[1,1,vocab]`` logits list — output-equivalent to :meth:`step_batch`'s Design-A loop
        (gated in ``parity/dsv4_batched_attention_test.py``)."""
        cfg = self.cfg
        eps, hc, iters, heps = cfg.norm_eps, cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps
        b = int(ids_b.shape[0])
        h = hc_expand(self.embed_w[ids_b].astype(mx.bfloat16), hc)               # [B,1,hc,dim]
        max_len = max(offsets) + 1
        # #18: dispatch on the cache TYPE (not the flag) so an _ArenaCacheHandle takes the loop-kill
        # arena path (ONE scatter + ONE gather per layer) while a discrete DSV4Cache always keeps the
        # proven per-stream _LayerCache loop — even if the runtime was built with the arena on.
        arena_path = b > 0 and hasattr(caches[0], "row")
        # #153 M1+M2: a paged DSV4Cache (latent in a PagedLatentCacheView) takes the paged loop-kill via
        # _PagedKVArena (ONE block-table scatter + ONE gather over the shared manager) — on its DENSE
        # layers (M1) AND its COMPRESSED layers (M2, where only the LATENT is batched; the derived
        # ckv/ikv/ring stay per-stream so the paged boundary-snapshot lifecycle is unchanged). Gated on
        # self._paged_kv_batched (GRADUATED ON by default since #153 M3 — bit-exact vs the per-stream
        # paged loop, rule 4; revert via the single flag). M0 proved the batched scatter is bit-identical
        # to the per-stream write, so the latent round-trip is exact on both regimes.
        paged_path = (self._paged_kv_batched and b > 0 and not arena_path
                      and isinstance(caches[0].layers[0], _PagedLayerCache))
        rows = ([c.row for c in caches] if arena_path
                else list(range(b)) if paged_path else None)
        if paged_path:
            pmgr = caches[0].layers[0]._view._m                  # shared manager (same for all streams)
            pseqs = [c.layers[0]._view._seq for c in caches]     # each stream's SeqHandle (same across layers)
        for i, p in enumerate(self.layers):
            cos, sin = self._inner._rope(i, max_len)                             # full RoPE tables
            dense = cfg.compress_ratio(i) == 0
            res = h                                                              # attention sub-block
            x, post, comb = hc_pre(h, p["hc_attn_fn"], p["hc_attn_scale"], p["hc_attn_base"],
                                   hc, iters, eps, heps)
            x = _rms_w(x, p["attn_norm"], eps)
            if arena_path:
                latent = self._arena_set.latent[i]
                if dense:
                    x = decode_step_dense_batched(x, p["attn"], cfg, i, None, cos, sin, offsets,
                                                  arena=latent, rows=rows)
                else:
                    x = decode_step_compressed_batched(x, p["attn"], cfg, i, None, cos, sin, offsets,
                                                       arena=latent, rows=rows,
                                                       comp=self._arena_set.comp[i])
            elif paged_path:                                     # #153 M1 (dense) + M2 (compressed)
                latent = _PagedKVArena(pmgr, pseqs, i)           # ONE block-table scatter + ONE gather
                if dense:
                    x = decode_step_dense_batched(x, p["attn"], cfg, i, None, cos, sin, offsets,
                                                  arena=latent, rows=rows)
                else:                                            # latent batched; derived ckv/ikv/ring
                    lcs = [caches[s][i] for s in range(b)]       # stay per-stream (snapshot lifecycle)
                    x = decode_step_compressed_batched(x, p["attn"], cfg, i, lcs, cos, sin, offsets,
                                                       arena=latent, rows=rows)   # comp=None -> hybrid
            else:
                lcs = [caches[s][i] for s in range(b)]                           # this layer per stream
                if dense:
                    x = decode_step_dense_batched(x, p["attn"], cfg, i, lcs, cos, sin, offsets)
                else:
                    x = decode_step_compressed_batched(x, p["attn"], cfg, i, lcs, cos, sin, offsets)
            h = hc_post(x, res, post, comb)
            h = self._ffn_batched(h, ids_b, p, i)                               # batched MoE sub-block
        mx.eval(h)
        logits = self._inner._head(h)                                            # [B,1,vocab]
        return [logits[s:s + 1] for s in range(b)]

    # --- single-stream __call__ delegating to inner ---------------------------
    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        """Single-stream forward — delegate to the inner :class:`DSV4ResidentModel`.

        Same signature as :meth:`DSV4ResidentModel.__call__` so the batched runtime is a true
        drop-in (the class docstring's promise). The prefill + per-path-verify + commit-replay
        paths inside :func:`quanta.dsv4.spec.spec_generate_tree` use this single-stream surface
        regardless of ``batched=`` value (only the per-position verify in the ``batched=True``
        branch dispatches via :meth:`batch_step`). Use :meth:`step_batch` directly for the
        multi-stream batched-decode surface (per-stream offsets); use :meth:`batch_step` for
        the shared-offset spec-verify surface."""
        return self._inner(token_ids, caches=caches, offset=offset, capture_layers=capture_layers)

    # --- shared-offset batched step for tree-spec batched verify -------------
    def batch_step(
        self,
        tokens,
        *,
        caches: list[DSV4Cache],
        offset: int,
        capture_layer: int | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """One batched decode step with a SHARED offset across all ``B = len(tokens)`` streams —
        the verify shape :func:`quanta.dsv4.spec.batch_verify` drives for batched tree-spec
        (docs/batched_tree_verify.md).

        Returns ``(logits, captured)``:

        * ``logits`` — ``[B, 1, vocab]`` per-stream logits at this verify position.
        * ``captured`` — ``[B, 1, hc, dim]`` HC residual after layer ``capture_layer`` (or ``None``
          if not requested). This is the MAIN-model hidden the spec loop carries through as
          ``prev_hidden`` for the next MTP build.

        Differences from :meth:`step_batch`:

        * single position (``T == 1``) per call — the spec's verify loops :math:`depth+1` calls;
        * every stream shares ``offset`` (replicas built from the same prefix advance lock-step),
          where :meth:`step_batch` carries per-stream offsets for ragged agentic-loop streams;
        * optionally captures a layer's HC residual for the MTP feature.

        Reuses the same per-stream attention + batched-MoE Design-A internals (``_attn_per_stream``
        + ``_ffn_batched``) so the per-stream attention reads the SAME ``decode_step_dense`` /
        ``decode_step_compressed`` the single-stream runtime uses (and the MoE call is the same
        ``dsv4_moe`` over the stacked ``[B, 1, hc, dim]`` — the bandwidth win that makes batched
        verify cheaper than ``W^D`` sequential per-path forwards).

        Numerically equivalent to running each stream through the single-stream runtime at the same
        offset with its own cache — gated model-free in
        ``parity/dsv4_batched_tree_verify_test.py`` against a stub main model.
        """
        b = len(tokens)
        if b < 1:
            raise ValueError("batch_step: empty stream batch (need >= 1 stream)")
        if b > self.max_batch:
            raise ValueError(f"batch_step: B={b} exceeds max_batch={self.max_batch}")
        if len(caches) != b:
            raise ValueError(
                f"batch_step: len(caches)={len(caches)} != len(tokens)={b}"
            )
        # Each replica must already sit at the shared offset (fail loud — rule 6 / no silent drift):
        # the verify loop only makes sense when every stream advances the same absolute position.
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

        cfg = self.cfg
        # Per-stream input ids + HC residual init — same surface as step_batch, just for T=1.
        ids_b = mx.array([int(t) for t in tokens], dtype=mx.int32)[:, None]    # [B, 1]
        slice_h: list[mx.array] = [
            hc_expand(self.embed_w[ids_b[k:k + 1]].astype(mx.bfloat16), cfg.hc_mult)
            for k in range(b)
        ]                                                                       # B × [1,1,hc,dim]

        captured: mx.array | None = None
        for i, p in enumerate(self.layers):
            # 1) per-stream attention through the replica's own cache (bounded B-loop, rule-3 OK).
            slice_h = [
                self._attn_per_stream(slice_h[k], p, i, caches[k], offset) for k in range(b)
            ]
            # 2) stack → batched MoE → split (the routed/shared expert weight reads amortize over
            # all B streams in ONE dsv4_moe call — the win batched verify exists to harvest).
            h_stack = mx.concatenate(slice_h, axis=0)                           # [B,1,hc,dim]
            h_stack = self._ffn_batched(h_stack, ids_b, p, i)
            if capture_layer is not None and i == capture_layer:
                captured = h_stack                                              # [B,1,hc,dim]
            slice_h = [h_stack[k:k + 1] for k in range(b)]

        # Materialize this step's grown caches + residuals (one eval per token, rule-8 boundary).
        mx.eval(*slice_h)

        # Final HC head + norm + lm_head over the stacked [B,1,hc,dim] — single matmul.
        final_h = mx.concatenate(slice_h, axis=0)                               # [B,1,hc,dim]
        logits = self._inner._head(final_h)                                     # [B,1,vocab]
        return logits, captured
