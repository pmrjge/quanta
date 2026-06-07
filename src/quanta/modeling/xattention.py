"""XAttention — training-free block-sparse prefill attention (Xu et al., 2025).

Attacks attention's O(T²) by scoring each (query-block, key-block) pair with a cheap
**antidiagonal** proxy, then keeping only the minimal set of blocks whose normalized
score reaches a threshold (nucleus selection). Lossy ⇒ gated by the perplexity gate,
never numeric parity (CLAUDE.md). Default off; with ``threshold=1.0`` it keeps every
causal block and is bit-equivalent to dense (the parity sanity check).

Antidiagonal scoring: within a ``block``×``block`` tile of QKᵀ, the main antidiagonal
band (row-phase + col-phase ≈ const) captures the local/diagonal attention mass. We
sum each ``stride``-wide phase group of Q and of K, reverse K's phase order, and take
a phase-aligned dot product — so the block score is one small matmul of phase-summed,
phase-reversed vectors (cost ≈ 1/stride² of full QKᵀ). This is the scoring *signal*;
realizing the matching FLOP/memory savings (skip unscored QKᵀ, gather kept blocks) is
a later phase — here the selection drives an additive mask so its quality is
measurable through the existing attention path.

Per-head: selection and the returned mask are ``[B, H, T, S]`` (XAttention is
per-head). The mask is O(T·S) — fine at the moderate context the quality gate uses;
long-context memory needs the gathered execution path (deferred).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx

NEG_INF = float("-inf")


@dataclass(frozen=True)
class HeadSpec:
    """One query head's selector **kind + its own selection params** (M5 per-head params).

    M4's ``head_selectors`` routes each head to a selector *kind* but shares one set of params across
    every head of that kind. M5 lets each head carry its OWN params: a length-``num_query_heads`` tuple
    of ``HeadSpec`` on :class:`XAttnConfig` (``head_specs``) names, per head, both the kind and the
    params its kind reads (``xattn`` → ``threshold``, ``ashape`` → ``local``, ``vslash`` → ``vert`` /
    ``slash``; the others are ignored for that head). Frozen + hashable so distinct specs dedupe to the
    bounded per-spec loop in :func:`_select_keep_per_head_specs` (rule 3 — the loop is over DISTINCT
    specs, never heads).

    All three kinds carry **freely per-head** params. ``vslash`` shares the one global probe (computed
    once over the whole sequence and threaded into every gather chunk — that is what keeps gather ==
    mask), but the probe now returns param-INDEPENDENT masses (:func:`vertical_slash_index`) and the
    top-``vert`` / top-``slash`` cut is applied per spec in :func:`select_keep`, so two heads can read
    the same probe yet keep different ``vert``/``slash``. ``ashape``/``xattn`` select locally per spec.
    (Earlier the probe baked the top-k in, forcing one vslash vert/slash per config; that pin is gone.)
    """

    kind: str = "xattn"
    threshold: float = 0.9
    local: int = 8
    vert: int = 8
    slash: int = 8

    def __post_init__(self) -> None:
        if self.kind not in ("xattn", "ashape", "vslash"):
            raise ValueError(f"HeadSpec.kind must be 'xattn', 'ashape', or 'vslash', got {self.kind!r}")
        if self.local < 1:
            raise ValueError(f"HeadSpec.local must be >= 1, got {self.local}")
        if self.vert < 1 or self.slash < 1:
            raise ValueError(f"HeadSpec.vert/slash must be >= 1, got vert={self.vert}, slash={self.slash}")


@dataclass(frozen=True)
class XAttnConfig:
    """Block-sparse prefill config. ``threshold`` is the nucleus mass to retain.

    Defaults are the **runtime** config: the bounded block-gather path with a kept-block
    budget. ``gather`` selects the execution path (both output-equivalent given the same
    selection): True (default) gathers only the selected K/V blocks per query block and
    attends over those — O(T·max_kept·block), chunked to fit ``max_alloc_gb``, the path
    that makes long-context prefill cheap and can't OOM; False builds an additive
    ``[B,H,T,S]`` mask and runs full SDPA — correct but O(T²), only for the
    quality-measurement / parity path at moderate context.
    """

    block: int = 128
    stride: int = 16
    threshold: float = 0.9
    min_seq: int = 256  # below this, prefill runs dense (sparsity not worth it)
    gather: bool = True
    # cap on kept blocks/query → at most budget*block keys attended; bounds long-context
    # gather memory and only binds beyond ~budget*block tokens (so it never touches the
    # moderate-context quality results). 64 → ≤8192 keys/query. None = uncapped nucleus.
    budget: int | None = 64
    max_alloc_gb: float = 8.0  # chunks sized to fit this; gather fails loud rather than exceed it
    # --- selector: which block-importance pattern picks the kept set (execution is shared) -------
    # "xattn"  — antidiagonal block scoring + nucleus selection (the validated default).
    # "ashape" — MInference A-shape: attention sink (block 0) + a ``local``-block causal window,
    #            purely positional (no scoring). Static/lossier; MInference assigns it per-head.
    # "vslash" — MInference vertical-slash: a global pattern from an online probe of the LAST query
    #            block's attention to all keys (MInference §3) — top-``vert`` vertical key-blocks
    #            (columns everyone attends) ∪ top-``slash`` slash block-offset bands (diagonals at a
    #            fixed query-minus-key offset) — applied to every query block. Data-dependent but the
    #            pattern is built ONCE from the last block (prefill-only; decode stays dense).
    selector: str = "xattn"
    local: int = 8  # A-shape local window width in blocks (incl. the diagonal); selector="ashape" only
    vert: int = 8   # vertical-slash: # of vertical key-blocks kept globally; selector="vslash" only
    slash: int = 8  # vertical-slash: # of slash block-offset bands kept globally; selector="vslash" only
    # --- per-head assignment (M4): route each query head to its OWN selector (offline-assigned). ----
    # None ⇒ uniform ``selector`` for every head (every path above is byte-for-byte unchanged). When
    # set, a length-``num_query_heads`` tuple naming each head's selector kind ("xattn"/"ashape"/
    # "vslash"); ``selector`` is then ignored and ``select_keep`` routes per head — each head's keep is
    # byte-identical to the uniform keep for its assigned kind (a pure routing layer over the validated
    # selectors). Heads sharing a kind share that kind's params (``threshold``/``local``/``vert``/
    # ``slash``); per-head *params* are a later refinement. Built offline by :func:`assign_head_selectors`.
    head_selectors: tuple[str, ...] | None = None
    # --- per-head *params* (M5): each head carries its OWN selector kind + params (not just a kind). --
    # None ⇒ fall back to ``head_selectors`` (M4) / uniform ``selector`` — every path unchanged. When set,
    # a length-``num_query_heads`` tuple of :class:`HeadSpec` (kind + that kind's params), taking
    # precedence over ``head_selectors`` (setting both is rejected). ``select_keep`` dispatches per head
    # via :func:`_select_keep_per_head_specs` — each head's keep is byte-identical to the uniform keep for
    # its spec (pure routing). All kinds' params are freely per-head: the vslash probe returns
    # param-independent masses and the per-head top-``vert``/``slash`` cut is applied in
    # :func:`select_keep`, so heads sharing the one global probe can still keep different vert/slash.
    # Built offline by :func:`assign_head_specs` (kernel-aware FLOP-budgeted search).
    head_specs: tuple[HeadSpec, ...] | None = None
    # --- per-head-GROUPED gather (M9-speed "fold"): the block-gather sizes its work by ONE global
    # ``max_kept`` = the densest head's kept-block count, so a per-head assignment that mixes a cheap
    # static pattern (ashape, kept ~3% at long ctx) with a dense one (xattn nucleus, kept ~63%) makes
    # EVERY head pay the dense head's budget — combining does not fold the speed. When this is True AND a
    # per-head config is set (``head_specs`` / ``head_selectors``), :func:`gather_sparse_attention`
    # instead partitions heads by their DISTINCT spec and gathers each group at its OWN ``max_kept`` (a
    # bounded loop over distinct specs, rule 3), so the cheap-pattern heads run cheap. Output-equivalent
    # to the naive per-head gather (each head attends the SAME kept blocks); default True ⇒ the grouped
    # fold for per-head configs (**GRADUATED** — equivalence proven bit-exact by the model-free
    # ``internlm2_grouped_gather_test``, so rule 4's "naive until parity is proven" is satisfied and the
    # faster fold is now the default). Pass ``grouped_gather=False`` for the naive single-``max_kept``
    # path. Only affects ``gather=True`` + a per-head config; the uniform-selector (the fold guard is
    # False ⇒ no-op) and mask paths are unchanged.
    grouped_gather: bool = True

    def __post_init__(self) -> None:
        if self.block % self.stride != 0:
            raise ValueError(f"block {self.block} must be divisible by stride {self.stride}")
        if self.budget is not None and self.budget < 1:
            raise ValueError(f"budget must be >= 1, got {self.budget}")
        if self.selector not in ("xattn", "ashape", "vslash"):
            raise ValueError(f"selector must be 'xattn', 'ashape', or 'vslash', got {self.selector!r}")
        if self.local < 1:
            raise ValueError(f"local must be >= 1, got {self.local}")
        if self.vert < 1 or self.slash < 1:
            raise ValueError(f"vert/slash must be >= 1, got vert={self.vert}, slash={self.slash}")
        if self.head_selectors is not None:
            if len(self.head_selectors) == 0:
                raise ValueError("head_selectors must be a non-empty tuple (or None for uniform)")
            unknown = sorted(set(self.head_selectors) - {"xattn", "ashape", "vslash"})
            if unknown:
                raise ValueError(f"head_selectors has unknown selector kind(s) {unknown}; "
                                 "valid kinds: 'xattn', 'ashape', 'vslash'")
        if self.head_specs is not None:
            if self.head_selectors is not None:
                raise ValueError("set at most one of head_specs (per-head params) / head_selectors "
                                 "(per-head kind) — both route per head")
            if len(self.head_specs) == 0:
                raise ValueError("head_specs must be a non-empty tuple (or None)")
            for sp in self.head_specs:
                if not isinstance(sp, HeadSpec):
                    raise ValueError(f"head_specs entries must be HeadSpec, got {type(sp).__name__}")
                # NOTE: vslash specs may now carry per-head vert/slash. The global probe returns
                # param-INDEPENDENT masses (:func:`vertical_slash_index`) and the top-vert/slash cut is
                # applied per spec in :func:`select_keep`, so two heads share one probe yet keep
                # different vert/slash — the config's vert/slash no longer constrain a vslash spec.


# Runtime default sparse config: bounded block-gather XAttention prefill (gather=True,
# budget=64, chunked). Wired as the default for KimiModel.__call__ / generate so prefill
# is sparse by default — pass sparse=None for the exact dense path. Frozen ⇒ safe to share
# as a default argument. Only engages at prefills >= min_seq (256 tokens); shorter
# sequences (e.g. the parity/ppl harnesses) run dense via the min_seq gate.
DEFAULT_SPARSE = XAttnConfig()


def _pad_to(x: mx.array, mult: int, axis: int) -> mx.array:
    """Zero-pad ``x`` along ``axis`` up to a multiple of ``mult``."""
    n = x.shape[axis]
    pad = (-n) % mult
    if pad == 0:
        return x
    widths = [(0, 0)] * x.ndim
    widths[axis] = (0, pad)
    return mx.pad(x, widths)


def block_scores(q: mx.array, k: mx.array, scale: float, block: int, stride: int) -> mx.array:
    """Antidiagonal block-importance scores ``[B, H, Tq, Tk]`` from ``q``/``k`` ``[B,H,*,D]``."""
    bsz, h, _, d = q.shape
    qp = _pad_to(q, block, axis=2)
    kp = _pad_to(k, block, axis=2)
    tq, tk = qp.shape[2] // block, kp.shape[2] // block
    p = block // stride

    # [B,H,Tblk,P,stride,D] → sum the stride dim → phase-summed [B,H,Tblk,P,D]
    qb = qp.reshape(bsz, h, tq, p, stride, d).sum(axis=4)
    kb = kp.reshape(bsz, h, tk, p, stride, d).sum(axis=4)
    kb = kb[:, :, :, ::-1, :]  # reverse phase order → antidiagonal alignment

    qf = qb.reshape(bsz, h, tq, p * d)
    kf = kb.reshape(bsz, h, tk, p * d)
    # / (block*stride) = / (P*stride²) averages the P*stride² summed q·k pairs back to a
    # per-token-logit magnitude, so the block softmax has a sane temperature (not one-hot).
    return (qf @ mx.swapaxes(kf, -1, -2)) * (scale / (block * stride))  # [B,H,Tq,Tk]


def select_blocks(scores: mx.array, threshold: float, q_offset: int = 0) -> mx.array:
    """Nucleus block selection → bool keep-mask ``[B,H,Tq,Tk]`` (causal; diag+sink forced).

    ``q_offset`` is the global block index of the first query row — used when scoring a
    chunk of query blocks against all keys, so causality and the forced diagonal use the
    query's *global* position. ``q_offset=0`` is the whole-sequence case.
    """
    tq, tk = scores.shape[-2], scores.shape[-1]
    i = mx.arange(q_offset, q_offset + tq)[:, None]
    j = mx.arange(tk)[None, :]
    causal = j <= i  # block-level causality (global query index)

    masked = mx.where(causal, scores, NEG_INF)
    probs = mx.softmax(masked.astype(mx.float32), axis=-1)

    order = mx.argsort(-probs, axis=-1)
    sorted_p = mx.take_along_axis(probs, order, axis=-1)
    # keep block k if the mass strictly before it ≤ threshold (minimal nucleus reaching it);
    # ≤ (not <) makes threshold=1.0 keep every block even when softmax saturates to one-hot.
    keep_sorted = (mx.cumsum(sorted_p, axis=-1) - sorted_p) <= threshold
    inv = mx.argsort(order, axis=-1)
    keep = mx.take_along_axis(keep_sorted, inv, axis=-1)

    forced = (j == i) | (j == 0)  # always keep the diagonal (local) and block 0 (sink)
    return (keep | forced) & causal


def ashape_keep(tq: int, tk: int, q_offset: int, local: int) -> mx.array:
    """A-shape (StreamingLLM / MInference) block keep-mask ``[Tq, Tk]`` bool — purely positional.

    Query block ``i`` (global index ``q_offset + row``) keeps the **sink** (block 0) and a causal
    **local window** of the last ``local`` key blocks ending at its own diagonal block — i.e. the
    blocks ``{0} ∪ {i-local+1, …, i}``. No scores: A-shape is a fixed sink+window pattern (the
    static MInference selector), not data-dependent nucleus selection. ``local=1`` ⇒ just the
    diagonal + sink (the tightest A-shape — XAttention's force-kept set with no nucleus blocks).
    """
    i = mx.arange(q_offset, q_offset + tq)[:, None]   # [Tq,1] global query-block index
    j = mx.arange(tk)[None, :]                         # [1,Tk] key-block index
    causal = j <= i
    window = (j > i - local) & causal                  # last ``local`` blocks incl. the diagonal
    return (window | (j == 0)) & causal                # ∪ sink (block 0), re-clamped causal


def vertical_slash_index(
    q: mx.array, k: mx.array, scale: float, cfg: XAttnConfig
) -> tuple[mx.array, mx.array]:
    """MInference vertical-slash **probe masses** from the LAST query block (MInference §3).

    Runs the *last* (real) query block's attention over all keys, then decomposes that probe into two
    **param-independent** masses — the raw per-key-block and per-block-offset attention mass — from
    which the vertical-slash keep is selected. The top-``vert`` / top-``slash`` cut does NOT live here;
    it is applied in :func:`select_keep` from these masses, so a head can apply its OWN ``vert``/``slash``
    to the one shared probe (that is what lets per-head vslash *params* vary while every head still reads
    a single global probe — gather == mask):

    * **vertical** — ``key_mass[b,h,kb]`` = per-key-*column* attention mass summed over the probe
      queries, pooled to key blocks. :func:`select_keep` keeps the top-``vert`` of these (vertical
      stripes — columns everyone attends); sink block 0 is force-kept separately.
    * **slash** — ``slash_mass[b,h,o]`` = per-token-*offset* mass (query-pos − key-pos) summed over the
      probe, pooled to block-offsets ``o = δ // block``. :func:`select_keep` keeps the top-``slash`` of
      these (diagonal bands); offset 0 (the diagonal) is force-kept separately.

    Returns ``(key_mass [B,H,Tk] float, slash_mass [B,H,Tq] float)`` — the two raw masses, no
    thresholding (``key_mass`` doubles as the gather-budget priority). The probe is one ``[B,H,lp,S]``
    attention matrix (``lp`` = last-block size ≤ ``block``); a plain ``q@kᵀ`` matmul (not
    ``mx.fast.sdpa``) because the *attention weights* are needed, not the output. When the full probe
    fits ``max_alloc_gb`` (the short-doc default) it is computed single-shot below; past that (long
    context, 100K+) it is taken in KEY CHUNKS via :func:`_vertical_slash_index_chunked` (online/flash
    softmax) so peak memory is one chunk and it never OOMs — the masses are output-equivalent up to fp
    reassociation. Data-dependent but built ONCE from the last block (a prefill-only pattern; decode
    stays dense); the per-(query,key) execution mask the selection feeds is still strictly causal, so no
    future *content* leaks into earlier positions.
    """
    bsz, h, t, _ = q.shape
    s = k.shape[2]
    blk = cfg.block
    tq = (t + blk - 1) // blk
    tk = (s + blk - 1) // blk
    p0 = (tq - 1) * blk                          # first token of the last (real) query block
    q_last = q[:, :, p0:, :]                      # [B,H,lp,D]
    lp = q_last.shape[2]

    gb = bsz * h * lp * s * 4 / 1e9               # fp32 probe attention [B,H,lp,S]
    if gb > cfg.max_alloc_gb:
        # Long-context probe: the full [B,H,lp,S] attention exceeds the budget (OOM at 100K+). Take the
        # probe softmax in KEY CHUNKS (online/flash) so peak memory is one chunk, not O(S). The masses
        # equal this single-shot path up to fp reassociation (gated model-free). The short-doc default
        # below (gb <= max_alloc_gb) is left byte-for-byte unchanged, so the M1–M6 gates are bit-identical.
        return _vertical_slash_index_chunked(q_last, k, scale, cfg, p0, lp, t, s, tq, tk)

    qpos = mx.arange(p0, p0 + lp)[:, None]        # [lp,1] abs query positions
    kpos = mx.arange(s)[None, :]                  # [1,S]
    sc = (q_last @ mx.swapaxes(k, -1, -2)) * scale            # [B,H,lp,S] raw probe scores
    sc = mx.where(kpos <= qpos, sc, NEG_INF)                  # causal
    a = mx.softmax(sc.astype(mx.float32), axis=-1)           # [B,H,lp,S] probe attention

    # vertical: per-key-block column mass over the probe queries (param-independent — no top-k here).
    col = _pad_to(a.sum(axis=2), blk, axis=2)                # [B,H,tk*blk]
    key_mass = col.reshape(bsz, h, tk, blk).sum(axis=3)      # [B,H,tk]

    # slash: per-block-offset mass (qpos − kpos), param-independent (no top-k here). gather a[r, p0+r-δ]
    # for each probe row r and offset δ — the offset-δ key of that row — then sum over r. One gather +
    # one sum (no per-token loop); invalid (non-causal) entries zeroed; pooled to block-offsets δ//block.
    n_off = t
    r = mx.arange(lp)[:, None]
    dl = mx.arange(n_off)[None, :]
    c_idx = (p0 + r - dl).astype(mx.int32)                   # [lp,n_off] key index for offset dl
    valid = (dl <= p0 + r) & (c_idx >= 0)                    # causal & in-range
    gi = mx.broadcast_to(mx.clip(c_idx, 0, s - 1)[None, None], (bsz, h, lp, n_off))
    g = mx.where(valid[None, None], mx.take_along_axis(a, gi, axis=-1), 0.0)  # [B,H,lp,n_off]
    slash_tok = _pad_to(g.sum(axis=2), blk, axis=2)          # [B,H,tq*blk]
    slash_mass = slash_tok.reshape(bsz, h, tq, blk).sum(axis=3)   # [B,H,tq] per block-offset
    return key_mass, slash_mass


def _vertical_slash_index_chunked(
    q_last: mx.array, k: mx.array, scale: float, cfg: XAttnConfig,
    p0: int, lp: int, t: int, s: int, tq: int, tk: int,
) -> tuple[mx.array, mx.array]:
    """Key-chunked long-context vertical-slash probe (the M7 lever) — the same param-independent masses
    as the single-shot :func:`vertical_slash_index`, at O(one key chunk) peak memory instead of O(S).

    The single-shot probe materializes the whole ``[B,H,lp,S]`` attention; past ``max_alloc_gb`` that
    OOMs at 100K+. Here the probe softmax over keys is taken in key chunks via the standard
    **online-softmax (flash) two pass**: pass 1 accumulates the per-probe-row running max ``m[r]`` and
    normalizer ``l[r]`` over key chunks (peak one ``[B,H,lp,Sc]`` chunk); pass 2 recomputes each chunk's
    FINAL normalized probs ``a = exp(sc - m)/l`` and accumulates the two masses:

    * **vertical** ``key_mass[b,h,kb]`` — per-key-block column mass (sum over probe rows), written into
      this chunk's key blocks (chunks cover disjoint whole key blocks).
    * **slash** ``slash_mass[b,h,ob]`` — per-block-offset mass. Each chunk contributes a bounded offset
      window ``δ = p0 + r - key`` (``key`` in ``[ks,ke)``); windows from adjacent chunks overlap in
      ``δ``-space, so they accumulate (``+=``) into the per-token ``slash_tok`` before pooling to blocks.

    Output-equivalent to single-shot up to fp reassociation of the key reduction. The two chunk loops are
    the sanctioned coarse, bounded chunked-prefill loops (like :func:`gather_sparse_attention`), never a
    per-token/per-key hot loop (rule 3). Returns ``(key_mass [B,H,tk], slash_mass [B,H,tq])`` fp32.
    """
    bsz, h = q_last.shape[0], q_last.shape[1]
    blk = cfg.block
    # key chunk = the largest whole-block span whose [B,H,lp,Sc] probe (+ its exp) fits max_alloc_gb
    # (1.5x margin); >= one block. The chunk boundary lands on a key-block edge so vertical pools cleanly.
    per_blk = 1.5 * bsz * h * lp * blk * 4
    nblk_chunk = max(1, int(cfg.max_alloc_gb * 1e9 // per_blk))
    sc_chunk = nblk_chunk * blk
    qpos = mx.arange(p0, p0 + lp)[:, None]                    # [lp,1] abs probe-query positions

    # ---- pass 1: online-softmax (flash) running max m[r] + normalizer denom[r] over key chunks ----
    m = mx.full((bsz, h, lp), NEG_INF, dtype=mx.float32)
    denom = mx.zeros((bsz, h, lp), dtype=mx.float32)
    for ks in range(0, s, sc_chunk):
        ke = min(ks + sc_chunk, s)
        kpos = mx.arange(ks, ke)[None, :]
        sc = ((q_last @ mx.swapaxes(k[:, :, ks:ke, :], -1, -2)) * scale).astype(mx.float32)
        sc = mx.where(kpos <= qpos, sc, NEG_INF)             # causal
        m_new = mx.maximum(m, sc.max(axis=-1))
        denom = denom * mx.exp(m - m_new) + mx.exp(sc - m_new[..., None]).sum(axis=-1)
        m = m_new
        mx.eval(m, denom)
    inv_denom = (1.0 / denom)[..., None]                     # [B,H,lp,1]

    # ---- pass 2: accumulate the masses from the final, globally-normalized probs ----
    key_mass = mx.zeros((bsz, h, tk), dtype=mx.float32)
    slash_tok = mx.zeros((bsz, h, tq * blk), dtype=mx.float32)   # per-token offset, summed over rows
    r = mx.arange(lp)[:, None]
    for ks in range(0, s, sc_chunk):
        ke = min(ks + sc_chunk, s)
        kpos = mx.arange(ks, ke)[None, :]
        sc = ((q_last @ mx.swapaxes(k[:, :, ks:ke, :], -1, -2)) * scale).astype(mx.float32)
        sc = mx.where(kpos <= qpos, sc, NEG_INF)
        a = mx.exp(sc - m[..., None]) * inv_denom            # [B,H,lp,Sc] final global-softmax probs
        # vertical: per-key-block column mass (sum over probe rows) into this chunk's disjoint key blocks
        col = _pad_to(a.sum(axis=2), blk, axis=2)            # [B,H, ceil(Sc/blk)*blk]
        nbc = col.shape[2] // blk
        kb0 = ks // blk
        key_mass[:, :, kb0:kb0 + nbc] = (
            key_mass[:, :, kb0:kb0 + nbc] + col.reshape(bsz, h, nbc, blk).sum(axis=3)
        )
        # slash: bounded offset window δ = p0 + r - key for key in [ks,ke); accumulate (windows overlap).
        d_hi = p0 + (lp - 1) - ks
        d_lo = max(0, p0 - (ke - 1))
        w = d_hi - d_lo + 1
        dwin = mx.arange(d_lo, d_lo + w)[None, :]            # [1,w]
        c_idx = (p0 + r - dwin).astype(mx.int32)             # [lp,w] key index for each offset
        valid = (c_idx >= ks) & (c_idx < ke) & (dwin <= p0 + r)
        gi = mx.broadcast_to(mx.clip(c_idx - ks, 0, ke - ks - 1)[None, None], (bsz, h, lp, w))
        g = mx.where(valid[None, None], mx.take_along_axis(a, gi, axis=-1), 0.0)   # [B,H,lp,w]
        slash_tok[:, :, d_lo:d_lo + w] = slash_tok[:, :, d_lo:d_lo + w] + g.sum(axis=2)
        mx.eval(key_mass, slash_tok)
    slash_mass = slash_tok.reshape(bsz, h, tq, blk).sum(axis=3)    # [B,H,tq] per block-offset
    return key_mass, slash_mass


def _uses_vslash(cfg: XAttnConfig) -> bool:
    """True iff any head uses the vertical-slash selector — so the caller precomputes its ONE global
    index. Uniform: ``selector == "vslash"``. Per-head: ``"vslash"`` appears in ``head_selectors`` (M4)
    or as any ``HeadSpec.kind`` in ``head_specs`` (M5)."""
    if cfg.head_specs is not None:
        return any(sp.kind == "vslash" for sp in cfg.head_specs)
    if cfg.head_selectors is not None:
        return "vslash" in cfg.head_selectors
    return cfg.selector == "vslash"


def _select_keep_per_head(
    q: mx.array, k: mx.array, scale: float, cfg: XAttnConfig, q_offset: int,
    index: tuple[mx.array, mx.array] | None,
) -> tuple[mx.array, mx.array]:
    """Per-head selector dispatch → ``(keep, rank)`` ``[B,H,Tq,Tk]`` routing each head to its kind.

    ``cfg.head_selectors`` names each of the ``H`` query heads' selector kind. Computes each *distinct*
    kind's ``(keep, rank)`` for ALL heads (a bounded loop over the ≤3 selector KINDS present — never a
    per-head/per-token hot loop, rule 3), stacks them ``[n_kind,B,H,Tq,Tk]``, and selects per head the
    slice from that head's assigned kind with one ``take_along_axis`` over a ``[1,B,H,Tq,Tk]`` index. So
    head ``h``'s keep is byte-identical to the uniform keep for ``head_selectors[h]`` — the per-head path
    adds no new selection math, only routing. Vertical-slash's global ``index`` (threaded by the caller
    over the whole sequence) is forwarded to the vslash sub-selection so per-head gather == mask.
    """
    h = q.shape[1]
    if len(cfg.head_selectors) != h:
        raise ValueError(
            f"head_selectors has {len(cfg.head_selectors)} entries but attention has {h} query heads"
        )
    kinds = sorted(set(cfg.head_selectors))            # ≤3 distinct kinds; deterministic order
    keeps: list[mx.array] = []
    ranks: list[mx.array] = []
    for s in kinds:                                    # bounded over selector KINDS, not heads
        sub = replace(cfg, head_selectors=None, selector=s)
        kp, rk = select_keep(q, k, scale, sub, q_offset, index if s == "vslash" else None)
        keeps.append(kp)
        ranks.append(rk.astype(mx.float32))
    stacked_keep = mx.stack(keeps, axis=0)             # [n_kind, B, H, Tq, Tk]
    stacked_rank = mx.stack(ranks, axis=0)
    pick = mx.array([kinds.index(s) for s in cfg.head_selectors], dtype=mx.int32)  # [H] kind index
    ind = mx.broadcast_to(pick.reshape(1, 1, h, 1, 1), (1, *stacked_keep.shape[1:]))
    keep = mx.take_along_axis(stacked_keep, ind, axis=0)[0]
    rank = mx.take_along_axis(stacked_rank, ind, axis=0)[0]
    return keep, rank


def _select_keep_per_head_specs(
    q: mx.array, k: mx.array, scale: float, cfg: XAttnConfig, q_offset: int,
    index: tuple[mx.array, mx.array] | None,
) -> tuple[mx.array, mx.array]:
    """Per-head **param** dispatch → ``(keep, rank)`` ``[B,H,Tq,Tk]`` routing each head to its spec.

    Generalizes :func:`_select_keep_per_head` (M4, kind-only) to per-head *params*: ``cfg.head_specs``
    is a length-``H`` tuple of :class:`HeadSpec` (kind + params). Computes each *distinct* spec's
    ``(keep, rank)`` for ALL heads (a bounded loop over the DISTINCT specs present — the search-grid
    size, never a per-head/per-token hot loop, rule 3), stacks them ``[n_spec,B,H,Tq,Tk]``, and selects
    per head the slice from that head's spec with one ``take_along_axis``. So head ``h``'s keep is
    byte-identical to the uniform keep for ``head_specs[h]``'s (kind, params) — pure routing, no new
    selection math. The vslash global ``index`` (param-independent masses, threaded by the caller) is
    forwarded to vslash sub-selections; each vslash spec cuts its OWN top-``vert``/``slash`` from the
    shared masses (so two heads can keep different vert/slash) while per-head gather == mask.
    """
    h = q.shape[1]
    specs = cfg.head_specs
    if len(specs) != h:
        raise ValueError(f"head_specs has {len(specs)} entries but attention has {h} query heads")
    uniq: list[HeadSpec] = []
    for sp in specs:                                   # distinct specs in first-seen (deterministic) order
        if sp not in uniq:
            uniq.append(sp)
    keeps: list[mx.array] = []
    ranks: list[mx.array] = []
    for sp in uniq:                                    # bounded over DISTINCT specs, not heads
        sub = replace(cfg, head_specs=None, head_selectors=None, selector=sp.kind,
                      threshold=sp.threshold, local=sp.local, vert=sp.vert, slash=sp.slash)
        kp, rk = select_keep(q, k, scale, sub, q_offset, index if sp.kind == "vslash" else None)
        keeps.append(kp)
        ranks.append(rk.astype(mx.float32))
    stacked_keep = mx.stack(keeps, axis=0)             # [n_spec, B, H, Tq, Tk]
    stacked_rank = mx.stack(ranks, axis=0)
    pick = mx.array([uniq.index(sp) for sp in specs], dtype=mx.int32)   # [H] spec index per head
    ind = mx.broadcast_to(pick.reshape(1, 1, h, 1, 1), (1, *stacked_keep.shape[1:]))
    keep = mx.take_along_axis(stacked_keep, ind, axis=0)[0]
    rank = mx.take_along_axis(stacked_rank, ind, axis=0)[0]
    return keep, rank


def select_keep(
    q: mx.array, k: mx.array, scale: float, cfg: XAttnConfig, q_offset: int = 0,
    index: tuple[mx.array, mx.array] | None = None,
) -> tuple[mx.array, mx.array]:
    """Selector dispatch → ``(keep, rank)`` block masks ``[B, H, Tq, Tk]`` for ``cfg.selector``.

    ``keep`` is the bool block keep-mask; ``rank`` is the per-block priority used to order the kept
    blocks when the gather budget cap binds (higher = gathered first). The selectors share the
    downstream execution (additive mask / block gather) verbatim — they differ ONLY here:

    * ``"xattn"`` — antidiagonal block scores → nucleus selection; ``rank`` is the score itself, so
      a binding budget keeps the highest-scoring blocks. This branch is **byte-for-byte the
      pre-selector path** (``select_blocks(block_scores(…))``), so XAttention quality is unchanged.
    * ``"ashape"`` — positional sink + ``local``-block window; ``rank`` is block recency (key-block
      index ``j``), so a binding budget keeps the nearest-to-diagonal local blocks.
    * ``"vslash"`` — global vertical key-blocks ∪ slash block-offset bands from a last-block probe.
      The probe (:func:`vertical_slash_index`) returns param-independent ``(key_mass, slash_mass)``; the
      top-``cfg.vert`` / top-``cfg.slash`` cut is applied HERE, so a per-head vslash spec can read the one
      shared probe yet keep its OWN vert/slash. The masses are GLOBAL, so the caller computes them once
      over the whole sequence and threads them in via ``index`` (every chunked gather call re-cuts the
      same masses — that is what keeps gather == mask). ``rank`` is the per-key-block mass. When ``index``
      is None the masses are computed here from ``(q, k)`` (the whole-sequence case).

    ``q_offset`` is the global block index of the first query row (the chunked gather path passes its
    chunk origin so causality/window use global positions); ``q_offset=0`` is the whole-sequence case.

    When ``cfg.head_specs`` is set (M5), dispatch is **per head with per-head params**: each head uses
    its own ``HeadSpec``'s (kind, params) selection (:func:`_select_keep_per_head_specs`). Else when
    ``cfg.head_selectors`` is set (M4), dispatch is per head by *kind* (shared params,
    :func:`_select_keep_per_head`): the returned ``keep[:, h]`` equals the uniform keep for
    ``head_selectors[h]``. Both None (default) is the uniform path below, unchanged.
    """
    if cfg.head_specs is not None:
        return _select_keep_per_head_specs(q, k, scale, cfg, q_offset, index)
    if cfg.head_selectors is not None:
        return _select_keep_per_head(q, k, scale, cfg, q_offset, index)
    if cfg.selector == "ashape":
        bsz, h = q.shape[0], q.shape[1]
        tq = (q.shape[2] + cfg.block - 1) // cfg.block
        tk = (k.shape[2] + cfg.block - 1) // cfg.block
        keep = mx.broadcast_to(ashape_keep(tq, tk, q_offset, cfg.local)[None, None], (bsz, h, tq, tk))
        rank = mx.broadcast_to(mx.arange(tk, dtype=mx.float32)[None, None, None, :], (bsz, h, tq, tk))
        return keep, rank
    if cfg.selector == "vslash":
        bsz, h = q.shape[0], q.shape[1]
        blk = cfg.block
        tq = (q.shape[2] + blk - 1) // blk
        tk = (k.shape[2] + blk - 1) // blk
        if index is None:
            index = vertical_slash_index(q, k, scale, cfg)
        key_mass, slash_mass = index
        # Cut the top-``cfg.vert`` vertical key-blocks (sink block 0 excluded — force-kept) and the
        # top-``cfg.slash`` block-offset bands (offset 0, the diagonal, excluded — force-kept) HERE from
        # the shared, param-independent masses — so a per-head vslash spec applies its OWN vert/slash to
        # the one global probe (the masses are byte-identical across heads; only the cut differs).
        nkb = key_mass.shape[-1]                                # global # key blocks
        kk = mx.arange(nkb)[None, None, :]
        vscore = mx.where(kk == 0, NEG_INF, key_mass)
        vrank = mx.argsort(mx.argsort(-vscore, axis=-1), axis=-1)   # rank of each block (0 = largest)
        vert_keep = vrank < min(cfg.vert, nkb)                  # [B,H,nkb] bool
        noff = slash_mass.shape[-1]                             # global # block-offsets
        oo = mx.arange(noff)[None, None, :]
        sscore = mx.where(oo == 0, NEG_INF, slash_mass)
        srank = mx.argsort(mx.argsort(-sscore, axis=-1), axis=-1)
        slash_keep = srank < min(cfg.slash, noff)              # [B,H,noff] bool
        i = mx.arange(q_offset, q_offset + tq)[:, None]
        j = mx.arange(tk)[None, :]
        causal = j <= i
        off = mx.clip((i - j).astype(mx.int32), 0, noff - 1)    # [tq,tk] block-offset
        slsh = mx.take(slash_keep, off, axis=-1)                # [B,H,tq,tk] slash band membership
        vert = vert_keep[:, :, None, :tk]                       # [B,H,1,tk] broadcast over query rows
        forced = (j == i) | (j == 0)                            # diagonal (local) + sink (block 0)
        keep = (vert | slsh | forced) & causal                 # [B,H,tq,tk]
        rank = mx.broadcast_to(key_mass[:, :, None, :tk], (bsz, h, tq, tk))
        return keep, rank
    scores = block_scores(q, k, scale, cfg.block, cfg.stride)
    return select_blocks(scores, cfg.threshold, q_offset), scores


def assign_head_selectors(
    errors: mx.array, cand_kinds: list[str], tol: float
) -> tuple[str, ...]:
    """Offline per-head pattern assignment → a ``head_selectors`` tuple for :class:`XAttnConfig`.

    ``errors`` ``[C, H]`` is each candidate selector's per-head approximation error vs dense attention,
    with candidate rows ordered **cheapest-kernel → most-accurate** (``cand_kinds[c]`` names row ``c``'s
    selector kind). For each head, route it to the *cheapest* candidate whose error ≤ ``tol`` (the
    MInference principle — the lightest pattern that still recalls that head's attention); if none
    qualifies, fall back to the last (most-accurate) candidate. Pure / positional (no model state) so
    the policy is unit-testable; the per-head ``errors`` are measured offline on a calibration forward
    (dense inputs) by the ppl harness. ``mx.argmax`` over the boolean within-tol column returns the
    FIRST (cheapest) qualifying candidate.
    """
    c = errors.shape[0]
    within = errors <= tol                             # [C, H]
    any_ok = mx.any(within, axis=0)                    # [H]
    first_ok = mx.argmax(within.astype(mx.int32), axis=0)   # [H] first True row (0 if none → fallback)
    choice = mx.where(any_ok, first_ok, c - 1).astype(mx.int32)
    return tuple(cand_kinds[int(i)] for i in choice.tolist())


def assign_head_specs(
    errors: mx.array, costs: mx.array, candidates: list[HeadSpec], budget: float
) -> tuple[HeadSpec, ...]:
    """Kernel-aware FLOP-budgeted per-head search → a ``head_specs`` tuple for :class:`XAttnConfig`.

    The **dual** of :func:`assign_head_selectors`: instead of *cheapest within an error tol*, pick per
    head the **most accurate** candidate whose kernel-aware cost is within a FLOP ``budget`` (MInference's
    actual setup — the best pattern+params a head can afford). ``errors`` ``[C, H]`` is each candidate's
    per-head approximation error vs dense; ``costs`` ``[C]`` is each candidate's kernel-aware cost (e.g.
    measured average kept blocks per query + a per-kind selection-kernel constant), candidates ordered
    **cheapest → most-accurate**. For each head: among candidates with ``cost ≤ budget`` choose the
    minimum-error one (ties → the cheaper, since rows ascend in cost); if none is affordable (budget below
    the cheapest cost), fall back to the globally cheapest candidate. Pure / positional (no model state)
    so the policy is unit-testable; the per-head ``errors`` and ``costs`` are measured offline on a
    calibration forward (dense inputs) by the ppl harness.
    """
    affordable = (costs <= budget)[:, None]                 # [C, 1] (cost is head-independent)
    any_aff = mx.any(affordable, axis=0)                    # [1] → broadcasts over heads
    big = mx.max(errors) + 1.0                              # push unaffordable candidates past every error
    score = mx.where(affordable, errors, big)              # [C, H]
    acc = mx.argmin(score, axis=0)                          # [H] most-accurate affordable candidate
    cheap = mx.argmin(costs)                                # [] globally cheapest (the no-budget fallback)
    choice = mx.where(any_aff, acc, cheap).astype(mx.int32)  # [H]
    return tuple(candidates[int(i)] for i in choice.tolist())


def additive_mask(
    keep: mx.array, q_len: int, kv_len: int, block: int, dtype: mx.Dtype
) -> mx.array:
    """Expand block keep-mask to a token additive mask ``[B,H,q_len,kv_len]`` (incl. causal)."""
    block_add = mx.where(keep, mx.array(0.0, dtype), mx.array(NEG_INF, dtype))  # [B,H,Tq,Tk]
    expanded = mx.repeat(mx.repeat(block_add, block, axis=-2), block, axis=-1)
    expanded = expanded[:, :, :q_len, :kv_len]

    a = mx.arange(q_len)[:, None]
    b = mx.arange(kv_len)[None, :]
    causal = mx.where(b <= a, mx.array(0.0, dtype), mx.array(NEG_INF, dtype))  # [q_len,kv_len]
    return expanded + causal[None, None]


def sparse_prefill_mask(
    q: mx.array, k: mx.array, scale: float, cfg: XAttnConfig
) -> mx.array:
    """End-to-end: antidiagonal scores → nucleus select → additive token mask ``[B,H,T,S]``.

    The mask itself is O(T²·H) — fine at the moderate context of the quality gate, but it
    will OOM at long context. Guarded by ``cfg.max_alloc_gb`` (fail loud); use
    ``gather=True`` for long context.
    """
    bsz, h, t = q.shape[0], q.shape[1], q.shape[2]
    s = k.shape[2]
    gb = bsz * h * t * s * 3 / 1e9  # bool mask + bf16 additive mask ≈ 3 bytes/elem
    if gb > cfg.max_alloc_gb:
        raise MemoryError(
            f"sparse_prefill_mask [B,H,T,S] ~= {gb:.1f} GB (T={t}, heads={h}) exceeds "
            f"max_alloc_gb={cfg.max_alloc_gb}; enable XAttnConfig.gather=True for long context."
        )
    # vertical-slash (uniform or per-head) needs ONE global pattern (from the last query block's probe);
    # xattn/ashape select locally so they pass index=None and select inside select_keep.
    index = vertical_slash_index(q, k, scale, cfg) if _uses_vslash(cfg) else None
    keep, _ = select_keep(q, k, scale, cfg, 0, index)
    return additive_mask(keep, t, s, cfg.block, q.dtype)


def _chunk_blocks(bsz: int, h: int, blk: int, d: int, vd: int, cap: int, max_gb: float) -> int:
    """Largest #query-blocks whose per-chunk peak (mask + gathered K/V) fits in ``max_gb``.

    Conservative (1.5×): if even a single query block can't fit, raise — fail loud, no OOM.
    """
    per_block = 1.5 * bsz * h * cap * blk * (blk * 3 + d * 2 + vd * 2)  # mask + k_sel + v_sel bytes
    if per_block > max_gb * 1e9:
        raise MemoryError(
            f"one query block needs ~{per_block / 1e9:.1f} GB (heads={h}, max_kept={cap} x {blk}) "
            f"> max_alloc_gb={max_gb}. Lower XAttnConfig.budget. Refusing to allocate."
        )
    return max(1, int(max_gb * 1e9 // per_block))


def gather_sparse_attention(
    q: mx.array, k: mx.array, v: mx.array, scale: float, cfg: XAttnConfig
) -> mx.array:
    """Block-gathered sparse prefill attention → ``[B,H,T,Vd]`` (the long-context speed path).

    For each query block, gather only its selected K/V blocks and attend over those —
    O(T·max_kept·block), not O(T²). Output-equivalent to ``sparse_prefill_mask`` + dense
    SDPA (dropped blocks contribute zero either way). Query blocks are processed in chunks
    sized to fit ``cfg.max_alloc_gb``, and **each chunk is materialized + evaluated on its
    own**, so peak memory is one chunk — never the whole O(T²) mask. The selection is also
    chunked (each chunk scores only its query blocks against keys up to its causal horizon),
    so nothing global is O((T/blk)²) either. The chunk loop is the sanctioned coarse,
    bounded chunked-prefill loop — not a per-token/per-block hot loop.

    Per-head "fold" (``cfg.grouped_gather``): a per-head assignment otherwise sizes the gather by ONE
    global ``max_kept`` = the densest head's, so a few dense heads slow every head. With
    ``grouped_gather=True`` the heads are partitioned by distinct spec and each group gathered at its own
    ``max_kept`` (:func:`_gather_grouped_per_head`) — output-equivalent, but the cheap-pattern heads run
    cheap. Default True (**GRADUATED** — equivalence proven bit-exact by ``internlm2_grouped_gather_test``,
    so rule 4 is satisfied); pass ``grouped_gather=False`` for the naive single-``max_kept`` path below.
    """
    if cfg.grouped_gather and (cfg.head_specs is not None or cfg.head_selectors is not None):
        return _gather_grouped_per_head(q, k, v, scale, cfg)
    bsz, h, t, d = q.shape
    vd = v.shape[-1]
    blk = cfg.block
    qp, kp, vp = _pad_to(q, blk, 2), _pad_to(k, blk, 2), _pad_to(v, blk, 2)
    tp = qp.shape[2]
    nb = tp // blk  # number of blocks (query == key, prefill)

    kb = kp.reshape(bsz, h, nb, blk, d)
    vb = vp.reshape(bsz, h, nb, blk, vd)
    cap = nb if cfg.budget is None else min(cfg.budget, nb)
    cb = _chunk_blocks(bsz, h, blk, d, vd, cap, cfg.max_alloc_gb)  # query blocks per chunk
    inf = mx.array(float("inf"))

    # vertical-slash builds ONE global pattern from the last query block's probe (over the *unpadded*
    # q/k, so the probe is the last REAL block) and applies it to every chunk — computed once here,
    # not per chunk, so all chunks select identically (that is what keeps gather == mask). xattn and
    # ashape select locally per chunk (index stays None). Per-head: fires when ANY head uses vslash.
    index = vertical_slash_index(q, k, scale, cfg) if _uses_vslash(cfg) else None

    outs: list[mx.array] = []
    for c0 in range(0, nb, cb):
        c1 = min(c0 + cb, nb)  # query blocks [c0,c1); causal horizon = key blocks [0,c1)
        m = c1 - c0
        q_chunk = qp[:, :, c0 * blk : c1 * blk, :]  # [B,H,m*blk,d]

        keep, rank = select_keep(q_chunk, kp[:, :, : c1 * blk, :], scale, cfg, c0, index)  # [B,H,m,c1]
        counts = mx.sum(keep.astype(mx.int32), axis=-1)  # [B,H,m]
        max_k = min(int(mx.max(counts).item()), cap)
        capped = mx.minimum(counts, max_k)  # [B,H,m]

        ig = mx.arange(c0, c1)[:, None]
        jg = mx.arange(c1)[None, :]
        forced = (jg == ig) | (jg == 0)  # [m,c1] diagonal (global) + sink
        ord_score = mx.where(forced, inf, mx.where(keep, rank, -inf))
        idx = mx.argsort(-ord_score, axis=-1)[..., :max_k].astype(mx.int32)  # [B,H,m,max_k]
        valid = mx.arange(max_k)[None, None, None, :] < capped[..., None]  # [B,H,m,max_k]

        # gather selected blocks from the first c1 key/value blocks
        gi = mx.broadcast_to(idx.reshape(bsz, h, m * max_k)[..., None, None], (bsz, h, m * max_k, blk, d))
        k_sel = mx.take_along_axis(kb[:, :, :c1], gi, axis=2).reshape(bsz, h, m, max_k * blk, d)
        gi_v = mx.broadcast_to(idx.reshape(bsz, h, m * max_k)[..., None, None], (bsz, h, m * max_k, blk, vd))
        v_sel = mx.take_along_axis(vb[:, :, :c1], gi_v, axis=2).reshape(bsz, h, m, max_k * blk, vd)

        kc = mx.arange(blk)
        abs_k = (idx[..., None] * blk + kc[None, None, None, None, :]).reshape(bsz, h, m, max_k * blk)
        valid_k = mx.broadcast_to(valid[..., None], (bsz, h, m, max_k, blk)).reshape(bsz, h, m, max_k * blk)
        abs_q = mx.arange(c0, c1)[:, None] * blk + mx.arange(blk)[None, :]  # [m,blk] global positions
        allowed = valid_k[:, :, :, None, :] & (abs_k[:, :, :, None, :] <= abs_q[None, None, :, :, None])
        mask = mx.where(allowed, mx.array(0.0, q.dtype), mx.array(NEG_INF, q.dtype))

        b2 = bsz * h * m
        out_c = mx.fast.scaled_dot_product_attention(
            q_chunk.reshape(b2, 1, blk, d),
            k_sel.reshape(b2, 1, max_k * blk, d),
            v_sel.reshape(b2, 1, max_k * blk, vd),
            scale=scale,
            mask=mask.reshape(b2, 1, blk, max_k * blk),
        ).reshape(bsz, h, m, blk, vd)
        mx.eval(out_c)  # materialize + free this chunk's mask / gathered K/V before the next
        outs.append(out_c)

    out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)  # [B,H,nb,blk,vd]
    return out.reshape(bsz, h, tp, vd)[:, :, :t, :]


def _gather_grouped_per_head(
    q: mx.array, k: mx.array, v: mx.array, scale: float, cfg: XAttnConfig
) -> mx.array:
    """Per-head-GROUPED block-gather (the M9-speed "fold") → ``[B,H,T,Vd]``, output-equivalent to the
    naive per-head :func:`gather_sparse_attention` but each distinct-spec head-group runs at its OWN
    ``max_kept``.

    The naive per-head gather sizes its work by ONE global ``max_kept`` = the densest head's kept-block
    count (it builds a rectangular ``[B,H,m,max_kept,blk]`` gather), so a mixed assignment makes the cheap
    heads pay the dense head's budget. Here the heads are partitioned by their DISTINCT spec (``head_specs``
    ⇒ HeadSpec, else ``head_selectors`` ⇒ kind); each group is sliced out (``mx.take`` over the head axis),
    gathered with that group's UNIFORM selector (its own tight ``max_kept``), and the group outputs are
    un-permuted back to the original head order. Bit-equivalent: head ``i`` attends exactly its kept blocks
    either way (the naive path's extra ``-inf`` mask slots contribute nothing to the softmax). The loop is
    bounded over DISTINCT specs (≤ the search-grid size, never per-head/per-token — rule 3).
    """
    h = q.shape[1]
    specs = cfg.head_specs
    if specs is not None:
        keys: list = list(specs)
        uniforms = {sp: replace(cfg, head_specs=None, head_selectors=None, grouped_gather=False,
                                selector=sp.kind, threshold=sp.threshold, local=sp.local,
                                vert=sp.vert, slash=sp.slash) for sp in set(specs)}
    else:
        keys = list(cfg.head_selectors)
        uniforms = {kd: replace(cfg, head_specs=None, head_selectors=None, grouped_gather=False,
                                selector=kd) for kd in set(keys)}
    if len(keys) != h:
        raise ValueError(f"grouped_gather: per-head config has {len(keys)} entries but {h} query heads")
    groups: dict = {}                                          # distinct spec → original head indices
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    order: list[int] = []
    outs: list[mx.array] = []
    for key, idxs in groups.items():                          # bounded over DISTINCT specs, not heads
        hi = mx.array(idxs, dtype=mx.int32)
        outs.append(gather_sparse_attention(mx.take(q, hi, axis=1), mx.take(k, hi, axis=1),
                                            mx.take(v, hi, axis=1), scale, uniforms[key]))
        order.extend(idxs)
    out_perm = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)   # [B,H,T,Vd] grouped order
    inv = mx.argsort(mx.array(order, dtype=mx.int32))          # restore original head order
    return mx.take(out_perm, inv, axis=1)
