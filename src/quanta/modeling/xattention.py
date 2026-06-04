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

from dataclasses import dataclass

import mlx.core as mx

NEG_INF = float("-inf")


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
) -> tuple[mx.array, mx.array, mx.array]:
    """MInference vertical-slash index from an online probe of the LAST query block (MInference §3).

    Runs the *last* (real) query block's attention over all keys, then decomposes that probe two
    ways to pick a single GLOBAL block pattern that is then applied to every query block:

    * **vertical** — per-key-*column* mass summed over the probe queries, pooled to key blocks; the
      top-``cfg.vert`` key blocks are kept as vertical stripes (columns everyone attends). Sink block
      0 is excluded from the top-k (it is force-kept anyway), so the budget is spent on real columns.
    * **slash** — per-token-*offset* mass (query-pos − key-pos) summed over the probe, pooled to
      block-offsets; the top-``cfg.slash`` block-offsets are kept as diagonal bands. Offset 0 (the
      diagonal) is excluded (force-kept). A token-offset δ is binned to block-offset ``δ // block``.

    Returns ``(vert_keep [B,H,Tk] bool, slash_keep [B,H,Tq] bool, key_mass [B,H,Tk] float)`` — the two
    boolean selections plus the per-key-block probe mass, reused as the gather-budget priority. The
    probe is one ``[B,H,lp,S]`` attention matrix (``lp`` = last-block size ≤ ``block``); a plain
    ``q@kᵀ`` matmul (not ``mx.fast.sdpa``) because the *attention weights* are needed, not the output.
    Guarded by ``max_alloc_gb`` (fail loud, never OOM). Data-dependent but built ONCE from the last
    block — a prefill-only pattern (decode stays dense); the per-(query,key) execution mask the
    selection feeds is still strictly causal, so no future *content* leaks into earlier positions.
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
        raise MemoryError(
            f"vertical_slash probe [B,H,{lp},{s}] ~= {gb:.1f} GB exceeds max_alloc_gb="
            f"{cfg.max_alloc_gb}; the long-context probe needs key-chunking (not yet implemented)."
        )

    qpos = mx.arange(p0, p0 + lp)[:, None]        # [lp,1] abs query positions
    kpos = mx.arange(s)[None, :]                  # [1,S]
    sc = (q_last @ mx.swapaxes(k, -1, -2)) * scale            # [B,H,lp,S] raw probe scores
    sc = mx.where(kpos <= qpos, sc, NEG_INF)                  # causal
    a = mx.softmax(sc.astype(mx.float32), axis=-1)           # [B,H,lp,S] probe attention

    # vertical: column mass over probe queries → per-key-block; sink (block 0) out of the top-k.
    col = _pad_to(a.sum(axis=2), blk, axis=2)                # [B,H,tk*blk]
    key_mass = col.reshape(bsz, h, tk, blk).sum(axis=3)      # [B,H,tk]
    kk = mx.arange(tk)[None, None, :]
    vscore = mx.where(kk == 0, NEG_INF, key_mass)
    vrank = mx.argsort(mx.argsort(-vscore, axis=-1), axis=-1)  # rank of each block (0 = largest)
    vert_keep = vrank < min(cfg.vert, tk)                    # [B,H,tk] bool

    # slash: token-offset mass (qpos − kpos) → per block-offset; diagonal (offset 0) out of the top-k.
    # gather a[r, p0+r-δ] for each probe row r and offset δ — the offset-δ key of that row — then
    # sum over r. One gather + one sum (no per-token loop); invalid (non-causal) entries zeroed.
    n_off = t
    r = mx.arange(lp)[:, None]
    dl = mx.arange(n_off)[None, :]
    c_idx = (p0 + r - dl).astype(mx.int32)                   # [lp,n_off] key index for offset dl
    valid = (dl <= p0 + r) & (c_idx >= 0)                    # causal & in-range
    gi = mx.broadcast_to(mx.clip(c_idx, 0, s - 1)[None, None], (bsz, h, lp, n_off))
    g = mx.where(valid[None, None], mx.take_along_axis(a, gi, axis=-1), 0.0)  # [B,H,lp,n_off]
    slash_tok = _pad_to(g.sum(axis=2), blk, axis=2)          # [B,H,tq*blk]
    slash_off = slash_tok.reshape(bsz, h, tq, blk).sum(axis=3)   # [B,H,tq] per block-offset
    oo = mx.arange(tq)[None, None, :]
    sscore = mx.where(oo == 0, NEG_INF, slash_off)
    srank = mx.argsort(mx.argsort(-sscore, axis=-1), axis=-1)
    slash_keep = srank < min(cfg.slash, tq)                  # [B,H,tq] bool
    return vert_keep, slash_keep, key_mass


def select_keep(
    q: mx.array, k: mx.array, scale: float, cfg: XAttnConfig, q_offset: int = 0,
    index: tuple[mx.array, mx.array, mx.array] | None = None,
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
    * ``"vslash"`` — global vertical key-blocks ∪ slash block-offset bands from a last-block probe
      (:func:`vertical_slash_index`). The pattern is GLOBAL, so the caller computes it once over the
      whole sequence and threads it in via ``index`` (the chunked gather path reuses the same global
      index for every chunk — that is what keeps gather == mask). ``rank`` is the per-key-block probe
      mass. When ``index`` is None it is computed here from ``(q, k)`` (the whole-sequence case).

    ``q_offset`` is the global block index of the first query row (the chunked gather path passes its
    chunk origin so causality/window use global positions); ``q_offset=0`` is the whole-sequence case.
    """
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
        vert_keep, slash_keep, key_mass = index
        i = mx.arange(q_offset, q_offset + tq)[:, None]
        j = mx.arange(tk)[None, :]
        causal = j <= i
        off = mx.clip((i - j).astype(mx.int32), 0, slash_keep.shape[-1] - 1)   # [tq,tk] block-offset
        slsh = mx.take(slash_keep, off, axis=-1)                # [B,H,tq,tk] slash band membership
        vert = vert_keep[:, :, None, :tk]                       # [B,H,1,tk] broadcast over query rows
        forced = (j == i) | (j == 0)                            # diagonal (local) + sink (block 0)
        keep = (vert | slsh | forced) & causal                 # [B,H,tq,tk]
        rank = mx.broadcast_to(key_mass[:, :, None, :tk], (bsz, h, tq, tk))
        return keep, rank
    scores = block_scores(q, k, scale, cfg.block, cfg.stride)
    return select_blocks(scores, cfg.threshold, q_offset), scores


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
    # vertical-slash needs ONE global pattern (from the last query block's probe); xattn/ashape
    # select locally so they pass index=None and select inside select_keep.
    index = vertical_slash_index(q, k, scale, cfg) if cfg.selector == "vslash" else None
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
    """
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
    # ashape select locally per chunk (index stays None).
    index = vertical_slash_index(q, k, scale, cfg) if cfg.selector == "vslash" else None

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
