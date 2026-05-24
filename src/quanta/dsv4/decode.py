"""DeepSeek-V4 single-token (decode) attention — incremental, output-equivalent to prefill.

This is the decode half of task #77: the per-token stepper for all three DSV4 attention regimes,
plus the cache state they need. It is **parity-first** — every step is numerically equivalent to the
prefill paths (:func:`quanta.dsv4.attention.attention_dense`,
:func:`quanta.dsv4.indexer.attention_compressed`) evaluated at the same absolute position. Wherever
possible it *reuses* the prefill helpers (``project_qkv``, ``sdpa_window_sink``, ``output_proj``,
``compressor_prefill``'s exact pooling arithmetic) rather than reimplementing the math.

The three regimes (selected by ``cfg.compress_ratio(layer_id)``):

* **ratio 0** — pure sliding-window (``attention_dense``): append the new latent KV to the per-layer
  stream and run the windowed-sink SDPA for the single query.
* **ratio 128** — compressed, no indexer: window KV **plus** all causally-visible compressed KV.
* **ratio 4** — compressed + Lightning-Indexer (DSA): window KV **plus** the top-``index_topk``
  compressed KV selected by the indexer.

**Compressor decode state machine.** A compressed token ``c`` pools the ``ratio`` positions
``[c*ratio,(c+1)*ratio)`` (overlap regime ratio==4 also pools the previous window's ``ratio``
positions) and becomes causally visible to a query at absolute position ``i`` iff
``c < (i+1)//ratio`` — i.e. exactly when its window has fully completed at or before ``i``. So decode
keeps a small **ring of the last ``coff*ratio`` raw hidden vectors**; each time a window boundary is
crossed (``(offset+1) % ratio == 0``) it pools one new compressed token (and, on ratio-4 layers, one
indexer compressed token) from the ring and appends it to the compressed cache — making it visible to
the very query that closed the window, matching prefill's causal count ``(P+1)//ratio``.

No Python loops on the compute path: the only loop is the caller's decode-step loop. All cache state
is tiny (the ring is bounded by ``coff*ratio``; the latent/compressed streams grow with context like
the Kimi/Nemotron caches). ``truncate(length)`` makes the state bit-identical to having only ever
fed ``length`` tokens (speculative-decode rollback). Gated model-free in
``parity/dsv4_decode_attn_test.py``.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.dsv4.attention import (
    _rms_w,
    output_proj,
    project_qkv,
    rope_partial,
    sdpa_window_sink,
)
from quanta.dsv4.config import DeepSeekV4Config

_NEG = -1e30


# --- per-layer decode cache --------------------------------------------------
class _LayerCache:
    """Per-layer decode state. Holds the growing latent KV stream (all regimes), and — on compressed
    layers — the compressed KV stream(s) plus the raw-hidden ring that drives the pooling state
    machine. ``offset`` is the number of tokens already consumed by this layer."""

    __slots__ = ("kv", "ckv", "ikv", "ring", "ratio")

    def __init__(self) -> None:
        self.kv: mx.array | None = None     # [B,S,head_dim]  roped latent KV (single MQA head)
        self.ckv: mx.array | None = None    # [B,ncomp,head_dim] compressed KV (ratio 4/128 layers)
        self.ikv: mx.array | None = None    # [B,ncomp,index_head_dim] indexer compressed KV (ratio 4)
        self.ring: mx.array | None = None   # [B,r,dim] raw hidden of the last ``coff*ratio`` positions
        self.ratio: int = 0                 # compression ratio (0 = dense layer; set on first append)

    def n_comp(self) -> int:
        return 0 if self.ckv is None else self.ckv.shape[1]


class DSV4Cache:
    """Decode cache for a DSV4 attention stack: one :class:`_LayerCache` per attention block.

    Mirrors the update/``truncate``/``offset`` ergonomics of :class:`quanta.cache.MLACache` and
    :class:`quanta.nemotron.attention.KVCache`. ``offset`` reports the shared decode position (every
    attention layer advances in lock-step), derived from the latent stream length so it is exact and
    survives ``truncate``.
    """

    def __init__(self, n_layers: int) -> None:
        self.layers: list[_LayerCache] = [_LayerCache() for _ in range(n_layers)]

    def __getitem__(self, i: int) -> _LayerCache:
        return self.layers[i]

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def offset(self) -> int:
        """Number of tokens already cached (positions consumed). 0 before the first append.

        Every attention layer advances in lock-step, so any populated layer reports the same value;
        we read the first populated one (robust to a cache that drives only a subset of layers)."""
        for lc in self.layers:
            if lc.kv is not None:
                return lc.kv.shape[1]
        return 0

    def truncate(self, length: int) -> None:
        """Roll every layer back to exactly the state after consuming ``length`` tokens (drop rejected
        speculative drafts). The latent/compressed streams slice cleanly (per-position storage). The
        compressed stream is kept to ``length//ratio`` tokens — the count of windows that had completed
        by position ``length-1``, matching prefill — and the raw-hidden ring drops the rolled-back tail
        positions. Rollback must keep the ring deep enough to still reconstruct the next window's
        overlap (always true for spec-decode draft suffixes); a deeper rollback fails loudly (rule 6)
        rather than silently pooling a wrong window."""
        if length < 0:
            raise ValueError(f"truncate length {length} < 0")
        old = self.offset
        if length >= old:
            return
        drop = old - length                                      # rolled-back tail positions
        for lc in self.layers:
            if lc.kv is None:
                continue
            lc.kv = lc.kv[:, :length]
            if lc.ratio:                                         # compressed layer
                keep = length // lc.ratio                        # windows completed by position length-1
                lc.ckv = None if keep == 0 else lc.ckv[:, :keep]
                if lc.ikv is not None:
                    lc.ikv = None if keep == 0 else lc.ikv[:, :keep]
                # ring ends at ``old``; drop the rolled-back tail so it ends at ``length``.
                if length == 0:
                    lc.ring = None
                    continue
                lc.ring = lc.ring[:, :-drop] if drop < lc.ring.shape[1] else lc.ring[:, :0]
                # The next window to pool is ``keep`` (positions [keep*ratio,(keep+1)*ratio)); its
                # overlap ``prev`` window starts at (keep-1)*ratio. Guard that the trimmed ring (which
                # now ends at ``length``) still reaches that earliest needed raw position.
                need = max(0, (keep - (1 if _layer_overlap(lc) else 0)) * lc.ratio)
                if lc.ring.shape[1] < length - need:
                    raise ValueError(
                        f"truncate({length}) rolls back {drop} tokens — too deep to reconstruct the "
                        f"compressor window from the bounded raw-hidden ring (have "
                        f"{lc.ring.shape[1]} positions, need back to {need}). Spec-decode draft "
                        f"suffixes are within bounds; enlarge the ring for deeper rollback.")


def _layer_overlap(lc: _LayerCache) -> bool:
    """Whether the layer's compressor uses overlapping windows (ratio==4 ⟺ overlap; ratio-128 layers
    are non-overlapping)."""
    return lc.ratio == 4


# --- compressor pooling (one window) — bit-identical to compressor_prefill ----
def _project_window(ring_slice: mx.array, wkv: mx.array, wgate: mx.array
                    ) -> tuple[mx.array, mx.array]:
    """Per-position raw projections of a window's hidden vectors (float32, as ``compressor_prefill``):
    ``kv = x @ wkv.T``, ``score = x @ wgate.T``. ``ring_slice``: ``[B,ratio,dim]``."""
    xf = ring_slice.astype(mx.float32)
    return xf @ wkv.T, xf @ wgate.T


def _pool_one_window(cur: mx.array, prev: mx.array | None, ape: mx.array, norm_w: mx.array,
                     wkv: mx.array, wgate: mx.array, *, ratio: int, head_dim: int,
                     rope_head_dim: int, eps: float, cos_c: mx.array, sin_c: mx.array,
                     overlap: bool) -> mx.array:
    """Pool one completed window into a single compressed KV vector ``[B,1,head_dim]`` — the exact
    per-window arithmetic of :func:`quanta.dsv4.compressor.compressor_prefill` (gated-softmax pool +
    weighted RMSNorm + partial RoPE at the window-start position). ``cur``: this window's raw hidden
    ``[B,ratio,dim]``; ``prev``: the previous window's raw hidden ``[B,ratio,dim]`` (overlap only;
    ``None`` for window 0 or the non-overlap regime). ``cos_c``/``sin_c``: RoPE row at absolute
    position ``c*ratio`` (``[1,rope_head_dim/2]``)."""
    kv_cur, score_cur = _project_window(cur, wkv, wgate)         # [B,ratio,coff*hd]
    b = cur.shape[0]
    if overlap:
        d = head_dim
        # prev slots: window c-1 first-half projection; cur slots: window c second-half projection.
        if prev is None:                                         # window 0 — pad (kv=0, score=-inf)
            prev_kv = mx.zeros((b, ratio, d), dtype=kv_cur.dtype)
            prev_sc = mx.full((b, ratio, d), float("-inf"), dtype=score_cur.dtype)
        else:
            kv_prev, score_prev = _project_window(prev, wkv, wgate)
            prev_kv = kv_prev[..., :d]
            prev_sc = score_prev[..., :d] + ape[:, :d]           # ape added before overlap in prefill
        kv_win = mx.concatenate([prev_kv, kv_cur[..., d:]], axis=1)            # [B,2*ratio,hd]
        score_win = mx.concatenate([prev_sc, score_cur[..., d:] + ape[:, d:]], axis=1)
        pool_axis = 1
    else:
        kv_win = kv_cur                                          # [B,ratio,hd]
        score_win = score_cur + ape                             # ape [ratio, hd]
        pool_axis = 1
    pooled = mx.sum(kv_win * mx.softmax(score_win, axis=pool_axis), axis=pool_axis, keepdims=True)
    pooled = _rms_w(pooled, norm_w, eps)                         # [B,1,head_dim]
    return rope_partial(pooled, cos_c, sin_c, rope_head_dim)


def _maybe_pool(lc: _LayerCache, p: dict, cfg: DeepSeekV4Config, layer_id: int, ratio: int,
                offset: int, cos: mx.array, sin: mx.array) -> None:
    """If position ``offset`` closes a window, pool one main (and, on ratio-4 layers, one indexer)
    compressed token from the ring and append it to the cache. ``cos``/``sin`` are the full RoPE
    tables for ``[0, offset+1)``."""
    if (offset + 1) % ratio != 0:
        return
    c = offset // ratio                                          # window index just completed
    overlap = cfg.overlap(layer_id)
    cur = lc.ring[:, -ratio:]
    prev = lc.ring[:, -2 * ratio:-ratio] if (overlap and lc.ring.shape[1] >= 2 * ratio) else None
    cos_c, sin_c = cos[c * ratio:c * ratio + 1], sin[c * ratio:c * ratio + 1]

    cp = p["compressor"]
    ck = _pool_one_window(cur, prev, cp["ape"].astype(mx.float32), cp["norm"].astype(mx.float32),
                          cp["wkv"].astype(mx.float32), cp["wgate"].astype(mx.float32),
                          ratio=ratio, head_dim=cfg.head_dim, rope_head_dim=cfg.rope_head_dim,
                          eps=cfg.norm_eps, cos_c=cos_c, sin_c=sin_c, overlap=overlap)
    lc.ckv = ck if lc.ckv is None else mx.concatenate([lc.ckv, ck], axis=1)

    if cfg.has_indexer(layer_id):
        icp = p["indexer"]["compressor"]
        ik = _pool_one_window(cur, prev, icp["ape"].astype(mx.float32), icp["norm"].astype(mx.float32),
                              icp["wkv"].astype(mx.float32), icp["wgate"].astype(mx.float32),
                              ratio=4, head_dim=cfg.index_head_dim, rope_head_dim=cfg.rope_head_dim,
                              eps=cfg.norm_eps, cos_c=cos_c, sin_c=sin_c, overlap=True)
        lc.ikv = ik if lc.ikv is None else mx.concatenate([lc.ikv, ik], axis=1)


def _push_ring(lc: _LayerCache, x_t: mx.array, ratio: int, overlap: bool) -> None:
    """Append the new hidden vector to the raw-hidden ring, trimmed to the last ``coff*ratio``
    positions (the minimum needed to pool the next window and roll back within it)."""
    cap = (2 if overlap else 1) * ratio
    lc.ring = x_t if lc.ring is None else mx.concatenate([lc.ring, x_t], axis=1)
    if lc.ring.shape[1] > cap:
        lc.ring = lc.ring[:, -cap:]


# --- decode steppers ---------------------------------------------------------
def decode_step_dense(x_t: mx.array, p: dict, cfg: DeepSeekV4Config, layer_id: int,
                      cache: DSV4Cache, cos: mx.array, sin: mx.array, offset: int) -> mx.array:
    """One token through the **ratio-0** (pure sliding-window) regime. ``x_t``: ``[B,1,dim]``;
    ``p``: the loader ``attention(layer_id)`` dict; ``cos``/``sin``: the full RoPE tables for absolute
    positions ``[0, offset+1)`` (``offset`` is this token's absolute position). Returns ``[B,1,dim]``
    — the value ``attention_dense`` would return for this position."""
    lc = cache[layer_id]
    cs, sn = cos[offset:offset + 1], sin[offset:offset + 1]
    _, q, kv = project_qkv(x_t, p, cfg, cs, sn)                  # q [B,1,H,hd], kv [B,1,hd]
    lc.kv = kv if lc.kv is None else mx.concatenate([lc.kv, kv], axis=1)
    o = sdpa_window_sink(q.astype(mx.float32), lc.kv.astype(mx.float32),
                         p["attn_sink"].astype(mx.float32), cfg.attn_scale,
                         cfg.sliding_window, offset)
    return output_proj(o, p, cfg, cs, sn)


def decode_step_compressed(x_t: mx.array, p: dict, cfg: DeepSeekV4Config, layer_id: int,
                           cache: DSV4Cache, cos: mx.array, sin: mx.array, offset: int) -> mx.array:
    """One token through the **compressed** regime (ratio 4 with Lightning-Indexer / ratio 128
    without): append to the window, advance the compressor pooling state when a window fills, select
    the compressed tokens (indexer top-k for ratio 4, all-causal for ratio 128) and combine —
    output-equivalent to :func:`quanta.dsv4.indexer.attention_compressed` at this position. ``x_t``:
    ``[B,1,dim]``; ``cos``/``sin``: full RoPE tables for ``[0, offset+1)``. Returns ``[B,1,dim]``."""
    lc = cache[layer_id]
    ratio = cfg.compress_ratio(layer_id)
    lc.ratio = ratio
    cs, sn = cos[offset:offset + 1], sin[offset:offset + 1]
    qr, q, kv = project_qkv(x_t, p, cfg, cs, sn)

    # window latent KV + ring of raw hidden (drives pooling); pool if this token closes a window.
    lc.kv = kv if lc.kv is None else mx.concatenate([lc.kv, kv], axis=1)
    _push_ring(lc, x_t, ratio, cfg.overlap(layer_id))
    _maybe_pool(lc, p, cfg, layer_id, ratio, offset, cos, sin)

    qf = q.astype(mx.float32)
    kvf = lc.kv.astype(mx.float32)
    sink, scale = p["attn_sink"].astype(mx.float32), cfg.attn_scale

    # window scores: single query at abs pos ``offset`` attends keys in (offset-window, offset].
    sc = mx.einsum("bqhd,bsd->bqhs", qf, kvf) * scale            # [B,1,H,S]
    ki = mx.arange(lc.kv.shape[1])[None, :]
    win = (ki <= offset) & (ki > offset - cfg.sliding_window)    # [1,S]
    sc = sc + mx.where(win, 0.0, _NEG)[None, :, None, :]
    kv_all = kvf

    if lc.ckv is not None:
        ckv = lc.ckv.astype(mx.float32)
        ncomp = ckv.shape[1]
        if cfg.has_indexer(layer_id):
            sel = _decode_indexer_select(x_t, qr, lc, p["indexer"], cfg, cs, sn, ncomp)  # [1,1,ncomp]
        else:
            sel = mx.ones((1, 1, ncomp), dtype=mx.bool_)        # all cached are causally visible
        sc_c = mx.einsum("bqhd,btd->bqht", qf, ckv) * scale     # [B,1,H,ncomp]
        sc_c = sc_c + mx.where(sel, 0.0, _NEG)[:, :, None, :]
        sc = mx.concatenate([sc, sc_c], axis=-1)
        kv_all = mx.concatenate([kvf, ckv], axis=1)

    m = mx.max(sc, axis=-1, keepdims=True)
    ex = mx.exp(sc - m)
    denom = mx.sum(ex, axis=-1) + mx.exp(sink[None, None, :] - m[..., 0])
    o = mx.einsum("bqht,btd->bqhd", ex, kv_all) / denom[..., None]
    return output_proj(o, p, cfg, cs, sn)


def _decode_indexer_select(x_t: mx.array, qr: mx.array, lc: _LayerCache, idx_p: dict,
                           cfg: DeepSeekV4Config, cs: mx.array, sn: mx.array, ncomp: int) -> mx.array:
    """Boolean ``[1,1,ncomp]`` mask of compressed tokens the indexer selects for the single decode
    query (top-``index_topk`` by the Lightning-Indexer score). All cached ``ikv`` are causally valid
    (only window-completed tokens are appended), so this mirrors :func:`indexer_select` with the
    causal mask already satisfied."""
    inh, ihd, rd = cfg.index_n_heads, cfg.index_head_dim, cfg.rope_head_dim
    b = x_t.shape[0]
    qb = (qr @ idx_p["wq_b"].T).reshape(b, 1, inh, ihd)
    qb = rope_partial(qb, cs, sn, rd).astype(mx.float32)
    ikv = lc.ikv.astype(mx.float32)                             # [B,ncomp,index_head_dim]
    weights = (x_t @ idx_p["weights_proj"].T).astype(mx.float32) * (ihd ** -0.5 * inh ** -0.5)
    score = mx.einsum("bqhd,btd->bqht", qb, ikv)               # [B,1,inh,ncomp]
    score = (mx.maximum(score, 0.0) * weights[..., None]).sum(axis=2)   # [B,1,ncomp]
    k = min(cfg.index_topk, ncomp)
    if k >= ncomp:
        return mx.ones((b, 1, ncomp), dtype=mx.bool_)
    thr = mx.sort(score, axis=-1)[..., ncomp - k][..., None]   # k-th largest
    return score >= thr
