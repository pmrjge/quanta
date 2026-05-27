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

from quanta.cache_quant import dequantize_last_axis, quantize_last_axis
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
    machine. ``offset`` is the number of tokens already consumed by this layer.

    Two storage modes for ``kv`` (latent KV, the big stream) and ``ckv`` (compressed KV) — #123:

    * ``quantized=True`` (default since #133 / #123): the streams are stored as **affine int8**
      per-token, per-group over ``head_dim`` (last axis) via :mod:`quanta.cache_quant`.
      The ``kv`` / ``ckv`` properties dequantize to bf16 on read so the SDPA path is unchanged.
      Steady-state memory drops from 16 bpp → ~8.25 bpp; pays a per-step dequant cost (the
      same trade as the Kimi MLA cache since #47 and the GLM/MiniMax/Qwen3.5 caches since #122).
    * ``quantized=False``: bf16 streams (the historical mode; kept for parity gates).

    ``ikv`` (indexer compressed KV) stays bf16 in both modes — it's the smaller ratio-4 stream
    and is matmul'd into a top-k selection mask rather than read by SDPA, so the dequant overhead
    is not worth the modest memory it would save.
    """

    __slots__ = ("ring", "ratio", "quantized", "group_size", "max_rollback",
                 "_kv_bf16", "_kv_q", "_kv_s", "_kv_b",
                 "_ckv_bf16", "_ckv_q", "_ckv_s", "_ckv_b",
                 "ikv")

    def __init__(self, *, quantized: bool = True, group_size: int = 128,
                 max_rollback: int = 1) -> None:
        self.quantized = quantized
        self.group_size = group_size
        # max_rollback enlarges the raw-hidden ring so :meth:`DSV4Cache.truncate` can drop more than
        # one draft suffix at once (k≥2 chained spec-decode in :mod:`quanta.dsv4.spec`). Default 1
        # = the k=1 spec ceiling — bigger only adds a few extra dim-sized rows per compressed layer.
        self.max_rollback = max(1, int(max_rollback))
        # bf16 streams (always-on for ikv; used for kv/ckv when ``quantized=False``)
        self._kv_bf16: mx.array | None = None
        self._ckv_bf16: mx.array | None = None
        self.ikv: mx.array | None = None    # [B,ncomp,index_head_dim] indexer compressed KV (ratio 4)
        # int8 codes + per-group scales/biases (when ``quantized=True``); appended along seq axis
        # so the trio still shares the leading [B, ncomp_or_S] prefix.
        self._kv_q: mx.array | None = None
        self._kv_s: mx.array | None = None
        self._kv_b: mx.array | None = None
        self._ckv_q: mx.array | None = None
        self._ckv_s: mx.array | None = None
        self._ckv_b: mx.array | None = None
        # state-machine fields
        self.ring: mx.array | None = None   # [B,r,dim] raw hidden of the last ``coff*ratio`` positions
        self.ratio: int = 0                 # compression ratio (0 = dense layer; set on first append)

    # --- length probes that avoid dequantizing just to read a shape ----------
    def kv_length(self) -> int:
        """Number of tokens in the latent KV stream (== ``self.kv.shape[1]`` had it been bf16)."""
        if self._kv_q is not None:
            return self._kv_q.shape[1]
        if self._kv_bf16 is not None:
            return self._kv_bf16.shape[1]
        return 0

    def n_comp(self) -> int:
        if self._ckv_q is not None:
            return self._ckv_q.shape[1]
        if self._ckv_bf16 is not None:
            return self._ckv_bf16.shape[1]
        return 0

    # --- KV / CKV access (bf16 view; int8 codes are the source of truth) -----
    @property
    def kv(self) -> mx.array | None:
        if not self.quantized:
            return self._kv_bf16
        if self._kv_q is None:
            return None
        return dequantize_last_axis(self._kv_q, self._kv_s, self._kv_b,
                                    self.group_size, dtype=mx.bfloat16)

    @kv.setter
    def kv(self, value: mx.array | None) -> None:
        """Re-store the FULL stream. Used by :meth:`DSV4Cache.truncate` and a few tests; the hot
        decode path uses :meth:`append_kv` so it only quantizes the new chunk."""
        if value is None:
            self._kv_bf16 = None
            self._kv_q = self._kv_s = self._kv_b = None
            return
        if not self.quantized:
            self._kv_bf16 = value
            self._kv_q = self._kv_s = self._kv_b = None
        else:
            self._kv_q, self._kv_s, self._kv_b = quantize_last_axis(value, self.group_size)
            self._kv_bf16 = None

    @property
    def ckv(self) -> mx.array | None:
        if not self.quantized:
            return self._ckv_bf16
        if self._ckv_q is None:
            return None
        return dequantize_last_axis(self._ckv_q, self._ckv_s, self._ckv_b,
                                    self.group_size, dtype=mx.bfloat16)

    @ckv.setter
    def ckv(self, value: mx.array | None) -> None:
        if value is None:
            self._ckv_bf16 = None
            self._ckv_q = self._ckv_s = self._ckv_b = None
            return
        if not self.quantized:
            self._ckv_bf16 = value
            self._ckv_q = self._ckv_s = self._ckv_b = None
        else:
            self._ckv_q, self._ckv_s, self._ckv_b = quantize_last_axis(value, self.group_size)
            self._ckv_bf16 = None

    def _resolve_quant(self, head_dim: int) -> None:
        """First-append check: ``mx.quantize`` requires ``head_dim`` divisible by ``group_size``,
        and supports ``group_size ∈ {32, 64, 128}``. Real DSV4 head_dim=128 so the default
        ``group_size=128`` works. Tiny synthetic tests with head_dim<32 cannot quantize at all —
        disable quantization explicitly (loud, no silent fallback)."""
        if not self.quantized:
            return
        valid = (32, 64, 128)
        if self.group_size in valid and head_dim % self.group_size == 0:
            return
        # auto-shrink to the largest valid group that divides head_dim
        for g in (128, 64, 32):
            if g <= head_dim and head_dim % g == 0:
                self.group_size = g
                return
        # head_dim too small (or not a multiple of 32) — quantization is not possible.
        self.quantized = False

    def append_kv(self, kv_new: mx.array) -> None:
        """Hot-path append: quantize only the new chunk (single token at decode) and append along
        the seq axis. The full bf16 stream is produced on read via the :attr:`kv` property."""
        if self._kv_q is None and self._kv_bf16 is None:
            self._resolve_quant(kv_new.shape[-1])
        if not self.quantized:
            self._kv_bf16 = (kv_new if self._kv_bf16 is None
                             else mx.concatenate([self._kv_bf16, kv_new], axis=1))
            return
        kq, ks, kb = quantize_last_axis(kv_new, self.group_size)
        if self._kv_q is None:
            self._kv_q, self._kv_s, self._kv_b = kq, ks, kb
        else:
            self._kv_q = mx.concatenate([self._kv_q, kq], axis=1)
            self._kv_s = mx.concatenate([self._kv_s, ks], axis=1)
            self._kv_b = mx.concatenate([self._kv_b, kb], axis=1)

    def append_ckv(self, ckv_new: mx.array) -> None:
        """Hot-path append for the compressed KV stream (one pooled token per closed window)."""
        if self._ckv_q is None and self._ckv_bf16 is None:
            self._resolve_quant(ckv_new.shape[-1])
        if not self.quantized:
            self._ckv_bf16 = (ckv_new if self._ckv_bf16 is None
                              else mx.concatenate([self._ckv_bf16, ckv_new], axis=1))
            return
        cq, cs, cb = quantize_last_axis(ckv_new, self.group_size)
        if self._ckv_q is None:
            self._ckv_q, self._ckv_s, self._ckv_b = cq, cs, cb
        else:
            self._ckv_q = mx.concatenate([self._ckv_q, cq], axis=1)
            self._ckv_s = mx.concatenate([self._ckv_s, cs], axis=1)
            self._ckv_b = mx.concatenate([self._ckv_b, cb], axis=1)

    def truncate_kv(self, length: int) -> None:
        """Slice the latent KV stream to the first ``length`` tokens (rollback)."""
        if not self.quantized:
            if self._kv_bf16 is not None:
                self._kv_bf16 = self._kv_bf16[:, :length]
            return
        if self._kv_q is not None:
            self._kv_q = self._kv_q[:, :length]
            self._kv_s = self._kv_s[:, :length]
            self._kv_b = self._kv_b[:, :length]

    def truncate_ckv(self, keep: int) -> None:
        """Slice the compressed KV stream to the first ``keep`` windows (rollback)."""
        if keep == 0:
            self.ckv = None
            return
        if not self.quantized:
            if self._ckv_bf16 is not None:
                self._ckv_bf16 = self._ckv_bf16[:, :keep]
            return
        if self._ckv_q is not None:
            self._ckv_q = self._ckv_q[:, :keep]
            self._ckv_s = self._ckv_s[:, :keep]
            self._ckv_b = self._ckv_b[:, :keep]


def _layer_shallow_copy(lc: _LayerCache) -> _LayerCache:
    """Per-layer shallow copy that shares array references with ``lc`` (lossless under MLX's
    immutable arrays: subsequent ``append_kv`` / ``truncate_kv`` create new arrays, leaving the
    originals — and any other shallow-copy sibling — untouched). Drives
    :meth:`DSV4Cache.replicate` for the batched tree-spec verify (docs/batched_tree_verify.md)."""
    new = _LayerCache(quantized=lc.quantized, group_size=lc.group_size,
                      max_rollback=lc.max_rollback)
    new.ratio = lc.ratio
    new._kv_bf16 = lc._kv_bf16
    new._kv_q = lc._kv_q
    new._kv_s = lc._kv_s
    new._kv_b = lc._kv_b
    new._ckv_bf16 = lc._ckv_bf16
    new._ckv_q = lc._ckv_q
    new._ckv_s = lc._ckv_s
    new._ckv_b = lc._ckv_b
    new.ikv = lc.ikv
    new.ring = lc.ring
    return new


class DSV4Cache:
    """Decode cache for a DSV4 attention stack: one :class:`_LayerCache` per attention block.

    Mirrors the update/``truncate``/``offset`` ergonomics of :class:`quanta.cache.MLACache` and
    :class:`quanta.nemotron.attention.KVCache`. ``offset`` reports the shared decode position (every
    attention layer advances in lock-step), derived from the latent stream length so it is exact and
    survives ``truncate``.

    The ``quantized`` / ``group_size`` arguments propagate to every :class:`_LayerCache` — int8
    KV storage (default since #123) for the long-context memory win.
    """

    def __init__(self, n_layers: int, *, quantized: bool = True, group_size: int = 128,
                 max_rollback: int = 1) -> None:
        self.layers: list[_LayerCache] = [
            _LayerCache(quantized=quantized, group_size=group_size, max_rollback=max_rollback)
            for _ in range(n_layers)
        ]

    def replicate(self, b: int) -> list["DSV4Cache"]:
        """Return ``b`` parallel decode caches, each initially sharing this cache's prefix state.

        MLX arrays are immutable, so the replicas can share references to the prefix tensors at
        zero cost — subsequent ``append_kv`` / ``append_ckv`` / ``truncate`` on any replica creates
        new arrays only, leaving the originals (and every other replica) untouched. This is the
        structural-sharing form of ``cache.replicate(B)`` for the batched tree-spec verify
        (docs/batched_tree_verify.md): each enumerated draft path advances its own replica, the
        original prefix stays read-only, and the B-wide verify amortizes the routed-MoE weight reads
        across all B paths in one batched MoE call (via :class:`DSV4BatchedResidentModel.batch_step`).

        The picked path's replica is discarded after the round — the commit-forward re-feeds the
        accepted prefix through the original (un-replicated) cache, exactly as today. So nothing of
        the replica state is persisted past the round, and the original cache is bit-identical to
        what a sequential per-path verify would have left it at (gated in
        ``parity/dsv4_batched_tree_verify_test.py``'s "cache invariance" assertion).
        """
        if b < 1:
            raise ValueError(f"replicate(B) requires B >= 1 (got {b})")
        return [self._copy() for _ in range(b)]

    def _copy(self) -> "DSV4Cache":
        """Shallow per-layer copy of every ``_LayerCache`` (array references shared; MLX
        immutability makes divergent appends/truncates lossless)."""
        new = self.__new__(DSV4Cache)
        new.layers = [_layer_shallow_copy(lc) for lc in self.layers]
        return new

    def __getitem__(self, i: int) -> _LayerCache:
        return self.layers[i]

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def offset(self) -> int:
        """Number of tokens already cached (positions consumed). 0 before the first append.

        Every attention layer advances in lock-step, so any populated layer reports the same value;
        we read the first populated one (robust to a cache that drives only a subset of layers).
        Reads the int8 codes' shape directly so we don't dequantize just to probe length."""
        for lc in self.layers:
            n = lc.kv_length()
            if n > 0:
                return n
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
            if lc.kv_length() == 0:
                continue
            lc.truncate_kv(length)
            if lc.ratio:                                         # compressed layer
                keep = length // lc.ratio                        # windows completed by position length-1
                lc.truncate_ckv(keep)
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
    lc.append_ckv(ck)

    if cfg.has_indexer(layer_id):
        icp = p["indexer"]["compressor"]
        ik = _pool_one_window(cur, prev, icp["ape"].astype(mx.float32), icp["norm"].astype(mx.float32),
                              icp["wkv"].astype(mx.float32), icp["wgate"].astype(mx.float32),
                              ratio=4, head_dim=cfg.index_head_dim, rope_head_dim=cfg.rope_head_dim,
                              eps=cfg.norm_eps, cos_c=cos_c, sin_c=sin_c, overlap=True)
        lc.ikv = ik if lc.ikv is None else mx.concatenate([lc.ikv, ik], axis=1)


def _push_ring(lc: _LayerCache, x_t: mx.array, ratio: int, overlap: bool) -> None:
    """Append the new hidden vector to the raw-hidden ring, trimmed to the last
    ``coff*ratio + (max_rollback-1)`` positions — the minimum needed to pool the next window AND
    roll back up to ``max_rollback`` tokens within it (k≥2 spec-decode drops a full chained suffix
    when every draft is rejected). ``max_rollback`` defaults to 1 ⇒ classic k=1 sizing."""
    cap = (2 if overlap else 1) * ratio + (lc.max_rollback - 1)
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
    lc.append_kv(kv)                                             # int8 (default) or bf16
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
    lc.append_kv(kv)
    _push_ring(lc, x_t, ratio, cfg.overlap(layer_id))
    _maybe_pool(lc, p, cfg, layer_id, ratio, offset, cos, sin)

    qf = q.astype(mx.float32)
    kvf = lc.kv.astype(mx.float32)
    sink, scale = p["attn_sink"].astype(mx.float32), cfg.attn_scale

    # window scores: single query at abs pos ``offset`` attends keys in (offset-window, offset].
    sc = mx.einsum("bqhd,bsd->bqhs", qf, kvf) * scale            # [B,1,H,S]
    ki = mx.arange(lc.kv_length())[None, :]
    win = (ki <= offset) & (ki > offset - cfg.sliding_window)    # [1,S]
    sc = sc + mx.where(win, 0.0, _NEG)[None, :, None, :]
    kv_all = kvf

    if lc.n_comp() > 0:
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
