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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from quanta.cache_quant import BITS, dequantize_last_axis, quantize_last_axis
from quanta.dsv4.attention import (
    _rms_w,
    gather_rope_rows,
    output_proj,
    output_proj_b,
    project_qkv,
    project_qkv_b,
    rope_partial,
    rope_partial_b,
    sdpa_window_sink,
    sdpa_window_sink_batched,
)
from quanta.dsv4.config import DeepSeekV4Config

if TYPE_CHECKING:  # type-only — the paged latent store is duck-typed at runtime (no import cycle)
    from quanta.paged.paged_kv_cache import PagedKVCacheManager, PagedLatentCacheView, SeqHandle

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


# --- #152/#175 paged latent KV + content-addressed derived-state snapshots ----------------------
#
# DSV4's prefix-sharing splits the per-layer decode state two ways (the #174 hybrid pattern, adapted):
#
#   * the **latent KV** stream (``_LayerCache.kv``, the dominant byte cost, append-only, the same
#     int8-along-head_dim layout :class:`quanta.nemotron.attention.KVCache` uses) is PAGED — shared
#     across concurrent / multi-turn requests via :class:`~quanta.paged.PagedKVCacheManager`'s
#     single-stream codec, so a common prompt prefix's latent KV is stored once;
#   * the **derived** compressed-KV / indexer-KV / raw-hidden ring (a deterministic function of the
#     prefix's RAW hidden states — the compressor pools raw hidden, NOT the latent KV, so they can't
#     be recomputed from the shared latent) are kept per-stream and content-addressed at block
#     boundaries via :class:`~quanta.paged.RecurrentPrefixCache`. On a prefix hit the boundary
#     snapshot is restored and the suffix pools only its OWN windows, seeded by the restored ring —
#     the ring already holds the ``coff*ratio`` raw-hidden tail every regime needs to pool the first
#     post-boundary (possibly boundary-straddling) window, so correctness is independent of any
#     block_size↔ratio alignment (gated in ``parity/dsv4_paged_latent_test.py``).


class _PagedLayerCache(_LayerCache):
    """A :class:`_LayerCache` whose LATENT KV lives in a shared
    :class:`~quanta.paged.PagedLatentCacheView` (prefix blocks dedup'd across requests/turns), while
    the derived compressed-KV / indexer-KV / raw-hidden ring stay per-stream (recomputed over the
    suffix from a restored boundary snapshot). The decode steppers are UNCHANGED — they call
    ``append_kv`` / read ``kv`` / read ``kv_length`` / pool the ring exactly as before; only the latent
    storage swaps to the paged view here. ckv/ikv/ring + their append/truncate are inherited verbatim.
    """

    __slots__ = ("_view",)

    def __init__(self, view: "PagedLatentCacheView", *, quantized: bool = True,
                 group_size: int = 128, max_rollback: int = 1) -> None:
        super().__init__(quantized=quantized, group_size=group_size, max_rollback=max_rollback)
        self._view = view

    @property
    def kv(self) -> mx.array | None:
        if self._view.offset == 0:
            return None
        return self._view.current()                 # gather prefix blocks + suffix -> bf16 latent

    @kv.setter
    def kv(self, value: mx.array | None) -> None:
        raise RuntimeError("paged latent KV is written via append_kv() and paged blocks; direct set "
                           "is unsupported (truncate via the manager).")

    def kv_length(self) -> int:
        return self._view.offset

    def append_kv(self, kv_new: mx.array) -> None:
        self._view.append(kv_new)                   # write-only; the kv property re-gathers on read

    def truncate_kv(self, length: int) -> None:
        self._view.truncate(length)


def paged_cache(manager: "PagedKVCacheManager", seq: "SeqHandle", n_layers: int, *,
                quantized: bool = True, group_size: int = 128, max_rollback: int = 1) -> DSV4Cache:
    """A :class:`DSV4Cache` whose every layer's latent KV is a paged view into ``manager`` (ALL DSV4
    layers are attention, so every latent stream is paged). Derived ckv/ikv/ring stay per-stream.
    ``quantized``/``group_size`` MUST match the discrete cache's settled values (the runtime threads
    the latent's :func:`quanta.dsv4.batched_runtime._latent_quant` result) so the paged round-trip is
    bit-identical to the discrete stream."""
    obj = DSV4Cache.__new__(DSV4Cache)
    obj.layers = [
        _PagedLayerCache(manager.view_one(seq, i), quantized=quantized,
                         group_size=group_size, max_rollback=max_rollback)
        for i in range(n_layers)
    ]
    return obj


@dataclass(frozen=True)
class _DerivedSnapshot:
    """Opaque per-layer derived-state snapshot at a block boundary (the payload carried in
    :class:`~quanta.paged.RecurrentPrefixCache` for DSV4). Holds everything the suffix pooling needs
    that ISN'T the paged latent: the compressed-KV stream (int8 codes or bf16), the indexer KV, and
    the raw-hidden ring. MLX arrays are immutable, so capturing references is a lossless snapshot —
    a later append on the live cache creates new arrays and leaves these untouched. ``None`` for a
    dense (ratio-0) layer, which has no derived state."""

    ratio: int
    quantized: bool
    group_size: int
    ckv_q: Any
    ckv_s: Any
    ckv_b: Any
    ckv_bf16: Any
    ikv: Any
    ring: Any


def snapshot_derived(cache: DSV4Cache) -> list[_DerivedSnapshot | None]:
    """Capture each layer's derived (compressed-KV / indexer-KV / ring) state at the current boundary
    — the per-layer list stored in the recurrent prefix cache. Dense layers contribute ``None``."""
    out: list[_DerivedSnapshot | None] = []
    for lc in cache.layers:
        if lc.ratio == 0 and lc.ring is None:        # dense layer: latent only, no derived state
            out.append(None)
            continue
        out.append(_DerivedSnapshot(
            ratio=lc.ratio, quantized=lc.quantized, group_size=lc.group_size,
            ckv_q=lc._ckv_q, ckv_s=lc._ckv_s, ckv_b=lc._ckv_b, ckv_bf16=lc._ckv_bf16,
            ikv=lc.ikv, ring=lc.ring))
    return out


def restore_derived(cache: DSV4Cache, payload: list[_DerivedSnapshot | None] | None) -> None:
    """Restore a :func:`snapshot_derived` payload into ``cache`` (in place) before a suffix prefill —
    seeds each compressed layer's ckv/ikv/ring so the suffix resumes pooling exactly where the prefix
    left off (bit-identical to a continuous decode). ``None`` payload / per-layer ``None`` is a no-op
    (fresh / dense layer)."""
    if payload is None:
        return
    for lc, snap in zip(cache.layers, payload, strict=True):
        if snap is None:
            continue
        lc.ratio = snap.ratio
        lc.quantized = snap.quantized
        lc.group_size = snap.group_size
        lc._ckv_q, lc._ckv_s, lc._ckv_b = snap.ckv_q, snap.ckv_s, snap.ckv_b
        lc._ckv_bf16 = snap.ckv_bf16
        lc.ikv = snap.ikv
        lc.ring = snap.ring


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


# --- batched single-token decode steppers (per-stream offsets) ----------------
# The per-stream-offset siblings of the steppers above: B ragged-offset decode streams run through
# ONE projection / ONE windowed-sink SDPA per layer instead of a Python loop over streams. The only
# remaining per-stream work is the bounded cache append/ckv-append/ring-push (IO, rule-3) and the
# window-closing pool, which is data-dependent per stream. Output-equivalent to looping the
# single-stream steppers: B=1 is bit-exact (no padding), B≥2 is greedy-exact (the pad+mask SDPA
# reorders the softmax reduction → argmax-stable bf16 ULPs). Gated in parity/dsv4_batched_attention_test.
def _pad_stack(arrs: list[mx.array | None], lmax: int | None = None) -> mx.array:
    """Stack ``B`` per-stream ``[1, L_b, D]`` streams → ``[B, lmax, D]`` (zero tail along seq). A
    ``None`` entry (a compressed/indexer stream that is still empty — no window has closed yet)
    becomes a zero ``[1, lmax, D]`` row. The window / visibility masks send every padded column to a
    large negative, so the padding is numerically inert. ``lmax`` defaults to the longest stream."""
    lengths = [0 if a is None else int(a.shape[1]) for a in arrs]
    if lmax is None:
        lmax = max(lengths)
    ref = next((a for a in arrs if a is not None), None)
    if ref is None:
        raise ValueError("_pad_stack: every stream is empty (no shape to infer)")
    hd, dtype = int(ref.shape[-1]), ref.dtype
    out = []
    for a, ln in zip(arrs, lengths, strict=True):
        if a is None:
            out.append(mx.zeros((1, lmax, hd), dtype=dtype))
        else:
            out.append(a if ln == lmax else mx.pad(a, [(0, 0), (0, lmax - ln), (0, 0)]))
    return mx.concatenate(out, axis=0)


# --- batched KV arena (#18): persistent max_batch-sized latent store ----------------------------
# Kills the per-stream KV-update loop in the batched decode steppers. Instead of B ragged per-stream
# _LayerCache streams (each quantizing + ``mx.concatenate``-growing per step) plus a ``_pad_stack``
# readback every step, one per-layer arena holds R = ``max_batch`` padded rows; the hot path writes
# all B active rows with ONE scatter (``arena[rows, cols, :] = codes``) and reads them with ONE gather
# (``mx.take``) + one batched dequant. Contents are bit-identical to ``_LayerCache``: the same
# :mod:`quanta.cache_quant` codec on the same input, and affine int-bits over the last axis is
# row-independent, so a batched ``[B,1,D]`` quantize equals B separate ``[1,1,D]`` quantizes row-for-row.
# Gated model-free in ``parity/dsv4_kv_arena_test.py``; flag-guarded (``kv_arena``) on
# :class:`~quanta.dsv4.batched_runtime.DSV4BatchedResidentModel`, default OFF until parity is green (rule 4).
def _grow_seq(arr: mx.array, cap: int) -> mx.array:
    """Return a ``[R, cap, X]`` copy of ``arr`` (``[R, old, X]``) with the live prefix slice-assigned
    in (the rest zero). Doubling growth for the arena's seq axis — amortized, not a hot op."""
    r, old, x = arr.shape
    out = mx.zeros((r, cap, x), dtype=arr.dtype)
    out[:, :old, :] = arr
    return out


class _KVArena:
    """One decoder layer's persistent batched latent-KV store: ``rows`` (== ``max_batch``) padded
    streams of affine int-bits codes (or bf16 when ``quantized=False``), each grown independently to
    ``lengths[row]`` tokens. Storage mirrors :class:`_LayerCache`'s latent trio (``_kv_q/_kv_s/_kv_b``
    via :mod:`quanta.cache_quant`) promoted to a leading ``rows`` axis: codes ``[R, L_cap, D/pack]``,
    scales/biases ``[R, L_cap, D/group]``. ``L_cap`` grows by doubling (slice-assign, amortized); the
    lazy first-write alloc learns ``D`` and the trio dtypes from the codec output."""

    __slots__ = ("rows", "group_size", "quantized", "bits", "l_cap", "lengths",
                 "_q", "_s", "_b", "_bf16")

    def __init__(self, rows: int, *, group_size: int, quantized: bool, bits: int = BITS) -> None:
        if rows < 1:
            raise ValueError(f"_KVArena rows must be >= 1 (got {rows})")
        self.rows = int(rows)
        self.group_size = int(group_size)
        self.quantized = bool(quantized)
        self.bits = int(bits)
        self.l_cap = 0
        self.lengths = [0] * self.rows        # per-row token count (bounded accounting, not a hot loop)
        self._q: mx.array | None = None
        self._s: mx.array | None = None
        self._b: mx.array | None = None
        self._bf16: mx.array | None = None

    # --- length probes -------------------------------------------------------
    def length(self, row: int) -> int:
        return self.lengths[row]

    def max_length(self, rows: list[int]) -> int:
        return max((self.lengths[r] for r in rows), default=0)

    # --- capacity (doubling) -------------------------------------------------
    def _ensure_q(self, need: int, q: mx.array, s: mx.array, b: mx.array) -> None:
        if self._q is None:
            cap = 1
            while cap < need:
                cap *= 2
            self._q = mx.zeros((self.rows, cap, q.shape[-1]), dtype=q.dtype)
            self._s = mx.zeros((self.rows, cap, s.shape[-1]), dtype=s.dtype)
            self._b = mx.zeros((self.rows, cap, b.shape[-1]), dtype=b.dtype)
            self.l_cap = cap
        elif need > self.l_cap:
            cap = self.l_cap
            while cap < need:
                cap *= 2
            self._q, self._s, self._b = (_grow_seq(self._q, cap), _grow_seq(self._s, cap),
                                         _grow_seq(self._b, cap))
            self.l_cap = cap

    def _ensure_bf16(self, need: int, kv_new: mx.array) -> None:
        if self._bf16 is None:
            cap = 1
            while cap < need:
                cap *= 2
            self._bf16 = mx.zeros((self.rows, cap, kv_new.shape[-1]), dtype=kv_new.dtype)
            self.l_cap = cap
        elif need > self.l_cap:
            cap = self.l_cap
            while cap < need:
                cap *= 2
            self._bf16 = _grow_seq(self._bf16, cap)
            self.l_cap = cap

    # --- writes --------------------------------------------------------------
    def append_row(self, row: int, kv_new: mx.array) -> None:
        """Per-row append (one stream, ``T`` tokens) — prefill / the multi-token tail via the view.
        ``kv_new``: ``[1, T, D]``. Slice-assigns row ``row`` at ``[length:length+T]`` (a bounded
        one-row write, not a hot per-stream loop)."""
        t = int(kv_new.shape[1])
        off = self.lengths[row]
        if self.quantized:
            q, s, b = quantize_last_axis(kv_new, self.group_size, self.bits)
            self._ensure_q(off + t, q, s, b)
            self._q[row, off:off + t, :] = q[0]
            self._s[row, off:off + t, :] = s[0]
            self._b[row, off:off + t, :] = b[0]
        else:
            self._ensure_bf16(off + t, kv_new)
            self._bf16[row, off:off + t, :] = kv_new[0]
        self.lengths[row] = off + t

    def append_batched(self, rows: list[int], kv_new: mx.array) -> None:
        """Hot-path batched append: ``kv_new`` ``[k, 1, D]`` for the ``k`` active rows. ONE quantize +
        ONE scatter at each row's current length, then advance lengths — no per-stream Python loop.
        Equivalent to :meth:`append_row` on each row (same codec; affine quant is row-independent)."""
        if int(kv_new.shape[1]) != 1:
            raise ValueError(f"append_batched expects T==1 (got {kv_new.shape[1]})")
        rows_arr = mx.array(rows, dtype=mx.int32)
        cols = mx.array([self.lengths[r] for r in rows], dtype=mx.int32)     # bounded accounting
        need = max(self.lengths[r] for r in rows) + 1
        if self.quantized:
            q, s, b = quantize_last_axis(kv_new, self.group_size, self.bits)
            self._ensure_q(need, q, s, b)
            self._q[rows_arr, cols, :] = q[:, 0, :]
            self._s[rows_arr, cols, :] = s[:, 0, :]
            self._b[rows_arr, cols, :] = b[:, 0, :]
        else:
            self._ensure_bf16(need, kv_new)
            self._bf16[rows_arr, cols, :] = kv_new[:, 0, :]
        for r in rows:                          # bounded accounting (<= max_batch), no tensor compute
            self.lengths[r] += 1

    # --- reads ---------------------------------------------------------------
    def read_row(self, row: int) -> mx.array | None:
        """Dequantized ``[1, length, D]`` bf16 for one row (the view's ``kv`` property); ``None`` if
        empty. Bit-identical to :attr:`_LayerCache.kv` for the same stream."""
        n = self.lengths[row]
        if n == 0:
            return None
        if self.quantized:
            return dequantize_last_axis(self._q[row:row + 1, :n], self._s[row:row + 1, :n],
                                        self._b[row:row + 1, :n], self.group_size, bits=self.bits)
        return self._bf16[row:row + 1, :n]

    def read_batched(self, rows: list[int]) -> mx.array:
        """Gather the ``k`` active rows, slice to ``L_max = max(lengths[rows])`` and dequant →
        ``[k, L_max, D]`` bf16 — replaces ``_pad_stack([lc.kv ...])`` on the arena path. Rows shorter
        than ``L_max`` keep stale/zero tail codes that the SDPA window/pad mask sends to ``-inf``
        (numerically inert); every valid position is bit-identical to the per-stream stream."""
        l_max = self.max_length(rows)
        if l_max == 0:
            raise ValueError("read_batched: every active row is empty (no KV to read)")   # rule 6
        rows_arr = mx.array(rows, dtype=mx.int32)
        if self.quantized:
            q = mx.take(self._q, rows_arr, axis=0)[:, :l_max]
            s = mx.take(self._s, rows_arr, axis=0)[:, :l_max]
            b = mx.take(self._b, rows_arr, axis=0)[:, :l_max]
            return dequantize_last_axis(q, s, b, self.group_size, bits=self.bits)
        return mx.take(self._bf16, rows_arr, axis=0)[:, :l_max]

    # --- rollback / free-list reset ------------------------------------------
    def truncate_row(self, row: int, length: int) -> None:
        """Roll row ``row`` back to ``length`` tokens (spec-decode rollback). Cursor move only — the
        stale tail is never read (``read_row`` slices to ``length``) and is overwritten on next write."""
        if length < self.lengths[row]:
            self.lengths[row] = length

    def reset_row(self, row: int) -> None:
        """Free-list reset: drop row ``row`` to empty so a new stream can lease it (:meth:`free`)."""
        self.lengths[row] = 0


class _KVArenaSet:
    """The ``R``-row free-list + one :class:`_KVArena` per layer (latent KV). Owned by
    :class:`~quanta.dsv4.batched_runtime.DSV4BatchedResidentModel` when ``kv_arena=True``; a stream
    leases a row via :meth:`alloc` (the SAME row index across every layer) and returns it via
    :meth:`free` on release. Model-free constructible for the parity gate. (Compressed-layer
    ckv/ikv/ring arenas join the set in #18 M3.)"""

    def __init__(self, n_layers: int, rows: int, *, group_size: int, quantized: bool,
                 bits: int = BITS) -> None:
        if rows < 1:
            raise ValueError(f"_KVArenaSet rows must be >= 1 (got {rows})")
        if n_layers < 1:
            raise ValueError(f"_KVArenaSet n_layers must be >= 1 (got {n_layers})")
        self.rows = int(rows)
        self.latent: list[_KVArena] = [
            _KVArena(rows, group_size=group_size, quantized=quantized, bits=bits)
            for _ in range(n_layers)
        ]
        self._free: list[int] = list(reversed(range(rows)))     # pop() hands out 0,1,2,...

    def alloc(self) -> int:
        if not self._free:
            raise RuntimeError(f"_KVArenaSet: no free rows (all {self.rows} leased)")   # rule 6
        return self._free.pop()

    def free(self, row: int) -> None:
        if not 0 <= row < self.rows:
            raise ValueError(f"_KVArenaSet.free: row {row} out of range [0,{self.rows})")
        if row in self._free:
            raise RuntimeError(f"_KVArenaSet: double-free of row {row}")                 # rule 6
        for a in self.latent:
            a.reset_row(row)
        self._free.append(row)

    def __len__(self) -> int:
        return len(self.latent)

    def __getitem__(self, i: int) -> _KVArena:
        return self.latent[i]


class _ArenaLayerView(_LayerCache):
    """A :class:`_LayerCache` whose LATENT KV lives in a shared :class:`_KVArena` row (the batched
    arena, #18), while the derived ckv/ikv/ring stay per-object (inherited — batched into the arena set
    in M3). Mirrors the :class:`_PagedLayerCache` pattern: prefill / the single-stream decode path call
    ``append_kv`` / read ``kv`` / ``kv_length`` / ``truncate_kv`` exactly as before; only the latent
    storage swaps to the arena row here, so prefill seeds an arena row directly (no migration)."""

    __slots__ = ("_arena", "_row")

    def __init__(self, arena: _KVArena, row: int, *, quantized: bool = True,
                 group_size: int = 128, max_rollback: int = 1) -> None:
        super().__init__(quantized=quantized, group_size=group_size, max_rollback=max_rollback)
        self._arena = arena
        self._row = row

    @property
    def kv(self) -> mx.array | None:
        return self._arena.read_row(self._row)

    @kv.setter
    def kv(self, value: mx.array | None) -> None:
        raise RuntimeError("arena latent KV is written via append_kv(); direct set unsupported (#18)")

    def kv_length(self) -> int:
        return self._arena.length(self._row)

    def append_kv(self, kv_new: mx.array) -> None:
        self._arena.append_row(self._row, kv_new)

    def truncate_kv(self, length: int) -> None:
        self._arena.truncate_row(self._row, length)


def decode_step_dense_batched(x_t: mx.array, p: dict, cfg: DeepSeekV4Config, layer_id: int,
                              lcs: list[_LayerCache], cos: mx.array, sin: mx.array,
                              offsets: list[int]) -> mx.array:
    """Batched ratio-0 decode across ``B`` streams — per-stream-offset sibling of
    :func:`decode_step_dense`. ``x_t``: ``[B,1,dim]``; ``lcs[b]``: stream ``b``'s :class:`_LayerCache`
    for this layer; ``cos``/``sin``: full RoPE tables for ``[0, max(offsets)+1)``; ``offsets[b]``:
    stream ``b``'s absolute position. Returns ``[B,1,dim]``."""
    b = x_t.shape[0]
    cos_b, sin_b = gather_rope_rows(cos, sin, offsets)          # [B,1,rd/2] per-stream rows
    _, q, kv = project_qkv_b(x_t, p, cfg, cos_b, sin_b)         # q [B,1,H,hd], kv [B,1,hd]
    for s in range(b):                                          # bounded per-stream KV append (IO, rule-3)
        lcs[s].append_kv(kv[s:s + 1])
    kv_pad = _pad_stack([lc.kv for lc in lcs])                  # [B, L_max, hd]
    o = sdpa_window_sink_batched(q.astype(mx.float32), kv_pad.astype(mx.float32),
                                 p["attn_sink"].astype(mx.float32), cfg.attn_scale,
                                 cfg.sliding_window, offsets)
    return output_proj_b(o, p, cfg, cos_b, sin_b)


def _decode_indexer_select_batched(x_t: mx.array, qr: mx.array, lcs: list[_LayerCache], idx_p: dict,
                                   cfg: DeepSeekV4Config, cos_b: mx.array, sin_b: mx.array,
                                   ncomp_max: int, ncomps: list[int]) -> mx.array:
    """Batched Lightning-indexer top-k selection: ``[B,1,ncomp_max]`` bool mask of the compressed
    tokens each stream keeps. Per-stream-offset sibling of :func:`_decode_indexer_select` over the
    padded indexer-KV; padded columns (``t >= n_comp_b``) are scored ``-inf`` so they never win the
    per-stream top-``index_topk`` (k clamped to ``n_comp_b``). B=1 reproduces the single-stream mask."""
    inh, ihd, rd = cfg.index_n_heads, cfg.index_head_dim, cfg.rope_head_dim
    b = x_t.shape[0]
    qb = (qr @ idx_p["wq_b"].T).reshape(b, 1, inh, ihd)
    qb = rope_partial_b(qb, cos_b, sin_b, rd).astype(mx.float32)
    ikv = _pad_stack([lc.ikv for lc in lcs], ncomp_max).astype(mx.float32)   # [B,ncomp_max,index_head_dim]
    weights = (x_t @ idx_p["weights_proj"].T).astype(mx.float32) * (ihd ** -0.5 * inh ** -0.5)
    score = mx.einsum("bqhd,btd->bqht", qb, ikv)               # [B,1,inh,ncomp_max]
    score = (mx.maximum(score, 0.0) * weights[..., None]).sum(axis=2)   # [B,1,ncomp_max]
    ki = mx.arange(ncomp_max)[None, :]
    ncs = mx.array(ncomps, dtype=mx.int32)[:, None]
    valid = (ki < ncs)[:, None, :]                             # [B,1,ncomp_max]
    score = mx.where(valid, score, mx.array(float("-inf"), score.dtype))   # padding never selected
    # per-stream threshold = the k-th largest valid score (k = min(index_topk, n_comp_b)); ascending
    # sort puts the -inf padding first, so index ``ncomp_max - k`` is the k-th largest of the valids.
    ks = [min(cfg.index_topk, n) for n in ncomps]
    thr_idx = mx.array([min(max(ncomp_max - k, 0), ncomp_max - 1) for k in ks], dtype=mx.int32)
    sorted_sc = mx.sort(score, axis=-1)                        # [B,1,ncomp_max] ascending
    thr = mx.take_along_axis(sorted_sc, thr_idx[:, None, None], axis=-1)    # [B,1,1]
    return (score >= thr) & valid                              # exclude padding (redundant w/ -inf)


def decode_step_compressed_batched(x_t: mx.array, p: dict, cfg: DeepSeekV4Config, layer_id: int,
                                   lcs: list[_LayerCache], cos: mx.array, sin: mx.array,
                                   offsets: list[int]) -> mx.array:
    """Batched compressed decode across ``B`` streams (ratio-4 + indexer / ratio-128) —
    per-stream-offset sibling of :func:`decode_step_compressed`. Per-stream (rule-3): the latent-KV
    append, the raw-hidden ring push, and the window-closing compressor pool (``_maybe_pool``, which
    appends one compressed/indexer token when ``(offset_b+1) % ratio == 0``). Batched: ONE projection,
    and ONE softmax over the per-stream-padded window-latent ++ compressed keys with per-stream window
    / visibility / indexer-top-k masks (so each stream attends exactly the keys its single-stream step
    would). Returns ``[B,1,dim]``."""
    b = x_t.shape[0]
    ratio = cfg.compress_ratio(layer_id)
    overlap = cfg.overlap(layer_id)
    cos_b, sin_b = gather_rope_rows(cos, sin, offsets)         # [B,1,rd/2]
    qr, q, kv = project_qkv_b(x_t, p, cfg, cos_b, sin_b)       # qr [B,1,qlora], q [B,1,H,hd], kv [B,1,hd]
    for s in range(b):                                         # per-stream cache update + pool (IO, rule-3)
        lc = lcs[s]
        lc.ratio = ratio
        lc.append_kv(kv[s:s + 1])
        _push_ring(lc, x_t[s:s + 1], ratio, overlap)
        _maybe_pool(lc, p, cfg, layer_id, ratio, offsets[s], cos, sin)

    qf = q.astype(mx.float32)
    scale = cfg.attn_scale
    kv_pad = _pad_stack([lc.kv for lc in lcs]).astype(mx.float32)           # [B,L_max,hd]
    l_max = kv_pad.shape[1]
    sc = mx.einsum("bqhd,bsd->bqhs", qf, kv_pad) * scale                   # [B,1,H,L_max]
    ki = mx.arange(l_max)[None, :]
    off = mx.array(offsets, dtype=mx.int32)[:, None]
    win = (ki <= off) & (ki > off - cfg.sliding_window)                    # [B,L_max] window + pad mask
    sc = sc + mx.where(win, 0.0, _NEG)[:, None, None, :]
    kv_all = kv_pad

    ncomps = [lc.n_comp() for lc in lcs]
    ncomp_max = max(ncomps)
    if ncomp_max > 0:
        ckv = _pad_stack([lc.ckv for lc in lcs], ncomp_max).astype(mx.float32)   # [B,ncomp_max,hd]
        sc_c = mx.einsum("bqhd,btd->bqht", qf, ckv) * scale                # [B,1,H,ncomp_max]
        if cfg.has_indexer(layer_id):
            sel = _decode_indexer_select_batched(x_t, qr, lcs, p["indexer"], cfg, cos_b, sin_b,
                                                 ncomp_max, ncomps)        # [B,1,ncomp_max]
        else:                                                              # ratio-128: all cached visible
            ncs = mx.array(ncomps, dtype=mx.int32)[:, None]
            sel = (mx.arange(ncomp_max)[None, :] < ncs)[:, None, :]        # mask padded columns only
        sc_c = sc_c + mx.where(sel, 0.0, _NEG)[:, :, None, :]
        sc = mx.concatenate([sc, sc_c], axis=-1)
        kv_all = mx.concatenate([kv_pad, ckv], axis=1)

    m = mx.max(sc, axis=-1, keepdims=True)
    ex = mx.exp(sc - m)
    sink = p["attn_sink"].astype(mx.float32)
    denom = mx.sum(ex, axis=-1) + mx.exp(sink[None, None, :] - m[..., 0])
    o = mx.einsum("bqht,btd->bqhd", ex, kv_all) / denom[..., None]
    return output_proj_b(o, p, cfg, cos_b, sin_b)
