"""MiniMax-M2.7 single-token (decode) attention cache — incremental, output-equivalent to prefill.

MiniMax's regime is the **simplest** of every model in the repo: every layer is plain full-softmax
GQA (no MLA latent, no compressor, no Lightning-Indexer, no sliding window, no Mamba recurrence).
So the decode state is just a standard **growing KV cache** per layer, and the decode step is the
existing prefill attention restricted to a single query — :func:`quanta.minimax.attention.MiniMaxAttention`
*already* implements that cache-threaded path (append K/V for the new token, partial-RoPE the new
q/k at the cached offset, per-layer QK-norm, SDPA over the cached KV). This module therefore does
**not** re-derive the attention math; it supplies the cache container + a thin stepper that drives
that existing path, so a T-step incremental decode is output-equivalent to one prefill over the same
T tokens by construction (gated model-free in ``parity/minimax_decode_attn_test.py``).

* :class:`_LayerKVCache` — the per-layer growing K/V store. Subclasses
  :class:`quanta.minimax.attention.KVCache` (reusing its ``update``/``offset`` exactly, the same
  storage the prefill cache path writes) and adds :meth:`truncate` for lossless speculative-decode
  rollback.
* :class:`MiniMaxCache` — one ``_LayerKVCache`` per decoder layer with ``__getitem__`` / ``__len__`` /
  ``.offset`` / ``.truncate(length)``. ``truncate`` rolls every layer back to **exactly** the state
  after consuming ``length`` tokens (per-position K/V storage slices cleanly); it is lossless or it
  **raises** (rule 6) — a cache that cannot losslessly roll back a rejected draft is a silent
  correctness bug, so rolling *forward* past what was consumed fails loud instead of fabricating K/V.
* :func:`decode_step` — one token through one layer's attention, threading its ``_LayerKVCache``.

No Python loops on the compute path: the only loop is the caller's decode-step loop (one bounded
loop over generated tokens). All cache state is tiny (the K/V streams grow with context exactly like
the Nemotron / Kimi caches). **Tiny tensors only in the gate — never load the real model.**
"""

from __future__ import annotations

import mlx.core as mx

from quanta.minimax.attention import KVCache, MiniMaxAttention


class _LayerKVCache(KVCache):
    """Per-layer growing GQA K/V cache + lossless ``truncate`` for speculative-decode rollback.

    Inherits :class:`quanta.minimax.attention.KVCache`'s ``update`` / ``offset`` unchanged (the exact
    storage the prefill cache path writes ``[B, n_kv, S, head_dim]``); rollback is a clean prefix
    slice because K/V is stored per absolute position.
    """

    def truncate(self, length: int) -> None:
        """Roll this layer back to exactly the state after consuming ``length`` tokens.

        Drops the trailing ``offset - length`` positions from the K/V streams (a clean prefix slice —
        per-position storage). ``length == offset`` is a no-op; ``length > offset`` fails loud (cannot
        fabricate future K/V); ``length < 0`` fails loud. The slice is lossless by construction; there
        is no lossy fallback (rule 6)."""
        if length < 0:
            raise ValueError(f"truncate length {length} < 0")
        cur = 0 if self.k is None else self.k.shape[2]
        if length > cur:
            raise ValueError(f"truncate({length}) > cached length {cur}: cannot roll forward past "
                             f"consumed tokens (rule 6: no silent wrong-length state)")
        if length == cur:
            return
        if length == 0:
            self.k = self.v = None
            return
        self.k = self.k[:, :, :length]
        self.v = self.v[:, :, :length]


class MiniMaxCache:
    """Decode cache for the MiniMax GQA stack: one growing K/V cache per decoder layer.

    Mirrors the ``__getitem__`` / ``__len__`` / ``offset`` / ``truncate`` ergonomics of
    :class:`quanta.dsv4.decode.DSV4Cache` and :class:`quanta.nemotron.attention.KVCache`. Every layer
    advances in lock-step (one token completes all layers before the next begins), so ``offset`` reads
    the first populated layer's K/V length — exact and ``truncate``-stable.
    """

    def __init__(self, n_layers: int) -> None:
        if n_layers <= 0:
            raise ValueError(f"MiniMaxCache needs n_layers >= 1, got {n_layers}")
        self.layers: list[_LayerKVCache] = [_LayerKVCache() for _ in range(n_layers)]

    def __getitem__(self, i: int) -> _LayerKVCache:
        return self.layers[i]

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def offset(self) -> int:
        """Number of tokens already cached (positions consumed); 0 before the first append.

        Every attention layer advances in lock-step, so any populated layer reports the same value;
        read the first populated one (robust to a cache driving only a subset of layers)."""
        for lc in self.layers:
            if lc.k is not None:
                return lc.k.shape[2]
        return 0

    def truncate(self, length: int) -> None:
        """Roll the cache back to exactly the state after consuming ``length`` tokens (drop a rejected
        speculative draft). Lossless or it raises (rule 6): ``length`` past the cache's own ``offset``
        is a forward-roll and fails loud; otherwise every *populated* layer slices cleanly to
        ``length`` (unpopulated layers — a subset-driven cache — have nothing to roll back and are
        skipped)."""
        if length < 0:
            raise ValueError(f"truncate length {length} < 0")
        cur = self.offset
        if length > cur:
            raise ValueError(f"truncate({length}) > consumed length {cur}: cannot roll forward past "
                             f"consumed tokens (rule 6: no silent wrong-length state)")
        for lc in self.layers:
            if lc.k is None:                  # subset-driven cache: this layer never advanced
                continue
            lc.truncate(length)


def decode_step(x_t: mx.array, attn: MiniMaxAttention, cache: _LayerKVCache,
                *, use_fast: bool = True) -> mx.array:
    """One token through one layer's GQA attention, threading its growing K/V cache.

    ``x_t``: ``[B, 1, hidden]`` (the post-input-norm hidden for the new token). Delegates to the
    existing :class:`quanta.minimax.attention.MiniMaxAttention` cache path — which reads the absolute
    position from ``cache.offset``, partial-RoPE-s and QK-norms the new q/k, appends the new K/V, and
    runs SDPA over all cached KV — so the result equals the prefill attention at this absolute
    position by construction (no re-derived math). Returns ``[B, 1, hidden]`` (the attention output,
    pre-residual; the caller adds the residual, matching :meth:`MiniMaxBlock.__call__`)."""
    return attn(x_t, cache=cache, use_fast=use_fast)
