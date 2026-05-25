"""GLM-5.1 single-token (decode) cache + stepper — incremental, output-equivalent to prefill.

The decode half of the GLM serving stack (mirrors :mod:`quanta.dsv4.decode`). It is **parity-first**:
one decode step is numerically equal to the prefill block (:meth:`quanta.glm.model.GLMDecoderLayer.__call__`)
evaluated at the same absolute position, because the step path *reuses* the prefill helpers —
:meth:`quanta.glm.model.GLMDecoderLayer.step` threads the MLA latent KV / DSA indexer key through this
cache and calls :meth:`quanta.glm.attention.MLAAttention.step` + :meth:`quanta.glm.indexer.LightningIndexer.step_mask`
(no reimplemented attention/indexer math here — rule 1/4).

GLM's MLA decode mirrors :class:`quanta.cache.MLACache`: append the post-layernorm latent ``c_kv``
``[B,S,kv_lora]`` and the roped single-MQA-head ``k_pe`` ``[B,1,S,rope]`` per step (no per-head K/V
materialized). The DSA Lightning-Indexer keeps its own roped key stream ``[B,1,S,index_head_dim]``
(:class:`_IndexKeyCache`); each single-token query scores every cached key and selects the
top-``index_topk`` (its :meth:`step_mask` mirrors the DSV4 ``_decode_indexer_select``: all cached keys
are causally valid because one is appended per consumed token). Both streams grow in lock-step, so a
layer's ``offset`` is unambiguous and ``truncate`` slices them cleanly.

``GLMCache.truncate(length)`` makes the per-layer state **bit-identical** to having only ever fed
``length`` tokens — the speculative-decode rollback (drop a rejected draft). Per-position storage makes
the slice exact (rule 4); a negative length fails loud (rule 6). No Python loops on the compute path:
the only loops are the coarse per-layer / per-token decode loops in the caller
(:class:`quanta.glm.runtime.GLMResidentModel`). Gated model-free in ``parity/glm_decode_attn_test.py``.
"""

from __future__ import annotations

import mlx.core as mx


class _LayerKVCache:
    """Per-layer MLA latent KV cache: the post-layernorm latent ``c_kv`` ``[B,S,kv_lora]`` and the
    roped single-head ``k_pe`` ``[B,1,S,rope]`` (the ``update(c_kv, k_pe) -> (c_kv_all, k_pe_all)``
    protocol :meth:`quanta.glm.attention.MLAAttention.step` consumes; mirrors :class:`quanta.cache.MLACache`,
    bf16-only)."""

    __slots__ = ("c_kv", "k_pe")

    def __init__(self) -> None:
        self.c_kv: mx.array | None = None     # [B,S,kv_lora] post-layernorm latent
        self.k_pe: mx.array | None = None     # [B,1,S,rope] roped MQA rope key

    def update(self, c_kv_new: mx.array, k_pe_new: mx.array) -> tuple[mx.array, mx.array]:
        """Append the new token's latent + rope key; return the full streams."""
        self.c_kv = c_kv_new if self.c_kv is None else mx.concatenate([self.c_kv, c_kv_new], axis=1)
        self.k_pe = k_pe_new if self.k_pe is None else mx.concatenate([self.k_pe, k_pe_new], axis=2)
        return self.c_kv, self.k_pe


class _IndexKeyCache:
    """Per-layer DSA indexer key cache: the roped index keys ``[B,1,S,index_head_dim]`` (single MQA
    head). All cached keys are causally valid (one appended per consumed token), so the single decode
    query scores them all — mirroring prefill's causal score at that position."""

    __slots__ = ("k",)

    def __init__(self) -> None:
        self.k: mx.array | None = None

    def update(self, k_new: mx.array) -> mx.array:
        self.k = k_new if self.k is None else mx.concatenate([self.k, k_new], axis=2)
        return self.k


class _LayerCache:
    """A decoder layer's decode state: the MLA KV cache (``.kv``) and the indexer key cache (``.idx``),
    which grow in lock-step (one position each per consumed token)."""

    __slots__ = ("kv", "idx")

    def __init__(self) -> None:
        self.kv = _LayerKVCache()
        self.idx = _IndexKeyCache()


class GLMCache:
    """Decode cache for the GLM attention stack: one :class:`_LayerCache` per decoder layer.

    Matches the ``offset`` / ``truncate(length)`` ergonomics :mod:`quanta.glm.generate` /
    :mod:`quanta.glm.spec` expect (and :class:`quanta.cache.MLACache` / the DSV4 cache): every layer
    advances in lock-step, so ``offset`` reads any populated layer; ``truncate`` slices the per-position
    streams so the kept prefix is bit-identical to having only fed that many tokens (rule 4 rollback)."""

    def __init__(self, n_layers: int) -> None:
        self.layers: list[_LayerCache] = [_LayerCache() for _ in range(n_layers)]

    def __getitem__(self, i: int) -> _LayerCache:
        return self.layers[i]

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def offset(self) -> int:
        """Number of positions already cached (0 before the first append). Every layer advances in
        lock-step, so the first populated layer reports the shared value (robust to a cache driving a
        subset of layers)."""
        for lc in self.layers:
            if lc.kv.c_kv is not None:
                return lc.kv.c_kv.shape[1]
        return 0

    def truncate(self, length: int) -> None:
        """Roll every layer back to exactly the state after consuming ``length`` tokens (drop rejected
        speculative drafts). The latent / rope-key / indexer-key streams slice cleanly (per-position
        storage), so the kept prefix is bit-identical to having only fed ``length`` tokens. A negative
        length fails loud (rule 6); ``length >= offset`` is a no-op."""
        if length < 0:
            raise ValueError(f"truncate length {length} < 0")
        if length >= self.offset:
            return
        for lc in self.layers:
            if lc.kv.c_kv is None:
                continue
            lc.kv.c_kv = lc.kv.c_kv[:, :length]
            lc.kv.k_pe = lc.kv.k_pe[:, :, :length]
            if lc.idx.k is not None:
                lc.idx.k = lc.idx.k[:, :, :length]


def decode_step(layer, h_t: mx.array, layer_cache: _LayerCache, offset: int, *,
                use_fast: bool = False, use_indexer: bool = True) -> mx.array:
    """One decode token through ``layer`` at absolute position ``offset`` — output-equivalent to the
    prefill block at that position.

    Thin delegator to :meth:`quanta.glm.model.GLMDecoderLayer.step` (which threads ``layer_cache.kv`` /
    ``layer_cache.idx`` and reuses the prefill MLA + indexer helpers), so the decode math lives in
    :mod:`quanta.glm.attention` / :mod:`quanta.glm.indexer` and is never reimplemented here. ``h_t``:
    ``[B,1,dim]`` residual; returns ``[B,1,dim]``."""
    return layer.step(h_t, layer_cache, offset, use_fast=use_fast, use_indexer=use_indexer)
