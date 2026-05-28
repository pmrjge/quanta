"""Quanta-native block-paged KV cache with copy-on-write prefix sharing (#152).

mlx-lm-free reimplementation of a vLLM-style block pool (mirrors oMLX's ``cache/paged_cache.py``
design, rule 5) that lets concurrent / multi-turn agentic requests share a common prompt-prefix's KV
once instead of one private growing cache each. See :mod:`quanta.paged.paged_kv_cache`.

``PAGED_KV_DEFAULT`` stays ``False`` until the paged path is parity-green (paged == discrete, prefix
reuse == one-shot) AND the deferred real-model teacher-forced-ppl gate is green — the optimization is
kept behind a flag that defaults to the proven discrete path (rule 4).
"""

from __future__ import annotations

from quanta.paged.block_pool import BlockAllocator, CacheBlock, compute_block_hash
from quanta.paged.paged_kv_cache import (
    PagedCacheStats,
    PagedKVCacheManager,
    PagedKVCacheView,
    SeqHandle,
)
from quanta.paged.recurrent_cache import RecurrentCacheStats, RecurrentPrefixCache

PAGED_KV_DEFAULT = False

__all__ = [
    "PAGED_KV_DEFAULT",
    "BlockAllocator",
    "CacheBlock",
    "compute_block_hash",
    "PagedCacheStats",
    "PagedKVCacheManager",
    "PagedKVCacheView",
    "SeqHandle",
    "RecurrentCacheStats",
    "RecurrentPrefixCache",
]
