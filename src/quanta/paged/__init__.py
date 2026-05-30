"""Quanta-native block-paged KV cache with copy-on-write prefix sharing (#152).

mlx-lm-free reimplementation of a vLLM-style block pool (mirrors oMLX's ``cache/paged_cache.py``
design, rule 5) that lets concurrent / multi-turn agentic requests share a common prompt-prefix's KV
once instead of one private growing cache each. See :mod:`quanta.paged.paged_kv_cache`.

``PAGED_KV_DEFAULT`` was ``False`` until the paged path went parity-green; it is now ``True`` — paged ==
discrete is proven model-free (bit-exact) AND on the real model for all three keepers (Nemotron #174,
DSV4 #175 bit-exact, InternLM2.5 #176: top-1 10/10 greedy decode with prefix reuse + boundary-snapshot
restore), so the engine defaults to the prefix-sharing paged path (rule 4 satisfied). Models outside the
#152 paged scope (Qwen3.5) have no paged contract and stay on the proven discrete batched path —
``quanta.shim.omlx._make_batched_session`` forces them unpaged (rule 6: never paged_kv=True with no spec).
"""

from __future__ import annotations

from quanta.paged.block_pool import BlockAllocator, CacheBlock, compute_block_hash
from quanta.paged.paged_kv_cache import (
    PagedCacheStats,
    PagedKVCacheManager,
    PagedKVCacheView,
    PagedLatentCacheView,
    SeqHandle,
)
from quanta.paged.recurrent_cache import RecurrentCacheStats, RecurrentPrefixCache

PAGED_KV_DEFAULT = True

# #153: batched-paged KV decode — ONE block-table scatter + ONE gather across all B lock-step streams
# (the paged sibling of the #18 arena loop-kill), serving the paged keepers (Nemotron, DSV4,
# InternLM2.5). Default OFF until the steppers are wired + parity-green (rule 4). The storage primitives
# (``PagedKVCacheManager.write_*_batched`` / ``gather_*_batched``) already exist and are gated
# model-free in ``parity/dsv4_paged_batched_test.py`` (M0); the runtime/session reads this flag to pick
# the batched scatter/gather over the per-stream loop once dispatched (#153 M3 flips it for DSV4 +
# InternLM2.5). Nemotron has GRADUATED to its own ON default below — keep this shared flag OFF so
# flipping Nemotron does not preempt the DSV4 M3 regression.
PAGED_KV_BATCHED_DEFAULT = False

# Nemotron-scoped #153 default: GRADUATED to ON. The loop-kill is parity-proven model-free (bit-exact,
# ``parity/nemotron_batched_attention_test.py`` §D) AND on the real int4-g64 120B-A12B bake — greedy-exact
# vs the per-stream paged loop at B∈{1,32,48} with a measured **+18% decode tok/s at B=48** (+15% @ B=32,
# the prod operating point; ``parity/nemotron_paged_batched_bench.py``). The per-stream loop stops scaling
# past B=32 (122 tok/s @48 < 126 @32) while the loop-kill holds, so the win grows with B (rule 4 satisfied:
# parity proven + a real win ⇒ default ON, still a single flag to revert). Scoped to Nemotron because its
# attention KV is only the 8 ``*`` layers (the Mamba recurrent state is already batched); DSV4/InternLM2.5
# read the shared ``PAGED_KV_BATCHED_DEFAULT`` and stay OFF until their own milestones graduate.
NEMOTRON_PAGED_KV_BATCHED_DEFAULT = True

__all__ = [
    "PAGED_KV_DEFAULT",
    "PAGED_KV_BATCHED_DEFAULT",
    "NEMOTRON_PAGED_KV_BATCHED_DEFAULT",
    "BlockAllocator",
    "CacheBlock",
    "compute_block_hash",
    "PagedCacheStats",
    "PagedKVCacheManager",
    "PagedKVCacheView",
    "PagedLatentCacheView",
    "SeqHandle",
    "RecurrentCacheStats",
    "RecurrentPrefixCache",
]
