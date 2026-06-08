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
    manager_seq_of,
)
from quanta.paged.recurrent_cache import RecurrentCacheStats, RecurrentPrefixCache

PAGED_KV_DEFAULT = True

# #153: batched-paged KV decode — ONE block-table scatter + ONE gather across all B lock-step streams
# (the paged sibling of the #18 arena loop-kill), serving the paged keepers (Nemotron, DSV4,
# InternLM2.5). This SHARED flag now governs DSV4 ONLY, and has GRADUATED to ON (#153 M3): the DSV4
# steppers are wired (M1 dense / M2 compressed — latent-only batched via ``_PagedKVArena``, derived
# ckv/ikv/ring per-stream so the boundary-snapshot lifecycle is unchanged) and parity-green model-free —
# batched-paged == the per-stream paged loop BIT-EXACT through the real ``_DSV4BatchedSession`` decode
# (``parity/dsv4_paged_latent_test.py`` §C), and the M0 storage primitives
# (``PagedKVCacheManager.write_*_batched`` / ``gather_*_batched``) are gated in
# ``parity/dsv4_paged_batched_test.py``. M4 ✅ DONE (``parity/dsv4_paged_batched_bench.py``): loop ==
# loopkill BIT-exact on the real int4-g64 bake at B∈{1,32,48} AND +13% decode tok/s at B=32 & B=48 (the
# per-stream latent loop replaced by ONE scatter + ONE gather; smaller than #18 M5's unpaged arena/bat
# +37% since DSV4 batches only the latent — derived ckv/ikv/ring stay per-stream — and MoE dominates
# decode FLOPs). Rule 4 satisfied: parity proven ⇒ default ON, one flag to
# revert. (Nemotron AND InternLM2.5 graduated earlier via their own scoped ON defaults below, precisely
# so flipping them did not preempt this DSV4 M3 regression.)
PAGED_KV_BATCHED_DEFAULT = True

# Nemotron-scoped #153 default: GRADUATED to ON. The loop-kill is parity-proven model-free (bit-exact,
# ``parity/nemotron_batched_attention_test.py`` §D) AND on the real int4-g64 120B-A12B bake — greedy-exact
# vs the per-stream paged loop at B∈{1,32,48} with a measured **+18% decode tok/s at B=48** (+15% @ B=32,
# the prod operating point; ``parity/nemotron_paged_batched_bench.py``). The per-stream loop stops scaling
# past B=32 (122 tok/s @48 < 126 @32) while the loop-kill holds, so the win grows with B (rule 4 satisfied:
# parity proven + a real win ⇒ default ON, still a single flag to revert). Scoped to Nemotron because its
# attention KV is only the 8 ``*`` layers (the Mamba recurrent state is already batched); DSV4 reads the
# shared ``PAGED_KV_BATCHED_DEFAULT`` and stays OFF until its own milestones graduate (InternLM2.5 has its
# own scoped ON default below).
NEMOTRON_PAGED_KV_BATCHED_DEFAULT = True

# InternLM2.5-scoped #153 default: GRADUATED to ON. The loop-kill is parity-proven model-free (bit-exact,
# ``parity/internlm2_batched_attention_test.py`` §C) AND on the real int8-g64 7B-Chat-1M bake —
# greedy-exact vs the per-stream paged loop at B∈{1,32,48} with a measured **3.20x decode tok/s at B=32**
# (the prod operating point; 3.16x @ B=48; ``parity/internlm2_paged_batched_bench.py``). The win is far
# bigger than Nemotron's because InternLM2.5 is DENSE — ALL 32 layers are attention, so EVERY layer's
# per-stream KV ``.update()`` loop is killed (Nemotron trims only its 8 ``*`` attention layers). The
# per-stream loop stops scaling past B=32 (102 tok/s @48 < 104 @32) while the loop-kill holds flat at
# ~322–332 tok/s, so the win does not fade with B (rule 4 satisfied). Scoped like Nemotron's so DSV4 (the
# shared flag) is untouched until its own M3.
INTERNLM2_PAGED_KV_BATCHED_DEFAULT = True

__all__ = [
    "PAGED_KV_DEFAULT",
    "PAGED_KV_BATCHED_DEFAULT",
    "NEMOTRON_PAGED_KV_BATCHED_DEFAULT",
    "INTERNLM2_PAGED_KV_BATCHED_DEFAULT",
    "BlockAllocator",
    "CacheBlock",
    "compute_block_hash",
    "PagedCacheStats",
    "PagedKVCacheManager",
    "PagedKVCacheView",
    "PagedLatentCacheView",
    "SeqHandle",
    "manager_seq_of",
    "RecurrentCacheStats",
    "RecurrentPrefixCache",
]
