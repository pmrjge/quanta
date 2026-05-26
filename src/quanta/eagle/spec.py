"""EAGLE-3 spec-decode for Kimi-K2.6 — Kimi-specific wiring around the generic
:func:`quanta.eagle.spec_core.spec_generate` (lossless verify / accept / rollback).

Provides the Kimi forward signature (``sparse`` / ``absorbed`` kwargs), the :class:`MLACache`
factory, the explicit ``[c.c_kv, c.k_pe]`` ``mx.eval`` that controls memory under the 398 GiB
resident model, and the per-layer truncate that drops a rejected draft. Output is bit-identical to
plain greedy decode regardless of drafter quality (the drafter only changes *speed*, never
correctness).
"""

from __future__ import annotations

import mlx.core as mx

from quanta.cache import MLACache
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.spec_core import spec_generate as _spec_core
from quanta.modeling.xattention import DEFAULT_SPARSE

LAYERS = (10, 30, 50)


def spec_generate(
    model, drafter: EagleDrafter, embed: mx.array, head: mx.array, prompt_ids,
    *, max_new: int, k: int = 4, layers: tuple[int, ...] = LAYERS,
    quantized_kv: bool = True, sparse=DEFAULT_SPARSE, absorbed: bool = False,
    eos_id: int | None = None,
) -> tuple[list[int], dict]:
    """Lossless EAGLE spec-decode for Kimi-K2.6. Returns ``(tokens, stats)`` — ``stats['mean_accept']``
    is mean tokens emitted per target forward (1 = no speedup, ``k+1`` = perfect). ``absorbed=True``
    routes the verify through the absorbed MLA fast path (cheaper SDPA, same argmax → lossless
    preserved); ``absorbed=False`` (default) keeps the historical path the drafter was trained
    against."""
    n = model.cfg.num_hidden_layers
    caches = [MLACache(quantized=quantized_kv) for _ in range(n)]

    def forward_fn(ids, c, offset, capture_layers):
        return model(ids, caches=c, offset=offset, capture_layers=capture_layers,
                     absorbed=absorbed, sparse=sparse)

    def truncate_fn(c, length):
        for layer in c:
            layer.truncate(length)

    def cache_eval_fn(c):
        return [layer.c_kv for layer in c] + [layer.k_pe for layer in c]

    return _spec_core(forward_fn, caches, truncate_fn, drafter, embed, head, prompt_ids,
                     max_new=max_new, k=k, layers=layers, eos_id=eos_id,
                     cache_eval_fn=cache_eval_fn)
