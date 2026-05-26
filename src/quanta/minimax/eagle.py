"""EAGLE-3 spec-decode + capture wiring for MiniMax-M2.7 — MiniMax-specific glue around the
generic :func:`quanta.eagle.spec_core.spec_generate` and
:func:`quanta.eagle.capture.capture_features_to_shards_fn`.

MiniMax has **no native MTP** (62 layers, 0 ``nextn`` keys — see task #103), so the lossless
speedup path is EAGLE-3: train a one-layer transformer drafter on the resident model's captured
low / mid / high hidden states, then spec-decode by drafting ``k`` tokens with the drafter and
verifying with one target forward (output bit-identical to greedy regardless of drafter quality).

Defaults that match MiniMax-M2.7:
  * drafter dims — hidden=3072 (= MiniMax H), 24×128 heads (= H), 6144 SwiGLU (= 2H), RoPE θ=5e6
    (= MiniMax's rope_theta); ``n_feature_layers=3`` to match the (low, mid, high) capture;
  * capture layers — (10, 30, 50) of 62 by analogy with Kimi's (10, 30, 50) of 61. (Per-corpus
    tuning deferred — same model-free contract; the choice is a per-bake decision.)

The MiniMax runtime self-evals each token's KV growth (see
:meth:`quanta.minimax.runtime.MiniMaxResidentModel.__call__`), so the spec-decode loop here passes
no explicit ``cache_eval_fn`` (unlike Kimi, which needs explicit ``[c.c_kv, c.k_pe]`` eval for
memory control under the 398 GiB resident base).
"""

from __future__ import annotations

import mlx.core as mx

from quanta.eagle.artifact import DrafterConfig
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.spec_core import spec_generate as _spec_core
from quanta.minimax.decode import MiniMaxCache

__all__ = ["DEFAULT_CAPTURE_LAYERS", "MINIMAX_DRAFTER_CFG", "minimax_capture_forward", "spec_generate"]

DEFAULT_CAPTURE_LAYERS: tuple[int, int, int] = (10, 30, 50)  # low / mid / high of 62 decoder layers
MINIMAX_DRAFTER_CFG = DrafterConfig(
    hidden=3072, n_heads=24, head_dim=128, intermediate=6144,
    eps=1e-6, rope_base=5e6, n_feature_layers=3, layerscale_init=1e-4,
)


def minimax_capture_forward(model):
    """Forward callable for :func:`quanta.eagle.capture.capture_features_to_shards_fn` — takes a
    teacher-forced segment and returns ``(logits, caps)`` with caches=None (full prefill, no
    incremental state)."""
    def fwd(ids, capture_layers):
        return model(ids, caches=None, capture_layers=capture_layers)
    return fwd


def spec_generate(
    model, drafter: EagleDrafter, embed: mx.array, head: mx.array, prompt_ids,
    *, max_new: int, k: int = 4, layers: tuple[int, ...] = DEFAULT_CAPTURE_LAYERS,
    eos_id: int | None = None,
) -> tuple[list[int], dict]:
    """Lossless EAGLE-3 spec-decode for MiniMax-M2.7 (``MiniMaxResidentModel``). Returns
    ``(tokens, stats)`` — output is bit-identical to plain greedy regardless of drafter quality
    (the drafter only changes *speed*).

    Builds a :class:`MiniMaxCache` sized to the model and wires the MiniMax forward / truncate into
    :func:`quanta.eagle.spec_core.spec_generate`. The MiniMax runtime self-evals each token's KV
    growth, so no explicit ``cache_eval_fn`` is needed."""
    # int8 KV by default — matches the Kimi pattern (MLACache(quantized=True) since #47).
    cache = MiniMaxCache(model.num_layers, quantized=True)

    def forward_fn(ids, c, offset, capture_layers):
        return model(ids, caches=c, offset=offset, capture_layers=capture_layers)

    def truncate_fn(c, length):
        c.truncate(length)

    return _spec_core(forward_fn, cache, truncate_fn, drafter, embed, head, prompt_ids,
                     max_new=max_new, k=k, layers=layers, eos_id=eos_id)
