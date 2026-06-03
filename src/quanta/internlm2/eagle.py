"""EAGLE-3 spec-decode + capture wiring for InternLM2.5-7B-Chat-1M — InternLM2-specific glue around
the generic :func:`quanta.eagle.spec_core.spec_generate`, mirroring :mod:`quanta.minimax.eagle`.

InternLM2.5 is **dense** (32 attention layers, no MoE, no native MTP) — the only keeper without a
speculative path. EAGLE-3 is its lossless decode lever: train a one-layer transformer drafter on the
resident model's captured low / mid / high hidden states, then spec-decode by drafting ``k`` tokens
with the drafter and verifying with one target forward — output is bit-identical to plain greedy
regardless of drafter quality (the drafter only changes *speed*).

Defaults that match InternLM2.5-7B:
  * drafter dims — hidden=4096 (= InternLM2 H), 32×128 heads (= H), 8192 SwiGLU (= 2H), RoPE θ=5e7
    (= InternLM2's ``rope_theta``), eps=1e-5 (= ``rms_norm_eps``); ``n_feature_layers=3`` to match
    the (low, mid, high) capture;
  * capture layers — (8, 16, 24) of 32, low / mid / high. (Per-corpus tuning deferred — same
    model-free contract; the choice is a per-bake decision, as in :mod:`quanta.minimax.eagle`.)

The InternLM2 runtime tracks each token's KV growth in :class:`~quanta.internlm2.decode.InternLM2Cache`
(every forward mutates the per-layer KV in place), so the spec-decode loop here passes no explicit
``cache_eval_fn`` — like MiniMax, unlike Kimi (which needs explicit latent eval under its heavy
resident base).
"""

from __future__ import annotations

import mlx.core as mx

from quanta.eagle.artifact import DrafterConfig
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.spec_core import spec_generate as _spec_core

__all__ = ["DEFAULT_CAPTURE_LAYERS", "INTERNLM2_DRAFTER_CFG", "internlm2_capture_forward",
           "spec_generate"]

DEFAULT_CAPTURE_LAYERS: tuple[int, int, int] = (8, 16, 24)  # low / mid / high of 32 decoder layers
INTERNLM2_DRAFTER_CFG = DrafterConfig(
    hidden=4096, n_heads=32, head_dim=128, intermediate=8192,
    eps=1e-5, rope_base=5e7, n_feature_layers=3, layerscale_init=1e-4,
)


def internlm2_capture_forward(model):
    """Forward callable for :func:`quanta.eagle.capture.capture_features_to_shards_fn` — takes a
    teacher-forced segment and returns ``(logits, caps)`` with ``caches=None`` (full prefill, no
    incremental state)."""
    def fwd(ids, capture_layers):
        return model(ids, caches=None, capture_layers=capture_layers)
    return fwd


def spec_generate(
    model, drafter: EagleDrafter, embed: mx.array, head: mx.array, prompt_ids,
    *, max_new: int, k: int = 4, layers: tuple[int, ...] = DEFAULT_CAPTURE_LAYERS,
    eos_id: int | None = None, head_bits: int | None = None, head_group_size: int = 64,
) -> tuple[list[int], dict]:
    """Lossless EAGLE-3 spec-decode for InternLM2.5 (:class:`~quanta.internlm2.runtime.InternLM2ResidentModel`).
    Returns ``(tokens, stats)`` — output is bit-identical to plain greedy regardless of drafter
    quality (the drafter only changes *speed*).

    Builds a fresh :class:`~quanta.internlm2.decode.InternLM2Cache` via ``model.new_cache()`` and
    wires the InternLM2 forward / truncate into :func:`quanta.eagle.spec_core.spec_generate`. The
    runtime tracks KV growth itself, so no explicit ``cache_eval_fn`` is needed. ``head_bits`` (opt-in)
    quantizes the per-draft-step vocab projection — the dominant drafter cost — and is lossless-safe
    (draft-only; the target verifies with its own head)."""
    cache = model.new_cache()

    def forward_fn(ids, c, offset, capture_layers):
        return model(ids, caches=c, offset=offset, capture_layers=capture_layers)

    def truncate_fn(c, length):
        c.truncate(length)

    return _spec_core(forward_fn, cache, truncate_fn, drafter, embed, head, prompt_ids,
                      max_new=max_new, k=k, layers=layers, eos_id=eos_id,
                      head_bits=head_bits, head_group_size=head_group_size)
