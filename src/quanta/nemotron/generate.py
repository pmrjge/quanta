"""Autoregressive decode for Nemotron-H — prefill once, then an O(1)/step recurrence.

One Python loop, over decode steps (the bounded kind). Per layer the state that carries forward
is: a growing **KV cache** on attention layers, the **(ssm_state, conv_state)** recurrence on
mamba layers, and nothing on MoE. Prefill runs the chunked SSD + a single attention pass that
fills the caches (mamba ``conv_state=None`` ⇒ chunked); each decode step threads the updated
state back in and runs the mamba step recurrence + single-query attention.

``decode_state`` builds a from-scratch stepping state (zero conv-state so the step path engages
from the first token) — used by the prefill==decode parity gate.
"""

from __future__ import annotations

from collections.abc import Iterable

import mlx.core as mx

from quanta.generate import sample_logits
from quanta.nemotron.attention import KVCache
from quanta.nemotron.model import NemotronModel


def attn_caches(model: NemotronModel) -> list:
    """One KV cache per attention layer (``None`` on mamba/moe layers)."""
    return [KVCache() if k == "attention" else None for k in model.cfg.layers_block_type]


def decode_state(model: NemotronModel) -> tuple[list, list, list]:
    """From-scratch per-layer stepping state: KV cache on attention, **zero** conv-state on mamba
    (so the O(1) step path engages from token 0), ``None`` ssm-state (the step inits it to zero)."""
    cfg = model.cfg
    conv0 = mx.zeros((1, cfg.conv_kernel - 1, cfg.mamba_conv_dim))
    kinds = cfg.layers_block_type
    caches = attn_caches(model)
    ssm = [None] * len(kinds)
    conv = [conv0 if k == "mamba" else None for k in kinds]
    return caches, ssm, conv


def generate(
    model: NemotronModel,
    prompt_ids: Iterable[int],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_id: int | None = None,
    seed: int = 0,
) -> list[int]:
    """Generate up to ``max_new_tokens`` ids after ``prompt_ids``; stops early on ``eos_id``.

    Prefill uses the chunked SSD (mamba ``conv_state=None``) + a single attention pass that fills
    the KV caches; decode then threads ``(ssm, conv)`` and grows the caches one token at a time.
    """
    caches = attn_caches(model)
    logits, ssm, conv = model(mx.array(list(prompt_ids)), caches=caches)  # prefill (mamba chunked)
    key = mx.random.key(seed)
    out: list[int] = []
    for _ in range(max_new_tokens):  # sole loop: one decode step (one forward) per token
        key, sub = mx.random.split(key)
        tok = int(sample_logits(logits[0, -1], temperature=temperature, top_k=top_k, top_p=top_p, key=sub).item())
        if eos_id is not None and tok == eos_id:
            break
        out.append(tok)
        logits, ssm, conv = model(mx.array([tok]), caches=caches, ssm=ssm, conv=conv)
    return out
