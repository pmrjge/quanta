"""Autoregressive generation for Kimi-K2.6 — prefill once, then KV-cached decode.

Decode is inherently sequential (token *t+1* depends on *t*), so there is exactly one
Python loop: over decode **steps**. That loop is the same coarse, bounded kind as the
layer loop — one full forward per iteration, nothing per-element. Everything *inside* a
step is vectorized MLX:

* the per-step forward runs on the MLA latent cache via the **absorbed** decode path
  (attends the compressed ``c_kv`` directly — no per-head K/V materialized, no recompute
  of the prefix), so each step is O(1) in sequence length, not O(T);
* :func:`sample_logits` does temperature / top-k / top-p / categorical sampling over the
  whole 163K-wide vocab with ``argsort`` / ``cumsum`` / ``take_along_axis`` — no loops.

Prefill uses the expanded path (cheaper at Sq>1) and may use XAttention sparse prefill;
decode switches to absorbed. One ``mx.eval`` per step materializes the token + the grown
cache (``mx.async_eval`` can overlap steps on the resident runtime).
"""

from __future__ import annotations

from collections.abc import Iterable

import mlx.core as mx

from quanta.cache import MLACache
from quanta.model import KimiModel
from quanta.modeling.xattention import DEFAULT_SPARSE, XAttnConfig

NEG_INF = float("-inf")


def sample_logits(
    logits: mx.array,
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    key: mx.array | None = None,
) -> mx.array:
    """Sample one next token per row from ``[..., V]`` logits — fully vectorized, no loops.

    ``temperature == 0`` is greedy (argmax). Otherwise apply temperature, then optional
    top-k and top-p (nucleus) truncation, then categorical sampling.
    """
    if temperature <= 0.0:
        return mx.argmax(logits, axis=-1)

    logits = logits.astype(mx.float32) * (1.0 / temperature)
    v = logits.shape[-1]

    if 0 < top_k < v:
        kth = mx.sort(logits, axis=-1)[..., v - top_k : v - top_k + 1]  # kth-largest threshold
        logits = mx.where(logits < kth, NEG_INF, logits)

    if 0.0 < top_p < 1.0:
        order = mx.argsort(-logits, axis=-1)  # descending
        ordered = mx.take_along_axis(logits, order, axis=-1)
        probs = mx.softmax(ordered, axis=-1)
        before = mx.cumsum(probs, axis=-1) - probs  # mass strictly before each (keeps the crossing token)
        keep_ordered = before < top_p
        keep = mx.take_along_axis(keep_ordered, mx.argsort(order, axis=-1), axis=-1)  # scatter back
        logits = mx.where(keep, logits, NEG_INF)

    return mx.random.categorical(logits, axis=-1, key=key)


def generate(
    model: KimiModel,
    prompt_ids: Iterable[int],
    *,
    max_new_tokens: int,
    n_layers: int | None = None,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_id: int | None = None,
    seed: int = 0,
    sparse: XAttnConfig | None = DEFAULT_SPARSE,
    decode_absorbed: bool = True,
) -> list[int]:
    """Greedily/with-sampling generate up to ``max_new_tokens`` token ids after ``prompt_ids``.

    Stops early on ``eos_id``. ``sparse`` is XAttention prefill, **on by default** and applied
    to the prefill only (decode is single-token, so block-sparse prefill does not apply there
    and it engages only when the prompt reaches ``min_seq``); pass ``sparse=None`` for an exact
    dense prefill.

    Decode uses the absorbed-MLA path by default (``decode_absorbed=True``): decode-optimal and
    memory-light at long context (attends the compressed latent, no per-head K/V), bf16-close to
    expanded. The loop+cache itself is bit-exact vs one-shot recompute — pass
    ``decode_absorbed=False`` for an exact (heavier) decode.
    """
    n = model.cfg.num_hidden_layers if n_layers is None else n_layers
    caches = [MLACache() for _ in range(n)]

    # prefill: expanded MLA (+ optional sparse), fills the per-layer latent caches
    logits = model(
        mx.array(list(prompt_ids)), n_layers=n, use_fast=True, caches=caches, absorbed=False, sparse=sparse
    )
    mx.eval(logits, [c.c_kv for c in caches], [c.k_pe for c in caches])

    key = mx.random.key(seed)
    out: list[int] = []
    for _ in range(max_new_tokens):  # sole loop: one decode step (one forward) per token
        key, sub = mx.random.split(key)
        token = sample_logits(logits[0, -1], temperature=temperature, top_k=top_k, top_p=top_p, key=sub)
        tid = int(token.item())
        if eos_id is not None and tid == eos_id:
            break
        out.append(tid)
        logits = model(
            mx.array([tid]), n_layers=n, use_fast=True, caches=caches, offset=caches[0].offset,
            absorbed=decode_absorbed,
        )
        mx.eval(logits, [c.c_kv for c in caches], [c.k_pe for c in caches])

    return out
