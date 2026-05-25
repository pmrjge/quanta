"""Autoregressive generation for GLM-5.1 — seed the decode cache, then KV-cached decode.

Mirrors :mod:`quanta.dsv4.generate` (and the Kimi/Nemotron generators): two coarse, bounded loops
(over prompt positions to seed the cache, then over generated tokens), one full ``model(...)`` forward
per iteration, everything inside vectorized. Decode is sequential by nature; the single-token stepper
threads a per-stack cache (``quanta.glm.decode.GLMCache``, built lazily so this module imports before
that decode unit lands — the model-free gate passes its own cache).

``sample_logits`` is the shared sampler math (greedy at ``temperature==0``; else temperature → top-k →
top-p → min-p → seeded categorical) so the standalone generator and the oMLX shim agree token-for-token.
GLM's generation stop set is ``{<|endoftext|>, <|user|>, <|observation|>}`` (config ``eos_token_id``);
pass it via ``eos_id`` (an int or a collection).
"""

from __future__ import annotations

from collections.abc import Iterable

import mlx.core as mx

NEG_INF = -mx.inf


def _apply_top_p(logits: mx.array, top_p: float) -> mx.array:
    """Nucleus filter: keep the smallest prefix of descending-probability tokens whose cumulative mass
    (strictly before each) is < ``top_p`` (so the crossing token is kept). Vectorized, no loop."""
    if not 0.0 < top_p < 1.0:
        return logits
    order = mx.argsort(-logits, axis=-1)
    ordered = mx.take_along_axis(logits, order, axis=-1)
    probs = mx.softmax(ordered, axis=-1)
    before = mx.cumsum(probs, axis=-1) - probs
    keep_ordered = before < top_p
    keep = mx.take_along_axis(keep_ordered, mx.argsort(order, axis=-1), axis=-1)
    return mx.where(keep, logits, NEG_INF)


def _apply_min_p(logits: mx.array, min_p: float) -> mx.array:
    """Min-p filter: drop tokens whose probability is below ``min_p * max_prob``."""
    if min_p <= 0.0:
        return logits
    probs = mx.softmax(logits, axis=-1)
    return mx.where(probs < min_p * mx.max(probs, axis=-1, keepdims=True), NEG_INF, logits)


def sample_logits(
    logits: mx.array,
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    key: mx.array | None = None,
) -> mx.array:
    """Sample one next token per row from ``[..., V]`` logits — fully vectorized, no loops.

    ``temperature == 0`` is greedy (argmax). Otherwise apply temperature, then optional top-k, top-p
    (nucleus) and min-p truncation, then categorical sampling (``key`` ⇒ reproducible). Mirrors the
    oMLX shim's ``_sample`` math so the engine and the standalone generator agree token-for-token."""
    if temperature <= 0.0:
        return mx.argmax(logits, axis=-1)

    lg = logits.astype(mx.float32) * (1.0 / temperature)
    v = lg.shape[-1]
    if 0 < top_k < v:
        kth = mx.sort(lg, axis=-1)[..., v - top_k : v - top_k + 1]
        lg = mx.where(lg < kth, NEG_INF, lg)
    lg = _apply_min_p(_apply_top_p(lg, top_p), min_p)
    return mx.random.categorical(lg, axis=-1, key=key)


def _normalize_eos(eos_id) -> set[int]:
    """Accept an ``int`` eos, a collection of stop ids, or ``None`` → a set of ints."""
    if eos_id is None:
        return set()
    if isinstance(eos_id, (set, frozenset, tuple, list)):
        return {int(s) for s in eos_id if s is not None}
    return {int(eos_id)}


def generate(
    model,
    prompt_ids: Iterable[int],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    eos_id: int | None = None,
    seed: int = 0,
    cache=None,
) -> list[int]:
    """Generate up to ``max_new_tokens`` ids after ``prompt_ids``; stops early on ``eos_id``.

    Seeds the decode cache by stepping the prompt one token at a time (positions ``0..len-1``), then the
    bounded decode loop threads each new token at the next absolute position. ``model`` is a
    :class:`quanta.glm.runtime.GLMResidentModel` (or any object with ``.num_layers`` and the single-token
    ``__call__(token_ids, caches=, offset=)`` contract). ``eos_id`` may be an ``int`` or a collection of
    stop ids; the loop also stops at ``max_new_tokens`` so it can never run unbounded.

    ``cache`` is the per-stack decode cache; when ``None`` a fresh :class:`quanta.glm.decode.GLMCache`
    is built (imported lazily). Callers (and the model-free gate) may pass their own (e.g. a stub)."""
    ids = list(prompt_ids)
    if not ids:
        raise ValueError("prompt_ids is empty (need at least one token to prefill)")
    if cache is None:
        from quanta.glm.decode import GLMCache
        cache = GLMCache(model.num_layers)

    logits = None
    for pos, tid in enumerate(ids):
        logits = model(mx.array([tid]), caches=cache, offset=pos)
        mx.eval(logits)

    stop = _normalize_eos(eos_id)
    key = mx.random.key(seed)
    offset = len(ids)
    out: list[int] = []
    for _ in range(max_new_tokens):  # sole generation loop: one decode step (one forward) per token
        key, sub = mx.random.split(key)
        tok = int(sample_logits(logits[0, -1], temperature=temperature, top_k=top_k,
                                top_p=top_p, min_p=min_p, key=sub).item())
        if tok in stop:
            break
        out.append(tok)
        logits = model(mx.array([tok]), caches=cache, offset=offset)
        mx.eval(logits)
        offset += 1
    return out
