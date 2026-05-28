"""Generation loop + sampling for the resident InternLM2.5-7B-Chat-1M model.

Mirrors :mod:`quanta.qwen25.generate` minus DCA-aware chunking (InternLM2.5 uses dynamic-NTK,
not dual-chunk attention — there's no chunk-period to honor, and the NTK base scales smoothly
with the running sequence length). A single streaming loop: prefill once with the prompt,
sample one token at a time off the cached state, stop on the eos / stop set (or
``max_new_tokens``).

The sampler is the standard quanta stack: ``temperature`` / ``top_k`` / ``top_p`` / ``min_p`` /
``repetition_penalty`` — fully vectorized in MLX, no Python loop on logits (rule-3). ``seed``
makes greedy/stochastic decoding reproducible. Default ``temperature`` / ``top_p`` / ``top_k``
mirror common chat-model defaults (InternLM2's ``generation_config.json`` only ships the eos
set, not sampling params).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Iterable

import mlx.core as mx

from quanta.internlm2.runtime import InternLM2ResidentModel


def _prefill_chunked(model: InternLM2ResidentModel, prompt_ids: mx.array, cache,
                     chunk_size: int = 4096) -> mx.array:
    """Prefill ``prompt_ids`` into ``cache``; return the **last-position** logits ``[B, vocab]``.

    Chunked unconditionally (default ``chunk_size=4096``) so the transient ``[B, T, V]`` lm-head
    materialization never gets huge at long prompts. Each chunk passes
    ``offset = chunk_idx · chunk_size`` so RoPE / dynamic-NTK see absolute positions even though
    the cache grows incrementally. ``last_only=True`` slices the residual to its last row before
    the output head — at T=262144 with vocab=92544 the full ``[B, T, V]`` materialization would
    be ~48 GB transient, vs ~180 KB for the last row alone.
    """
    T = prompt_ids.shape[-1]
    if T <= chunk_size:
        return model(prompt_ids, cache=cache, last_only=True)[:, -1, :]
    last_row = None
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = prompt_ids[..., start:end]
        last_row = model(chunk, cache=cache, offset=start, last_only=True)[:, -1, :]
        mx.eval(last_row)                                                       # bound peak memory
    return last_row


def _apply_repetition_penalty(logits: mx.array, prev_ids: mx.array, penalty: float) -> mx.array:
    """Divide already-emitted-tokens' logits (positive) or multiply (negative) by ``penalty``.

    HuggingFace-compatible: ``logits[t] = logits[t] / penalty if logits[t] > 0 else logits[t] * penalty``
    applied for every token id in ``prev_ids``. Vectorized via a ``[V]`` seen-mask + ``mx.where``.
    """
    if penalty == 1.0 or prev_ids.size == 0:
        return logits
    vocab_idx = mx.arange(logits.shape[-1])                                # [V]
    seen_mask = (vocab_idx[None, :] == prev_ids[:, None]).any(axis=0)      # [V]
    penalized = mx.where(logits > 0, logits / penalty, logits * penalty)
    return mx.where(seen_mask, penalized, logits)


def _sample(logits: mx.array, *, temperature: float, top_k: int, top_p: float,
            min_p: float, key: mx.array) -> mx.array:
    """Sample one token id from ``[vocab]`` logits (caller has already applied any rep-penalty)."""
    if temperature <= 0.0:
        return mx.argmax(logits, axis=-1)

    logits = logits / temperature

    # min_p: drop tokens whose normalized prob is < min_p * max prob (vectorized).
    if min_p > 0.0:
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        threshold = probs.max() * min_p
        logits = mx.where(probs >= threshold, logits, mx.array(float("-inf"), logits.dtype))

    # top_k: zero out everything below the kth-largest logit.
    if top_k > 0 and top_k < logits.shape[-1]:
        kth = mx.sort(logits)[-top_k]
        logits = mx.where(logits >= kth, logits, mx.array(float("-inf"), logits.dtype))

    # top_p: nucleus — keep the smallest set whose cumulative prob ≥ top_p.
    if 0.0 < top_p < 1.0:
        sorted_logits = mx.sort(logits)[::-1]
        sorted_probs = mx.softmax(sorted_logits.astype(mx.float32), axis=-1)
        cum = mx.cumsum(sorted_probs, axis=-1)
        keep = cum <= top_p
        # Always keep at least the top-1 token (shift mask right by 1).
        keep = mx.concatenate([mx.array([True]), keep[:-1]])
        cutoff = mx.where(keep, sorted_logits, mx.array(float("inf"), sorted_logits.dtype)).min()
        logits = mx.where(logits >= cutoff, logits, mx.array(float("-inf"), logits.dtype))

    return mx.random.categorical(logits, key=key)


def generate(
    model: InternLM2ResidentModel,
    prompt_ids: list[int] | mx.array,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 20,
    top_p: float = 0.8,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    eos_ids: Iterable[int] | None = None,
    seed: int = 0,
    prefill_chunk: int = 4096,
) -> list[int]:
    """Greedy / stochastic generation. Returns the **newly generated** token ids (not the prompt).

    Defaults are common chat-model sampling values (InternLM2's ``generation_config.json`` only
    ships the eos set, not sampling params). ``temperature=0`` switches to greedy ``argmax``
    (top_k / top_p / min_p / seed become no-ops).

    ``eos_ids`` defaults to the model's :attr:`~InternLM2Config.eos_token_ids` (the
    ``generation_config.json`` stop set: ``</s>`` + ``<|im_end|>`` for InternLM2.5).
    """
    if not isinstance(prompt_ids, mx.array):
        prompt_ids = mx.array(prompt_ids, dtype=mx.int32)
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids[None]                                  # [1, T]

    stop = set(int(x) for x in (eos_ids if eos_ids is not None else model.cfg.eos_token_ids))
    cache = model.new_cache()

    logits = _prefill_chunked(model, prompt_ids, cache, chunk_size=prefill_chunk)
    key = mx.random.key(seed)
    out: list[int] = []
    history = list(int(x) for x in prompt_ids[0].tolist())

    for _ in range(max_new_tokens):
        lg = logits[0].astype(mx.float32)
        if repetition_penalty != 1.0 and history:
            lg = _apply_repetition_penalty(lg, mx.array(history, dtype=mx.int32),
                                           repetition_penalty)
        key, sub = mx.random.split(key)
        tok = int(_sample(lg, temperature=temperature, top_k=top_k, top_p=top_p,
                          min_p=min_p, key=sub).item())
        if tok in stop:
            break
        out.append(tok)
        history.append(tok)
        logits = model(mx.array([[tok]], dtype=mx.int32), cache=cache)[:, -1, :]

    return out


def stream_generate(
    model: InternLM2ResidentModel,
    prompt_ids: list[int] | mx.array,
    *,
    max_new_tokens: int = 256,
    **sampling_kwargs,
) -> Iterator[int]:
    """Same as :func:`generate` but yields token ids one-by-one as they are sampled.

    Use this for streaming server endpoints (oMLX shim, the Anthropic ``/v1/messages`` SSE path) so
    the consumer sees tokens as they're produced rather than only after the loop terminates.
    """
    if not isinstance(prompt_ids, mx.array):
        prompt_ids = mx.array(prompt_ids, dtype=mx.int32)
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids[None]

    stop = set(int(x) for x in (sampling_kwargs.pop("eos_ids", None) or model.cfg.eos_token_ids))
    cache = model.new_cache()
    prefill_chunk = int(sampling_kwargs.pop("prefill_chunk", 4096))
    logits = _prefill_chunked(model, prompt_ids, cache, chunk_size=prefill_chunk)

    temperature = float(sampling_kwargs.pop("temperature", 0.7))
    top_k = int(sampling_kwargs.pop("top_k", 20))
    top_p = float(sampling_kwargs.pop("top_p", 0.8))
    min_p = float(sampling_kwargs.pop("min_p", 0.0))
    rep_pen = float(sampling_kwargs.pop("repetition_penalty", 1.0))
    seed = int(sampling_kwargs.pop("seed", 0))

    key = mx.random.key(seed)
    history = list(int(x) for x in prompt_ids[0].tolist())

    for _ in range(max_new_tokens):
        lg = logits[0].astype(mx.float32)
        if rep_pen != 1.0 and history:
            lg = _apply_repetition_penalty(lg, mx.array(history, dtype=mx.int32), rep_pen)
        key, sub = mx.random.split(key)
        tok = int(_sample(lg, temperature=temperature, top_k=top_k, top_p=top_p,
                          min_p=min_p, key=sub).item())
        if tok in stop:
            return
        yield tok
        history.append(tok)
        logits = model(mx.array([[tok]], dtype=mx.int32), cache=cache)[:, -1, :]
