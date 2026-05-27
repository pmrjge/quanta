"""Generation loop + sampling for the resident Qwen2.5-14B-Instruct-1M model.

Mirrors :mod:`quanta.qwen35.generate` minus everything Qwen2.5 lacks (MTP / spec-decode). A single
streaming loop: prefill once with the prompt, sample one token at a time off the cached state, stop
on the eos / stop set (or ``max_new_tokens``).

The sampler is the standard quanta stack: ``temperature`` / ``top_k`` / ``top_p`` / ``min_p`` / a
``repetition_penalty`` (Qwen2.5's generation_config defaults to 1.05) — fully vectorized in MLX, no
Python loop on logits (rule-3). ``seed`` makes greedy/stochastic decoding reproducible.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Iterable

import mlx.core as mx

from quanta.qwen25.runtime import Qwen25ResidentModel


def _prefill(model: Qwen25ResidentModel, prompt_ids: mx.array, cache) -> mx.array:
    """Prefill ``prompt_ids`` into ``cache``; return the **last-position** logits ``[B, vocab]``.

    Single-shot when the prompt fits a DCA chunk (or DCA is off); chunked otherwise — each
    chunk gets ``offset = chunk_idx * chunk_size`` so its RoPE offset resets to 0 in chunk-local
    space and the cache stores intra-rotated K from every chunk uniformly. The DCA-at-decode path
    then mixes the per-chunk K with the right Q rotation (intra vs successor) per cache position.
    """
    cfg = model.cfg
    T = prompt_ids.shape[-1]
    cs = cfg.dca_chunk_size if cfg.use_dca else 0
    # ``last_only=True`` slices the residual to its last position before lm_head — at T=262144,
    # the full ``[B, T, V]`` materialization is ~78 GB transient, vs ~300 KB for the last row.
    if cs <= 0 or T <= cs:
        return model(prompt_ids, cache=cache, last_only=True)[:, -1, :]
    last_row = None
    for start in range(0, T, cs):
        end = min(start + cs, T)
        chunk = prompt_ids[..., start:end]
        last_row = model(chunk, cache=cache, offset=start, last_only=True)[:, -1, :]
        mx.eval(last_row)                                                       # bound peak memory
    return last_row


def _apply_repetition_penalty(logits: mx.array, prev_ids: mx.array, penalty: float) -> mx.array:
    """Divide already-emitted-tokens' logits (positive) or multiply (negative) by ``penalty``.

    HuggingFace-compatible: ``logits[t] = logits[t] / penalty if logits[t] > 0 else logits[t] * penalty``
    applied for every token id in ``prev_ids``. Vectorized via a ``[V]`` seen-mask + ``mx.where`` —
    MLX's ``.at[idx]`` indexer only supports atomic ops (``.add`` / ``.multiply`` / …), not bulk
    ``.set``, so a mask is the cleanest in-graph path.
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
        # Last logit to keep is the first index where cum > top_p (inclusive).
        keep = cum <= top_p
        # Always keep at least the top-1 token (shift mask right by 1).
        keep = mx.concatenate([mx.array([True]), keep[:-1]])
        cutoff = mx.where(keep, sorted_logits, mx.array(float("inf"), sorted_logits.dtype)).min()
        logits = mx.where(logits >= cutoff, logits, mx.array(float("-inf"), logits.dtype))

    return mx.random.categorical(logits, key=key)


def generate(
    model: Qwen25ResidentModel,
    prompt_ids: list[int] | mx.array,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 20,
    top_p: float = 0.8,
    min_p: float = 0.0,
    repetition_penalty: float = 1.05,
    eos_ids: Iterable[int] | None = None,
    seed: int = 0,
) -> list[int]:
    """Greedy / stochastic generation. Returns the **newly generated** token ids (not the prompt).

    Defaults match Qwen2.5's ``generation_config.json`` (temperature 0.7, top_k 20, top_p 0.8,
    repetition_penalty 1.05). ``temperature=0`` switches to greedy ``argmax`` (top_k / top_p /
    min_p / seed become no-ops).

    ``eos_ids`` defaults to the model's :attr:`~Qwen25Config.eos_token_ids` (the
    ``generation_config.json`` stop set: ``<|im_end|>`` + ``<|endoftext|>`` for Qwen2.5).
    """
    if not isinstance(prompt_ids, mx.array):
        prompt_ids = mx.array(prompt_ids, dtype=mx.int32)
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids[None]                                  # [1, T]

    stop = set(int(x) for x in (eos_ids if eos_ids is not None else model.cfg.eos_token_ids))
    cache = model.new_cache()

    # Prefill — single pass when the prompt fits a DCA chunk; chunked otherwise (each chunk's
    # RoPE offset resets so K is stored intra-rotated; at decode the DCA path then attends
    # across the full multi-chunk cache with the trained-window-bounded relative positions).
    logits = _prefill(model, prompt_ids, cache)                         # [1, vocab]
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
        # Decode step — feed the just-sampled token, get next-step logit.
        logits = model(mx.array([[tok]], dtype=mx.int32), cache=cache)[:, -1, :]

    return out


def stream_generate(
    model: Qwen25ResidentModel,
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
    logits = _prefill(model, prompt_ids, cache)

    temperature = float(sampling_kwargs.pop("temperature", 0.7))
    top_k = int(sampling_kwargs.pop("top_k", 20))
    top_p = float(sampling_kwargs.pop("top_p", 0.8))
    min_p = float(sampling_kwargs.pop("min_p", 0.0))
    rep_pen = float(sampling_kwargs.pop("repetition_penalty", 1.05))
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
