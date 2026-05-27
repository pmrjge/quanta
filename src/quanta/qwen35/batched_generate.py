"""Continuous-batching generator for Qwen3.5 — B concurrent streams over shared resident weights.

Drives :class:`quanta.qwen35.batched_runtime.Qwen35BatchedResidentModel.step_batch` for many
concurrent prompts. Per-stream sampling reuses :func:`quanta.qwen35.generate.sample_logits` exactly
(same temperature / top-k / top-p / min-p / seeded categorical math, so a stream emits the same
token at offset ``q`` whether it runs single-stream or batched).

The orchestrator (the agentic-loop server) consumes :func:`generate_batched` like this:

* a fresh :class:`quanta.qwen35.decode.Qwen35Cache` per stream (the model's ``make_caches()``);
* the prompt is consumed by :meth:`Qwen35BatchedResidentModel.prefill` ONCE per stream (full
  attention's prefill needs a per-stream offset / mask, so prefill is single-stream — Design A);
* then a single bounded outer loop steps every still-active stream **together**: each iteration
  is one :meth:`step_batch` call (one read of the routed-expert weights for all active streams),
  followed by per-stream sampling. A stream that hits eos / its ``max_new_tokens`` drops out; the
  loop ends when every stream is finished or the global budget is exhausted.

There is no rebatching / left-padding (Design A keeps per-stream caches: dropping a stream is just
removing it from the active list, no cache compaction). The outer loop is the only loop and is
bounded by the global ``max_new_tokens`` × ``B`` — within a step everything is the same vectorized
work the single-stream path does, plus the small per-stream mixer-step IO loop in ``step_batch``.

This module **does not** load the model. Pass it the batched runtime + per-stream prompts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import mlx.core as mx

from quanta.qwen35.decode import Qwen35Cache
from quanta.qwen35.generate import _normalize_eos, sample_logits


class _StreamState:
    """One active stream's per-step state: cache, current offset, the next token to feed, and the
    accumulated emitted tokens. ``done`` once it hit eos or its own ``max_new``.

    ``prompt_len`` is the absolute offset right after prefill (= len(prompt)), so the trailing
    cache-advance pass can detect streams whose last sampled token has been emitted (added to
    ``out``) but not yet fed to ``step_batch`` (cache aligned with ``out`` iff
    ``offset == prompt_len + len(out)``)."""

    __slots__ = ("cache", "offset", "next_tok", "out", "max_new", "stop", "prompt_len",
                 "temperature", "top_k", "top_p", "min_p", "key", "done")

    def __init__(self, cache: Qwen35Cache, next_tok: int, offset: int, *,
                 max_new: int, stop: set[int], temperature: float, top_k: int,
                 top_p: float, min_p: float, key: mx.array) -> None:
        self.cache = cache
        self.offset = offset                # absolute position the next decode will write
        self.prompt_len = offset             # remembered prefill end (== len(prompt)); used by the
                                             # trailing cache-advance pass to detect unfed tail tokens
        self.next_tok = next_tok            # token to feed in the next step (= the just-emitted one)
        self.out: list[int] = []
        self.max_new = int(max_new)
        self.stop = stop
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.min_p = float(min_p)
        self.key = key
        self.done = False


def _seed_one_stream(model, prompt_ids: list[int], cache: Qwen35Cache) -> tuple[int, mx.array]:
    """Consume a prompt into ``cache`` and return ``(first_sampled_token_unfiltered, last_logits)``.

    Uses :meth:`Qwen35BatchedResidentModel.prefill` which is the proven single-stream forward —
    each prompt is consumed independently into its own cache, so per-stream prefill is parity-trivial.
    """
    logits = model.prefill(prompt_ids, cache)             # [1,1,vocab] at last consumed position
    return logits


def generate_batched(
    model,
    prompts: Sequence[Iterable[int]],
    *,
    max_new_tokens: int,
    temperature: float | Sequence[float] = 0.0,
    top_k: int | Sequence[int] = 0,
    top_p: float | Sequence[float] = 1.0,
    min_p: float | Sequence[float] = 0.0,
    eos_id=None,
    seeds: int | Sequence[int] = 0,
    caches: Sequence[Qwen35Cache] | None = None,
) -> list[list[int]]:
    """Generate up to ``max_new_tokens`` per prompt; return per-prompt token lists.

    ``model``: a :class:`quanta.qwen35.batched_runtime.Qwen35BatchedResidentModel` (or any object
    exposing ``step_batch(...)`` + ``prefill(prompt_ids, state)`` + ``make_caches()`` + ``.cfg`` /
    ``.num_layers``). ``prompts`` is a sequence of prompt token-id iterables (one per stream).

    Per-stream sampling: ``temperature`` / ``top_k`` / ``top_p`` / ``min_p`` accept a scalar
    (applied to every stream) OR a per-stream sequence of the same length as ``prompts``. ``seeds``
    likewise accepts a single ``int`` (offset per stream so each stream has its own key) or a
    per-stream sequence. ``eos_id`` is a single ``int`` / collection of stop ids / ``None``,
    applied to every stream (mirrors :func:`quanta.qwen35.generate.generate`'s contract).

    ``caches`` lets the orchestrator pass pre-built per-stream caches (e.g. when continuing an
    earlier conversation); when ``None`` a fresh :class:`Qwen35Cache` is built per stream via
    ``model.make_caches()``.

    Per-stream output is bit-identical to running :func:`quanta.qwen35.generate.generate` on each
    prompt alone with the same args (the batched runtime's step is parity-gated equal to the
    single-stream step, and the sampler is the very same function).
    """
    prompts = [list(p) for p in prompts]
    b = len(prompts)
    if b == 0:
        return []
    for i, p in enumerate(prompts):
        if not p:
            raise ValueError(f"prompt {i} is empty (need >= 1 token to prefill)")

    # broadcast scalar samplers to per-stream sequences ----------------------
    def _broadcast(v, name: str, cast):
        if isinstance(v, (int, float)):
            return [cast(v)] * b
        seq = list(v)
        if len(seq) != b:
            raise ValueError(f"{name}: got {len(seq)} values for {b} streams")
        return [cast(x) for x in seq]
    temps = _broadcast(temperature, "temperature", float)
    topks = _broadcast(top_k, "top_k", int)
    topps = _broadcast(top_p, "top_p", float)
    minps = _broadcast(min_p, "min_p", float)
    if isinstance(seeds, int):
        seed_keys = [mx.random.key(int(seeds) + i) for i in range(b)]
    else:
        seed_seq = list(seeds)
        if len(seed_seq) != b:
            raise ValueError(f"seeds: got {len(seed_seq)} values for {b} streams")
        seed_keys = [mx.random.key(int(s)) for s in seed_seq]

    # build per-stream caches (or use what the caller passed) ----------------
    if caches is None:
        caches = [model.make_caches() for _ in range(b)]
    elif len(caches) != b:
        raise ValueError(f"caches: got {len(caches)} for {b} streams")

    stop = _normalize_eos(eos_id)

    # per-stream prefill (one stream at a time — full-attn prefill needs a per-stream offset; this
    # is the Design A trade-off, and matches what the single-stream contract already does). Sample
    # the first emitted token per stream from the last-position logits.
    streams: list[_StreamState] = []
    for i, p in enumerate(prompts):
        logits = _seed_one_stream(model, p, caches[i])         # [1,1,vocab]
        seed_keys[i], sub = mx.random.split(seed_keys[i])
        first_tok = int(sample_logits(logits[0, -1], temperature=temps[i], top_k=topks[i],
                                      top_p=topps[i], min_p=minps[i], key=sub).item())
        st = _StreamState(cache=caches[i], next_tok=first_tok, offset=len(p),
                          max_new=max_new_tokens, stop=stop,
                          temperature=temps[i], top_k=topks[i], top_p=topps[i], min_p=minps[i],
                          key=seed_keys[i])
        # the first sampled token is either emitted (and fed next step) or stopped here
        if first_tok in stop:
            st.done = True
        else:
            st.out.append(first_tok)
            if len(st.out) >= max_new_tokens:
                st.done = True
        streams.append(st)

    # continuous-batching outer loop: every iteration is ONE step_batch over the still-active streams.
    # Bounded by max_new_tokens (the only loop; no per-token inner loops on the hot path).
    for _step in range(max_new_tokens):
        active_idx = [i for i, s in enumerate(streams) if not s.done]
        if not active_idx:
            break
        active = [streams[i] for i in active_idx]
        tok_ids = [s.next_tok for s in active]
        active_caches = [s.cache for s in active]
        active_offsets = [s.offset for s in active]
        per_stream_logits = model.step_batch(tok_ids, active_caches, active_offsets)
        # per-stream sample + update; eval the materialized logits across all streams together so
        # the MoE call's outputs land in one graph eval, not B separate ones.
        mx.eval([lg for lg in per_stream_logits])
        for s, lg in zip(active, per_stream_logits, strict=True):
            s.key, sub = mx.random.split(s.key)
            tok = int(sample_logits(lg[0, -1], temperature=s.temperature, top_k=s.top_k,
                                    top_p=s.top_p, min_p=s.min_p, key=sub).item())
            s.offset += 1                # the token we just fed has been committed to the cache
            if tok in s.stop:
                s.done = True
                continue
            s.out.append(tok)
            if len(s.out) >= s.max_new:
                s.done = True
                continue
            s.next_tok = tok             # feed this token in the next batched step

    # Trailing cache-advance: feed each stream's last sampled-but-unfed token so the cache offset
    # ends at ``prompt_len + len(out)`` — bit-equivalent to single-stream :func:`generate` (which
    # is feed-then-sample, so the last emitted token is in its cache). Without this the
    # orchestrator's next ``step_batch`` on a continued conversation would desync (caught loudly
    # by ``step_batch``'s offset check — rule 6 — but still divergent). Streams that ended on eos
    # or returned empty are already aligned (the eos sample was discarded, never fed; the unfed
    # tail is only present when the stream ran to ``max_new`` on a non-eos token).
    pending = [s for s in streams if s.offset < s.prompt_len + len(s.out)]
    if pending:
        tok_ids = [s.out[-1] for s in pending]
        pending_caches = [s.cache for s in pending]
        pending_offsets = [s.offset for s in pending]
        advance_logits = model.step_batch(tok_ids, pending_caches, pending_offsets)
        mx.eval([lg for lg in advance_logits])    # commit the cache write; the logits are discarded
        for s in pending:
            s.offset += 1

    return [s.out for s in streams]
