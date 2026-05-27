"""Continuous-batching generation for DSV4 — many concurrent agent streams share resident weights.

Drives :class:`quanta.dsv4.batched_runtime.DSV4BatchedResidentModel`. Each input prompt becomes a
stream with its own :class:`quanta.dsv4.decode.DSV4Cache` and sampling key. The serving loop keeps
the active set saturated up to ``max_batch``: prompts admitted into freed slots prefill via the
single-stream path (the inner model — parity-correct), then decode in lock-step alongside other
active streams via :meth:`DSV4BatchedResidentModel.step_batch`. Streams retire on eos or
``max_new``; their completions go back to the caller in the input prompts' order.

Sampling math mirrors :func:`quanta.dsv4.generate.sample_logits` exactly (greedy at temperature=0;
otherwise temperature → optional top-k → top-p → min-p → seeded categorical). Per-stream sampling
keys (split off a single seed) make each stream's draws reproducible and independent.

The continuous-batching scheduler is the only loop structure — admit → step the active set → check
stop conditions → retire / admit. The hot path (the batched ``step_batch`` call) is the bandwidth
win; admission / retirement are rare bookkeeping (a few per generated tok at full saturation).
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from quanta.dsv4.batched_runtime import DSV4BatchedResidentModel
from quanta.dsv4.generate import _normalize_eos, sample_logits


class _Stream:
    """Per-stream serving state — prompt, cache, sampling key, running output, stop conditions."""

    __slots__ = ("idx", "cache", "key", "prompt", "out", "offset", "last_token", "max_new", "done")

    def __init__(self, idx: int, prompt: Sequence[int], cache, key: mx.array, max_new: int) -> None:
        self.idx = idx                                  # input-order index for the caller's output
        self.cache = cache                              # DSV4Cache
        self.key = key                                  # mx.random.key for this stream's sampling
        self.prompt = list(prompt)
        self.out: list[int] = []
        self.offset = 0                                 # next-token absolute position
        self.last_token: int | None = None              # the token to feed at the next step
        self.max_new = int(max_new)
        self.done = False


def batched_generate(
    model: DSV4BatchedResidentModel,
    prompts: Sequence[Sequence[int]],
    *,
    max_new: int,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    eos_ids: set[int] | None = None,
    seed: int = 0,
    max_batch: int | None = None,
) -> list[list[int]]:
    """Continuous-batching greedy / sampling decode — returns each prompt's completion in input order.

    Saturates the active set up to ``min(max_batch or model.max_batch, len(prompts))``: prompts are
    admitted into freed slots, prefill seeds their cache (single-stream, parity-correct), then they
    decode in lock-step with the rest of the active set via :meth:`step_batch`. A stream retires when
    it hits an eos in ``eos_ids`` OR reaches ``max_new`` generated tokens. The order of the returned
    completions matches the order of ``prompts``.

    ``eos_ids`` is the set of stop ids (``None`` ⇒ no early stop, decode runs the full ``max_new``).
    Per-stream sampling: a global ``seed`` is split per stream (so each stream's reproducible draw
    sequence is independent of how many other streams happen to share the batch). Greedy decode
    (``temperature == 0``) ignores the keys.
    """
    if max_new <= 0:
        raise ValueError(f"max_new must be > 0, got {max_new}")
    n = len(prompts)
    if n == 0:
        return []

    cap = model.max_batch if max_batch is None else int(max_batch)
    if cap < 1:
        raise ValueError(f"max_batch must be >= 1, got {cap}")
    cap = min(cap, model.max_batch)

    stops = _normalize_eos(eos_ids)
    outputs: list[list[int]] = [[] for _ in range(n)]

    # Per-stream sampling keys split off a master key; deterministic regardless of admission order.
    master = mx.random.key(int(seed))
    sub_keys = mx.random.split(master, n) if n > 0 else mx.array([])

    # Admission queue (input order) and active set; once a slot frees, the next pending stream admits.
    pending: list[int] = list(range(n))
    active: list[_Stream] = []

    def _admit_next() -> None:
        """Pop the next pending input, prefill its cache (single-stream — parity-correct), and add it
        to the active set. The first generated token is the argmax/sample of the prefill's final
        logits — the same rule the single-stream generator uses."""
        if not pending:
            return
        i = pending.pop(0)
        prompt = prompts[i]
        if len(prompt) == 0:
            raise ValueError(f"prompts[{i}] is empty (need >= 1 token to prefill)")
        cache = model.make_cache()
        # Prefill returns logits [1, T, vocab]; the next-token logits are the LAST position's row.
        logits = model.prefill(mx.array(list(prompt)), cache)
        mx.eval(logits)
        key = sub_keys[i] if n > 0 else mx.random.key(0)
        key, sub = mx.random.split(key)
        first = int(sample_logits(
            logits[0, -1], temperature=temperature, top_k=top_k, top_p=top_p,
            min_p=min_p, key=sub,
        ).item())
        st = _Stream(idx=i, prompt=prompt, cache=cache, key=key, max_new=max_new)
        st.offset = len(prompt)               # cache offset == prompt length after prefill
        if first in stops:
            # eos as the very first generated token → empty completion, retire immediately.
            outputs[i] = []
            return
        st.out.append(first)
        st.last_token = first
        # ``max_new`` counts emitted tokens INCLUDING the prefill-sampled first token (matching
        # :func:`quanta.dsv4.generate.generate`'s semantics of "tokens generated" being all sampled
        # outputs). If the first token already exhausts the quota, retire without entering decode.
        if len(st.out) >= st.max_new:
            outputs[i] = list(st.out)
            return
        active.append(st)

    # Seed the active set up to cap.
    while pending and len(active) < cap:
        _admit_next()

    # Continuous-batching main loop: step the active set in lock-step, then retire / admit.
    while active:
        # Each stream feeds its last sampled token at its next absolute position; step_batch grows
        # all caches by exactly one position.
        ids_per_stream = [mx.array([st.last_token]) for st in active]
        caches = [st.cache for st in active]
        offsets = [st.offset for st in active]
        logits_per_stream = model.step_batch(ids_per_stream, caches, offsets)
        mx.eval(logits_per_stream)

        next_active: list[_Stream] = []
        for st, logits in zip(active, logits_per_stream, strict=True):
            st.offset += 1
            st.key, sub = mx.random.split(st.key)
            tok = int(sample_logits(
                logits[0, -1], temperature=temperature, top_k=top_k, top_p=top_p,
                min_p=min_p, key=sub,
            ).item())
            if tok in stops:
                st.done = True
                outputs[st.idx] = list(st.out)
                continue
            st.out.append(tok)
            st.last_token = tok
            if len(st.out) >= st.max_new:
                st.done = True
                outputs[st.idx] = list(st.out)
                continue
            next_active.append(st)

        # Backfill freed slots from the pending queue (keeps the batch saturated).
        active = next_active
        while pending and len(active) < cap:
            _admit_next()

    return outputs
