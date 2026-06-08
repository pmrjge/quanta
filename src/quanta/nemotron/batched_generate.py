"""Continuous-batching decode for the Nemotron-H batched runtime.

Admits up to ``max_batch`` concurrent decode streams against a single shared resident weight
set: prefill each prompt into a freed slot (single-stream chunked prefill on the inner
resident model), then step all active slots in lockstep through :meth:`step_batch`. The
bandwidth win lives in the MoE call — ``mx.gather_qmm`` over a stacked ``[B, 1, dim]`` input
amortizes the always-on expert reads across streams. Per-stream stop conditions (eos / max
new) free a slot for the next pending prompt; a slot's per-stream state is reset on admit.

Reuses :func:`quanta.generate.sample_logits` for the per-stream sampler (same temperature /
top-k / top-p / min-p surface the single-stream Nemotron generate exposes). ``min_p`` is
forwarded to the sampler (drops tokens below ``min_p * max_prob``; ``min_p == 0`` is a no-op).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import mlx.core as mx

from quanta.generate import sample_logits
from quanta.nemotron.batched_runtime import NemotronBatchedResidentModel


def _normalize_eos(eos_ids) -> set[int]:
    """``eos_ids``: ``None`` | ``int`` | iterable[int] → frozen-style set of stop ids."""
    if eos_ids is None:
        return set()
    if isinstance(eos_ids, int):
        return {int(eos_ids)}
    return {int(e) for e in eos_ids}


class _Slot:
    """One active decode stream — its own prompt position, per-stream state, accumulated output.

    State (see :func:`quanta.nemotron.batched_runtime.make_stream_state`) is mutated in place by
    the per-step :meth:`NemotronBatchedResidentModel.step_batch` call; the slot just holds the
    triple + the next-token id + the rolling output list."""

    __slots__ = ("idx", "state", "next_id", "tokens", "stopped", "key")

    def __init__(self, idx: int, state, next_id: int, key: mx.array) -> None:
        self.idx = idx          # caller-side prompt index (so we can return outputs in order)
        self.state = state      # (caches, ssm, conv) — mutated in place by step_batch
        self.next_id = next_id  # the token to feed at the next step
        self.tokens: list[int] = []
        self.stopped = False
        self.key = key          # per-stream rng key (so seed-controlled outputs stay reproducible)


def batched_generate(
    model: NemotronBatchedResidentModel,
    prompts: Sequence[Iterable[int]],
    *,
    max_new: int,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    eos_ids=None,
    seed: int = 0,
) -> list[list[int]]:
    """Decode all ``prompts`` concurrently up to ``max_new`` tokens each; returns per-prompt id
    lists in caller order. Each prompt's per-stream state is admitted into a freed slot, prefilled,
    and stepped each round; per-stream stop fires on ``eos_ids`` or the per-stream new-token cap.

    ``temperature``/``top_k``/``top_p``/``min_p``: passed through to :func:`sample_logits` (``min_p``
    drops tokens below ``min_p * max_prob``; ``min_p == 0`` is a no-op). ``seed`` seeds a per-stream
    rng (so identical prompts under sampling still produce identical streams when given identical
    per-stream sub-keys — and the result is deterministic with respect to the admit order)."""
    if max_new <= 0:
        raise ValueError(f"max_new must be positive, got {max_new}")

    prompts_list: list[list[int]] = [list(p) for p in prompts]
    n_prompts = len(prompts_list)
    if n_prompts == 0:
        return []
    if any(len(p) == 0 for p in prompts_list):
        raise ValueError("batched_generate needs all prompts non-empty (mirrors the spec_generate gate)")

    stop = _normalize_eos(eos_ids)
    outputs: list[list[int]] = [[] for _ in range(n_prompts)]

    base_key = mx.random.key(int(seed))
    # split off one sub-key per prompt up front — guarantees per-stream sampler determinism
    # regardless of which slot a prompt is admitted into (admit-order-independence).
    keys = mx.random.split(base_key, n_prompts)

    # pending = prompts not yet admitted; slots = currently active streams.
    pending: list[int] = list(range(n_prompts))
    slots: list[_Slot] = []

    def _admit_one(pi: int) -> _Slot:
        """Prefill prompt ``pi`` into a fresh per-stream state and return its slot."""
        state = model.make_stream_state()
        logits = model.prefill(mx.array(prompts_list[pi]), state)
        mx.eval(logits)
        # initial sampler step on the prefill's last-position logits (one per stream).
        key_s, sub = mx.random.split(keys[pi])
        keys[pi] = key_s
        tok = int(sample_logits(logits[0, -1], temperature=temperature, top_k=top_k,
                                 top_p=top_p, min_p=min_p, key=sub).item())
        return _Slot(idx=pi, state=state, next_id=tok, key=keys[pi])

    # admit until max_batch or out of pending
    while pending and len(slots) < model.max_batch:
        slots.append(_admit_one(pending.pop(0)))

    while slots:
        # filter out any slots that hit their stop condition BEFORE the next forward — eos /
        # max_new check on the just-sampled ``next_id`` (mirrors single-stream nemotron.generate:
        # sample → bail-if-eos-or-cap BEFORE appending → otherwise append + forward → sample).
        active: list[_Slot] = []
        for s in slots:
            if s.next_id in stop or len(s.tokens) >= max_new:
                s.stopped = True
                continue
            s.tokens.append(s.next_id)
            # second stop check on max_new AFTER append: an emitted-but-non-eos token still
            # counts toward the cap; once tokens == max_new we don't take another forward.
            if len(s.tokens) >= max_new:
                s.stopped = True
                continue
            active.append(s)

        if active:  # only forward+sample for slots that still need a next token
            token_ids = [mx.array([s.next_id]) for s in active]
            stream_caches = [s.state for s in active]
            logits_list = model.step_batch(token_ids, stream_caches)
            for s, logits in zip(active, logits_list, strict=True):
                s.key, sub = mx.random.split(s.key)
                s.next_id = int(sample_logits(logits[0, -1], temperature=temperature, top_k=top_k,
                                               top_p=top_p, min_p=min_p, key=sub).item())

        # harvest finished slots, admit new pending ones
        for fs in slots:
            if fs.stopped:
                outputs[fs.idx] = fs.tokens[:max_new]
        slots = [s for s in slots if not s.stopped]
        while pending and len(slots) < model.max_batch:
            slots.append(_admit_one(pending.pop(0)))

    return outputs
