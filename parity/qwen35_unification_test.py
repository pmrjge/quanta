"""Gate: Qwen3.5 flows through the **unified** batched path — #152 step 2 (engine unify).

Before #152, Qwen3.5 batched serving lived in a bespoke :class:`Qwen35BatchedEngine` subclass
(token-ids-in ``batched_generate`` + ``spec_generate_batched``, still gated by
``parity/qwen35_omlx_engine_test.py``). #152 routes Qwen3.5 through the SAME
:class:`quanta.shim.omlx._BaseBatchedSession` slot adapter that already backs DSV4 (#145) and
Nemotron (#146): ``_make_batched_session`` now dispatches the ``qwen3_5`` model class to
:class:`_Qwen35BatchedSession`, and the engine's continuous-batching loop
(:meth:`QuantaOmlxEngine.batched_stream_generate`) drives all three through one Protocol — zero
change to the loop itself (the Protocol is the seam). This gate proves the new Qwen3.5 path works,
against a **fake** caller-owned-state runtime (~0 GB — safe while a big model is resident):

  (1) **hook contract** — ``_Qwen35BatchedSession`` formats the runtime call the Qwen3.5 way, which
      differs from its slot-mates: ``step_batch`` gets **plain int** token ids (DSV4 passes
      ``mx.array``), ``prefill`` gets a **list** of ids, and the per-stream ``offsets`` are the
      **real** ``Qwen35Cache.offset`` list (Nemotron passes ``None`` — its KVCache owns the offset).
      The fake records what it actually received; we assert all three. Also asserts the seam:
      ``_make_batched_session`` returns the injected session (no model load, no silent fallback).
  (2) **batched == single-stream** — B parallel identical streams produce the SAME token sequence as
      one stream alone on the same prompt (the deterministic absolute-position chain), so the batched
      optimization is output-equivalent (rule 4); every slot is released afterward (lifecycle).
  (3) **continuous batching** — more prompts than slots ⇒ freed slots admit the next pending prompt;
      all streams finish with the same output and the session ends with no resident caches.

The resident-model path (loads the int4-g64 Qwen3.5 artifact + the real
:class:`quanta.qwen35.batched_runtime.Qwen35BatchedResidentModel`) is the deferred GPU gate: a real
batched decode must match the single-stream decode token-for-token. Written here, run in a GPU
session — never concurrently with another large-resident job.

    uv run python -m parity.qwen35_unification_test
"""

from __future__ import annotations

import asyncio
from typing import Any

import mlx.core as mx

from quanta.shim.omlx import QuantaOmlxEngine, _Qwen35BatchedSession

# A Qwen3.5 artifact path string — NEVER loaded/detected here (the injected session + runtime make
# the batched path hermetic; start() no-ops when runtime+tokenizer are both provided).
QWEN_ART = "/Users/pmrj/models/Qwen3.5-quanta_int4g64"
EOS = 50  # generation eos (< vocab_size so the one-hot argmax stub can actually emit it)

# token id -> literal piece; reasoning / tool markers are ordinary tokens (parsed oMLX-side, raw here)
VOCAB = {100: "<think>", 101: "reason", 102: "</think>", 103: "<tool_call>", 104: "done"}
# argmax predictor keyed by the ABSOLUTE position of the fed token: feeding the token at position p
# predicts PRED[p]. Prompt is 3 ids ([5,6,7]) so prefill returns PRED[2] and decode walks 3..7.
PRED = {2: 100, 3: 101, 4: 102, 5: 103, 6: 104, 7: EOS}
EXPECT_TOKENS = [100, 101, 102, 103, 104, EOS]
EXPECT_TEXT = "<think>reason</think><tool_call>done"


class _FakeCache:
    """Per-stream :class:`quanta.qwen35.decode.Qwen35Cache` stand-in — only an ``offset`` (the
    absolute decode position), which is all :class:`_BaseBatchedSession` reads to drive a step."""

    def __init__(self) -> None:
        self.offset = 0


class _FakeQwen35Runtime:
    """Stub ``Qwen35BatchedResidentModel`` exposing the REAL caller-owned-state API the session drives:
    ``make_caches`` / ``prefill(ids, state)`` / ``step_batch(token_ids, caches, offsets)``.

    Slot-agnostic (the session owns slot<->cache bookkeeping): it advances each passed-in cache's
    offset and returns the deterministic-chain predictor logits (argmax ``PRED[pos]``, default eos),
    identical to a lone stream, so batched output is output-equivalent to single-stream (rule 4).

    It also captures the **Qwen3.5-specific call contract** on first use — the ids/token types and
    whether real per-stream offsets arrived — so the gate can assert ``_Qwen35BatchedSession``
    formats its runtime calls correctly (the distinction from the DSV4 / Nemotron sessions)."""

    def __init__(self, vocab_size: int = 200, n_layers: int = 3) -> None:
        self.num_layers = n_layers
        self._v = vocab_size
        self.prefills = 0
        self.steps: list[tuple[int, ...]] = []
        # contract probes (None until first observed)
        self.prefill_ids_was_list: bool | None = None
        self.step_tokens_all_int: bool | None = None
        self.step_offsets_were_real: bool | None = None

    def _row(self, nxt: int) -> mx.array:
        return (mx.arange(self._v) == nxt).astype(mx.float32) * 60.0 - 30.0

    def make_caches(self) -> _FakeCache:
        return _FakeCache()

    def prefill(self, prompt_ids, state: _FakeCache) -> mx.array:
        if self.prefill_ids_was_list is None:
            self.prefill_ids_was_list = isinstance(prompt_ids, list)  # Qwen35 _to_prefill_ids -> list
        n = len(list(prompt_ids))
        state.offset = n
        self.prefills += 1
        return mx.broadcast_to(self._row(PRED.get(n - 1, EOS)), (1, n, self._v))

    def step_batch(self, stream_token_ids, stream_caches, offsets) -> list[mx.array]:
        # capture the contract BEFORE advancing offsets (offsets must equal the pre-step positions).
        if self.step_tokens_all_int is None:
            self.step_tokens_all_int = all(isinstance(t, int) for t in stream_token_ids)
        if self.step_offsets_were_real is None:
            self.step_offsets_were_real = (
                offsets is not None and list(offsets) == [c.offset for c in stream_caches])
        self.steps.append(tuple(int(t) for t in stream_token_ids))
        out: list[mx.array] = []
        for cache in stream_caches:
            pos = cache.offset
            cache.offset = pos + 1
            out.append(mx.broadcast_to(self._row(PRED.get(pos, EOS)), (1, 1, self._v)))
        return out


class _Tok:
    """Minimal tokenizer: every prompt encodes to a fixed 3-token sequence so the absolute-offset
    chain aligns deterministically; decode maps the scripted ids to literal pieces (markers raw)."""

    eos_id = EOS
    stop_ids = (EOS,)

    def encode(self, text, *, add_bos=False, allow_special=False):
        return [5, 6, 7]

    def decode(self, ids, *, skip_special=True):
        return "".join(VOCAB.get(int(i), "") for i in ids)


def _engine(rt: _FakeQwen35Runtime, sess: _Qwen35BatchedSession) -> QuantaOmlxEngine:
    return QuantaOmlxEngine(QWEN_ART, runtime=rt, tokenizer=_Tok(), eos_token_ids={EOS},
                            batched_session=sess)


async def _collect_streams(engine, prompts, **kw) -> dict[int, list]:
    chunks: dict[int, list] = {i: [] for i in range(len(prompts))}
    async for sidx, chunk in engine.batched_stream_generate(prompts, **kw):
        chunks[sidx].append(chunk)
    return chunks


def run() -> None:
    ok = True

    # (1) hook contract + dispatch seam — drive ONE stream (B=1) so a prefill + >=1 step both fire.
    rt1 = _FakeQwen35Runtime()
    sess1 = _Qwen35BatchedSession(root=None, capacity=1, runtime=rt1)
    eng1 = _engine(rt1, sess1)
    seam_ok = eng1._make_batched_session(capacity=1) is sess1  # unified dispatch returns the session
    single = asyncio.run(_collect_streams(eng1, ["hi"], max_tokens=20, temperature=0.0))[0]
    single_last = single[-1]
    single_tokens, single_text = single_last.tokens, single_last.text
    contract_ok = (rt1.prefill_ids_was_list is True and rt1.step_tokens_all_int is True
                   and rt1.step_offsets_were_real is True)
    chain_ok = single_tokens == EXPECT_TOKENS and single_text == EXPECT_TEXT
    # stats seam: engine delegates to the session; OFF by default until a paged manager is wired in.
    stats_off = eng1.prefix_cache_enabled is False and eng1.get_cache_stats() is None
    good = seam_ok and contract_ok and chain_ok and stats_off and single_last.finish_reason == "stop"
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] hook contract: seam={seam_ok} "
          f"prefill_ids_list={rt1.prefill_ids_was_list} step_tokens_int={rt1.step_tokens_all_int} "
          f"offsets_real={rt1.step_offsets_were_real} stats_off={stats_off}")
    print(f"             single-stream tokens={single_tokens} text={single_text!r}")

    # (2) batched == single-stream: B identical streams equal the lone-stream chain; all slots freed.
    b = 4
    rt2 = _FakeQwen35Runtime()
    sess2 = _Qwen35BatchedSession(root=None, capacity=b, runtime=rt2)
    eng2 = _engine(rt2, sess2)
    chunks = asyncio.run(_collect_streams(eng2, ["go"] * b, max_tokens=20, temperature=0.0))
    per_stream_eq = all(
        chunks[i] and chunks[i][-1].tokens == single_tokens and chunks[i][-1].text == single_text
        and chunks[i][-1].finish_reason == single_last.finish_reason
        for i in range(b))
    prefills_ok = rt2.prefills == b           # session admitted all b streams
    released_ok = not sess2._caches           # every slot released after finishing
    good = per_stream_eq and prefills_ok and released_ok
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] batched==single: per_stream_eq={per_stream_eq} "
          f"prefills={rt2.prefills} caches_left={len(sess2._caches)}")

    # (3) continuous batching: prompts > capacity ⇒ freed slots admit the next pending prompt.
    n_streams, cap = 5, 2
    rt3 = _FakeQwen35Runtime()
    sess3 = _Qwen35BatchedSession(root=None, capacity=cap, runtime=rt3)
    eng3 = _engine(rt3, sess3)

    async def _collect_cb():
        seen: set[int] = set()
        finals: dict[int, Any] = {}
        async for sidx, chunk in eng3.batched_stream_generate(
                ["p"] * n_streams, max_tokens=20, temperature=0.0, batch_size=cap):
            seen.add(sidx)
            if chunk.finished:
                finals[sidx] = chunk
        return seen, finals

    seen, finals = asyncio.run(_collect_cb())
    cb_ok = (seen == set(range(n_streams)) and len(finals) == n_streams
             and all(finals[i].tokens == single_tokens for i in range(n_streams))
             and rt3.prefills == n_streams and not sess3._caches)
    ok = ok and cb_ok
    print(f"  [{'OK' if cb_ok else 'FAIL'}] continuous batching: streams_seen={sorted(seen)} "
          f"finished={sorted(finals)} cap={cap} prefills={rt3.prefills}")

    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
