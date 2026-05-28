"""Gate: the oMLX shim drives the Nemotron-H hybrid through its own decode stepper — NO model loaded.

Task #39 is the *engine*, not a parser. Kimi's MLA decode (one ``MLACache`` per layer + absorbed
single-token step) and Nemotron's hybrid decode (threaded ``(ssm, conv)`` + a per-attention-layer
``KVCache``, runtime returning ``(logits, ssm, conv)``) are different call conventions. The shim
factors that into a per-request ``_DecodeStepper`` so the shared sampling / detok / stop loop serves
both. This verifies, against a **fake** runtime (~0 GB — safe to run while a big model is resident),
that:
  (0) artifact ``model_type`` is detected (``nemotron_h`` / ``kimi_*``);
  (1) ``_make_stepper`` routes Nemotron -> ``_NemotronStepper`` and Kimi -> ``_MLAStepper``, and an
      unknown model class fails loud (CLAUDE.md #6 — never guess a decode convention);
  (2) ``stream_generate`` threads ``(ssm, conv)`` across steps (None at prefill, carried after),
      emits **raw** output — reasoning/tool markers (ordinary Nemotron tokens) pass through verbatim
      for oMLX's stock parsers — and stops on eos;
  (3) ``batched_stream_generate`` drives B parallel streams through a stub
      ``NemotronBatchedResidentModel`` — per-stream output matches single-stream ``stream_generate``
      on the same prompt (the batched optimization is output-equivalent — rule 4), freed slots admit
      the next pending prompt (continuous batching), and ``NemotronBatchedSession`` lazy-imports the
      sibling agent's module (#146) so the shim still compiles when it isn't merged;
  (4) the ``spec_k`` hook routes a ``spec_k>1`` single-stream request through
      ``quanta.nemotron.spec.spec_generate_k`` (stubbed — sibling agent #148 is in flight) and
      otherwise stays on the existing stepper loop.

The resident-model gen/ppl path (loads the 68 GB int4 artifact) is a separate gate, deferred so it
never runs concurrently with another large-resident job.

    uv run python -m parity.nemotron_omlx_engine_test
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import mlx.core as mx

from quanta.shim.omlx import (
    OmlxShimError,
    QuantaOmlxEngine,
    _MLAStepper,
    _NemotronBatchedSession,
    _NemotronStepper,
    detect_quanta_artifact,
)

NEM_ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"
KIMI_ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int2g64"
EOS = 11  # Nemotron <|im_end|> (chat/generation eos)

# token id -> literal piece; includes reasoning + tool markers (ordinary tokens in Nemotron)
VOCAB = {100: "<think>", 101: "reason", 102: "</think>", 103: "<tool_call>", 104: "done"}
SCRIPT = [100, 101, 102, 103, 104, EOS]  # forced argmax sequence, then eos


class _FakeRuntime:
    """Stands in for NemotronResidentModel: the ``(logits, ssm, conv)`` signature + a ``.cfg`` so
    ``attn_caches`` works, and ``num_layers`` for the MLA stepper. Records each call's phase + whether
    the threaded state was None (to prove the engine carries it forward)."""

    def __init__(self, kinds, vocab_size: int = 200, script=()) -> None:
        self.cfg = SimpleNamespace(layers_block_type=list(kinds))
        self._v = vocab_size
        self._script = list(script)
        self._i = 0
        self.calls: list[tuple[str, bool, bool]] = []  # (phase, ssm_is_none, conv_is_none)

    @property
    def num_layers(self) -> int:
        return len(self.cfg.layers_block_type)

    def __call__(self, token_ids, *, caches=None, ssm=None, conv=None, **kw):
        t = int(token_ids.shape[0])
        self.calls.append(("prefill" if t > 1 else "step", ssm is None, conv is None))
        nxt = self._script[self._i] if self._i < len(self._script) else EOS
        self._i += 1
        row = (mx.arange(self._v) == nxt).astype(mx.float32) * 60.0 - 30.0  # argmax == nxt
        logits = mx.broadcast_to(row, (1, t, self._v))
        n = self.num_layers
        return logits, [mx.zeros((1,))] * n, [mx.zeros((1,))] * n


class _FakeTok:
    """Minimal tokenizer: maps the scripted ids to literal pieces (incl. markers). No ``decode_bytes``
    / ``n_base`` so ``_Detok`` takes its string-fallback path — the path Nemotron actually uses."""

    eos_id = EOS
    stop_ids = (2, EOS)

    def encode(self, text, *, add_bos=False, allow_special=False):
        return [5, 6, 7]  # arbitrary prompt ids; never decoded

    def decode(self, ids, *, skip_special=True):
        return "".join(VOCAB.get(int(i), "") for i in ids)


async def _collect(engine, prompt, **kw):
    return [o async for o in engine.stream_generate(prompt, **kw)]


# Multi-step batched / spec stubs (no model load) -------------------------------------------------
# argmax at absolute offset; mirrors the SCRIPT used by the single-stream test so the batched-engine
# equivalence assertion can compare against the single-stream stub on the same prompt.
BATCH_PRED = {2: 100, 3: 101, 4: 102, 5: 103, 6: 104, 7: EOS}
BATCH_EXPECT_TOKENS = [100, 101, 102, 103, 104, EOS]


class _FakeCache:
    """A per-stream state stand-in for Nemotron's ``(caches, ssm, conv)`` triple — only an ``offset``
    (the runtime owns the real hybrid state; this stub routes purely by absolute position)."""

    def __init__(self) -> None:
        self.offset = 0


class _FakeBatchedRuntime:
    """Stub ``NemotronBatchedResidentModel`` exposing the REAL caller-owned-state API the session
    drives: ``make_stream_state`` / ``prefill(ids, state)`` / ``step_batch(token_ids, caches, offsets)``
    (Nemotron passes ``offsets=None`` — each KVCache owns its offset; here the stub state carries it).

    The session (not the runtime) owns slot<->state bookkeeping, so this stub is slot-agnostic: it
    advances each passed-in state's offset and returns logits by absolute position (argmax
    ``BATCH_PRED[position_of_fed_token]``), identical to the single-stream chain, so batched output is
    output-equivalent to single-stream (rule 4). ``prefills`` counts admits for the lifecycle check."""

    def __init__(self, vocab_size: int = 200, n_layers: int = 4) -> None:
        self.num_layers = n_layers
        self._v = vocab_size
        self.prefills = 0
        self.steps: list[tuple[int, ...]] = []

    def _row(self, nxt: int) -> mx.array:
        return (mx.arange(self._v) == nxt).astype(mx.float32) * 60.0 - 30.0

    def make_stream_state(self) -> _FakeCache:
        return _FakeCache()

    def prefill(self, prompt_ids, state: _FakeCache) -> mx.array:
        n = int(mx.array(prompt_ids).reshape(-1).shape[0])
        state.offset = n
        self.prefills += 1
        return mx.broadcast_to(self._row(BATCH_PRED.get(n - 1, EOS)), (1, n, self._v))

    def step_batch(self, stream_token_ids, stream_caches, offsets=None) -> list[mx.array]:
        self.steps.append(tuple(int(mx.array(t).reshape(-1)[0].item()) for t in stream_token_ids))
        out: list[mx.array] = []
        for state in stream_caches:
            pos = state.offset
            state.offset = pos + 1
            out.append(mx.broadcast_to(self._row(BATCH_PRED.get(pos, EOS)), (1, 1, self._v)))
        return out


class _SingleStreamBatchTok:
    """Tokenizer for the batched / spec gates: encodes any prompt to a fixed 3-token sequence so the
    absolute-offset chain in :class:`_FakeBatchedRuntime` aligns deterministically."""

    eos_id = EOS
    stop_ids = (EOS,)

    def encode(self, text, *, add_bos=False, allow_special=False):
        return [5, 6, 7]

    def decode(self, ids, *, skip_special=True):
        return "".join(VOCAB.get(int(i), "") for i in ids)


class _SingleStreamBatchRuntime:
    """Single-stream Nemotron stub used as the equivalence reference for the batched engine test.
    Mirrors :class:`_FakeRuntime` but uses BATCH_PRED + the absolute-offset chain so the single-stream
    output equals what each batched stream produces under the same prompt."""

    def __init__(self, vocab_size: int = 200, n_layers: int = 4) -> None:
        self.cfg = SimpleNamespace(layers_block_type=["mamba", "attention", "moe", "mamba"][:n_layers])
        self._v = vocab_size
        self._offset = 0

    @property
    def num_layers(self) -> int:
        return len(self.cfg.layers_block_type)

    def __call__(self, token_ids, *, caches=None, ssm=None, conv=None, **kw):
        del caches, ssm, conv
        t = int(token_ids.shape[0])
        nxt = BATCH_PRED.get(self._offset + t - 1, EOS)
        self._offset += t
        row = (mx.arange(self._v) == nxt).astype(mx.float32) * 60.0 - 30.0
        logits = mx.broadcast_to(row, (1, t, self._v))
        n = self.num_layers
        return logits, [mx.zeros((1,))] * n, [mx.zeros((1,))] * n


class _SpecStub:
    """Records the kwargs the engine passes to ``spec_generate_k`` and returns fixed tokens.

    Mirrors the DSV4 spec stub's surface so the Nemotron and DSV4 dispatchers can share a contract
    (rule 4: a sibling-agent function called the same way must accept the same arg shape)."""

    def __init__(self, fixed_tokens: list[int]) -> None:
        self.fixed_tokens = list(fixed_tokens)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, model, mtp, embed, head, prompt_ids, *, max_new, eos_id, k):
        del model, mtp, embed, head
        self.calls.append({"prompt_ids": list(prompt_ids), "max_new": max_new,
                           "eos_id": eos_id, "k": k})
        return list(self.fixed_tokens), {"k": k, "rounds": len(self.fixed_tokens), "mean_accept": 1.0}


def run() -> None:
    ok = True

    # (0) artifacts detected with the expected model_type
    nem, kim = detect_quanta_artifact(NEM_ART), detect_quanta_artifact(KIMI_ART)
    good = (nem is not None and nem.model_type == "nemotron_h"
            and kim is not None and (kim.model_type or "").startswith("kimi"))
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] detect model_type: "
          f"nemotron={nem.model_type if nem else None} kimi={kim.model_type if kim else None}")

    # (1) stepper dispatch on model_type (fake runtimes, no model load)
    kinds = ["mamba", "attention", "moe", "mamba"]
    nem_eng = QuantaOmlxEngine(NEM_ART, runtime=_FakeRuntime(kinds), tokenizer=_FakeTok(), eos_token_ids={EOS})
    kimi_eng = QuantaOmlxEngine(KIMI_ART, runtime=_FakeRuntime(["x", "x"]), tokenizer=_FakeTok(), eos_token_ids={EOS})
    s_nem, s_kimi = nem_eng._make_stepper(quantized_kv=True), kimi_eng._make_stepper(quantized_kv=True)
    good = isinstance(s_nem, _NemotronStepper) and isinstance(s_kimi, _MLAStepper)
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] stepper dispatch: "
          f"nemotron->{type(s_nem).__name__} kimi->{type(s_kimi).__name__}")

    # (1b) unknown model class fails loud rather than guessing a decode convention (CLAUDE.md #6)
    class _UnknownEng(QuantaOmlxEngine):
        @property
        def model_type(self):
            return "llama_surprise"

    ue = _UnknownEng(NEM_ART, runtime=_FakeRuntime(["x"]), tokenizer=_FakeTok(), eos_token_ids={EOS})
    try:
        ue._make_stepper(quantized_kv=True)
        good = False
    except OmlxShimError:
        good = True
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] unknown model_type -> OmlxShimError (no silent fallback)")

    # (2) full stream_generate over the fake hybrid: state threading + raw markers + eos stop
    fake = _FakeRuntime(kinds, script=SCRIPT)
    eng = QuantaOmlxEngine(NEM_ART, runtime=fake, tokenizer=_FakeTok(), eos_token_ids={EOS})
    outs = asyncio.run(_collect(eng, "hello", max_tokens=20, temperature=0.0))
    last = outs[-1]
    threaded = (fake.calls[0] == ("prefill", True, True)
                and all(c == ("step", False, False) for c in fake.calls[1:]))
    good = (last.tokens == SCRIPT and last.finished and last.finish_reason == "stop"
            and last.text == "<think>reason</think><tool_call>done"
            and "<think>" in last.text and "</think>" in last.text and "<tool_call>" in last.text
            and threaded and len(fake.calls) == len(SCRIPT))
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] stream_generate: tokens={last.tokens} "
          f"finish={last.finish_reason!r} threaded={threaded}")
    print(f"             raw text={last.text!r}")
    print(f"             runtime calls={fake.calls}")

    # (3) batched engine equivalence: B=4 identical streams equal single-stream on the same prompt
    b = 4
    batched_rt = _FakeBatchedRuntime()
    sess = _NemotronBatchedSession(root=None, capacity=b, runtime=batched_rt)
    eng_batched = QuantaOmlxEngine(NEM_ART, runtime=batched_rt, tokenizer=_SingleStreamBatchTok(),
                                    eos_token_ids={EOS}, batched_session=sess)
    async def _collect_batched():
        chunks: dict[int, list] = {i: [] for i in range(b)}
        async for sidx, chunk in eng_batched.batched_stream_generate(
                ["hi"] * b, max_tokens=20, temperature=0.0):
            chunks[sidx].append(chunk)
        return chunks
    batched_chunks = asyncio.run(_collect_batched())
    # Single-stream reference using the matching absolute-offset chain
    single_rt = _SingleStreamBatchRuntime()
    eng_single = QuantaOmlxEngine(NEM_ART, runtime=single_rt, tokenizer=_SingleStreamBatchTok(),
                                   eos_token_ids={EOS})
    single_last = asyncio.run(_collect(eng_single, "hi", max_tokens=20, temperature=0.0))[-1]
    per_stream_eq = all(
        batched_chunks[i] and batched_chunks[i][-1].tokens == single_last.tokens
        and batched_chunks[i][-1].text == single_last.text
        and batched_chunks[i][-1].finish_reason == single_last.finish_reason
        for i in range(b))
    prefills_ok = batched_rt.prefills == b               # session admitted all b streams
    released_ok = not sess._caches                       # every slot released after finishing
    good = per_stream_eq and prefills_ok and released_ok and single_last.tokens == BATCH_EXPECT_TOKENS
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] batched_stream_generate: per_stream_eq={per_stream_eq} "
          f"prefills={batched_rt.prefills} caches_left={len(sess._caches)} "
          f"single_tokens={single_last.tokens}")

    # (3b) continuous batching: prompts > capacity ⇒ freed slots admit the next pending prompt
    n_streams = 5
    cap = 2
    cb_rt = _FakeBatchedRuntime()
    cb_sess = _NemotronBatchedSession(root=None, capacity=cap, runtime=cb_rt)
    eng_cb = QuantaOmlxEngine(NEM_ART, runtime=cb_rt, tokenizer=_SingleStreamBatchTok(),
                                eos_token_ids={EOS}, batched_session=cb_sess)
    async def _collect_cb():
        seen_streams: set[int] = set()
        finals: dict[int, Any] = {}
        async for sidx, chunk in eng_cb.batched_stream_generate(
                ["p"] * n_streams, max_tokens=20, temperature=0.0, batch_size=cap):
            seen_streams.add(sidx)
            if chunk.finished:
                finals[sidx] = chunk
        return seen_streams, finals
    seen, finals = asyncio.run(_collect_cb())
    cb_ok = (seen == set(range(n_streams)) and len(finals) == n_streams
             and all(finals[i].tokens == single_last.tokens for i in range(n_streams))
             and cb_rt.prefills == n_streams and not cb_sess._caches)
    ok = ok and cb_ok
    print(f"  [{'OK' if cb_ok else 'FAIL'}] continuous batching: streams_seen={sorted(seen)} "
          f"finished={sorted(finals)} cap={cap}")

    # (4) multi-step MTP hook: spec_k>1 dispatches through quanta.nemotron.spec.spec_generate_k.
    # sibling agent #148 is in flight, so spec_generate_k may not yet exist on the worktree — install
    # a stub via monkeypatch and restore (or remove) afterward so the test is hermetic regardless.
    import quanta.nemotron.spec as _nem_spec
    stub_spec = _SpecStub(fixed_tokens=[100, 101, 102, EOS])
    had_attr = hasattr(_nem_spec, "spec_generate_k")
    original = getattr(_nem_spec, "spec_generate_k", None)
    _nem_spec.spec_generate_k = stub_spec
    try:
        # runtime needs mtp / embed_w / lm_head_w for the dispatcher
        rt_for_spec = SimpleNamespace(num_layers=4, mtp=object(),
                                       embed_w=mx.zeros((10, 4)), lm_head_w=mx.zeros((10, 4)))
        eng_spec = QuantaOmlxEngine(NEM_ART, runtime=rt_for_spec,
                                     tokenizer=_SingleStreamBatchTok(), eos_token_ids={EOS})
        spec_last = asyncio.run(_collect(eng_spec, "hi", max_tokens=20, temperature=0.0,
                                           spec_k=4))[-1]
        spec_called_k = bool(stub_spec.calls) and stub_spec.calls[0]["k"] == 4
        spec_called_max = bool(stub_spec.calls) and stub_spec.calls[0]["max_new"] == 20
        spec_tokens_ok = spec_last.tokens == [100, 101, 102, EOS]
        spec_text_ok = spec_last.text == "<think>reason</think>"
        spec_ok = spec_called_k and spec_called_max and spec_tokens_ok and spec_text_ok
        ok = ok and spec_ok
        print(f"  [{'OK' if spec_ok else 'FAIL'}] spec_k>1 dispatch: k_recorded="
              f"{stub_spec.calls[0]['k'] if stub_spec.calls else None} "
              f"tokens={spec_last.tokens} text={spec_last.text!r}")

        # (4b) spec_k==1 stays on the existing stepper (no spec call)
        stub_spec.calls.clear()
        fake2 = _FakeRuntime(kinds, script=SCRIPT)
        eng_no_spec = QuantaOmlxEngine(NEM_ART, runtime=fake2, tokenizer=_FakeTok(),
                                         eos_token_ids={EOS})
        non_spec_last = asyncio.run(_collect(eng_no_spec, "hello", max_tokens=20,
                                               temperature=0.0, spec_k=1))[-1]
        non_spec_ok = (non_spec_last.tokens == SCRIPT and not stub_spec.calls)
        ok = ok and non_spec_ok
        print(f"  [{'OK' if non_spec_ok else 'FAIL'}] spec_k==1 stays on stepper "
              f"(spec_calls={len(stub_spec.calls)})")
    finally:
        if had_attr:
            _nem_spec.spec_generate_k = original
        else:
            delattr(_nem_spec, "spec_generate_k")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
