"""Gate: the oMLX shim drives DeepSeek-V4-Flash through its own decode stepper — NO model loaded.

DSV4-Flash is a third model class behind the shim, with a decode convention unlike Kimi/DeepSeek-V3's
absorbed-MLA path or the Nemotron-H ``(ssm, conv)`` recurrence: one :class:`~quanta.dsv4.decode.DSV4Cache`
across all layers, seeded by stepping the prompt one token at a time, then a single-token decode at the
running offset. The shim factors that into a per-request ``_DSV4Stepper`` so the shared sampling /
detok / stop loop serves it unchanged. Against a **fake** runtime + temp artifact dirs (~0 GB — safe
to run while a big model is GPU-resident), this verifies:

  (0) ``detect_quanta_artifact`` reads ``model_type`` from a (synthetic) artifact;
  (1) ``_make_stepper`` routes ``deepseek_v4`` -> ``_DSV4Stepper`` — and, crucially, that the
      ``deepseek_v4`` model_type is NOT swallowed by the ``deepseek`` (V3 / Kimi-MLA) prefix it shares
      (the latent mis-routing bug); ``deepseek_v3`` still -> ``_MLAStepper``; unknown -> OmlxShimError;
  (2) ``stream_generate`` seeds the cache by stepping the prompt (offsets 0..len-1) then decodes
      (len, len+1, …) — proven via the runtime's recorded offsets — emits **raw** output (reasoning /
      tool markers pass through verbatim for oMLX's stock parsers) and stops on eos;
  (3) ``_DSV4TokenizerAdapter`` bridges the DSV4 tokenizer to the engine contract: ``encode`` accepts
      (and ignores) ``allow_special``; ``apply_chat_template`` renders a prompt through the real
      :func:`quanta.dsv4.encoding.encode_chat`, filtering the chat-control kwargs the renderer does not
      accept (``add_generation_prompt`` / ``tools`` / ``enable_thinking``); eos/stop/bos delegate;
  (4) ``batched_stream_generate`` drives B parallel streams through a stub
      ``DSV4BatchedResidentModel`` — per-stream output matches single-stream ``stream_generate`` on the
      same prompt (the batched optimization is output-equivalent — rule 4), freed slots admit the next
      pending prompt (continuous batching), and ``DSV4BatchedSession`` lazy-imports the sibling agent's
      module (#145) so the shim still compiles when it isn't merged;
  (5) the ``spec_k`` hook routes a ``spec_k>1`` single-stream request through
      ``quanta.dsv4.spec.spec_generate_k`` (stubbed) and otherwise stays on the existing stepper loop —
      proving the wiring without depending on a real MTP head.

The resident gen/ppl path (loads the real artifact) is a separate gate, deferred so it never runs
concurrently with another large-resident job:
    # from quanta.shim.omlx import QuantaOmlxEngine
    # eng = QuantaOmlxEngine("/Users/pmrj/models/DeepSeek-V4-Flash-quanta_<type>")
    # print(asyncio.run(eng.generate("Hello", max_tokens=32, temperature=0.7, seed=0)).text)

    uv run python -m parity.dsv4_omlx_engine_test
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx

from quanta.shim.omlx import (
    OmlxShimError,
    QuantaOmlxEngine,
    _DSV4BatchedSession,
    _DSV4Stepper,
    _DSV4TokenizerAdapter,
    _MLAStepper,
    detect_quanta_artifact,
)

EOS = 11  # fake generation eos
# token id -> literal piece; reasoning + tool markers are ordinary tokens in DSV4's raw stream
VOCAB = {100: "<think>", 101: "reason", 102: "</think>", 103: "<tool_call>", 104: "done"}
# prompt encodes to [5,6,7] (len 3); the forward at the last prompt pos (offset 2) predicts the first
# generated token, and each subsequent decode at offset 3,4,… predicts the next -> deterministic stream.
PRED = {2: 100, 3: 101, 4: 102, 5: 103, 6: 104, 7: EOS}
EXPECT_TOKENS = [100, 101, 102, 103, 104, EOS]
EXPECT_OFFSETS = [0, 1, 2, 3, 4, 5, 6, 7]
EXPECT_TEXT = "<think>reason</think><tool_call>done"


def _fake_artifact(model_type: str) -> str:
    """A synthetic quanta artifact dir holding ONLY config.json + manifest.json (no weights), enough
    for ``detect_quanta_artifact`` to report ``model_type``. Model-free, a few bytes."""
    d = Path(tempfile.mkdtemp(prefix="dsv4omlx_"))
    (d / "config.json").write_text(json.dumps({"text_config": {"model_type": model_type}}))
    (d / "manifest.json").write_text(json.dumps({"format": "quanta", "tensors": {}}))
    return str(d)


class _FakeDSV4Runtime:
    """Stands in for ``DSV4ResidentModel``: the single-token ``__call__(token_ids, caches=, offset=)``
    -> ``[1,t,vocab]`` contract + ``num_layers``. Records each call's offset so the test can prove the
    stepper seeds the prompt (offsets 0..len-1) then decodes (len, …). argmax at absolute ``offset`` is
    ``PRED[offset]`` (default eos)."""

    def __init__(self, pred: dict[int, int], *, vocab_size: int = 200, n_layers: int = 2) -> None:
        self.num_layers = n_layers
        self._v = vocab_size
        self._pred = dict(pred)
        self.offsets: list[int] = []

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        t = int(mx.array(token_ids).reshape(-1).shape[0])
        self.offsets.append(int(offset))
        nxt = self._pred.get(int(offset), EOS)
        row = (mx.arange(self._v) == nxt).astype(mx.float32) * 60.0 - 30.0  # argmax == nxt
        return mx.broadcast_to(row, (1, t, self._v))


class _FakeTok:
    """Minimal tokenizer for the engine path: a fixed 3-token prompt + marker detok. No ``decode_bytes``
    / ``n_base`` so ``_Detok`` takes its string-fallback path (the path DSV4 actually uses)."""

    eos_id = EOS
    stop_ids = (EOS,)

    def encode(self, text, *, add_bos=False, allow_special=False):
        return [5, 6, 7]

    def decode(self, ids, **kw):
        return "".join(VOCAB.get(int(i), "") for i in ids)


class _StubInnerTok:
    """Stub DeepSeekV4Tokenizer for the adapter test: records encode calls; arbitrary encode/decode."""

    bos_id = 1
    eos_id = EOS
    stop_ids = frozenset({EOS})

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text, *, add_bos=False):
        self.calls.append((text, add_bos))
        return ([self.bos_id] if add_bos else []) + [7, 8, 9]

    def decode(self, ids, *, skip_special_tokens=False):
        return "TXT:" + ",".join(str(int(i)) for i in ids)


async def _collect(engine, prompt, **kw):
    return [o async for o in engine.stream_generate(prompt, **kw)]


class _FakeBatchedRuntime:
    """Stub ``DSV4BatchedResidentModel`` for the batched-engine gate.

    Owns ``capacity`` decode slots, each with an independent (prompt_len + tokens-emitted) offset.
    Records every admit / step / release so the test can prove the engine's slot lifecycle is right.
    Returns deterministic per-slot logits whose argmax is the deterministic-chain predictor: at any
    slot, ``next = PRED[absolute_offset]`` (default eos). Because the stub ignores cache contents and
    routes solely by absolute offset, a single-stream stub running the same prompt would produce the
    same token sequence — exactly what the equivalence assertion needs."""

    def __init__(self, capacity: int, vocab_size: int = 200, n_layers: int = 2) -> None:
        self.capacity = capacity
        self.num_layers = n_layers
        self._v = vocab_size
        self._slot_offset: dict[int, int] = {}
        self.admits: list[tuple[int, tuple[int, ...]]] = []
        self.steps: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.releases: list[int] = []

    def _row(self, nxt: int) -> mx.array:
        return (mx.arange(self._v) == nxt).astype(mx.float32) * 60.0 - 30.0

    def prefill_slot(self, slot: int, prompt_ids):
        ids = tuple(int(t) for t in prompt_ids)
        self.admits.append((slot, ids))
        self._slot_offset[slot] = len(ids)
        return self._row(PRED.get(len(ids) - 1, EOS))

    def step_batch(self, slot_to_token):
        slots = tuple(sorted(slot_to_token))
        tokens = tuple(int(slot_to_token[s]) for s in slots)
        self.steps.append((slots, tokens))
        out: dict[int, mx.array] = {}
        for s in slots:
            self._slot_offset[s] += 1
            out[s] = self._row(PRED.get(self._slot_offset[s] - 1, EOS))
        return out

    def free_slot(self, slot: int) -> None:
        self.releases.append(slot)
        self._slot_offset.pop(slot, None)


class _SpecStub:
    """Records the (k, prompt_ids, max_new, eos_id) the engine passes to ``spec_generate_k`` and
    returns a fixed token sequence. ``calls`` exposes the kwargs for the assertions."""

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
    tmp: list[str] = []
    try:
        v4, v3, unk = _fake_artifact("deepseek_v4"), _fake_artifact("deepseek_v3"), _fake_artifact("llama_x")
        tmp += [v4, v3, unk]

        # (0) model_type detected from the synthetic artifact
        info = detect_quanta_artifact(v4)
        good = info is not None and info.model_type == "deepseek_v4"
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] detect model_type: {info.model_type if info else None}")

        # (1) stepper dispatch — deepseek_v4 -> _DSV4Stepper (NOT swallowed by the deepseek/V3 prefix)
        eng_v4 = QuantaOmlxEngine(v4, runtime=_FakeDSV4Runtime(PRED), tokenizer=_FakeTok(), eos_token_ids={EOS})
        eng_v3 = QuantaOmlxEngine(v3, runtime=SimpleNamespace(num_layers=2), tokenizer=_FakeTok(), eos_token_ids={EOS})
        s_v4 = eng_v4._make_stepper(quantized_kv=True)
        s_v3 = eng_v3._make_stepper(quantized_kv=True)
        good = isinstance(s_v4, _DSV4Stepper) and not isinstance(s_v4, _MLAStepper) and isinstance(s_v3, _MLAStepper)
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] stepper dispatch: deepseek_v4->{type(s_v4).__name__} "
              f"deepseek_v3->{type(s_v3).__name__} (v4 not mis-routed to MLA)")

        # (1b) unknown model class fails loud (CLAUDE.md #6 — never guess a decode convention)
        eng_unk = QuantaOmlxEngine(unk, runtime=SimpleNamespace(num_layers=2), tokenizer=_FakeTok(), eos_token_ids={EOS})
        try:
            eng_unk._make_stepper(quantized_kv=True)
            good = False
        except OmlxShimError:
            good = True
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] unknown model_type -> OmlxShimError (no silent fallback)")

        # (2) stream_generate: prompt-seed offsets, offset threading, raw markers, eos stop
        fake = _FakeDSV4Runtime(PRED)
        eng = QuantaOmlxEngine(v4, runtime=fake, tokenizer=_FakeTok(), eos_token_ids={EOS})
        last = asyncio.run(_collect(eng, "hello", max_tokens=20, temperature=0.0))[-1]
        seeded = fake.offsets[:3] == [0, 1, 2]
        good = (last.tokens == EXPECT_TOKENS and last.finished and last.finish_reason == "stop"
                and last.text == EXPECT_TEXT and seeded and fake.offsets == EXPECT_OFFSETS)
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] stream_generate: tokens={last.tokens} finish={last.finish_reason!r} "
              f"seeded={seeded} offsets={fake.offsets}")
        print(f"             raw text={last.text!r}")

        # (3) tokenizer adapter: allow_special passthrough, apply_chat_template kwarg filtering, delegation
        inner = _StubInnerTok()
        ad = _DSV4TokenizerAdapter(inner)
        enc = ad.encode("hi", add_bos=True, allow_special=False)  # allow_special accepted + ignored
        enc_ok = enc == [1, 7, 8, 9] and inner.calls[-1] == ("hi", True)
        dec_ok = ad.decode([100, 101]) == "TXT:100,101"
        attr_ok = ad.eos_id == EOS and set(ad.stop_ids) == {EOS} and ad.bos_id == 1
        msgs = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}]
        # the engine forwards add_generation_prompt / tools / enable_thinking; encode_chat accepts none
        # of them — a non-crashing string proves the adapter filters them out.
        s = ad.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, tools=None,
                                   enable_thinking=True, reasoning_effort="high")
        tmpl_str_ok = isinstance(s, str) and len(s) > 0
        ids = ad.apply_chat_template(msgs, tokenize=True)
        tmpl_ids_ok = isinstance(ids, list) and all(isinstance(i, int) for i in ids)
        good = enc_ok and dec_ok and attr_ok and tmpl_str_ok and tmpl_ids_ok
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] tokenizer adapter: encode={enc_ok} decode={dec_ok} "
              f"attrs={attr_ok} template(str)={tmpl_str_ok} template(ids)={tmpl_ids_ok}")

        # (4) batched engine equivalence: B=4 parallel streams equal single-stream on each prompt
        b = 4
        prompts = ["hi"] * b
        batched_rt = _FakeBatchedRuntime(capacity=b)
        sess = _DSV4BatchedSession(root=None, capacity=b, runtime=batched_rt)
        eng_batched = QuantaOmlxEngine(v4, runtime=batched_rt, tokenizer=_FakeTok(),
                                        eos_token_ids={EOS}, batched_session=sess)
        async def _collect_batched():
            chunks: dict[int, list] = {i: [] for i in range(b)}
            async for sidx, chunk in eng_batched.batched_stream_generate(
                    prompts, max_tokens=20, temperature=0.0):
                chunks[sidx].append(chunk)
            return chunks
        batched_chunks = asyncio.run(_collect_batched())
        # single-stream reference: same fake runtime contract -> same tokens
        eng_single = QuantaOmlxEngine(v4, runtime=_FakeDSV4Runtime(PRED), tokenizer=_FakeTok(),
                                       eos_token_ids={EOS})
        single_last = asyncio.run(_collect(eng_single, "hi", max_tokens=20, temperature=0.0))[-1]
        per_stream_eq = all(
            batched_chunks[i] and batched_chunks[i][-1].tokens == single_last.tokens
            and batched_chunks[i][-1].text == single_last.text
            and batched_chunks[i][-1].finish_reason == single_last.finish_reason
            for i in range(b))
        admits_ok = len(batched_rt.admits) == b and {s for s, _ in batched_rt.admits} == set(range(b))
        releases_ok = sorted(batched_rt.releases) == list(range(b))
        good = per_stream_eq and admits_ok and releases_ok
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] batched_stream_generate: per_stream_eq={per_stream_eq} "
              f"admits={len(batched_rt.admits)} releases={sorted(batched_rt.releases)}")

        # (4b) continuous batching: prompts > capacity ⇒ freed slots admit the next pending prompt
        n_streams = 5
        cap = 2
        cb_rt = _FakeBatchedRuntime(capacity=cap)
        cb_sess = _DSV4BatchedSession(root=None, capacity=cap, runtime=cb_rt)
        eng_cb = QuantaOmlxEngine(v4, runtime=cb_rt, tokenizer=_FakeTok(),
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
                 and all(finals[i].tokens == single_last.tokens for i in range(n_streams)))
        ok = ok and cb_ok
        print(f"  [{'OK' if cb_ok else 'FAIL'}] continuous batching: streams_seen={sorted(seen)} "
              f"finished={sorted(finals)} cap={cap}")

        # (5) multi-step MTP hook: spec_k>1 dispatches through quanta.dsv4.spec.spec_generate_k.
        # spec_generate_k is on main but may not yet be in every worktree, so we install a stub via
        # monkeypatch and restore (or remove) afterwards to keep this test hermetic.
        import quanta.dsv4.spec as _dsv4_spec
        stub_spec = _SpecStub(fixed_tokens=[100, 101, 102, EOS])
        had_attr = hasattr(_dsv4_spec, "spec_generate_k")
        original = getattr(_dsv4_spec, "spec_generate_k", None)
        _dsv4_spec.spec_generate_k = stub_spec
        try:
            # runtime needs mtp / embed_w / lm_head_w for the dispatcher
            rt_for_spec = SimpleNamespace(num_layers=2, mtp=object(),
                                          embed_w=mx.zeros((10, 4)), lm_head_w=mx.zeros((10, 4)))
            eng_spec = QuantaOmlxEngine(v4, runtime=rt_for_spec, tokenizer=_FakeTok(),
                                         eos_token_ids={EOS})
            spec_last = asyncio.run(_collect(eng_spec, "hi", max_tokens=20, temperature=0.0,
                                              spec_k=3))[-1]
            spec_called_k = stub_spec.calls and stub_spec.calls[0]["k"] == 3
            spec_called_max = stub_spec.calls and stub_spec.calls[0]["max_new"] == 20
            spec_tokens_ok = spec_last.tokens == [100, 101, 102, EOS]
            spec_text_ok = spec_last.text == "<think>reason</think>"  # EOS yields no text
            spec_ok = spec_called_k and spec_called_max and spec_tokens_ok and spec_text_ok
            ok = ok and spec_ok
            print(f"  [{'OK' if spec_ok else 'FAIL'}] spec_k>1 dispatch: k_recorded="
                  f"{stub_spec.calls[0]['k'] if stub_spec.calls else None} "
                  f"tokens={spec_last.tokens} text={spec_last.text!r}")

            # (5b) spec_k==1 stays on the existing stepper (no spec call)
            stub_spec.calls.clear()
            fake2 = _FakeDSV4Runtime(PRED)
            eng_no_spec = QuantaOmlxEngine(v4, runtime=fake2, tokenizer=_FakeTok(),
                                            eos_token_ids={EOS})
            non_spec_last = asyncio.run(_collect(eng_no_spec, "hi", max_tokens=20,
                                                  temperature=0.0, spec_k=1))[-1]
            non_spec_ok = (non_spec_last.tokens == EXPECT_TOKENS and not stub_spec.calls)
            ok = ok and non_spec_ok
            print(f"  [{'OK' if non_spec_ok else 'FAIL'}] spec_k==1 stays on stepper "
                  f"(spec_calls={len(stub_spec.calls)})")
        finally:
            if had_attr:
                _dsv4_spec.spec_generate_k = original
            else:
                delattr(_dsv4_spec, "spec_generate_k")
    finally:
        for d in tmp:
            shutil.rmtree(d, ignore_errors=True)

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
