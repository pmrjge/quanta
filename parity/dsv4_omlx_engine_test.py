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
      accept (``add_generation_prompt`` / ``tools`` / ``enable_thinking``); eos/stop/bos delegate.

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

import mlx.core as mx

from quanta.shim.omlx import (
    OmlxShimError,
    QuantaOmlxEngine,
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
    finally:
        for d in tmp:
            shutil.rmtree(d, ignore_errors=True)

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
