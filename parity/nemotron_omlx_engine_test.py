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
      for oMLX's stock parsers — and stops on eos.

The resident-model gen/ppl path (loads the 68 GB int4 artifact) is a separate gate, deferred so it
never runs concurrently with another large-resident job.

    uv run python -m parity.nemotron_omlx_engine_test
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import mlx.core as mx

from quanta.shim.omlx import (
    OmlxShimError,
    QuantaOmlxEngine,
    _MLAStepper,
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

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
