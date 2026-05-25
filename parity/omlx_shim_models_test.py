"""Gate: the oMLX shim drives GLM-5.1 / MiniMax-M2.7 / Qwen3.5 through their resident runtimes — NO model.

These three model classes share the DSV4 single-token decode contract (``__call__(token_ids, *, caches,
offset)`` grows a per-layer cache in place), so the shim serves them with one ``_SingleTokenStepper`` and
routes each ``model_type`` to the right decode cache. MiniMax/Qwen tokenizers render chat via
``render_chat`` and lack ``allow_special``, so they go through ``_RenderChatAdapter``; GLM's tokenizer
conforms directly. Against **fake** runtimes + temp artifact dirs (~0 GB — safe while a big model is
GPU-resident), this verifies:

  (0) ``detect_quanta_artifact`` reads each ``model_type`` (glm_moe_dsa / minimax_m2 / qwen3_5_moe_text);
  (1) ``_make_stepper`` routes each -> ``_SingleTokenStepper`` with the right cache (GLMCache /
      MiniMaxCache / the runtime's ``make_caches`` for the hybrid Qwen cache); unknown -> OmlxShimError;
  (2) ``stream_generate`` seeds the cache by stepping the prompt (offsets 0..len-1) then decodes
      (len, len+1, …) — proven via recorded offsets — emits **raw** output (``<think>`` / tool markers
      pass through verbatim for oMLX's parsers) and stops on eos;
  (3) ``_RenderChatAdapter`` bridges the MiniMax/Qwen tokenizers: ``encode`` accepts (and ignores)
      ``allow_special``; ``apply_chat_template`` renders through ``render_chat`` mapping ``thinking`` ->
      ``enable_thinking``; eos/stop/bos delegate (incl. a tokenizer that exposes only ``_stop_ids``).

The resident gen/ppl path (loads the real artifact) is deferred so it never runs concurrently with
another large-resident job.

    uv run python -m parity.omlx_shim_models_test
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
    _RenderChatAdapter,
    _SingleTokenStepper,
    detect_quanta_artifact,
)

EOS = 11
VOCAB = {100: "<think>", 101: "reason", 102: "</think>", 103: "<tool_call>", 104: "done"}
PRED = {2: 100, 3: 101, 4: 102, 5: 103, 6: 104, 7: EOS}  # argmax at absolute offset
EXPECT_TOKENS = [100, 101, 102, 103, 104, EOS]
EXPECT_OFFSETS = [0, 1, 2, 3, 4, 5, 6, 7]
EXPECT_TEXT = "<think>reason</think><tool_call>done"


def _fake_artifact(model_type: str, *, nested: bool) -> str:
    """Synthetic quanta artifact dir: config.json (+ manifest.json) enough for detect_quanta_artifact.
    ``nested`` puts model_type under ``text_config`` (Qwen) vs top-level (GLM/MiniMax). A few bytes."""
    d = Path(tempfile.mkdtemp(prefix="omlxmodels_"))
    cfg = {"text_config": {"model_type": model_type}} if nested else {"model_type": model_type}
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "manifest.json").write_text(json.dumps({"format": "quanta", "tensors": {}}))
    return str(d)


class _FakeRuntime:
    """Stands in for a resident model: single-token ``__call__`` -> ``[1,t,vocab]`` + ``num_layers`` +
    ``make_caches`` (records the call). Records each offset so the test proves the prompt-seed (0..len-1)
    then decode (len, …). argmax at absolute ``offset`` is ``PRED[offset]`` (default eos)."""

    def __init__(self, *, vocab_size: int = 200, n_layers: int = 2) -> None:
        self.num_layers = n_layers
        self._v = vocab_size
        self.offsets: list[int] = []
        self.made_cache = False

    def make_caches(self):
        self.made_cache = True
        return SimpleNamespace(tag="hybrid-cache")  # opaque to the stepper (runtime ignores it)

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        t = int(mx.array(token_ids).reshape(-1).shape[0])
        self.offsets.append(int(offset))
        nxt = PRED.get(int(offset), EOS)
        row = (mx.arange(self._v) == nxt).astype(mx.float32) * 60.0 - 30.0
        return mx.broadcast_to(row, (1, t, self._v))


class _FakeTok:
    eos_id = EOS
    stop_ids = (EOS,)

    def encode(self, text, *, add_bos=False, allow_special=False):
        return [5, 6, 7]

    def decode(self, ids, **kw):
        return "".join(VOCAB.get(int(i), "") for i in ids)


class _StubRenderTok:
    """Stub MiniMax/Qwen-style tokenizer for the adapter test: ``render_chat`` echoes its kwargs so the
    test can prove the thinking->enable_thinking mapping; exposes only ``_stop_ids`` (like Qwen)."""

    bos_id = None
    eos_id = EOS
    _stop_ids = (EOS,)

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text, *, add_bos=False):
        self.calls.append((text, add_bos))
        return [7, 8, 9]

    def decode(self, ids, *, skip_special_tokens=False):
        return "TXT:" + ",".join(str(int(i)) for i in ids)

    def render_chat(self, messages, *, add_generation_prompt=True, tools=None, **kwargs):
        return f"PROMPT think={kwargs.get('enable_thinking')} agp={add_generation_prompt} tools={tools}"


async def _collect(engine, prompt, **kw):
    return [o async for o in engine.stream_generate(prompt, **kw)]


def run() -> None:
    ok = True
    tmp: list[str] = []
    try:
        arts = {
            "glm": _fake_artifact("glm_moe_dsa", nested=False),
            "minimax": _fake_artifact("minimax_m2", nested=False),
            "qwen": _fake_artifact("qwen3_5_moe_text", nested=True),
        }
        unk = _fake_artifact("llama_x", nested=False)
        tmp += [*arts.values(), unk]

        # (0) model_type detection
        det = {k: (detect_quanta_artifact(p).model_type if detect_quanta_artifact(p) else None)
               for k, p in arts.items()}
        good = det == {"glm": "glm_moe_dsa", "minimax": "minimax_m2", "qwen": "qwen3_5_moe_text"}
        ok &= good
        print(f"  [{'OK' if good else 'FAIL'}] detect model_type: {det}")

        # (1) stepper dispatch -> _SingleTokenStepper with the right cache
        cache_names = {}
        qwen_made = False
        for k, p in arts.items():
            rt = _FakeRuntime()
            eng = QuantaOmlxEngine(p, runtime=rt, tokenizer=_FakeTok(), eos_token_ids={EOS})
            st = eng._make_stepper(quantized_kv=True)
            cache_names[k] = type(st._cache).__name__
            if k == "qwen":
                qwen_made = rt.made_cache
            good = isinstance(st, _SingleTokenStepper)
            ok &= good
        disp_ok = (cache_names["glm"] == "GLMCache" and cache_names["minimax"] == "MiniMaxCache"
                   and qwen_made)
        ok &= disp_ok
        print(f"  [{'OK' if disp_ok else 'FAIL'}] stepper dispatch caches={cache_names} qwen_make_caches={qwen_made}")

        # (1b) unknown model class fails loud
        eng_unk = QuantaOmlxEngine(unk, runtime=SimpleNamespace(num_layers=2), tokenizer=_FakeTok(),
                                   eos_token_ids={EOS})
        try:
            eng_unk._make_stepper(quantized_kv=True)
            ub = False
        except OmlxShimError:
            ub = True
        ok &= ub
        print(f"  [{'OK' if ub else 'FAIL'}] unknown model_type -> OmlxShimError")

        # (2) stream_generate (glm path): prompt-seed offsets, threading, raw markers, eos stop
        rt = _FakeRuntime()
        eng = QuantaOmlxEngine(arts["glm"], runtime=rt, tokenizer=_FakeTok(), eos_token_ids={EOS})
        last = asyncio.run(_collect(eng, "hello", max_tokens=20, temperature=0.0))[-1]
        seeded = rt.offsets[:3] == [0, 1, 2]
        gen_ok = (last.tokens == EXPECT_TOKENS and last.finished and last.finish_reason == "stop"
                  and last.text == EXPECT_TEXT and seeded and rt.offsets == EXPECT_OFFSETS)
        ok &= gen_ok
        print(f"  [{'OK' if gen_ok else 'FAIL'}] stream_generate tokens={last.tokens} "
              f"finish={last.finish_reason!r} offsets={rt.offsets} text={last.text!r}")

        # (3) _RenderChatAdapter
        inner = _StubRenderTok()
        ad = _RenderChatAdapter(inner)
        enc = ad.encode("hi", add_bos=False, allow_special=True)  # allow_special accepted + ignored
        enc_ok = enc == [7, 8, 9] and inner.calls[-1] == ("hi", False)
        dec_ok = ad.decode([100, 101]) == "TXT:100,101"
        attr_ok = ad.eos_id == EOS and tuple(ad.stop_ids) == (EOS,) and ad.bos_id is None
        msgs = [{"role": "user", "content": "hi"}]
        s = ad.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, tools=None,
                                   thinking=False)  # 'thinking' must map to enable_thinking
        map_ok = isinstance(s, str) and "think=False" in s
        ids = ad.apply_chat_template(msgs, tokenize=True, enable_thinking=True)
        ids_ok = isinstance(ids, list) and all(isinstance(i, int) for i in ids)
        ad_ok = enc_ok and dec_ok and attr_ok and map_ok and ids_ok
        ok &= ad_ok
        print(f"  [{'OK' if ad_ok else 'FAIL'}] adapter: encode={enc_ok} decode={dec_ok} attrs={attr_ok} "
              f"thinking_map={map_ok} template_ids={ids_ok}")
    finally:
        for d in tmp:
            shutil.rmtree(d, ignore_errors=True)

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
