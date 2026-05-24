"""Validate the oMLX shim standalone — no oMLX install (engine subclasses object fallback).

Checks the engine's KV-cached decode loop drives a stub runtime, applies generation kwargs
(greedy determinism, max_tokens, eos stop, temperature/top-k paths), and that the registry +
artifact detection route a quanta artifact to the quanta engine.

    uv run python -m parity.omlx_shim_test
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import mlx.core as mx

from quanta.omlx_patch import ENGINE_REGISTRY, _engine_type_for, register_engine
from quanta.shim.omlx import QuantaOmlxEngine, detect_quanta_artifact

VOCAB = 64


class StubRuntime:
    """Deterministic runtime: next-token logits peak at (last_token + 1) % VOCAB."""

    num_layers = 2

    def __call__(self, token_ids: mx.array, **kw) -> mx.array:
        ids = token_ids.reshape(-1)
        nxt = (int(ids[-1].item()) + 1) % VOCAB
        onehot = mx.zeros((VOCAB,)).at[nxt].add(10.0)
        return mx.broadcast_to(onehot, (1, ids.shape[0], VOCAB))


class StubTokenizer:
    eos_id = 999

    def encode(self, text: str, **kw):
        return [1, 2, 3]  # prompt; last id = 3

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class ScriptRuntime:
    """Emits a fixed token sequence (one per forward), independent of input."""

    num_layers = 1

    def __init__(self, script: list[int]) -> None:
        self.script, self.i = script, 0

    def __call__(self, token_ids: mx.array, **kw) -> mx.array:
        t = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        onehot = mx.zeros((VOCAB,)).at[t].add(10.0)
        return mx.broadcast_to(onehot, (1, token_ids.reshape(-1).shape[0], VOCAB))


class BytesTokenizer:
    """Byte-level stub: tokens 10+11 form one 3-byte char (世) split across two steps."""

    _b = {10: b"\xe4\xb8", 11: b"\x96", 12: b"!"}

    def encode(self, text: str, **kw):
        return [1]

    def decode_bytes(self, ids):
        return b"".join(self._b.get(int(i), b"") for i in ids)

    def decode(self, ids):
        return self.decode_bytes(ids).decode("utf-8", "replace")


async def _collect(agen):
    return [c async for c in agen]


def run() -> None:
    eng = QuantaOmlxEngine("stub", runtime=StubRuntime(), tokenizer=StubTokenizer())

    # greedy decode is deterministic: prompt last=3 -> 4,5,6,7
    out = asyncio.run(eng.generate("hi", max_tokens=4, temperature=0.0))
    greedy_ok = out.tokens == [4, 5, 6, 7] and out.completion_tokens == 4

    # max_tokens respected; sampled path runs and stays in-vocab
    s = asyncio.run(eng.generate("hi", max_tokens=3, temperature=0.8, top_k=5, top_p=0.9, min_p=0.05))
    sampled_ok = len(s.tokens) == 3 and all(0 <= t < VOCAB for t in s.tokens)

    # eos stop: a runtime that emits eos halts early
    class EosRuntime(StubRuntime):
        def __call__(self, token_ids, **kw):
            return mx.broadcast_to(mx.zeros((VOCAB,)).at[VOCAB - 1].add(10.0),
                                   (1, token_ids.reshape(-1).shape[0], VOCAB))

    eng2 = QuantaOmlxEngine("stub", runtime=EosRuntime(),
                            tokenizer=StubTokenizer(), eos_token_ids={VOCAB - 1})
    eos_out = asyncio.run(eng2.generate("hi", max_tokens=10, temperature=0.0))
    eos_ok = eos_out.finished and eos_out.completion_tokens == 1

    # byte-accurate streaming: a multi-byte char split across tokens never surfaces as � mid-stream
    beng = QuantaOmlxEngine("stub", runtime=ScriptRuntime([10, 11, 12]), tokenizer=BytesTokenizer())
    bchunks = asyncio.run(_collect(beng.stream_generate("hi", max_tokens=3, temperature=0.0)))
    pieces = [c.new_text for c in bchunks]
    stream_ok = (pieces == ["", "世", "!"] and bchunks[-1].text == "世!"
                 and "".join(pieces) == bchunks[-1].text)

    # finish_reason: exhausting max_tokens (no eos/stop) reports "length"
    lchunks = asyncio.run(_collect(eng.stream_generate("hi", max_tokens=4, temperature=0.0)))
    length_ok = (lchunks[-1].finish_reason == "length" and lchunks[-1].finished
                 and lchunks[-1].tokens == [4, 5, 6, 7])

    # stop string: the match (and anything after) is cut from the output; reason "stop"
    schunks = asyncio.run(_collect(eng.stream_generate("hi", max_tokens=10, temperature=0.0, stop=["6"])))
    stop_ok = (schunks[-1].text == "4 5 " and schunks[-1].finish_reason == "stop"
               and schunks[-1].finished and schunks[-1].tokens == [4, 5, 6])

    # registry + detection: a quanta artifact routes to the 'quanta' engine
    art = Path(tempfile.mkdtemp()) / "kimi-quanta_int3"
    art.mkdir(parents=True)
    (art / "manifest.json").write_text(json.dumps({"format": "quanta", "tensors": {}}))
    (art / "config.json").write_text(json.dumps({"text_config": {"model_type": "kimi_k2"}}))
    detect_ok = detect_quanta_artifact(art) is not None and detect_quanta_artifact(art).model_type == "kimi_k2"
    registry_ok = "quanta" in ENGINE_REGISTRY and _engine_type_for(art) == "quanta"
    callable(register_engine)  # public registration API exists

    print("\n=== oMLX shim (standalone) ===")
    print(f"greedy decode (cached, deterministic): {greedy_ok}  tokens={out.tokens}")
    print(f"sampled kwargs path (temp/top-k/p/min): {sampled_ok}")
    print(f"eos early stop                        : {eos_ok}")
    print(f"byte-accurate streaming (no � split) : {stream_ok}  pieces={pieces}")
    print(f"finish_reason length on max_tokens    : {length_ok}")
    print(f"stop-string truncation               : {stop_ok}  text={schunks[-1].text!r}")
    print(f"artifact detection                    : {detect_ok}")
    print(f"engine registry routes quanta         : {registry_ok}")
    assert all([greedy_ok, sampled_ok, eos_ok, stream_ok, length_ok, stop_ok, detect_ok, registry_ok])
    print("oMLX shim OK (engine loop + kwargs + streaming + registry)")


if __name__ == "__main__":
    run()
