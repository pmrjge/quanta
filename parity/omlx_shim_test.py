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
    print(f"artifact detection                    : {detect_ok}")
    print(f"engine registry routes quanta         : {registry_ok}")
    assert all([greedy_ok, sampled_ok, eos_ok, detect_ok, registry_ok])
    print("oMLX shim OK (engine loop + kwargs + registry)")


if __name__ == "__main__":
    run()
