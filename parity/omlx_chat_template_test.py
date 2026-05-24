"""Gate: oMLX-shim forwards chat-TEMPLATE controls (thinking / enable_thinking / reasoning_effort).

These are chat-template variables, not sampler knobs: the shim must route them into
``apply_chat_template`` (Kimi's template reads ``thinking``; Nemotron's reads ``enable_thinking``;
``reasoning_effort`` is forwarded for templates that use it, e.g. DSV4) — and must NOT leak sampler
kwargs (temperature, ...) into the template. Verified model-free with a recording tokenizer.

    uv run python -m parity.omlx_chat_template_test
"""

from __future__ import annotations

import asyncio

import mlx.core as mx

from quanta.shim.omlx import QuantaOmlxEngine

NEM_ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"


class _RecordingTok:
    """Captures the kwargs ``apply_chat_template`` receives; returns a trivial prompt."""

    eos_id = 11
    stop_ids = (11,)

    def __init__(self) -> None:
        self.seen: dict | None = None

    def apply_chat_template(self, messages, *, tools=None, tokenize=False,
                            add_generation_prompt=True, **kw):
        self.seen = {**kw, "tools": tools}
        return "PROMPT"

    def encode(self, text, *, add_bos=False, allow_special=False):
        return [1, 2]

    def decode(self, ids, *, skip_special=True):
        return ""


class _EosRuntime:
    """Hybrid-signature stub that emits eos immediately (so generation stops at step 0)."""

    def __init__(self) -> None:
        self.cfg = type("C", (), {"layers_block_type": ["mamba"]})()
        self.num_layers = 1

    def __call__(self, token_ids, *, caches=None, ssm=None, conv=None, **kw):
        t = int(token_ids.shape[0])
        row = (mx.arange(50) == 11).astype(mx.float32) * 60.0 - 30.0  # argmax == eos(11)
        return mx.broadcast_to(row, (1, t, 50)), [mx.zeros((1,))], [mx.zeros((1,))]


def _engine(tok: _RecordingTok) -> QuantaOmlxEngine:
    return QuantaOmlxEngine(NEM_ART, runtime=_EosRuntime(), tokenizer=tok, eos_token_ids={11})


def run() -> None:
    ok = True
    msgs = [{"role": "user", "content": "hi"}]

    # (1) chat forwards template controls (direct flags + chat_template_kwargs dict); NOT sampler kwargs
    tok = _RecordingTok()
    asyncio.run(_engine(tok).chat(msgs, enable_thinking=False, reasoning_effort="high",
                                  temperature=0.7, top_p=0.8,
                                  chat_template_kwargs={"truncate_history_thinking": False}))
    seen = tok.seen or {}
    good = (seen.get("enable_thinking") is False and seen.get("reasoning_effort") == "high"
            and seen.get("truncate_history_thinking") is False
            and "temperature" not in seen and "top_p" not in seen)
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] chat forwards template kwargs, sampler stays out: {seen}")

    # (2) default: no template controls forwarded (the template uses its own defaults)
    tok2 = _RecordingTok()
    asyncio.run(_engine(tok2).chat(msgs, temperature=0.0))
    seen2 = tok2.seen or {}
    good = "enable_thinking" not in seen2 and "reasoning_effort" not in seen2 and "thinking" not in seen2
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] no template kwargs by default: {seen2}")

    # (3) stream_chat forwards too (Kimi-style `thinking` flag)
    tok3 = _RecordingTok()

    async def _drive():
        async for _ in _engine(tok3).stream_chat(msgs, thinking=False):
            pass

    asyncio.run(_drive())
    good = (tok3.seen or {}).get("thinking") is False
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] stream_chat forwards thinking={(tok3.seen or {}).get('thinking')}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
