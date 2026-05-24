"""Gate: the quanta oMLX engine emits **raw** output that satisfies oMLX's parsing contract.

The engine does no reasoning/tool parsing itself — it streams raw text with the model's literal
markers (``</think>``, ``<|tool_calls_section_begin|>`` …) preserved, and oMLX's server splits them.
This test drives ``QuantaOmlxEngine`` with a scripted stub runtime + the real Kimi tokenizer, then
feeds the engine's raw output through a faithful copy of oMLX's ``extract_thinking``
(``omlx/api/thinking.py``) to confirm reasoning/content split exactly as oMLX would. Model-free
(tiktoken + jinja2, no resident model).

    uv run --with tiktoken --with jinja2 python -m parity.kimi_omlx_contract_test
"""

from __future__ import annotations

import asyncio
import re

import mlx.core as mx

from quanta.shim.omlx import QuantaOmlxEngine
from quanta.tokenizer import KimiTokenizer

ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int2g64"

# --- faithful copy of omlx/api/thinking.py::extract_thinking (the contract we must satisfy) ---
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_TAIL = re.compile(r"^(.*?)</think>", re.DOTALL)


def omlx_extract_thinking(text: str) -> tuple[str, str]:
    if not text:
        return ("", "")
    parts, remaining = [], text
    while True:
        m = _THINK.search(remaining)
        if not m:
            break
        parts.append(m.group(1))
        remaining = remaining[: m.start()] + remaining[m.end():]
    if parts:
        return ("\n".join(parts).strip(), remaining.strip())
    if "</think>" in text and "<think>" not in text:           # Kimi's shape: prompt opened <think>
        m = _THINK_TAIL.match(text)
        if m:
            return (m.group(1).strip(), text[m.end():].strip())
    if "<think>" in text and "</think>" not in text:
        idx = text.index("<think>")
        return ("", (text[:idx] + text[idx + len("<think>"):]).strip())
    return ("", text)


class _StubRuntime:
    def __init__(self, scripted: list[int], vocab: int, eos: int, num_layers: int = 1) -> None:
        self.scripted, self.vocab, self.eos, self.num_layers = scripted, vocab, eos, num_layers
        self._i = 0

        class _Cfg:
            bos_token_id = 163584

        self.cfg = _Cfg()

    def __call__(self, token_ids, **kw):
        nxt = self.scripted[self._i] if self._i < len(self.scripted) else self.eos
        self._i += 1
        return ((mx.arange(self.vocab) == nxt).astype(mx.float32) * 1e4).reshape(1, 1, self.vocab)


def _run(engine: QuantaOmlxEngine):
    async def _drive():
        chunks = []
        async for ch in engine.stream_generate("hi", max_tokens=400, temperature=0.0):
            chunks.append(ch)
        return chunks

    return asyncio.run(_drive())


def run() -> None:
    tk = KimiTokenizer(ART)
    vocab = tk.n_base + 256
    im_end = tk.special_tokens["<|im_end|>"]
    ok = True

    def engine_for(text: str) -> QuantaOmlxEngine:
        scripted = tk.encode(text, add_bos=False, allow_special=True) + [im_end]
        return QuantaOmlxEngine("stub", runtime=_StubRuntime(scripted, vocab, im_end),
                                tokenizer=tk, eos_token_ids=set(tk.stop_ids))

    # --- (a) reasoning + content: raw text keeps </think>; oMLX splits it correctly ---
    chunks = _run(engine_for("Let me think about it.</think>The answer is 42."))
    last = chunks[-1]
    raw_ok = last.text == "Let me think about it.</think>The answer is 42."
    reasoning, content = omlx_extract_thinking(last.text)
    split_ok = reasoning == "Let me think about it." and content == "The answer is 42."
    stream_ok = "".join(c.new_text for c in chunks) == last.text  # deltas assemble to raw text
    good = raw_ok and split_ok and stream_ok and last.tool_calls is None
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] reasoning+content: raw={raw_ok} oMLX-split={split_ok} "
          f"stream-assembles={stream_ok}")
    print(f"        raw={last.text!r} -> reasoning={reasoning!r} content={content!r}")

    # --- (b) plain answer, no </think> emitted: all content ---
    last = _run(engine_for("Just the answer."))[-1]
    reasoning, content = omlx_extract_thinking(last.text)
    good = last.text == "Just the answer." and reasoning == "" and content == "Just the answer."
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] no-think: raw={last.text!r} -> content={content!r}")

    # --- (c) tool call: markers pass through raw (engine sets no tool_calls; oMLX-side parser TBD) ---
    tool = ("checking</think>one sec<|tool_calls_section_begin|><|tool_call_begin|>"
            "functions.get_weather:0<|tool_call_argument_begin|>{\"location\": \"SF\"}"
            "<|tool_call_end|><|tool_calls_section_end|>")
    last = _run(engine_for(tool))[-1]
    markers_present = ("</think>" in last.text and "<|tool_calls_section_begin|>" in last.text
                       and "functions.get_weather:0" in last.text and "<|tool_call_end|>" in last.text)
    good = markers_present and last.tool_calls is None  # raw pass-through, no engine-side parsing
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] tool: raw markers present={markers_present} "
          f"engine_tool_calls={last.tool_calls} (parsing deferred to oMLX)")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
