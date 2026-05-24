"""Gate: Nemotron's reasoning/tool output is handled by oMLX's STOCK parsers — no custom patcher.

Unlike Kimi (special-token ``<|tool_calls_section_begin|>`` markup → the ``kimi_tools`` autopatch),
Nemotron-3 emits ``<think>...</think>`` reasoning + Qwen/Llama-style XML tool calls
(``<tool_call><function=name><parameter=key>value</parameter></function></tool_call>``). Both are
already covered by oMLX's ``extract_thinking`` and ``_parse_xml_tool_calls``, and the markers are
ordinary tokens that ``clean_special_tokens`` leaves intact. This verifies that against the REAL oMLX
with the quanta autopatch armed — confirming (a) no Nemotron parser is needed and (b) the Kimi patch
correctly *delegates* Nemotron's non-special markup to oMLX's original registry.

Serving Nemotron through oMLX still needs its own engine (Mamba/GQA generation, ≠ the MLA shim loop —
task #39); this gate only locks down the parsing contract.

    uv run --extra omlx python -m parity.nemotron_omlx_contract_test
"""

from __future__ import annotations

import json

import quanta.omlx_patch as patch

patch.install()  # arm the quanta autopatch (Kimi tool parser) — Nemotron markup must delegate past it

from omlx.api import thinking, tool_calling  # noqa: E402  (must follow patch.install)

# Nemotron tool call exactly as chat_template.jinja renders it.
TOOL = ("<tool_call>\n<function=get_weather>\n<parameter=location>\nSF\n</parameter>\n"
        "</function>\n</tool_call>")


def run() -> None:
    ok = True

    # (0) the Kimi autopatch is live on the real module (Nemotron must still work through it)
    good = getattr(tool_calling, patch.PATCH_MARKER, None) == patch.PATCH_VERSION
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] quanta autopatch armed on real omlx.api.tool_calling")

    # (1) reasoning: model emits {reasoning}\n</think>\n{content} (prompt opened <think>)
    reasoning, content = thinking.extract_thinking("Let me reason about it.\n</think>\nThe answer is 4.")
    good = reasoning == "Let me reason about it." and content == "The answer is 4."
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] stock extract_thinking: reasoning={reasoning!r} content={content!r}")

    # (2) tool call: Nemotron XML parsed by oMLX (delegated past the Kimi patch -> _parse_xml_tool_calls)
    cleaned, tcs = tool_calling.parse_tool_calls("Sure.\n" + TOOL, tokenizer=None, tools=None)
    good = (tcs is not None and len(tcs) == 1 and tcs[0].function.name == "get_weather"
            and json.loads(tcs[0].function.arguments) == {"location": "SF"} and cleaned == "Sure.")
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] stock parse_tool_calls: "
          f"{tcs[0].function.name if tcs else None} args={tcs[0].function.arguments if tcs else None} cleaned={cleaned!r}")

    # (3) full server path: split reasoning, then parse tools from the remainder
    full = ("planning the call.\n</think>\nOn it.\n<tool_call>\n<function=add>\n<parameter=a>\n1\n"
            "</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>")
    r, rest = thinking.extract_thinking(full)
    cleaned, tcs = tool_calling.parse_tool_calls(rest, tokenizer=None, tools=None)
    good = (r == "planning the call." and cleaned == "On it." and tcs is not None and len(tcs) == 1
            and tcs[0].function.name == "add" and json.loads(tcs[0].function.arguments) == {"a": 1, "b": 2})
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] combined: reasoning={r!r} content={cleaned!r} "
          f"tool={tcs[0].function.name if tcs else None} args={tcs[0].function.arguments if tcs else None}")

    # (4) non-tool plain answer is untouched
    c2, t2 = tool_calling.parse_tool_calls("Just a normal answer.", tokenizer=None, tools=None)
    good = t2 is None
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] plain answer -> no tool calls (t2={t2})")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
