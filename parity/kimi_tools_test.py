"""Gate: oMLX-side Kimi tool-call parser — pure parser + autopatch wiring, model-free.

(a) ``parse_kimi_tool_calls`` extracts ``functions.{name}:{idx}`` calls + cleaned content, and returns
    ``None`` for non-Kimi text (so the original oMLX registry handles it).
(b) ``quanta.omlx_patch._patch_tool_calling`` wraps ``parse_tool_calls`` so Kimi markup yields oMLX
    ``ToolCall`` objects (function name from the id, id kept native for round-trip) while every other
    format delegates to the original. Exercised against fake ``omlx`` modules (oMLX not installed).

    uv run python -m parity.kimi_tools_test
"""

from __future__ import annotations

import sys
import types

import quanta.omlx_patch as patch
from quanta.shim.kimi_tools import (
    TC_ARG_BEGIN,
    TC_BEGIN,
    TC_END,
    TC_SECTION_BEGIN,
    TC_SECTION_END,
    parse_kimi_tool_calls,
)


def _section(*calls: tuple[str, str]) -> str:
    body = "".join(f"{TC_BEGIN}{cid}{TC_ARG_BEGIN}{args}{TC_END}" for cid, args in calls)
    return f"{TC_SECTION_BEGIN}{body}{TC_SECTION_END}"


def run() -> None:
    ok = True

    # --- (a) pure parser ------------------------------------------------------
    text = "one sec" + _section(("functions.get_weather:0", '{"location": "SF"}'),
                                 ("functions.add:1", '{"a": 1, "b": 2}'))
    cleaned, calls = parse_kimi_tool_calls(text)
    got = [(c["name"], c["arguments"], c["id"]) for c in calls]
    good = cleaned == "one sec" and got == [
        ("get_weather", '{"location": "SF"}', "functions.get_weather:0"),
        ("add", '{"a": 1, "b": 2}', "functions.add:1")]
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] parse 2 calls: cleaned={cleaned!r} calls={got}")

    good = parse_kimi_tool_calls("just an answer, no tools") is None
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] non-Kimi text -> None (delegates to original)")

    cleaned, calls = parse_kimi_tool_calls(
        "hmm" + TC_SECTION_BEGIN + TC_BEGIN + "functions.x:0" + TC_ARG_BEGIN + "{}")  # truncated
    good = cleaned == "hmm" and calls == []
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] truncated section -> cleaned={cleaned!r} calls={calls}")

    # --- (b) autopatch wiring (fake omlx modules) -----------------------------
    class FunctionCall:
        def __init__(self, name, arguments):
            self.name, self.arguments = name, arguments

    class ToolCall:
        def __init__(self, id, type, function):
            self.id, self.type, self.function = id, type, function

    for nm in ("omlx", "omlx.api"):
        sys.modules.setdefault(nm, types.ModuleType(nm))
    fake_models = types.ModuleType("omlx.api.openai_models")
    fake_models.ToolCall, fake_models.FunctionCall = ToolCall, FunctionCall
    sys.modules["omlx.api.openai_models"] = fake_models

    tc_mod = types.ModuleType("omlx.api.tool_calling")
    tc_mod.parse_tool_calls = lambda text, tokenizer=None, tools=None: ("ORIG:" + text, None)
    patch._patch_tool_calling(tc_mod)

    cleaned, tcs = tc_mod.parse_tool_calls(
        "here you go" + _section(("functions.get_weather:0", '{"location": "SF"}')), None, None)
    good = (cleaned == "here you go" and tcs is not None and len(tcs) == 1
            and tcs[0].function.name == "get_weather"
            and tcs[0].function.arguments == '{"location": "SF"}'
            and tcs[0].id == "functions.get_weather:0" and tcs[0].type == "function")
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] patched parse_tool_calls: Kimi -> ToolCall("
          f"{tcs[0].function.name if tcs else None})")

    delegated = tc_mod.parse_tool_calls("plain content", None, None)
    good = delegated == ("ORIG:plain content", None)
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] non-Kimi delegates to original: {delegated}")

    good = "omlx.api.tool_calling" in patch._TARGETS
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] omlx.api.tool_calling registered as autopatch target")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
