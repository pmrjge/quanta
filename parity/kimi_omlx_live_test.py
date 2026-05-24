"""Live integration test against the REAL oMLX package (no resident model).

Validates that the Kimi-on-oMLX wiring works against oMLX's actual code, not a replica:
  (1) the autopatch import hook patches the real ``omlx.api.tool_calling`` when it loads;
  (2) oMLX's own ``extract_thinking`` splits the engine's raw ``</think>`` output;
  (3) oMLX's own (patched) ``parse_tool_calls`` turns Kimi tool markup into ``ToolCall`` objects,
      and the full ``extract_tool_calls_with_thinking`` path the server uses returns them;
  (4) ``QuantaOmlxEngine`` really subclasses oMLX's ``BaseEngine`` and the int2-g64 artifact is
      detected/dispatched.

No model is loaded (≈0 GB) — this is purely the integration contract.

    uv run --extra omlx --with tiktoken python -m parity.kimi_omlx_live_test
"""

from __future__ import annotations

import quanta.omlx_patch as patch

patch.install()  # arm the import hook BEFORE importing the oMLX targets

from omlx.api import thinking, tool_calling          # noqa: E402  (must follow patch.install)
from omlx.engine.base import BaseEngine              # noqa: E402

from quanta.shim.kimi_tools import (                 # noqa: E402
    TC_ARG_BEGIN,
    TC_BEGIN,
    TC_END,
    TC_SECTION_BEGIN,
    TC_SECTION_END,
)
from quanta.shim.omlx import QuantaOmlxEngine, detect_quanta_artifact  # noqa: E402

ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int2g64"
SECTION = (TC_SECTION_BEGIN + TC_BEGIN + "functions.get_weather:0" + TC_ARG_BEGIN
           + '{"location": "SF"}' + TC_END + TC_SECTION_END)


def run() -> None:
    ok = True

    # (1) autopatch applied to the REAL omlx.api.tool_calling
    good = getattr(tool_calling, patch.PATCH_MARKER, None) == patch.PATCH_VERSION
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] autopatch applied to real omlx.api.tool_calling")

    # (2) oMLX's own extract_thinking splits the engine's raw reasoning output
    reasoning, content = thinking.extract_thinking("Let me think.</think>The answer is 42.")
    good = reasoning == "Let me think." and content == "The answer is 42."
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] real extract_thinking: reasoning={reasoning!r} content={content!r}")

    # (3a) oMLX's own (patched) parse_tool_calls extracts Kimi tools -> real ToolCall objects
    cleaned, tcs = tool_calling.parse_tool_calls("one sec" + SECTION, tokenizer=None, tools=None)
    good = (cleaned == "one sec" and tcs and len(tcs) == 1
            and tcs[0].function.name == "get_weather"
            and tcs[0].id == "functions.get_weather:0")
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] real parse_tool_calls (patched): "
          f"{tcs[0].function.name if tcs else None} args={tcs[0].function.arguments if tcs else None}")

    # (3b) non-Kimi text still flows through oMLX's original registry (delegation intact)
    c2, t2 = tool_calling.parse_tool_calls("plain answer, no tools", tokenizer=None, tools=None)
    good = t2 is None
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] non-Kimi delegates to original registry (tool_calls={t2})")

    # (3c) the full server path: extract_tool_calls_with_thinking over raw reasoning+content+tools
    try:
        r, c = thinking.extract_thinking("thinking hard</think>here you go" + SECTION)
        extraction = tool_calling.extract_tool_calls_with_thinking(
            r, c, tokenizer=None, tools=[{"type": "function", "function": {"name": "get_weather"}}])
        good = (extraction.tool_calls and extraction.tool_calls[0].function.name == "get_weather"
                and extraction.cleaned_text.strip() == "here you go")
        print(f"  [{'OK' if good else 'FAIL'}] real extract_tool_calls_with_thinking: "
              f"calls={[t.function.name for t in (extraction.tool_calls or [])]} text={extraction.cleaned_text!r}")
    except Exception as e:  # tokenizer-dependent helper; report rather than hard-fail
        good = False
        print(f"  [WARN] extract_tool_calls_with_thinking raised ({type(e).__name__}: {e}) — parse_tool_calls path already verified")
    ok = ok and good

    # (4) engine really subclasses oMLX BaseEngine + artifact detected/dispatched (no model load)
    sub = issubclass(QuantaOmlxEngine, BaseEngine)
    info = detect_quanta_artifact(ART)
    eng = QuantaOmlxEngine(ART)
    good = sub and info is not None and (info.model_type or "").startswith("kimi") and isinstance(eng, BaseEngine)
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] QuantaOmlxEngine<:BaseEngine={sub} "
          f"artifact model_type={info.model_type if info else None}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
