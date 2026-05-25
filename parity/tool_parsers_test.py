"""Gate: the quanta tool-call parsers + the unified dispatcher the oMLX patch installs — pure, no model.

The quanta engine emits raw output, so each model's tool markup reaches oMLX as literal text. oMLX's
registry natively covers xml/json/gemma/glm/qwen but NOT Kimi's special-token section or MiniMax's
``<minimax:tool_call>`` wrapper. :func:`quanta.shim.tool_parsers.parse_quanta_tool_calls` tries the
quanta parsers (Kimi/MiniMax/GLM/Qwen) and returns the first match, else ``None`` so the patch delegates
to oMLX's original parser. This verifies:

  (1) each parser extracts ``{id,name,arguments}`` from its own format (arguments a JSON string), with
      typed values recovered (numbers/objects via JSON, strings raw) and the section stripped from the
      cleaned text;
  (2) the parsers are **strict** — they never swallow another model's markup: GLM requires ``<arg_key>``;
      Qwen requires a JSON body with ``name``; so a Nemotron ``<function=…>`` XML tool call (and plain
      prose) yields ``None`` from the dispatcher → the oMLX patch delegates it (the losslessness the
      ``nemotron_omlx_contract_test`` depends on);
  (3) the dispatcher routes each format to the right parser and is order-robust (GLM vs Qwen both use
      ``<tool_call>`` — disambiguated by inner shape).

    uv run python -m parity.tool_parsers_test
"""

from __future__ import annotations

import json

from quanta.shim.tool_parsers import (
    parse_glm_tool_calls,
    parse_minimax_tool_calls,
    parse_qwen_tool_calls,
    parse_quanta_tool_calls,
)

MINIMAX = ('before <minimax:tool_call>\n<invoke name="get_weather">'
           '<parameter name="location">Tokyo</parameter>'
           '<parameter name="opts">{"units": "c"}</parameter></invoke>\n</minimax:tool_call> after')
GLM = ('think <tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>Tokyo</arg_value>\n'
       '<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>')
QWEN = 'sure <tool_call>\n{"name": "get_weather", "arguments": {"location": "Tokyo", "days": 3}}\n</tool_call>'
KIMI = ("<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
        '<|tool_call_argument_begin|>{"location": "Tokyo"}<|tool_call_end|><|tool_calls_section_end|>')
# Nemotron / oMLX-native XML — MUST NOT be parsed by any quanta parser (delegate to oMLX).
NEMOTRON = '<tool_call><function=get_weather><parameter=location>Tokyo</parameter></function></tool_call>'
PLAIN = "no tools here, just prose with <think>reasoning</think>."


def _check(label: str, result, *, name: str, args_subset: dict, cleaned_excludes: str) -> bool:
    if result is None:
        print(f"  [FAIL] {label}: returned None")
        return False
    cleaned, calls = result
    if not calls or calls[0]["name"] != name:
        print(f"  [FAIL] {label}: name={calls[0]['name'] if calls else None!r} (want {name!r})")
        return False
    args = json.loads(calls[0]["arguments"])
    for k, v in args_subset.items():
        if args.get(k) != v:
            print(f"  [FAIL] {label}: arg {k}={args.get(k)!r} (want {v!r})")
            return False
    if cleaned_excludes in cleaned:
        print(f"  [FAIL] {label}: cleaned text still contains {cleaned_excludes!r}: {cleaned!r}")
        return False
    print(f"  [OK] {label}: name={name} args={args} cleaned={cleaned!r}")
    return True


def run() -> None:
    ok = True

    # (1) each parser on its own format — typed values + section stripped
    ok &= _check("minimax", parse_minimax_tool_calls(MINIMAX), name="get_weather",
                 args_subset={"location": "Tokyo", "opts": {"units": "c"}},  # str raw, object via JSON
                 cleaned_excludes="<minimax:tool_call>")
    ok &= _check("glm", parse_glm_tool_calls(GLM), name="get_weather",
                 args_subset={"location": "Tokyo", "days": 3},  # 3 recovered as int
                 cleaned_excludes="<tool_call>")
    ok &= _check("qwen", parse_qwen_tool_calls(QWEN), name="get_weather",
                 args_subset={"location": "Tokyo", "days": 3}, cleaned_excludes="<tool_call>")

    # (2) strictness / delegation — the dispatcher returns None for non-quanta markup
    nem = parse_quanta_tool_calls(NEMOTRON)
    g1 = nem is None
    ok &= g1
    print(f"  [{'OK' if g1 else 'FAIL'}] Nemotron <function=> delegates (dispatcher None): {nem}")
    g2 = parse_quanta_tool_calls(PLAIN) is None
    ok &= g2
    print(f"  [{'OK' if g2 else 'FAIL'}] plain prose -> None (no false tool call)")
    # GLM parser must reject Qwen's JSON block (no <arg_key>); Qwen parser must reject GLM's body.
    g3 = parse_glm_tool_calls(QWEN) is None and parse_qwen_tool_calls(GLM) is None
    ok &= g3
    print(f"  [{'OK' if g3 else 'FAIL'}] GLM/Qwen parsers don't cross-match each other's format")
    # Qwen parser must reject Nemotron XML (body isn't JSON with a name)
    g4 = parse_qwen_tool_calls(NEMOTRON) is None and parse_glm_tool_calls(NEMOTRON) is None
    ok &= g4
    print(f"  [{'OK' if g4 else 'FAIL'}] GLM/Qwen parsers reject Nemotron <function=> XML")

    # (3) dispatcher routes each format to the right parser
    routes = {
        "minimax": parse_quanta_tool_calls(MINIMAX),
        "glm": parse_quanta_tool_calls(GLM),
        "qwen": parse_quanta_tool_calls(QWEN),
        "kimi": parse_quanta_tool_calls(KIMI),
    }
    g5 = all(r is not None and r[1] and r[1][0]["name"] == "get_weather" for r in routes.values())
    ok &= g5
    print(f"  [{'OK' if g5 else 'FAIL'}] dispatcher routes all four formats -> get_weather")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
