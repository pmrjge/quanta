"""Real-model eval: Qwen3.6-35B-A3B orchestrator tool-calling + structured-JSON fidelity.

The orchestrator's job in the agentic app is to (a) pick the right function and emit a well-formed tool
call and (b) produce schema-valid structured JSON on demand. This drives the BAKED int4 artifact through
the EXACT #26 serving surface — :class:`quanta.shim.omlx.QuantaOmlxEngine` →
:class:`_Qwen35BatchedSession`, whose ``admit`` pins the dynamic-YaRN factor from the per-stream budget
before batched decode — renders tool prompts via the checkpoint's ``chat_template.jinja`` (the
qwen3_coder XML tool-call form), greedily decodes (temperature 0 ⇒ deterministic), and parses/validates
the output. It is an end-to-end ARBITER of the orchestrator (like teacher-forced ppl is for coherence),
not a model-free unit test.

Tool-call format the template instructs (see ``chat_template.jinja``):

    <tool_call>
    <function=get_current_weather>
    <parameter=location>
    Paris, France
    </parameter>
    <parameter=unit>
    celsius
    </parameter>
    </function>
    </tool_call>

HEAVY: loads the ~21 GB resident model. Run SOLO (one model at a time — OOM-reboot hazard), after
verifying GPU memory is clear:

    uv run python -u -m parity.qwen36_tool_call_eval
"""

from __future__ import annotations

import asyncio
import json
import re

import mlx.core as mx

from quanta.shim.omlx import QuantaOmlxEngine

ART = "/Users/pmrj/models/Qwen3.6-35B-A3B-quanta_int4g64"

# --- tool schemas (standard OpenAI function shape; the template just ``tojson``s each) -------------
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string",
                             "description": "City and country, e.g. 'Paris, France'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                         "description": "Temperature unit to report in"},
            },
            "required": ["location", "unit"],
        },
    },
}
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the public web for up-to-date information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
        },
    },
}
CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression and return the numeric result.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string",
                                          "description": "An arithmetic expression, e.g. '47 * 89'"}},
            "required": ["expression"],
        },
    },
}

# --- qwen3_coder XML tool-call parser -------------------------------------------------------------
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNC_RE = re.compile(r"<function\s*=\s*([^>\n]+?)\s*>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter\s*=\s*([^>\n]+?)\s*>\s*(.*?)\s*</parameter>", re.DOTALL)


def parse_tool_calls(text: str) -> tuple[list[dict], bool]:
    """Extract ``[{name, arguments:{k:v}}]`` from the qwen3_coder XML form. Returns ``(calls, wrapped)``
    where ``wrapped`` is True iff every ``<function=...>`` sat inside a ``<tool_call>`` envelope (the
    template's required nesting). Parses bare ``<function=...>`` blocks too (lenient) so a missing
    envelope is reported as a format issue rather than hiding the call."""
    calls: list[dict] = []
    for body in _TOOLCALL_RE.findall(text):
        for name, fn_body in _FUNC_RE.findall(body):
            calls.append({"name": name.strip(),
                          "arguments": {k.strip(): v.strip() for k, v in _PARAM_RE.findall(fn_body)}})
    wrapped = bool(calls)
    if not calls:  # lenient: maybe emitted <function=...> without the <tool_call> envelope
        for name, fn_body in _FUNC_RE.findall(text):
            calls.append({"name": name.strip(),
                          "arguments": {k.strip(): v.strip() for k, v in _PARAM_RE.findall(fn_body)}})
        wrapped = False
    return calls, wrapped


def extract_json(text: str) -> dict | None:
    """First parseable top-level JSON object in ``text`` (tolerates prose / code fences around it)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


# --- eval cases -----------------------------------------------------------------------------------
def _weather_check(text: str) -> tuple[bool, str]:
    calls, wrapped = parse_tool_calls(text)
    if not calls:
        return False, "no tool call emitted"
    c = calls[0]
    if c["name"] != "get_current_weather":
        return False, f"wrong function {c['name']!r}"
    args = c["arguments"]
    loc_ok = "paris" in args.get("location", "").lower()
    unit_ok = args.get("unit", "").lower().strip() == "celsius"
    ok = loc_ok and unit_ok
    return ok, (f"call={c['name']} location={args.get('location')!r} unit={args.get('unit')!r} "
                f"wrapped={wrapped}")


def _search_check(text: str) -> tuple[bool, str]:
    calls, wrapped = parse_tool_calls(text)
    if not calls:
        return False, "no tool call emitted"
    c = calls[0]
    if c["name"] != "web_search":
        return False, f"selected wrong function {c['name']!r} (expected web_search)"
    q = c["arguments"].get("query", "")
    ok = "mars" in q.lower() and len(q) > 3
    return ok, f"call={c['name']} query={q!r} wrapped={wrapped}"


def _no_tool_check(text: str) -> tuple[bool, str]:
    calls, _ = parse_tool_calls(text)
    if calls:
        return False, f"hallucinated tool call {calls[0]['name']!r} for a general-knowledge Q"
    return "paris" in text.lower(), f"answered inline (no call); says-Paris={'paris' in text.lower()}"


def _json_check(text: str) -> tuple[bool, str]:
    obj = extract_json(text)
    if obj is None:
        return False, "no parseable JSON object"
    name_ok = isinstance(obj.get("name"), str) and obj["name"].strip().lower() == "alice"
    age_ok = isinstance(obj.get("age"), (int, float)) and int(obj["age"]) == 30
    ok = name_ok and age_ok
    return ok, f"json={obj} name_ok={name_ok} age_ok={age_ok}"


CASES = [
    {
        "name": "tool/weather (extract args)", "hard": True, "max_tokens": 160,
        "tools": [WEATHER_TOOL],
        "messages": [{"role": "user",
                      "content": "What is the current weather in Paris, France? Report it in celsius."}],
        "check": _weather_check,
    },
    {
        "name": "tool/selection (3 tools)", "hard": True, "max_tokens": 160,
        "tools": [WEATHER_TOOL, SEARCH_TOOL, CALC_TOOL],
        "messages": [{"role": "user",
                      "content": "Find the latest news about NASA's Mars rover."}],
        "check": _search_check,
    },
    {
        "name": "tool/abstain (no call needed)", "hard": False, "max_tokens": 96,
        "tools": [WEATHER_TOOL, SEARCH_TOOL],
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "check": _no_tool_check,
    },
    {
        "name": "json/schema adherence", "hard": True, "max_tokens": 128,
        "tools": None,
        "messages": [
            {"role": "system", "content": "You are a JSON API. Reply with ONLY a single JSON object, "
                                          "no prose, no code fences."},
            {"role": "user", "content": 'Return a JSON object with keys "name" (string) and "age" '
                                        '(number) describing a person named Alice who is 30 years old.'},
        ],
        "check": _json_check,
    },
]


async def _run() -> bool:
    mx.set_wired_limit(int(60 * 1024 ** 3))  # 21 GB model + activations; plenty of headroom
    engine = QuantaOmlxEngine(ART)
    await engine.start()
    tok = engine._tokenizer
    print(f"\n=== Qwen3.6-35B-A3B orchestrator: tool-call + JSON eval (model_type={engine.model_type}) ===")
    print(f"    serving via _Qwen35BatchedSession (default capacity B={engine._default_capacity(1)})\n")

    hard_ok = True
    for case in CASES:
        prompt = tok.apply_chat_template(case["messages"], tokenize=False,
                                         add_generation_prompt=True, tools=case["tools"],
                                         enable_thinking=False)
        outs = await engine.batched_generate(
            [prompt],
            per_request=[{"max_tokens": case["max_tokens"], "temperature": 0.0, "add_bos": False}],
        )
        text = outs[0].text
        ok, detail = case["check"](text)
        tag = "OK" if ok else ("FAIL" if case["hard"] else "INFO")
        if case["hard"]:
            hard_ok &= ok
        print(f"  [{tag}] {case['name']}: {detail}")
        snippet = text.strip().replace("\n", "\\n")
        print(f"        raw[:200]: {snippet[:200]}")

    print("\nPASS" if hard_ok else "\nFAIL")
    return hard_ok


def run() -> None:
    ok = asyncio.run(_run())
    assert ok, "orchestrator tool-call/JSON eval failed on a hard case"


if __name__ == "__main__":
    run()
