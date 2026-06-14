"""Model-free M3-7a gate: the MiniMax-M3-VL oMLX output parsers (reasoning + nested-XML tool calls).

The first sub-milestone of the MiniMax-M3 oMLX serving shim is the model's *output* parsing surface —
the analog of Qwen3.5's N3-2 (qwen3_coder tool parser + qwen3 reasoning parser). M3 emits a markup the
in-tree parsers do NOT cover:

* **Reasoning** is wrapped in ``<mm:think>…</mm:think>`` (vocab ids 200059/200060), NOT the bare
  ``<think>``/``</think>`` (200050/200051) the vocab also carries.
* **Tool calls** are a *namespace-prefixed recursive nested XML*: the section is
  ``]<]minimax[>[<tool_call>`` … ``]<]minimax[>[</tool_call>`` (``]<]minimax[>[`` = ns_token, id 200058),
  each call is ``ns<invoke name="NAME">`` … ``ns</invoke>``, and the arguments are produced by the chat
  template's recursive ``to_xml`` macro (mapping → ``ns<k>…ns</k>``, list → ``ns<item>…ns</item>``, bool
  → ``true``/``false``, other scalar → raw text). This is RICHER than M2.7's flat
  ``<minimax:tool_call>``/``<parameter name=>`` form and than Qwen3-Coder's ``<function=…>`` form.

The parity discipline (CLAUDE.md Methodology): build a **reference renderer** that reproduces the jinja
``to_xml`` / invoke / section rendering exactly, then assert the parser inverts it for a battery of arg
shapes (flat, typed, nested mapping, list of scalars, list of dicts — the template's own example). A
green round-trip proves :func:`parse_minimax_m3_tool_calls` is the inverse of the on-disk template's
renderer without needing the 233 GiB weights. Also gated: reasoning shapes, response formatting,
Protocol conformance, markup disjointness vs the sibling parsers, and the legacy-dispatcher routing.

No real weights, no torch/PIL — pure-Python on the parser module; runs in the model-free sweep.

    uv run python -m parity.minimax_m3_tools_test
"""

from __future__ import annotations

import json

from quanta.shim.tool_parsers import (
    M3_NS,
    MiniMaxM3ReasoningParser,
    MiniMaxM3ToolParser,
    ReasoningParser,
    ToolParser,
    parse_glm_tool_calls,
    parse_minimax_m3_tool_calls,
    parse_minimax_tool_calls,
    parse_quanta_tool_calls,
    parse_qwen3_coder_tool_calls,
    parse_qwen_tool_calls,
)

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


# --- reference renderer: a faithful re-implementation of the chat_template.jinja macros --------------
# ``to_xml(val, ns)``: mapping → ns<k>to_xml(v)ns</k> per non-None key; iterable (non-str) →
# ns<item>to_xml(item)ns</item> per element; bool → tojson; None → '' (skipped upstream); else → str.
def _to_xml(val: object) -> str:
    ns = M3_NS
    if isinstance(val, dict):
        return "".join(f"{ns}<{k}>{_to_xml(v)}{ns}</{k}>" for k, v in val.items() if v is not None)
    if isinstance(val, bool):                       # NB: bool before int/Iterable — bool is an int
        return json.dumps(val)                      # → "true"/"false"
    if isinstance(val, (list, tuple)):
        return "".join(f"{ns}<item>{_to_xml(item)}{ns}</item>" for item in val)
    return str(val)


def _render_invoke(name: str, args: dict) -> str:
    ns = M3_NS
    body = "".join(f"{ns}<{k}>{_to_xml(v)}{ns}</{k}>" for k, v in args.items() if v is not None)
    return f'{ns}<invoke name="{name}">{body}{ns}</invoke>\n'


def _render_section(*calls: tuple[str, dict]) -> str:
    ns = M3_NS
    return f"{ns}<tool_call>\n" + "".join(_render_invoke(n, a) for n, a in calls) + f"{ns}</tool_call>"


def _roundtrip(name: str, args: dict) -> dict:
    """Render ``args`` through the reference renderer, parse them back, return the recovered dict."""
    cleaned, calls = parse_minimax_m3_tool_calls("pre " + _render_section((name, args)) + " post")
    _ck(cleaned == "pre  post", f"cleaned text wrong for {name}: {cleaned!r}")
    _ck(len(calls) == 1 and calls[0]["name"] == name, f"call shape wrong for {name}: {calls}")
    return json.loads(calls[0]["arguments"])


def run() -> None:
    rp = MiniMaxM3ReasoningParser()
    tp = MiniMaxM3ToolParser()

    # === (A) tool-call round-trips: parser inverts the reference to_xml renderer ====================
    # flat scalars + typed recovery (int / float / bool / string-with-spaces)
    a1 = {"location": "San Francisco", "days": 3, "ratio": 0.5, "metric": True}
    _ck(_roundtrip("get_weather", a1) == a1, "flat typed args did not round-trip")

    # nested mapping
    a2 = {"window": {"start": 1, "end": 9}, "label": "q3"}
    _ck(_roundtrip("query", a2) == a2, "nested mapping did not round-trip")

    # list of scalars
    a3 = {"tags": ["red", "green", "blue"], "n": 3}
    _ck(_roundtrip("tag", a3) == a3, "list-of-scalars did not round-trip")

    # list of dicts — the template's own documented example shape
    a4 = {"steps": [{"id": 1, "act": "go"}, {"id": 2, "act": "stop"}]}
    _ck(_roundtrip("plan", a4) == a4, "list-of-dicts did not round-trip")

    # deeply nested: dict → list → dict → list
    a5 = {"tree": {"nodes": [{"children": [1, 2]}, {"children": []}]}}
    rt5 = _roundtrip("build", a5)
    # the populated branch round-trips exactly; an EMPTY list renders ``ns<children>ns</children>`` —
    # byte-identical to an empty mapping or an empty scalar — so it is irrecoverably an empty leaf
    # ("") on parse. This collapse is intrinsic to the markup (no on-disk distinction), documented
    # here rather than worked around: a tool whose schema needs "empty list" must encode it explicitly.
    _ck(rt5["tree"]["nodes"][0] == {"children": [1, 2]}, f"deep nesting failed: {rt5}")
    _ck(rt5["tree"]["nodes"][1]["children"] == "", f"empty list → empty leaf '' expected: {rt5}")

    # None-valued args are skipped by the renderer (template: ``if v is not none``) → absent on parse
    a6 = {"keep": "yes", "drop": None}
    _ck(_roundtrip("f", a6) == {"keep": "yes"}, "None arg must be omitted, not rendered as 'null'")

    # === (B) multiple invokes / multiple sections / id numbering ====================================
    text_multi = ("ok" + _render_section(("a", {"x": 1}), ("b", {"y": 2}))
                  + _render_section(("c", {"z": 3})) + "!")
    cleaned, calls = parse_minimax_m3_tool_calls(text_multi)
    _ck(cleaned == "ok!", f"multi-section cleaned wrong: {cleaned!r}")
    _ck([c["id"] for c in calls] == ["a:0", "b:1", "c:2"], f"id numbering wrong: {[c['id'] for c in calls]}")
    _ck([json.loads(c["arguments"]) for c in calls] == [{"x": 1}, {"y": 2}, {"z": 3}], "multi args wrong")

    # === (C) string vs number coercion (the documented _typed ambiguity, shared house behavior) =====
    #  a value rendered as bare digits is recovered as a number (json.loads); this matches the M2.7 /
    #  qwen parsers' _typed and is the accepted lossy behavior. "007" (invalid JSON) stays a string.
    rt = _roundtrip("g", {"a": "42", "b": "007", "c": "hello world"})
    _ck(rt == {"a": 42, "b": "007", "c": "hello world"}, f"_typed coercion changed: {rt}")

    # === (D) truncated / empty / absent sections (rule 6: drop the tail, never half-parse) ==========
    ct, cc = parse_minimax_m3_tool_calls("hi" + M3_NS + "<tool_call>" + M3_NS + '<invoke name="x">'
                                         + M3_NS + "<a>1" + M3_NS + "</a>")   # no closing </tool_call>
    _ck(ct == "hi" and cc == [], f"truncated section not dropped: {(ct, cc)}")
    _ck(parse_minimax_m3_tool_calls("just prose, no tools") is None, "absent section must return None")
    cz, czc = parse_minimax_m3_tool_calls("x" + _render_section() + "y")     # empty section, 0 invokes
    _ck(cz == "xy" and czc == [], f"empty section wrong: {(cz, czc)}")

    # === (E) reasoning parser shapes ===============================================================
    _ck(rp.parse("<mm:think>R</mm:think>A") == {"reasoning": "R", "answer": "A"}, "explicit block")
    _ck(rp.parse("R</mm:think>A") == {"reasoning": "R", "answer": "A"}, "bare-opener shape")
    _ck(rp.parse("</mm:think>A") == {"reasoning": "", "answer": "A"}, "disabled-mode empty reasoning")
    _ck(rp.parse("<mm:think>R only") == {"reasoning": "R only", "answer": ""}, "truncated reasoning")
    _ck(rp.parse("plain answer") == {"reasoning": None, "answer": "plain answer"}, "no markers → None")
    _ck(rp.parse("") == {"reasoning": None, "answer": ""}, "empty text")
    multi = rp.parse("<mm:think>one</mm:think>mid<mm:think>two</mm:think>end")
    _ck(multi == {"reasoning": "one\ntwo", "answer": "midend"}, f"multi-block join wrong: {multi}")
    #  M3 reasoning must NOT fire on the bare <think> tokens the vocab also carries (those are 200050/1)
    _ck(rp.parse("<think>not mm</think>x") == {"reasoning": None, "answer": "<think>not mm</think>x"},
        "must not treat bare <think> as M3 reasoning")

    # === (F) response formatting + Protocol conformance ============================================
    _ck(tp.format_tool_response("get_weather:0", "22C sunny") == "<response>22C sunny</response>",
        "tool response wrapper wrong")
    try:
        tp.format_tool_response("", "x")
        _ck(False, "empty tool_call_id accepted")
    except ValueError:
        _ck(True, "empty tool_call_id refused (rule 6)")
    _ck(isinstance(rp, ReasoningParser), "MiniMaxM3ReasoningParser must conform to ReasoningParser")
    _ck(isinstance(tp, ToolParser), "MiniMaxM3ToolParser must conform to ToolParser")
    _ck(tp.parse_tool_calls("no tools here") == [], "ToolParser.parse_tool_calls → [] on no tools")
    _ck(tp.parse_tool_calls(_render_section(("k", {"a": 1})))[0]["name"] == "k", "class parse delegates")

    # === (G) markup disjointness: M3 parser ignores sibling formats, and they ignore M3 ============
    m27 = '<minimax:tool_call><invoke name="x"><parameter name="a">1</parameter></invoke></minimax:tool_call>'
    qwen = '<tool_call>{"name": "x", "arguments": {"a": 1}}</tool_call>'
    glm = "<tool_call>x<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>"
    coder = "<tool_call><function=x><parameter=a>1</parameter></function></tool_call>"
    m3 = _render_section(("x", {"a": 1}))
    _ck(parse_minimax_m3_tool_calls(m27) is None, "M3 parser must ignore M2.7 markup")
    _ck(parse_minimax_m3_tool_calls(qwen) is None, "M3 parser must ignore Hermes markup")
    _ck(parse_minimax_m3_tool_calls(glm) is None, "M3 parser must ignore GLM markup")
    _ck(parse_minimax_m3_tool_calls(coder) is None, "M3 parser must ignore Qwen3-Coder markup")
    _ck(parse_minimax_tool_calls(m3) is None, "M2.7 parser must ignore M3 markup")
    _ck(parse_qwen_tool_calls(m3) is None, "Hermes parser must ignore M3 markup")
    _ck(parse_glm_tool_calls(m3) is None, "GLM parser must ignore M3 markup")
    _ck(parse_qwen3_coder_tool_calls(m3) is None, "Qwen3-Coder parser must ignore M3 markup")

    # === (H) legacy dispatcher routes M3 to the M3 parser =========================================
    routed = parse_quanta_tool_calls("see" + m3 + "ok")
    _ck(routed is not None and routed[0] == "seeok", f"dispatcher cleaned wrong: {routed}")
    _ck(routed[1][0]["name"] == "x" and json.loads(routed[1][0]["arguments"]) == {"a": 1},
        "dispatcher must route M3 markup to parse_minimax_m3_tool_calls")
    #  and a non-M3 sibling format is NOT captured by the M3 entry (Hermes still routes to qwen)
    rq = parse_quanta_tool_calls(qwen)
    _ck(rq is not None and rq[1][0]["name"] == "x", "Hermes must still route through the dispatcher")

    print(f"PARITY-CHECKS: {_N}")
    print(f"PASS — MiniMax-M3-VL oMLX parsers: tool-call round-trips invert the reference to_xml renderer "
          f"(flat/typed/nested/list-of-dicts/None-skip/multi-section), <mm:think> reasoning shapes, "
          f"<response> formatting, Protocol conformance, markup disjointness + dispatcher routing "
          f"({_N} checks).")


if __name__ == "__main__":
    run()
