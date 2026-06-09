"""Model-free gate: the Qwen3-Coder / Nex-N2-Pro tool-call + reasoning parsers (pure, no model).

Nex-N2-Pro (agentic Qwen3.5) renders tool calls as the nested-XML "pythonic" form its chat template
prescribes — ``<tool_call><function=NAME><parameter=KEY>value</parameter>…</function></tool_call>`` —
NOT the Hermes ``<tool_call>{json}</tool_call>`` form :class:`Qwen3ToolParser` handles. Reasoning is the
*pre-opened* ``<think>`` shape: the generation prompt ends with a bare ``<think>`` so the model's output
is ``{reasoning}\n</think>\n\n{answer}`` (no opening tag in the output). This gate locks down both:

  (A) :class:`Qwen3CoderToolParser` extracts ``{id,name,arguments}`` from the nested-XML form —
      single/multi param, typed values (int/object via JSON, strings raw), multi-line values, multiple
      calls, prose-before-call stripped from ``cleaned``, empty list on no-markup; ``format_tool_response``
      round-trips the id and renders the ``<tool_response>`` block; conforms to the ``ToolParser`` Protocol.

  (B) **Strictness** — the parser never swallows another model's markup: Hermes JSON, GLM ``<arg_key>``,
      MiniMax ``<minimax:tool_call>``, and plain prose all yield ``[]``. It DOES match the byte-identical
      Nemotron markup (correct — same format), which is exactly why it is kept OUT of the global
      dispatcher (next).

  (C) **Dispatcher exclusion (the design contract)** — ``parse_quanta_tool_calls`` returns ``None`` for
      this nested-XML form, so serving keeps DELEGATING it to oMLX's stock ``_parse_xml_tool_calls``
      (the path ``parity/nemotron_omlx_contract_test`` gates for this exact markup). Were the parser
      registered in the dispatcher it would silently re-route Nemotron's delegated tool calls. The
      existing Hermes/GLM parsers also reject the nested-XML form (mutual exclusivity).

  (D) :class:`Qwen3ReasoningParser` splits the pre-opened ``<think>`` shape (and the explicit /
      truncated / no-reasoning variants), and composes with the tool parser the way the oMLX server
      does (split reasoning, then parse tools from the remainder).

    uv run python -m parity.qwen3_coder_tool_parser_test
"""

from __future__ import annotations

import json

from quanta.shim.tool_parsers import (
    Qwen3CoderToolParser,
    Qwen3ReasoningParser,
    ToolParser,
    parse_glm_tool_calls,
    parse_quanta_tool_calls,
    parse_qwen3_coder_tool_calls,
    parse_qwen_tool_calls,
)

_N = 0  # PARITY-CHECKS counter


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


# --- the exact Nex-N2-Pro chat_template.jinja tool-call shape (newlines included) ------------------
SINGLE = ("Let me check that.\n<tool_call>\n<function=get_weather>\n<parameter=location>\n"
          "San Francisco\n</parameter>\n</function>\n</tool_call>")
MULTI_PARAM = ("<tool_call>\n<function=get_weather>\n<parameter=location>\nTokyo\n</parameter>\n"
               "<parameter=days>\n3\n</parameter>\n<parameter=opts>\n{\"units\": \"c\"}\n</parameter>\n"
               "</function>\n</tool_call>")
MULTILINE = ("<tool_call>\n<function=run_code>\n<parameter=code>\ndef f():\n    return 1\n\n"
             "print(f())\n</parameter>\n</function>\n</tool_call>")
TWO_CALLS = ("<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n"
             "</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=mul>\n<parameter=a>\n"
             "3\n</parameter>\n<parameter=b>\n4\n</parameter>\n</function>\n</tool_call>")

# foreign markup the strict parser must reject (-> [])
HERMES = 'sure <tool_call>\n{"name": "get_weather", "arguments": {"location": "Tokyo"}}\n</tool_call>'
GLM = ('think <tool_call>get_weather\n<arg_key>location</arg_key>\n<arg_value>Tokyo</arg_value>\n'
       '</tool_call>')
MINIMAX = ('x <minimax:tool_call>\n<invoke name="get_weather">'
           '<parameter name="location">Tokyo</parameter></invoke>\n</minimax:tool_call>')
PLAIN = "no tools here, just prose."
# byte-identical to Nemotron's chat-template tool call (parity/nemotron_omlx_contract_test TOOL).
NEMOTRON = ("<tool_call>\n<function=get_weather>\n<parameter=location>\nSF\n</parameter>\n"
            "</function>\n</tool_call>")


def run() -> None:
    parser = Qwen3CoderToolParser()
    rp = Qwen3ReasoningParser()

    # --- (A) Qwen3CoderToolParser: conformance + extraction ---------------------------------------
    _ck(isinstance(parser, ToolParser),
        "Qwen3CoderToolParser must satisfy the @runtime_checkable ToolParser Protocol")

    c1 = parser.parse_tool_calls(SINGLE)
    _ck(len(c1) == 1 and c1[0]["name"] == "get_weather"
        and json.loads(c1[0]["arguments"]) == {"location": "San Francisco"},
        f"single-param extract wrong: {c1}")

    c2 = parser.parse_tool_calls(MULTI_PARAM)
    a2 = json.loads(c2[0]["arguments"])
    _ck(len(c2) == 1 and a2 == {"location": "Tokyo", "days": 3, "opts": {"units": "c"}},
        f"typed-value recovery wrong (int/object/str): {a2!r}")  # 3 -> int, opts -> object, loc -> str

    c3 = parser.parse_tool_calls(MULTILINE)
    _ck(json.loads(c3[0]["arguments"])["code"] == "def f():\n    return 1\n\nprint(f())",
        f"multi-line value not preserved: {c3}")

    c4 = parser.parse_tool_calls(TWO_CALLS)
    _ck(len(c4) == 2 and c4[0]["name"] == "add" and c4[1]["name"] == "mul"
        and c4[0]["id"] != c4[1]["id"]
        and json.loads(c4[0]["arguments"]) == {"a": 1, "b": 2}
        and json.loads(c4[1]["arguments"]) == {"a": 3, "b": 4},
        f"two-call extract wrong: {c4}")

    # prose-before-the-call is stripped from cleaned text (free function returns (cleaned, calls))
    res = parse_qwen3_coder_tool_calls(SINGLE)
    _ck(res is not None and res[0] == "Let me check that." and "<tool_call>" not in res[0],
        f"cleaned text wrong: {res[0] if res else None!r}")

    _ck(parser.parse_tool_calls(PLAIN) == [],
        "no markup must yield [] (NOT None — emptiness is a successful no-op)")

    # format_tool_response: shape + id round-trip + loud fail on empty id
    resp = parser.format_tool_response(c1[0]["id"], '{"temp": 14}')
    _ck(resp == '<tool_response>\n{"temp": 14}\n</tool_response>', f"tool_response shape wrong: {resp!r}")
    bad_id = False
    try:
        parser.format_tool_response("", "x")
    except ValueError:
        bad_id = True
    _ck(bad_id, "empty tool_call_id must raise ValueError (rule 6 — no silent bad id)")

    # --- (B) strictness: never swallow another model's markup -------------------------------------
    _ck(parser.parse_tool_calls(HERMES) == [], "must reject Hermes JSON <tool_call>{json}</tool_call>")
    _ck(parser.parse_tool_calls(GLM) == [], "must reject GLM <arg_key>/<arg_value> markup")
    _ck(parser.parse_tool_calls(MINIMAX) == [], "must reject MiniMax <minimax:tool_call> wrapper")
    _ck(parser.parse_tool_calls(PLAIN) == [], "must reject plain prose")
    # the byte-identical Nemotron markup IS matched (same format) — this is correct, and is the reason
    # the parser is kept out of the global dispatcher (checked in C).
    cn = parser.parse_tool_calls(NEMOTRON)
    _ck(len(cn) == 1 and cn[0]["name"] == "get_weather"
        and json.loads(cn[0]["arguments"]) == {"location": "SF"},
        f"nested-XML parser must handle the (shared) Nemotron form too: {cn}")

    # --- (C) dispatcher exclusion: serving keeps delegating the shared XML form to oMLX -----------
    _ck(parse_quanta_tool_calls(SINGLE) is None,
        "qwen3_coder XML must NOT be claimed by the legacy dispatcher (else it re-routes Nemotron)")
    _ck(parse_quanta_tool_calls(NEMOTRON) is None,
        "Nemotron delegation contract: dispatcher returns None for the shared <function=> XML")
    # the other <tool_call> parsers also reject the nested-XML form (mutual exclusivity)
    _ck(parse_qwen_tool_calls(SINGLE) is None, "Hermes parser must reject nested-XML (body isn't JSON)")
    _ck(parse_glm_tool_calls(SINGLE) is None, "GLM parser must reject nested-XML (no <arg_key>)")

    # --- (D) Qwen3ReasoningParser on the pre-opened <think> shape ---------------------------------
    # pre-opened (Nex default): prompt ends with bare <think>; output = {reasoning}\n</think>\n\n{answer}
    pre = rp.parse("\nLet me work it out step by step.\n</think>\n\nThe answer is 42.")
    _ck(pre == {"reasoning": "Let me work it out step by step.", "answer": "The answer is 42."},
        f"pre-opened <think> split wrong: {pre}")
    # explicit block, truncated opener, no reasoning
    _ck(rp.parse("a<think>R</think>b") == {"reasoning": "R", "answer": "ab"}, "explicit block wrong")
    _ck(rp.parse("<think>partial, cut off") == {"reasoning": "partial, cut off", "answer": ""},
        "truncated opener wrong")
    none = rp.parse("just an answer, no thinking")
    _ck(none["reasoning"] is None and none["answer"] == "just an answer, no thinking",
        "no reasoning span -> reasoning=None (NOT empty string)")
    # thinking-disabled: template emits <think>\n\n</think> in the PROMPT, so the output has no markers
    dis = rp.parse("Direct answer with no reasoning markers.")
    _ck(dis["reasoning"] is None, "thinking-disabled output (no markers) -> reasoning=None")

    # compose the two the way the oMLX server does: split reasoning, then parse tools from the remainder
    turn = ("Plan the call.\n</think>\n\nOn it.\n<tool_call>\n<function=add>\n<parameter=a>\n1\n"
            "</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>")
    split = rp.parse(turn)
    tail = parse_qwen3_coder_tool_calls(split["answer"])
    _ck(split["reasoning"] == "Plan the call." and tail is not None and tail[0] == "On it."
        and tail[1][0]["name"] == "add" and json.loads(tail[1][0]["arguments"]) == {"a": 1, "b": 2},
        f"reasoning+tool compose wrong: reasoning={split['reasoning']!r} tail={tail}")

    print(f"PARITY-CHECKS: {_N}")
    print(f"PASS — Qwen3-Coder/Nex-N2-Pro tool parser ({len(c4)}-call XML, typed/multiline) + pre-opened "
          f"<think> reasoning + dispatcher-exclusion (Nemotron delegation preserved): {_N} checks.")


if __name__ == "__main__":
    run()
