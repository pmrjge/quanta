"""Tool-call parsers for the quanta models that oMLX's built-in registry does not cover, plus a
unified dispatcher the oMLX patch installs.

The quanta engine emits **raw** output (special/control tokens render as their literal marker
strings), so a model's tool-call markup reaches oMLX as plain text that ``clean_special_tokens``
leaves intact. oMLX's ``parse_tool_calls`` registry natively handles xml/json/gemma/glm/qwen, but:

* **Kimi-K2.6** uses ``<|tool_calls_section_begin|>`` special tokens — no registry entry
  (:mod:`quanta.shim.kimi_tools`).
* **MiniMax-M2.7** wraps Anthropic-style ``<invoke>`` calls in a model-specific
  ``<minimax:tool_call>…</minimax:tool_call>`` section — no registry entry.
* **GLM-5.1** (``<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>``) and
  **Qwen3.5** (``<tool_call>{json}</tool_call>``) *are* oMLX-native, but we ship strict parsers for
  them too so serving never silently depends on oMLX's exact registry regexes matching these
  checkpoints (rule 6).

:func:`parse_quanta_tool_calls` tries each parser in order and returns the first match, else ``None``
(so the oMLX patch delegates unmatched text — e.g. Nemotron's ``<function=…>`` XML — to the original
registry parser). Every parser is **strict** about its inner shape so it never swallows another
model's markup: the GLM parser requires ``<arg_key>``; the Qwen parser requires a JSON body with a
``name``. This keeps the Nemotron/registry delegation intact (gated in
``parity/nemotron_omlx_contract_test.py`` and ``parity/tool_parsers_test.py``).

Pure (no torch/mlx/omlx). Each returns ``(cleaned_text, calls)`` where ``calls`` is a list of
``{"id", "name", "arguments"}`` (``arguments`` a JSON string), matching :mod:`quanta.shim.kimi_tools`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from quanta.shim.kimi_tools import parse_kimi_tool_calls

ParseResult = tuple[str, list[dict]]

# --- MiniMax-M2.7: <minimax:tool_call> <invoke name="x"><parameter name="k">v</parameter></invoke> ---
MM_SECTION_BEGIN = "<minimax:tool_call>"
MM_SECTION_END = "</minimax:tool_call>"
_MM_SECTION_RE = re.compile(re.escape(MM_SECTION_BEGIN) + r"(?P<body>.*?)" + re.escape(MM_SECTION_END),
                            re.DOTALL)
_MM_INVOKE_RE = re.compile(r"<invoke\s+name=\"(?P<name>[^\"]+)\">(?P<args>.*?)</invoke>", re.DOTALL)
_MM_PARAM_RE = re.compile(r"<parameter\s+name=\"(?P<key>[^\"]+)\">(?P<val>.*?)</parameter>", re.DOTALL)


def _typed(raw: str):
    """Best-effort recover the original typed value: the template renders non-string args as JSON and
    string args raw, so ``json.loads`` recovers numbers/bools/objects and falls back to the raw text."""
    s = raw.strip()
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return raw


def parse_minimax_tool_calls(text: str) -> ParseResult | None:
    """Parse MiniMax ``<minimax:tool_call>`` sections (XML ``<invoke>``/``<parameter>`` inner form)."""
    if MM_SECTION_BEGIN not in text:
        return None
    calls: list[dict] = []
    for sec in _MM_SECTION_RE.finditer(text):
        for inv in _MM_INVOKE_RE.finditer(sec.group("body")):
            name = inv.group("name").strip()
            args = {p.group("key").strip(): _typed(p.group("val")) for p in _MM_PARAM_RE.finditer(inv.group("args"))}
            calls.append({"id": f"{name}:{len(calls)}", "name": name,
                          "arguments": json.dumps(args, ensure_ascii=False)})
    cleaned = _MM_SECTION_RE.sub("", text)
    if MM_SECTION_BEGIN in cleaned:  # unterminated (truncated) section → drop the tail
        cleaned = cleaned[: cleaned.index(MM_SECTION_BEGIN)]
    return cleaned.strip(), calls


# --- GLM-5.1: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call> --------------
_GLM_CALL_RE = re.compile(r"<tool_call>(?P<body>.*?)</tool_call>", re.DOTALL)
_GLM_PAIR_RE = re.compile(r"<arg_key>(?P<key>.*?)</arg_key>\s*<arg_value>(?P<val>.*?)</arg_value>", re.DOTALL)


def parse_glm_tool_calls(text: str) -> ParseResult | None:
    """Parse GLM ``<tool_call>`` blocks. Strict: only fires when a block carries ``<arg_key>`` (so it
    never swallows Qwen's JSON blocks or Nemotron's ``<function=…>`` form)."""
    if "<tool_call>" not in text or "<arg_key>" not in text:
        return None
    calls: list[dict] = []
    for m in _GLM_CALL_RE.finditer(text):
        body = m.group("body")
        pairs = list(_GLM_PAIR_RE.finditer(body))
        if not pairs:
            continue  # a <tool_call> without arg pairs is not GLM markup — skip it
        name = body[: pairs[0].start()].strip()
        args = {p.group("key").strip(): _typed(p.group("val")) for p in pairs}
        calls.append({"id": f"{name}:{len(calls)}", "name": name,
                      "arguments": json.dumps(args, ensure_ascii=False)})
    if not calls:
        return None
    cleaned = _GLM_CALL_RE.sub("", text)
    return cleaned.strip(), calls


# --- Qwen3.5: <tool_call>{"name": ..., "arguments": {...}}</tool_call> (Hermes) ---------------------
_QWEN_CALL_RE = re.compile(r"<tool_call>(?P<body>.*?)</tool_call>", re.DOTALL)


def parse_qwen_tool_calls(text: str) -> ParseResult | None:
    """Parse Qwen/Hermes ``<tool_call>{json}</tool_call>`` blocks. Strict: the body must be a JSON
    object with a ``name`` key (so Nemotron's ``<function=…>`` XML and GLM's ``<arg_key>`` form fall
    through to ``None``)."""
    if "<tool_call>" not in text:
        return None
    calls: list[dict] = []
    matched = False
    for m in _QWEN_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group("body").strip())
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        matched = True
        args = obj.get("arguments", obj.get("parameters", {}))
        args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        calls.append({"id": f"{obj['name']}:{len(calls)}", "name": str(obj["name"]), "arguments": args_str})
    if not matched:
        return None
    cleaned = _QWEN_CALL_RE.sub("", text)
    return cleaned.strip(), calls


# --- unified dispatcher ----------------------------------------------------------------------------
_PARSERS: tuple[Callable[[str], ParseResult | None], ...] = (
    parse_kimi_tool_calls,        # <|tool_calls_section_begin|> (special tokens)
    parse_minimax_tool_calls,     # <minimax:tool_call> wrapper
    parse_glm_tool_calls,         # <tool_call> + <arg_key>
    parse_qwen_tool_calls,        # <tool_call> + JSON body
)


def parse_quanta_tool_calls(text: str) -> ParseResult | None:
    """Try each quanta tool-call parser in order; return the first ``(cleaned, calls)`` match, else
    ``None`` so the caller delegates to oMLX's original registry parser (xml/json/gemma/glm/qwen)."""
    for parser in _PARSERS:
        result = parser(text)
        if result is not None:
            return result
    return None
