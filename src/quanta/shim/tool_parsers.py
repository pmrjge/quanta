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

Parsers contract
----------------
Two ``@runtime_checkable`` Protocols formalize the surface a new model's parsers must implement:

* :class:`ReasoningParser` — splits a raw assistant turn into a reasoning span (the ``<think>…</think>``
  body, if any) and the visible answer.
* :class:`ToolParser` — extracts OpenAI-style tool-call dicts from a raw assistant turn and shapes a
  tool-response message body back into the model's expected format.

These exist because the engine emits RAW output (special-token markers preserved) — the integrator's
parser is what bridges the model's markup to the OpenAI server surface. Each Qwen3.5 / Kimi parser
below is exposed as a class conforming to one of these Protocols so a new model integrator can read
exactly one docstring (the Protocol) and have a checklist of what to implement. Stateless by
construction (no per-request state held on the parser), so a single instance is safe to reuse across
requests. Nemotron has no quanta-side parser (oMLX's registry covers ``<function=…>`` XML natively),
and is intentionally NOT made to conform here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from quanta.shim.kimi_tools import parse_kimi_tool_calls

ParseResult = tuple[str, list[dict]]


# --- parsers contract (Protocols) ------------------------------------------------------------------
@runtime_checkable
class ReasoningParser(Protocol):
    """A reasoning-span parser. Implementations split a raw assistant turn into ``reasoning`` (the
    contents of the ``<think>…</think>`` block, or any model-specific equivalent) and ``answer`` (the
    visible text). Stateless — no per-request state on the parser instance.

    The contract is a single method:

    * :meth:`parse` takes a raw assistant turn (with the model's markers preserved — e.g. literal
      ``</think>`` for Kimi/Qwen, ``[REASONING]`` for some checkpoints) and returns a dict with at
      minimum ``{"reasoning": str | None, "answer": str}``. ``reasoning`` is ``None`` (NOT empty
      string) when the turn carried no reasoning span, so a downstream consumer can disambiguate
      "no reasoning was emitted" from "reasoning was emitted and was empty".

    The engine itself never invokes a ReasoningParser (it streams raw text); this contract is for
    consumers shaping responses on the oMLX server side, mirroring :class:`ToolParser`."""

    def parse(self, text: str) -> dict: ...


@runtime_checkable
class ToolParser(Protocol):
    """A tool-call parser. Implementations extract OpenAI-style tool-call dicts from a raw assistant
    turn and shape a tool-response message body back into the model's expected format. Stateless.

    Two methods:

    * :meth:`parse_tool_calls` takes the raw assistant text and returns a list of OpenAI-style
      ``{"id", "name", "arguments"}`` dicts (``arguments`` is a JSON string). Returns an empty list
      when no tool call is present (NOT ``None`` — emptiness is a successful no-op, not a parse
      failure; mismatched markup raises rather than returning ``None`` so it is never silently
      swallowed, rule 6).

    * :meth:`format_tool_response` takes a tool-call id + a string content and returns the body the
      next assistant turn's prompt needs (model-specific: e.g. a ``<tool_response>…</tool_response>``
      block for Qwen, or a ``<|tool_response|>`` section for Kimi). The id must round-trip through
      :meth:`parse_tool_calls` -> :meth:`format_tool_response`.

    The free function :func:`parse_quanta_tool_calls` is the older legacy dispatcher (returns the
    Kimi/MiniMax/GLM/Qwen ``(cleaned, calls)`` tuple or ``None``) and is kept for the oMLX patch's
    delegation path — new code should use a ToolParser class instead."""

    def parse_tool_calls(self, text: str) -> list[dict]: ...

    def format_tool_response(self, tool_call_id: str, content: str) -> str: ...

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


# --- per-model parser classes (conform to ReasoningParser / ToolParser) ----------------------------
#
# These wrap the legacy free-function parsers in classes that explicitly conform to the Protocols
# above so the contract is visible at use sites and a new model integrator can copy the pattern.
# They hold no state and add no behavior — they are an additive surface (rule 6: no silent change).

# Qwen3.5 reasoning block: ``<think>…</think>`` (the bare ``<think>`` opener is the template's default
# when ``enable_thinking=True``). The shape mirrors the oMLX server's ``extract_thinking`` exactly so a
# round-trip through this parser is the same split the server would do — but exposed via the contract.
_THINK_FULL_RE = re.compile(r"<think>(?P<body>.*?)</think>", re.DOTALL)
_THINK_TAIL_RE = re.compile(r"^(?P<body>.*?)</think>", re.DOTALL)


class Qwen3ReasoningParser:
    """Reasoning-span parser for Qwen3.5 — splits ``<think>…</think>`` from the visible answer.

    Conforms to :class:`ReasoningParser`. Handles three cases the chat template can produce:

    * ``"… <think>R</think>A …"`` — explicit reasoning block (the standard case);
    * ``"R</think>A"`` — the template's *bare* ``<think>`` opener was followed by reasoning, then the
      model closed it; this is the on-by-default Qwen3.5 prefix shape (and the Kimi/oMLX server uses
      the same fallback);
    * ``"<think>R"`` — model opened reasoning but never closed it (truncated output); reasoning is
      whatever was emitted, the answer is empty.

    When no ``<think>``/``</think>`` is present, ``reasoning=None`` (NOT ``""``) so a consumer can
    distinguish "no reasoning was emitted" from "reasoning was emitted and was empty".
    """

    def parse(self, text: str) -> dict:
        if not text:
            return {"reasoning": None, "answer": ""}
        # explicit <think>…</think> blocks: concatenate all of them, strip them from the answer.
        parts: list[str] = []
        remaining = text
        while True:
            m = _THINK_FULL_RE.search(remaining)
            if not m:
                break
            parts.append(m.group("body"))
            remaining = remaining[: m.start()] + remaining[m.end() :]
        if parts:
            return {"reasoning": "\n".join(parts).strip(), "answer": remaining.strip()}
        # bare-opener shape: "R</think>A" — closing tag with no opener.
        if "</think>" in text and "<think>" not in text:
            m = _THINK_TAIL_RE.match(text)
            if m:
                return {"reasoning": m.group("body").strip(), "answer": text[m.end():].strip()}
        # truncated: "<think>R" — opener but no closer.
        if "<think>" in text and "</think>" not in text:
            idx = text.index("<think>")
            return {"reasoning": text[idx + len("<think>"):].strip(),
                    "answer": text[:idx].strip()}
        return {"reasoning": None, "answer": text}


class Qwen3ToolParser:
    """Tool-call parser for Qwen3.5 (Hermes ``<tool_call>{json}</tool_call>``).

    Conforms to :class:`ToolParser`. Delegates the parse to :func:`parse_qwen_tool_calls` (strict:
    requires a JSON body with a ``name`` key, so it never swallows GLM/Nemotron markup — gated in
    :mod:`parity.tool_parsers_test`). The response formatter renders a Qwen3.5 ``<tool_response>``
    block: the next assistant prompt's tool-result rendering uses this exact shape via the chat
    template's ``role == "tool"`` branch.
    """

    def parse_tool_calls(self, text: str) -> list[dict]:
        result = parse_qwen_tool_calls(text)
        return list(result[1]) if result is not None else []

    def format_tool_response(self, tool_call_id: str, content: str) -> str:
        # ``tool_call_id`` is the round-trippable id from :meth:`parse_tool_calls` (e.g. "get_weather:0");
        # Qwen's chat template embeds the role-tool body verbatim, so the wrapping markup is here.
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError(f"tool_call_id must be a non-empty str (got {tool_call_id!r})")
        return f"<tool_response>\n{content}\n</tool_response>"


class KimiToolParser:
    """Tool-call parser for Kimi-K2.6 (``<|tool_calls_section_begin|>…<|tool_calls_section_end|>``).

    Conforms to :class:`ToolParser`. Delegates the parse to :func:`parse_kimi_tool_calls` (the
    existing pure-Python parser in :mod:`quanta.shim.kimi_tools`, gated in
    :mod:`parity.kimi_tools_test`). The response formatter renders the tool result inside Kimi's
    ``<|tool_response_begin|>…<|tool_response_end|>`` special-token wrapper — the inverse of the
    section-begin/end pair the model emits. Stateless, additive — does NOT change parser behavior.
    """

    def parse_tool_calls(self, text: str) -> list[dict]:
        result = parse_kimi_tool_calls(text)
        return list(result[1]) if result is not None else []

    def format_tool_response(self, tool_call_id: str, content: str) -> str:
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError(f"tool_call_id must be a non-empty str (got {tool_call_id!r})")
        # Kimi's chat template wraps a tool result in <|tool_response_begin|>{id}<|tool_response_argument_begin|>
        # {content}<|tool_response_end|> in the role == "tool" branch (see ~/models/Kimi-K2.6/chat_template.jinja);
        # exposing the inverse shape here keeps the round-trip (parse -> format -> next-prompt) explicit.
        return ("<|tool_response_begin|>" + tool_call_id
                + "<|tool_response_argument_begin|>" + content
                + "<|tool_response_end|>")


__all__ = [
    "KimiToolParser",
    "MM_SECTION_BEGIN",
    "MM_SECTION_END",
    "ParseResult",
    "Qwen3ReasoningParser",
    "Qwen3ToolParser",
    "ReasoningParser",
    "ToolParser",
    "parse_glm_tool_calls",
    "parse_kimi_tool_calls",
    "parse_minimax_tool_calls",
    "parse_quanta_tool_calls",
    "parse_qwen_tool_calls",
]
