"""Kimi-K2.6 tool-call parser — consumed by the oMLX-side patch (not by the engine).

The engine emits **raw** output, so a Kimi tool call reaches oMLX as literal text::

    <|tool_calls_section_begin|>
      <|tool_call_begin|>functions.{name}:{idx}<|tool_call_argument_begin|>{json}<|tool_call_end|> ...
    <|tool_calls_section_end|>

oMLX's ``clean_special_tokens`` strips only a fixed set (``<|im_end|>`` etc.) and leaves these markers
intact, so its ``parse_tool_calls`` sees them — but its built-in parser registry has no Kimi format.
:func:`quanta.omlx_patch` wraps ``omlx.api.tool_calling.parse_tool_calls`` with this parser so Kimi
tool calls are extracted on the oMLX side (the function name is carried by the id, ``functions.{name}:{idx}``,
which is kept as the OpenAI tool-call ``id`` so it round-trips natively through the chat template).

Pure (no torch/mlx/omlx) — gated in ``parity/kimi_tools_test.py``.
"""

from __future__ import annotations

import re

TC_SECTION_BEGIN = "<|tool_calls_section_begin|>"
TC_SECTION_END = "<|tool_calls_section_end|>"
TC_BEGIN = "<|tool_call_begin|>"
TC_ARG_BEGIN = "<|tool_call_argument_begin|>"
TC_END = "<|tool_call_end|>"

_CALL_RE = re.compile(
    re.escape(TC_BEGIN) + r"(?P<id>.*?)" + re.escape(TC_ARG_BEGIN) + r"(?P<args>.*?)" + re.escape(TC_END),
    re.DOTALL,
)
_SECTION_RE = re.compile(re.escape(TC_SECTION_BEGIN) + r".*?" + re.escape(TC_SECTION_END), re.DOTALL)


def _name_from_id(tid: str) -> str:
    """``functions.get_weather:0`` -> ``get_weather`` (robust to dotted names; falls back to the id)."""
    s = tid[len("functions."):] if tid.startswith("functions.") else tid
    return s.rsplit(":", 1)[0] if ":" in s else s


def parse_kimi_tool_calls(text: str) -> tuple[str, list[dict]] | None:
    """Parse Kimi tool-call markup from ``text``.

    Returns ``(cleaned_text, calls)`` where ``calls`` is a list of ``{"id", "name", "arguments"}``
    dicts (``arguments`` is the raw JSON string the model emitted), or ``None`` when no Kimi tool
    section is present (so the caller can delegate to the original parser). ``cleaned_text`` is ``text``
    with the tool section removed (an unterminated/truncated section is cut from its begin marker)."""
    if TC_SECTION_BEGIN not in text:
        return None
    calls: list[dict] = []
    for m in _CALL_RE.finditer(text):
        tid = m.group("id").strip()
        calls.append({"id": tid, "name": _name_from_id(tid), "arguments": m.group("args").strip()})
    cleaned = _SECTION_RE.sub("", text)
    if TC_SECTION_BEGIN in cleaned:  # unterminated section (truncated output) — drop the tail
        cleaned = cleaned[: cleaned.index(TC_SECTION_BEGIN)]
    return cleaned.strip(), calls
