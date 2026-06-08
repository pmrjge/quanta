"""Model-free gate: the oMLX shim's reasoning-parser conformance after #150 landed the formal Protocol.

#150 put the formal ``ReasoningParser`` Protocol (``parse(text) -> {"reasoning", "answer"}``) and a
concrete ``Qwen3ReasoningParser`` in :mod:`quanta.shim.tool_parsers`. Before this reconciliation the
shim (:mod:`quanta.shim.omlx`) still carried a stale pre-#150 stub whose contract method was
``extract`` (returning a tuple), and ``_conformant_reasoning_parsers`` checked ``hasattr(p, "extract")``
— a latent mismatch (harmless only because the registered tuple is empty). This gate proves the
reconciliation:

  A. **the Protocol landed** — it is importable + ``@runtime_checkable``, and the concrete
     ``Qwen3ReasoningParser`` satisfies it via ``parse`` (the new dict surface), not the old ``extract``.
  B. **the engine registry is empty + validates** — DSV4 / Nemotron use stock markup, so
     ``_conformant_reasoning_parsers()`` returns ``()`` without raising, and the extracted conformance
     helper accepts a real parser.
  C. **a malformed parser fails loud** — a parser missing ``parse`` (e.g. the OLD ``extract``-only
     surface) raises ``OmlxShimError`` naming ``.parse`` (rule 6 — never silently use a bad parser).

    uv run python -m parity.omlx_reasoning_parser_test
"""

from __future__ import annotations

from quanta.shim.omlx import (
    OmlxShimError,
    _assert_reasoning_conformant,
    _conformant_reasoning_parsers,
)
from quanta.shim.tool_parsers import Qwen3ReasoningParser, ReasoningParser


def test_protocol_landed_and_concrete_conforms() -> None:
    p = Qwen3ReasoningParser()
    assert isinstance(p, ReasoningParser), \
        "Qwen3ReasoningParser must satisfy the @runtime_checkable ReasoningParser Protocol (via .parse)"
    out = p.parse("before<think>R</think>after")
    assert out == {"reasoning": "R", "answer": "beforeafter"}, f"unexpected parse: {out}"
    none = p.parse("no markup here")
    assert none["reasoning"] is None and none["answer"] == "no markup here", \
        "no reasoning span -> reasoning=None (NOT empty string)"
    print("A formal ReasoningParser Protocol landed; Qwen3ReasoningParser conforms via .parse  ok")


def test_engine_registry_empty_and_conformant() -> None:
    parsers = _conformant_reasoning_parsers()
    assert parsers == (), f"DSV4/Nemotron use stock markup -> expected no registered parsers, got {parsers!r}"
    _assert_reasoning_conformant((Qwen3ReasoningParser(),))  # a real parser passes the contract check
    print("B engine reasoning-parser registry empty (stock markup) + conformance accepts .parse  ok")


def test_nonconformant_fails_loud() -> None:
    class _Bad:  # the OLD pre-#150 surface (extract, no parse) — must now be rejected.
        def extract(self, text: str): ...

    raised = False
    try:
        _assert_reasoning_conformant((_Bad(),))
    except OmlxShimError as e:
        raised = "parse" in str(e)
    assert raised, "a parser without .parse must raise OmlxShimError naming .parse"
    print("C non-conformant parser (extract-only, no .parse) fails loud (names .parse)  ok")


def run() -> None:
    test_protocol_landed_and_concrete_conforms()
    test_engine_registry_empty_and_conformant()
    test_nonconformant_fails_loud()
    print("PASS — oMLX reasoning-parser contract reconciled with #150's formal Protocol "
          "(parse-based, concrete conforms, registry empty+validated, malformed fails loud)")


if __name__ == "__main__":
    run()
