"""Gate: DeepSeek-V4 chat encoding == the checkpoint's gold fixtures, + reasoning_effort=max default.

Validates the vendored :mod:`quanta.dsv4.encoding` against the 4 gold input/output fixtures shipped
in the checkpoint's ``encoding/tests`` (byte-exact ``encode_messages`` + a parse roundtrip), then
checks the quanta policy wrapper :func:`encode_chat` prepends the maximum-reasoning-effort prefix by
default (and drops it when overridden).

    uv run python -m parity.dsv4_encoding_test
"""

from __future__ import annotations

import json
from pathlib import Path

from quanta.dsv4.encoding import (
    REASONING_EFFORT_MAX,
    encode_chat,
    encode_messages,
    parse_message_from_completion_text,
)

TESTS = Path("/Users/pmrj/models/DeepSeek-V4-Flash/encoding/tests")


def run() -> None:
    ok = True

    def check(tag, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'OK' if cond else 'FAIL'}] {tag}")

    # --- gold fixtures ---------------------------------------------------------
    print("=== gold encode fixtures ===")
    # case 1: thinking + tools (tools attached to system message)
    td = json.loads((TESTS / "test_input_1.json").read_text())
    msgs1 = td["messages"]
    msgs1[0]["tools"] = td["tools"]
    gold1 = (TESTS / "test_output_1.txt").read_text()
    check("case 1 thinking+tools", encode_messages(msgs1, thinking_mode="thinking") == gold1)

    # cases 2,3 thinking ; case 4 chat
    for n, mode in ((2, "thinking"), (3, "thinking"), (4, "chat")):
        msgs = json.loads((TESTS / f"test_input_{n}.json").read_text())
        gold = (TESTS / f"test_output_{n}.txt").read_text()
        check(f"case {n} {mode}", encode_messages(msgs, thinking_mode=mode) == gold)

    # --- parse roundtrip (case 2 final assistant turn) -------------------------
    print("=== parse roundtrip ===")
    prompt2 = encode_messages(json.loads((TESTS / "test_input_2.json").read_text()), thinking_mode="thinking")
    marker = "<｜Assistant｜><think>"
    parsed = parse_message_from_completion_text(prompt2[prompt2.rfind(marker) + len(marker):],
                                                thinking_mode="thinking")
    check("parsed final turn fields", parsed["role"] == "assistant"
          and parsed["content"] == "The capital of France is Paris." and parsed["tool_calls"] == [])

    # --- reasoning_effort=max default (quanta policy) --------------------------
    print("=== encode_chat reasoning_effort=max default ===")
    conv = [{"role": "user", "content": "What is 2+2?"}]
    default = encode_chat(conv)                                    # thinking + max by default
    none = encode_chat(conv, reasoning_effort=None)
    explicit = encode_messages(conv, thinking_mode="thinking", reasoning_effort="max")
    check("default prepends REASONING_EFFORT_MAX", default.startswith(
        "<｜begin▁of▁sentence｜>" + REASONING_EFFORT_MAX))
    check("default ends in <think> (thinking mode)", default.endswith("<｜Assistant｜><think>"))
    check("reasoning_effort=None drops the prefix", REASONING_EFFORT_MAX not in none)
    check("default == explicit max call", default == explicit)

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
