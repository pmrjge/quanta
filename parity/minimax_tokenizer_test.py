"""Gate: MiniMax-M2.7 tokenizer — model-free, offline, ~0 GB (reads only ``tokenizer.json`` +
``tokenizer_config.json`` + ``chat_template.jinja`` + ``config.json``; no model weights).

Checks the runtime :class:`quanta.minimax.tokenizer.MiniMaxTokenizer` (no torch/transformers on the
hot path) on:

* encode/decode round-trip on ascii + unicode strings (``decode(encode(s)) == s``);
* BOS behavior follows the checkpoint's ``add_bos_token`` (absent -> default ``False``): bare ``encode``
  prepends nothing, ``encode(add_bos=True)`` prepends ``bos_id`` exactly once;
* ``stop_ids`` equals the grounded generation eos id set (200020);
* chat rendering with ``add_generation_prompt=True`` contains the expected assistant turn markers.

When the offline HF ``tokenizers`` reference is installed it ADDITIONALLY asserts bit-exact ``encode``
parity (rule 5 keeps ``tokenizers`` off the runtime path); without it those extra checks are skipped but
the model-free gate above still runs in full.

    uv run --with jinja2 --with tokenizers python -m parity.minimax_tokenizer_test
"""

from __future__ import annotations

import os
import sys

from quanta.minimax.tokenizer import MiniMaxTokenizer

MINIMAX_DIR = os.environ.get("QUANTA_MINIMAX_DIR", "/Users/pmrj/models/MiniMax-M2.7")

# grounded ids (config.json / generation_config.json), confirmed from the checkpoint
BOS_ID = 200019
EOS_ID = 200020

CASES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def f(x):\n    return x ** 2  # squared\n",
    "café naïve — résumé, 数字 123456, emoji 🚀🔥 mixed",
    "   leading and  multiple   spaces\t\ttabs\n\n\nnewlines   ",
    "Numbers: 0 1 22 333 4444 55555 and 1,234,567.89",
    "你好，世界！こんにちは。カタカナ ABC mixed 漢字テスト",
    "<think>reason</think>answer<minimax:tool_call>x</minimax:tool_call>",
    "",
    "a",
]


def run() -> None:
    tj = os.path.join(MINIMAX_DIR, "tokenizer.json")
    if not os.path.isfile(tj):
        print(f"SKIP — no tokenizer.json at {MINIMAX_DIR} (run on the host with the MiniMax checkpoint)")
        return

    tok = MiniMaxTokenizer.from_pretrained(MINIMAX_DIR)
    ok = True

    # --- (a) ids: bos / eos / stop set ---------------------------------------
    ids_ok = tok.bos_id == BOS_ID and tok.eos_id == EOS_ID and tuple(tok.stop_ids) == (EOS_ID,)
    ok = ok and ids_ok
    print(f"  [{'OK' if ids_ok else 'FAIL'}] bos={tok.bos_id} eos={tok.eos_id} "
          f"stop_ids={tuple(tok.stop_ids)} vocab={tok.vocab_size}")

    # --- (b) encode/decode round-trip on ascii + unicode ---------------------
    n_rt = 0
    for s in CASES:
        rt = tok.decode(tok.encode(s))
        if rt == s:
            n_rt += 1
        else:
            ok = False
            print(f"  [FAIL] round-trip ({s[:30]!r}): got {rt[:40]!r}")
    print(f"  [{'OK' if n_rt == len(CASES) else 'FAIL'}] decode(encode(s)) round-trip: {n_rt}/{len(CASES)}")

    # --- (c) BOS behavior follows add_bos_token (default False) ---------------
    base = tok.encode("hello world")
    no_bos = base and base[0] != BOS_ID                  # default add_bos_token=False -> no leading bos
    with_bos = tok.encode("hello world", add_bos=True) == [BOS_ID] + base
    bos_ok = bool(no_bos) and with_bos
    ok = ok and bos_ok
    print(f"  [{'OK' if bos_ok else 'FAIL'}] add_bos: default prepends nothing, add_bos=True prepends "
          f"bos once")

    # --- (d) chat rendering: assistant turn markers + generation prompt -------
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    rendered = tok.render_chat(messages, add_generation_prompt=True)
    chat_ids = tok.encode_chat(messages, add_generation_prompt=True)
    markers_ok = (
        "]~b]ai" in rendered                 # assistant turn opener
        and "]~b]user" in rendered           # user turn marker
        and rendered.rstrip("\n").endswith("<think>")   # generation prompt tail
        and BOS_ID in chat_ids and EOS_ID in chat_ids   # bos marker + turn-end (eos) tokenized
    )
    no_gp = tok.render_chat(messages, add_generation_prompt=False)
    gp_adds_ai = (not no_gp.rstrip("\n").endswith("<think>")) and rendered != no_gp
    chat_ok = markers_ok and gp_adds_ai
    ok = ok and chat_ok
    print(f"  [{'OK' if chat_ok else 'FAIL'}] chat render: assistant marker present, gen-prompt adds "
          f"<think> opener ({len(chat_ids)} ids)")

    # --- (e) optional bit-exact parity vs HF tokenizers (offline reference) ---
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("  [SKIP] bit-exact vs HF `tokenizers` (run with: uv run --with tokenizers ...)")
    else:
        ref = Tokenizer.from_file(tj)
        n_enc = n_dec = 0
        for s in CASES:
            mine = tok.encode(s)
            gold = ref.encode(s, add_special_tokens=False).ids
            if mine == gold:
                n_enc += 1
            else:
                ok = False
                print(f"  [FAIL] encode mismatch ({s[:30]!r}): mine[:8]={mine[:8]} gold[:8]={gold[:8]}")
            for skip in (False, True):
                if tok.decode(gold, skip_special_tokens=skip) == ref.decode(
                        gold, skip_special_tokens=skip):
                    n_dec += 1
                else:
                    ok = False
                    print(f"  [FAIL] decode(skip={skip}) mismatch ({s[:30]!r})")
        print(f"  [{'OK' if n_enc == len(CASES) else 'FAIL'}] encode bit-exact vs ref: "
              f"{n_enc}/{len(CASES)}")
        print(f"  [{'OK' if n_dec == 2 * len(CASES) else 'FAIL'}] decode matches ref: "
              f"{n_dec}/{2 * len(CASES)} (both skip modes)")

    print("PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    run()
