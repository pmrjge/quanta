"""Gate: the pure-Python DeepSeek-V4 BPE tokenizer == the HF ``tokenizers`` reference, bit-exact.

Diffs :class:`quanta.dsv4.tokenizer.DeepSeekV4Tokenizer` (runtime, no torch/transformers) against
``tokenizers.Tokenizer.from_file(tokenizer.json)`` (offline oracle) across a wide battery: every
pre-tokenizer branch (digit runs of varied length, CJK/kana, emoji/symbols, punctuation, code,
whitespace/newlines, combining marks), real prose from the checkpoint READMEs, the four chat gold
prompts shipped in ``encoding/tests``, special-token passthrough, BOS handling, decode round-trips
(both skip modes), and the ``encode_chat`` path (reasoning_effort="max" default).

    uv run --with tokenizers python -m parity.dsv4_tokenizer_test
"""

from __future__ import annotations

import os

from tokenizers import Tokenizer

from quanta.dsv4 import encoding
from quanta.dsv4.tokenizer import DeepSeekV4Tokenizer

ART = "/Users/pmrj/models/DeepSeek-V4-Flash"


def _first_diff(a: list[int], b: list[int]) -> str:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return f"@{i}: mine={a[i]} ref={b[i]} (ctx mine={a[max(0,i-2):i+3]} ref={b[max(0,i-2):i+3]})"
    if len(a) != len(b):
        return f"len mine={len(a)} ref={len(b)} (tail mine={a[n:n+4]} ref={b[n:n+4]})"
    return "identical"


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _corpus() -> list[str]:
    cases = [
        "",
        " ",
        "hello world",
        " leading space",
        "trailing space ",
        "The quick brown fox jumps over 1234 lazy dogs.",
        "你好，世界！こんにちは。カタカナ ABC mixed 漢字テスト",
        "def f(x):\n    return x**2  # comment\n\n\ttab\tindent",
        "Mix123abc456 and 7890123 digits 1 12 123 1234 12345 999999999",
        "Émojis: 🤖🚀👨‍👩‍👧‍👦 and symbols €£¥ ©®™ — café naïve résumé",
        "tabs\tand   multiple    spaces\n\n\nand newlines\r\n\r\nCRLF",
        "URL https://example.com/path?q=1&x=2#frag and email a.b@c.io",
        "punct!!!??? ...---***   (nested [brackets] {curly} <angle>)",
        "combining: áèî and ZWJ‍joiner and RTL العربية",
        "MixedCaseCamelCaseSCREAMING and snake_case and kebab-case-words",
        "math: ∑∫∂√∞ ≤≥≠± αβγδ ΩΦΨ and superⁿ subₙ",
        # strings embedding special tokens (must map to ids directly):
        "<｜begin▁of▁sentence｜>system here<｜User｜>hi<｜Assistant｜><think>reason</think>ok<｜end▁of▁sentence｜>",
        "before<｜DSML｜>middle｜DSML｜after<｜place▁holder▁no▁0｜>tail",
    ]
    # real prose / markdown corpora from the checkpoint (English + structure + code fences)
    for p in (os.path.join(ART, "README.md"),
              os.path.join(ART, "encoding", "README.md")):
        if os.path.exists(p):
            cases.append(_read(p))
    # the four chat gold prompts (specials + reasoning-effort prefix + tool DSML)
    tdir = os.path.join(ART, "encoding", "tests")
    for i in (1, 2, 3, 4):
        gp = os.path.join(tdir, f"test_output_{i}.txt")
        if os.path.exists(gp):
            cases.append(_read(gp))
    return cases


def run() -> None:
    ref = Tokenizer.from_file(os.path.join(ART, "tokenizer.json"))
    tok = DeepSeekV4Tokenizer.from_pretrained(ART)
    ok = True

    # --- (a) vocab / id-space sanity -----------------------------------------
    good = tok.vocab_size == ref.get_vocab_size() == 129280 and tok.bos_id == 0 and tok.eos_id == 1
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] vocab_size={tok.vocab_size} bos={tok.bos_id} eos={tok.eos_id}")

    # --- (b) encode parity (bit-exact ids) -----------------------------------
    n_enc = 0
    for s in _corpus():
        mine = tok.encode(s)
        rids = ref.encode(s, add_special_tokens=False).ids
        if mine != rids:
            ok = False
            label = repr(s[:48]) + ("..." if len(s) > 48 else "")
            print(f"  [FAIL] encode {label}  {_first_diff(mine, rids)}")
        else:
            n_enc += 1
    print(f"  [{'OK' if n_enc == len(_corpus()) else 'FAIL'}] encode bit-exact on {n_enc}/{len(_corpus())} corpus cases")

    # --- (c) decode parity (both skip modes) ---------------------------------
    n_dec = 0
    corpus = _corpus()
    for s in corpus:
        rids = ref.encode(s, add_special_tokens=False).ids
        for skip in (False, True):
            md = tok.decode(rids, skip_special_tokens=skip)
            rd = ref.decode(rids, skip_special_tokens=skip)
            if md != rd:
                ok = False
                print(f"  [FAIL] decode(skip={skip}) {repr(s[:40])}  mine={md[:60]!r} ref={rd[:60]!r}")
            else:
                n_dec += 1
    print(f"  [{'OK' if n_dec == 2 * len(corpus) else 'FAIL'}] decode matches reference on {n_dec}/{2*len(corpus)} (both skip modes)")

    # --- (d) round-trip: decode(encode(s)) recovers s on clean text ----------
    clean = "The quick brown fox 1234. 你好世界 café — code: x=1; y=2."
    rt = tok.decode(tok.encode(clean))
    good = rt == clean
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] round-trip decode(encode) recovers clean text")

    # --- (e) BOS handling -----------------------------------------------------
    base = tok.encode("hello world")
    good = tok.encode("hello world", add_bos=True) == [tok.bos_id] + base
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] add_bos prepends bos id once")

    # --- (f) encode_chat (reasoning_effort=max default) parity ---------------
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2? Explain in 123 ways, with 你好 and code `x=1`."},
    ]
    prompt = encoding.encode_chat(messages)                       # max-effort default
    rids = ref.encode(prompt, add_special_tokens=False).ids
    mine = tok.encode_chat(messages)
    good = mine == rids and mine[0] == tok.bos_id and encoding.REASONING_EFFORT_MAX.split("\n")[0] in prompt
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] encode_chat == ref ({len(mine)} ids, bos-first, max-effort prefix present)  {_first_diff(mine, rids)}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
