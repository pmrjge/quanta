"""Gate: GLM-5.1 tokenizer is bit-exact vs the HF ``tokenizers`` reference — offline, model-free, ~0 GB.

Builds :class:`quanta.glm.tokenizer.GLMTokenizer` and the reference ``tokenizers.Tokenizer`` from the
SAME ``tokenizer.json`` and checks ``encode`` matches token-for-token across ascii / unicode / code /
whitespace / digits / embedded special-token strings (comparing against ``add_special_tokens=False`` so
the post-processor's auto prefix is excluded — our ``encode`` adds none), plus byte-level decode
round-trips. Reads only the 20 MB ``tokenizer.json`` (no model weights); the ``tokenizers`` lib is an
offline reference (rule 5 keeps it off the runtime path). SKIPs cleanly if the checkpoint is absent.

    uv run --with tokenizers python -m parity.glm_tokenizer_test
"""

from __future__ import annotations

import os
import sys

from quanta.glm.tokenizer import GLMTokenizer

GLM_DIR = os.environ.get("QUANTA_GLM_DIR", "/Users/pmrj/models/GLM-5.1")

CASES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def f(x):\n    return x ** 2  # squared\n",
    "café naïve — résumé, 数字 123456, emoji 🚀🔥 mixed",
    "   leading and  multiple   spaces\t\ttabs\n\n\nnewlines   ",
    "Numbers: 0 1 22 333 4444 55555 and 1,234,567.89",
    "<|system|>You are helpful.<|user|>Hi there<|assistant|>",
    "[gMASK]<sop>prefix then <|observation|> tool output",
    "",
    "a",
]


def run() -> None:
    tj = os.path.join(GLM_DIR, "tokenizer.json")
    if not os.path.isfile(tj):
        print(f"SKIP — no tokenizer.json at {GLM_DIR} (run on the host with the GLM checkpoint)")
        return
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("SKIP — `tokenizers` not available (run with: uv run --with tokenizers ...)")
        return

    tok = GLMTokenizer.from_pretrained(GLM_DIR)
    ref = Tokenizer.from_file(tj)
    ok = True

    # eos / stop set
    stop_ok = tok.eos_id == 154820 and set(tok.stop_ids) == {154820, 154827, 154829}
    ok = ok and stop_ok
    print(f"  [{'OK' if stop_ok else 'FAIL'}] eos={tok.eos_id} stop_ids={sorted(tok.stop_ids)} vocab={tok.vocab_size}")

    # encode bit-exactness
    n_enc_ok = 0
    for s in CASES:
        mine = tok.encode(s)
        gold = ref.encode(s, add_special_tokens=False).ids
        match = mine == gold
        n_enc_ok += match
        if not match:
            ok = False
            print(f"  [FAIL] encode mismatch ({s[:30]!r}): mine[:8]={mine[:8]} gold[:8]={gold[:8]} "
                  f"len {len(mine)} vs {len(gold)}")
    print(f"  [{'OK' if n_enc_ok == len(CASES) else 'FAIL'}] encode bit-exact: {n_enc_ok}/{len(CASES)} cases")

    # decode round-trip (byte-level is lossless for valid utf-8) + matches reference decode
    n_dec_ok = 0
    for s in CASES:
        ids = tok.encode(s)
        rt = tok.decode(ids)
        ref_dec = ref.decode(ids, skip_special_tokens=False)
        if rt == s and rt == ref_dec:
            n_dec_ok += 1
        else:
            ok = False
            print(f"  [FAIL] decode ({s[:30]!r}): rt=={s==rt} ref_match={rt==ref_dec}")
    print(f"  [{'OK' if n_dec_ok == len(CASES) else 'FAIL'}] decode round-trip + ref-match: {n_dec_ok}/{len(CASES)}")

    print("PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    run()
