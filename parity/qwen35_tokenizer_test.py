"""Gate: Qwen3.5 tokenizer is bit-exact vs the HF ``tokenizers`` reference — offline, model-free, ~0 GB.

Builds :class:`quanta.qwen35.tokenizer.Qwen35Tokenizer` and the reference ``tokenizers.Tokenizer`` from
the SAME ``tokenizer.json`` and checks ``encode`` matches token-for-token across ascii / unicode (incl.
NFC-relevant combining marks) / code / whitespace / digits / embedded special-token strings (comparing
against ``add_special_tokens=False`` so the post-processor adds nothing — and confirming no BOS is
prepended), plus byte-level decode round-trips. It then renders a chat with reasoning on/off via the
checkpoint ``chat_template.jinja`` (jinja2) and asserts the ``<|im_start|>assistant`` opener and the
``<think>`` reasoning markers, and that ``stop_ids`` equals the grounded ``{<|im_end|>, <|endoftext|>}``.

Reads only the ~13 MB ``tokenizer.json`` + small config/jinja files (NO model weights); ``tokenizers``
is an offline reference (rule 5 keeps it off the runtime path). SKIPs cleanly if the checkpoint is
absent. Runnable as:

    uv run --with tokenizers --with jinja2 python -m parity.qwen35_tokenizer_test
"""

from __future__ import annotations

import os
import sys

from quanta.qwen35.tokenizer import Qwen35Tokenizer

QWEN_DIR = os.environ.get("QUANTA_QWEN35_DIR", "/Users/pmrj/models/Qwen3.5-397B-A17B")

# Grounded ids (read from tokenizer_config.json / generation_config.json): eos = <|im_end|>=248046;
# generation stop set = {<|im_end|>, <|endoftext|>} = {248046, 248044}; pad = <|endoftext|>=248044.
IM_END_ID = 248046
ENDOFTEXT_ID = 248044
EXPECTED_STOP = {IM_END_ID, ENDOFTEXT_ID}

CASES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def f(x):\n    return x ** 2  # squared\n",
    "café naïve — résumé, 数字 123456, emoji 🚀🔥 mixed",
    "café naïve",  # NFD combining marks -> must NFC-normalize to match the reference
    "   leading and  multiple   spaces\t\ttabs\n\n\nnewlines   ",
    "Numbers: 0 1 22 333 4444 55555 and 1,234,567.89",
    "<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n",
    "think tag: <think>\nreason\n</think>\n\nanswer",
    "tool: <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
    "",
    "a",
]

# Round-trippable subset for decode(encode(s)) == s (excludes the explicit-NFD case, which the
# tokenizer normalizes — so it round-trips to the NFC form, not the original bytes).
ROUNDTRIP_CASES = [s for s in CASES if s != "café naïve"]


def run() -> None:
    tj = os.path.join(QWEN_DIR, "tokenizer.json")
    if not os.path.isfile(tj):
        print(f"SKIP — no tokenizer.json at {QWEN_DIR} (run on the host with the Qwen3.5 checkpoint)")
        return
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("SKIP — `tokenizers` not available (run with: uv run --with tokenizers ...)")
        return

    tok = Qwen35Tokenizer.from_pretrained(QWEN_DIR)
    ref = Tokenizer.from_file(tj)
    ok = True

    # eos / stop set (grounded) + no BOS configured
    stop_ok = (tok.eos_id == IM_END_ID and set(tok.stop_ids) == EXPECTED_STOP
               and isinstance(tok.stop_ids, tuple) and tok.bos_id is None)
    ok = ok and stop_ok
    print(f"  [{'OK' if stop_ok else 'FAIL'}] eos={tok.eos_id} stop_ids={sorted(tok.stop_ids)} "
          f"pad={tok.pad_id} bos={tok.bos_id} vocab={tok.vocab_size}")

    # encode bit-exactness vs reference (add_special_tokens=False — our encode adds no prefix/BOS)
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

    # explicitly assert NO BOS is prepended (first id is real content, never a sentinel)
    no_bos = tok.encode("Hello, world!") == ref.encode("Hello, world!", add_special_tokens=False).ids
    no_bos = no_bos and tok.encode("Hello", add_bos=True) == tok.encode("Hello")
    ok = ok and no_bos
    print(f"  [{'OK' if no_bos else 'FAIL'}] no BOS prepended (add_bos is a no-op)")

    # decode round-trip (byte-level is lossless for valid utf-8) + matches reference decode
    n_dec_ok = 0
    for s in ROUNDTRIP_CASES:
        ids = tok.encode(s)
        rt = tok.decode(ids)
        ref_dec = ref.decode(ids, skip_special_tokens=False)
        if rt == s and rt == ref_dec:
            n_dec_ok += 1
        else:
            ok = False
            print(f"  [FAIL] decode ({s[:30]!r}): rt=={s == rt} ref_match={rt == ref_dec}")
    print(f"  [{'OK' if n_dec_ok == len(ROUNDTRIP_CASES) else 'FAIL'}] decode round-trip + ref-match: "
          f"{n_dec_ok}/{len(ROUNDTRIP_CASES)}")

    # chat template: reasoning ON (default) -> assistant opener + a bare <think> opener
    try:
        chat = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        ids_on = tok.encode_chat(chat, add_generation_prompt=True, enable_thinking=True)
        text_on = tok.decode(ids_on)
        think_open_seq = tok.encode("<|im_start|>assistant\n<think>\n")
        on_ok = ("<|im_start|>assistant" in text_on and "<think>" in text_on
                 and "</think>" not in text_on  # reasoning-on emits an *open* think block only
                 and _ends_with(ids_on, think_open_seq))
        ok = ok and on_ok
        print(f"  [{'OK' if on_ok else 'FAIL'}] chat add_generation_prompt + thinking-on "
              f"(assistant + open <think>)")

        # reasoning OFF -> the empty <think>\n\n</think>\n\n block is present
        ids_off = tok.encode_chat(chat, add_generation_prompt=True, enable_thinking=False)
        text_off = tok.decode(ids_off)
        empty_block_seq = tok.encode("<think>\n\n</think>\n\n")
        off_ok = ("<|im_start|>assistant" in text_off
                  and "<think>\n\n</think>" in text_off
                  and _contains(ids_off, empty_block_seq))
        ok = ok and off_ok
        print(f"  [{'OK' if off_ok else 'FAIL'}] chat thinking-off (empty </think> block)")
    except RuntimeError as e:
        ok = False
        print(f"  [FAIL] encode_chat: {e}")
    except ImportError:
        print("  [SKIP] encode_chat — `jinja2` not available (run with: --with jinja2)")

    print("PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


def _ends_with(seq: list[int], tail: list[int]) -> bool:
    return len(tail) > 0 and seq[-len(tail):] == tail


def _contains(seq: list[int], sub: list[int]) -> bool:
    if not sub:
        return True
    return any(seq[i:i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))


if __name__ == "__main__":
    run()
