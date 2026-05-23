"""Validate KimiTokenizer against the upstream tiktoken/specials/chat-template.

Confirms: 163,584 base ids + named specials at the right offsets; the eval path is
byte-identical (control-token strings encode as ordinary text under the default
allow_special=False, and round-trip); chat mode maps control tokens to special ids;
the generation stop set is {[EOS], <|im_end|>, [EOT]}; and apply_chat_template renders
the upstream chat_template.jinja exactly.

    uv run --with tiktoken python -m parity.tokenizer_test
"""

from __future__ import annotations

from quanta.tokenizer import KimiTokenizer

MODEL = "/Users/pmrj/models/Kimi-K2.6"


def run() -> None:
    tok = KimiTokenizer(MODEL)

    # base/special offsets (from tokenizer_config.json added_tokens_decoder)
    offsets_ok = (
        tok.n_base == 163584
        and tok.special_tokens["[BOS]"] == 163584
        and tok.special_tokens["[EOS]"] == 163585
        and tok.special_tokens["<|im_end|>"] == 163586
        and tok.special_tokens["[EOT]"] == 163593
        and tok.special_tokens["[PAD]"] == 163839
    )

    # generation stop set = tokenizer [EOS] + chat <|im_end|> + end-of-turn [EOT]
    stop_ok = tok.stop_ids == {163585, 163586, 163593}

    # eval path: plain prose round-trips, BOS prefix applied, all base ids
    prose = "The capital of France is Paris. 2 + 2 = 4."
    ids = tok.encode(prose, add_bos=True)
    roundtrip_ok = (
        ids[0] == 163584
        and all(0 <= t < 163584 for t in ids[1:])
        and tok.decode(ids[1:]) == prose
    )

    # byte-identical eval behaviour: a control-token *string* must encode as ordinary
    # text (never the special id) under the default allow_special=False
    text_mode = tok.encode("<|im_end|>", add_bos=False, allow_special=False)
    special_mode = tok.encode("<|im_end|>", add_bos=False, allow_special=True)
    special_map_ok = (
        163586 not in text_mode and len(text_mode) > 1 and special_mode == [163586]
    )

    # chat template renders the upstream jinja exactly (one user turn, default thinking)
    rendered = tok.apply_chat_template([{"role": "user", "content": "hi"}])
    expected = "<|im_user|>user<|im_middle|>hi<|im_end|><|im_assistant|>assistant<|im_middle|><think>"
    template_ok = rendered == expected

    # tokenized chat: control tokens map to special ids, NO bos, ends at <think>
    chat_ids = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=True)
    chat_tok_ok = (
        chat_ids[0] == tok.special_tokens["<|im_user|>"]
        and 163586 in chat_ids
        and chat_ids[-1] == tok.special_tokens["<think>"]
        and 163584 not in chat_ids  # no BOS on the chat path
    )

    print("\n=== KimiTokenizer (upstream parity) ===")
    print(f"base/special offsets                 : {offsets_ok}  n_base={tok.n_base}")
    print(f"stop set {{[EOS],<|im_end|>,[EOT]}}      : {stop_ok}  {sorted(tok.stop_ids)}")
    print(f"eval round-trip (base ids + BOS)      : {roundtrip_ok}")
    print(f"control str: text vs special mapping  : {special_map_ok}")
    print(f"apply_chat_template == upstream jinja : {template_ok}")
    print(f"tokenized chat (specials, no BOS)     : {chat_tok_ok}")
    if not template_ok:
        print(f"  rendered={rendered!r}")
    assert all([offsets_ok, stop_ok, roundtrip_ok, special_map_ok, template_ok, chat_tok_ok])
    print("KimiTokenizer OK (offsets + eval-identical + chat specials + template)")


if __name__ == "__main__":
    run()
