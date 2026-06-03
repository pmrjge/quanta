---
name: project-tokenizer-eos
description: "Kimi-K2.6 tokenizer facts — tiktoken 163584+256, the two distinct eos, dual-mode encode + jinja chat template, self-contained artifact files"
metadata:
  node_type: memory
  type: project
  originSessionId: e88fadb0-5f2c-48da-82a1-e14175af6551
---

Source tokenizer (verified byte-identical to upstream commit `7eb5002f`): tiktoken BPE,
**163,584 base ids** + **256 reserved specials** at `[163584, 163840)`, named by
`tokenizer_config.json` `added_tokens_decoder`. Our `_PAT_STR` == upstream
`tokenization_kimi.py` `pat_str` (char-for-char).

## Two distinct eos — do not "fix" one into the other
- `[BOS]` = **163584**
- `[EOS]` = **163585** (the tokenizer's nominal eos_token)
- `<|im_end|>` = **163586** = the model's **generation** eos (`config.json` /
  `generation_config.json` `eos_token_id`) — the token the model emits to end a turn
- `[EOT]` = **163593**; `[PAD]` = 163839
Generation/serving stops on `{163585, 163586, 163593}`. (Earlier CLAUDE.md said
`eos=163585` — that's the tokenizer [EOS], not wrong, just not the generation eos.)

## KimiTokenizer dual-mode (src/quanta/tokenizer.py)
- `encode(..., allow_special=False)` (default): control-token strings encoded as ordinary
  text (`allowed_special=set(), disallowed_special=()`). **Byte-identical** to the old
  perplexity/bake-calibration path — every existing caller uses this.
- `encode(..., allow_special=True)`: control tokens map to special ids — chat path only.
- `apply_chat_template`: renders the upstream `chat_template.jinja` via Jinja2
  (`ImmutableSandboxedEnvironment(trim_blocks, lstrip_blocks)` + a `tojson` filter, matching
  transformers). Lazy-imported jinja2 (in env as 3.1.6; ships with the omlx extra) so the
  eval/hot path stays dep-free. Tool decls use the template's JSON branch (we don't vendor the
  upstream TS encoder). Chat prompts are encoded with **no BOS** (template carries structure).
- `decode` keeps base-only filtering (`< n_base`) → clean user-facing text.

## Shim wiring (src/quanta/shim/omlx.py)
`start()` unions `tokenizer.stop_ids` into `_eos`. `chat`/`stream_chat` set
`add_bos=not templated`, `allow_special=templated`; `stream_generate` pops those kwargs.

## Self-contained serving artifact
`run_bake.py` copies **tiktoken.model + tokenizer_config.json + chat_template.jinja** into the
artifact (chat needs the config's special names + the template). Any artifact baked before this
fix (the 2026-05-23 int3) has only tiktoken.model — copy the other two in for its chat path to work.

## Raw-output serving (shim → oMLX)
The shim serves **raw output**: control ids render back to literal markers via the tokenizer's
`id_to_special` map, and reasoning/tool parsing happens **oMLX-side**, not engine-side — see
[[project-omlx-serving-contract]]. `decode` still does base-only filtering for clean user-facing text.
