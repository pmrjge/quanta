---
name: project-omlx-serving-contract
description: "How quanta serves through oMLX — raw-output engine + import-hook autopatch; reasoning/tools are parsed oMLX-side, not engine-side"
metadata:
  node_type: memory
  type: project
  originSessionId: e88fadb0-5f2c-48da-82a1-e14175af6551
---

How quanta plugs into the **oMLX** serving host (github.com/jundot/omlx), which has no plugin seam
of its own. Hard-won from reading oMLX source (curl'd; `gh`/WebFetch wouldn't serve it).

**Autopatch (`quanta/omlx_patch.py`).** A `MetaPathFinder` import hook patches three oMLX modules
the moment they import — importing quanta does NOT import oMLX. `_TARGETS`:
- `omlx.model_discovery` — tag quanta artifacts with `engine_type='quanta'` (+ `model_type='llm'`).
- `omlx.engine_pool` — `_load_engine` routes a registered `engine_type` to its factory, else
  delegates to the original (oMLX dispatches on a hard-coded chain, so we own the runtime).
- `omlx.api.tool_calling` — wrap `parse_tool_calls` (Kimi parser, below).
Guarded by `PATCH_MARKER`/`PATCH_VERSION`; env `QUANTA_OMLX_AUTOPATCH` toggles it. Console script
`quanta-omlx` arms the patch then hands off to oMLX's CLI. Engines self-register via
`register_engine(engine_type, detector, factory)`; the quanta engine subclasses oMLX `BaseEngine`.

**The contract: the engine emits RAW output.** Special/control tokens are rendered as **literal
marker strings** (`<think>`, `<|tool_calls_section_begin|>`, …), NOT parsed engine-side. This was a
*corrected mistake*: oMLX does reasoning/tool extraction **itself** from the text and **ignores**
engine-side reasoning fields. So `OmlxGenerationOutput` carries no `reasoning_content`; the shim
(`quanta/shim/omlx.py`) is a thin raw loop (`_Detok` uses `tokenizer.id_to_special` for byte-exact
markers). The earlier engine-side `kimi_format.py` parser was deleted.

**oMLX-side parsing (what the host does to our raw text).** Reasoning:
`extract_thinking(clean_special_tokens(text))`. `clean_special_tokens` (`SPECIAL_TOKENS_PATTERN`)
strips only a **fixed set** (`<|im_end|>`,`<|im_start|>`,`<|endoftext|>`,`<|end|>`,`<|eot_id|>`,
`<|start_header_id|>`,`<|end_header_id|>`,`</s>`,`<s>`,`<pad>`,`[PAD]`,`[SEP]`,`[CLS]`) — it
**preserves** `<think>` and tool markers. Tools: `parse_tool_calls`; its built-in registry covers
xml/json/gemma/glm/qwen formats. (`OutputParserSession`/`detect_output_parser` are used only by
oMLX's own dflash/batched engines, NOT by custom engines.)

**Per-model.**
- **Kimi-K2.6** needs a tool-parser patch — its tool markup uses *special tokens*
  (`<|tool_calls_section_begin|>…<|tool_call_begin|>functions.{name}:{idx}<|tool_call_argument_begin|>{json}<|tool_call_end|>…<|tool_calls_section_end|>`)
  with no entry in oMLX's registry. `quanta/shim/kimi_tools.parse_kimi_tool_calls` extracts them
  (pure, no torch/mlx/omlx); the patch tries Kimi first and **delegates everything else** to the
  original parser. Function name rides the OpenAI tool-call `id` (`functions.{name}:{idx}`) so it
  round-trips through the chat template. Reasoning (`<think>`) needs no patch. Gates:
  `parity/kimi_omlx_contract_test.py`, `parity/kimi_tools_test.py`, `parity/kimi_omlx_live_test.py`.
- **Nemotron** needs **NO patch at all** — `<think>` reasoning + Qwen/Llama XML tools
  (`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`) are oMLX-native
  (stock `extract_thinking` + `_parse_xml_tool_calls`); markers are ordinary tokens that survive
  `clean_special_tokens`. `parity/nemotron_omlx_contract_test.py` proves it delegates *past* the
  armed Kimi patch.

**Lesson.** Map to the model's own encoding and patch oMLX's parser **registry**, not the engine.
Serving a model still needs its own runtime engine (Nemotron's Mamba/GQA loop ≠ Kimi's MLA shim —
task #39); this contract only covers tokens-in / raw-text-out + how the host parses it.
