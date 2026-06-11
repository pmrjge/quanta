---
name: project-nex-n2-pro
description: "Nex-N2-Pro = post-trained Qwen3.5-397B-A17B (qwen3_5_moe) — the EXACT arch quanta.qwen35 already targets, so validate-at-scale+bake (Super→Ultra), NOT build-from-scratch. int4-g64 SHIPPED (214 GiB, ~lossless +0.68% ppl). N0–N3-2 done; ACTIVE task = N3 serving. Handover PLAN_nex_n2_pro.md."
metadata:
  node_type: memory
  type: project
  originSessionId: be1e7097-a051-4573-af5f-0995c6587155
---

**What:** `nex-agi/Nex-N2-Pro` (739 GiB/122 shards bf16 at `~/models/Nex-N2-Pro`) is the post-trained
**Qwen3.5-397B-A17B** (`qwen3_5_moe`) — the EXACT architecture the in-tree **`quanta.qwen35`** module
already targets (60L hybrid: **45 Gated-DeltaNet linear + 15 gated-GQA full**; 512 experts top-10 +
shared; partial mRoPE 0.25, θ1e7; dynamic-YaRN-to-1M already coded). So this is **validate-at-scale +
bake (the Super→Ultra pattern)**, NOT build-from-scratch — the qwen35 forward was already correct from
the Qwen3.6-35B keeper ([[project-qwen35-experts]]). The active task; handover **`PLAN_nex_n2_pro.md`**.
One model resident at a time; real-weight gates SOLO ([[feedback-memory-safety]]).

**N0 (37e19f2) — groundwork (model-free/header-only, no 739 GB load).** The fit-test caught two real
Nex-vs-35B-contract divergences: **(1) EOS** — Nex ships **no `generation_config.json`** + config eos is
the lone `<|endoftext|>` 248044 (never ends a turn); `from_pretrained` now derives the ChatML stop set
**{248046 `<|im_end|>`, 248044}** from the tokenizer (35B path byte-unchanged), and the bake
**synthesizes** a correct `generation_config.json`. **(2) MTP** — Nex declares `mtp_num_hidden_layers=1`
but ships **ZERO `mtp.*` weights** ⇒ `num_mtp_modules→0` by index presence ⇒ **native-MTP spec-decode is
N/A for Nex** (bake `include_mtp=False`). **1M-in-config (user requires it first-class):**
`_bake_long_context` writes **standard HF YaRN** (`rope_type=yarn`/`factor=4`/`original_max=262144`) +
raises `max_position_embeddings` to **1,010,000**; `from_pretrained` reads `yarn_original_max` from
`rope.original_max` (DECOUPLED from `max_position_embeddings`) so the served window declares 1M while the
dynamic-YaRN baseline stays 262144 (`eff@8k=1.0`, `eff@1M=3.85`). `quant_policy.py`: 1038 text tensors =
**453 dense / 465 int8 / 120 expert_int4** (+333 vision excluded). FIT: int4-g64 ≈ 214.1 / int6-g64 ≈
304.1 GiB — both ≤ 490.4. ([[project-tokenizer-eos]])

**N1 (4620897) — layer parity @ 397B vs an independent `transformers` `Qwen3_5Moe` reference**
(transformers 5.9.0 ships `qwen3_5_moe`), SOLO/layer-streamed: **deltanet Δ1.95e-6**, **attn Δ2.10e-6**
(partial-mRoPE + **doubled-`q_proj` sigmoid output gate** + per-head `(1+w)` q/k norm), **moe** router
top-10 **set-exact** (softmax + `norm_topk_prob` renorm — confirmed **NOT** DeepSeek sigmoid/noaux_tc),
**block Δ1.9e-6**. All fp32 cross-impl at machine precision — **no forward bug surfaced** (N1 is the
at-scale re-gate, the Super→Ultra pattern). The `(1+w)` fold is `runtime.py:_one_plus` (layer/q/k/final
norms, **NOT** the gated-DeltaNet norm — [[project-forward-bug-resolved]] discipline).

**N2 (1b0c43c) — int4-g64 baked → SHIP.** `run_bake_nex_n2_pro_int4g64.py` (2.7 min, data-free RTN) →
`~/models/Nex-N2-Pro-quanta_int4g64`: **214.1 GiB/25 shards**, counts {int8 465, expert_int4 120, dense
453} (== the N0 projection exactly), MTP excluded, **config declares the 1M window** + synthesized
`generation_config.json` + tokenizer copied (self-contained). int6-g64 also baked (304.1 GiB) via a new
**`expert_bits` knob** (default 4; int4 path byte-identical, int6 = same recipe at a wider MLX-affine
grid). Every bake is **self-contained AS CODE** — `_audit_self_contained` fail-loud (no symlinks, no path
leaks, relative weight_map, all shards present — [[feedback-selfcontained-artifact]]). **ppl arbiter**
(SOLO, 3 streamed forwards over 645-tok prose, one block resident at a time): **bf16 5.0386 / int4 5.0729
(+0.68%) / int6 5.0237 (−0.30%)**. **int4-RTN ~lossless** (the Nemotron-Ultra +0.3% finding reproduces on
a bf16 source — [[project-nemotron-ultra]]); int6 recovers <1pp for +90 GiB. **SHIP int4-g64 (214 GiB).**
Teacher-forced ppl is THE arbiter; top-1 agree ~0.95 is the noisy secondary signal (bf16-ULP near-ties).

**N3-1 (7e2c817) — resident + batched serving re-gate @ 397B** (ONE 214.7 GiB load): the served kernels
(packed-int4 routed experts via `gather_qmm` + int8 mixer via `quantized_matmul`) teacher-forced give
**ppl 5.0715 == the streamed dequant int4 ref 5.0729 (Δ −0.03%** — packed fuses the dequant at full
precision, marginally beating the ref's bf16 pre-round). The **#153 loop-kill** `step_batch` is
loop==loopkill **greedy-exact at B∈{1,4,8,16,32}** incl. the chunked B=16/32 regime; B=1 batched==single
(Design-A). Throughput **B=1 14.0 → B=32 55.7 agg tok/s (3.98× batching)**, loop-kill **1.15→1.50×**;
resident 215→261 GiB (~1.5 GiB/stream, 229 GiB headroom ⇒ B can go far higher). ([[project-paged-batched-153]])

**N3-2 (7b7813f) — `qwen3_coder` tool parser + `qwen3` reasoning parser** (serving recipe
`--reasoning-parser qwen3 --tool-call-parser qwen3_coder`). Tool calls render as nested-XML
`<tool_call>\n<function=NAME>\n<parameter=KEY>\nvalue\n</parameter>…</function>\n</tool_call>`; reasoning
**pre-opens a bare `<think>`** (output `{reasoning}\n</think>\n\n{answer}`). **DECISIVE finding: this tool
markup is BYTE-IDENTICAL to Nemotron-3's** (reasoning = same pre-opened `<think>`) — both already handled
by oMLX's stock `_parse_xml_tool_calls`/`extract_thinking`, the path quanta **DELEGATES** (gated by
`nemotron_omlx_contract_test`), so Nex's tool+reasoning serving already *functions*. Per rule 6 still ship
the strict quanta-owned `Qwen3CoderToolParser` — the **ONE quanta parser deliberately kept OUT of the
global `parse_quanta_tool_calls` dispatcher** (registering it would silently re-route Nemotron's delegated
markup, indistinguishable by text). Additive only; gate `parity/qwen3_coder_tool_parser_test.py` (24
checks) + a conformance/exclusion block in `qwen35_omlx_engine_test`; sweep 100/100.
([[project-omlx-serving-contract]])

**Remaining N3 (over the int4-g64 artifact):** **1M long-doc/needle gate** (the YaRN arbiter) →
**paged-KV + prefix caching** (only the **15 full-attn layers hold KV**; the 45 Gated-DeltaNet layers are
O(1) recurrent state — the Nemotron hybrid-paging pattern, [[project-model-targets]]) → **MInference
sparse-prefill** on the 15 full-attn layers (the InternLM2.5 M0–M10 substrate transfers,
[[project-internlm2-minference]]) → **fused/batched Gated-DeltaNet decode-step** (the Nemotron
`BATCHED_FUSED_SSD_STEP` +36%@B32 pattern → `gdn_step`, [[batched-serving-operating-point]]) →
**multi-stream batched decode past B=32**.

**Cadence (standing):** single thread, NO subagents, commit each milestone, then STOP for the user to
compact.
