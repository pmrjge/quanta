# PLAN — MiniMax-M3-VL serving runtime + int6 bake (full VL)

Handover for the **MiniMax-M3-VL** build. Single-thread cadence (NO subagents, commit each
milestone, STOP for the user to compact). This is the project's only forward model
(see `roadmap-collapsed-minimax-only` memory; CLAUDE.md "Current state").

## What it is (empirically confirmed — M0 fit-test)

`~/models/MiniMax-M3` = **MiniMax-M3-VL** (`minimax_m3_vl`,
`MiniMaxM3SparseForConditionalGeneration`), **809.5 GiB bf16 / 59 shards / 23,416 tensors**. This is
a **different architecture** from the in-tree `quanta.minimax` module (which targets the OLD
**M2.7** `minimax_m2`: 62L all-MoE, full softmax, 256 experts, no shared expert, fp8 source — M2.7
source is gone from disk). So M3 is a **real build**, not validate-at-scale. The M3 code is added
**additively** (`*_m3.py`); the M2.7 files are left intact (retire later if desired).

**Text backbone (`MiniMaxM3SparseForCausalLM`), 60 layers:**
- **Layers 0–2 dense** (`mlp.{gate,up,down}_proj`, width 12288); **layers 3–59 MoE**
  (`block_sparse_moe.*`). Split = the explicit `moe_layer_freq` list.
- **GQA** 64 q / 4 kv heads, head_dim 128 (q_proj [8192,6144], k/v [512,6144], o [6144,8192]);
  **partial RoPE** rotary_dim 64 θ=5e6; **per-head QK-norm** (q/k_norm [128]); **Gemma `(1+w)`
  norms** (`use_gemma_norm`). ≈ `quanta.qwen35` attention.
- **Native TRAINED block-sparse attention**, layers 3–59: `self_attn.index_{q,k}_proj` +
  `index_{q,k}_norm` (4 index query heads dim 128 + shared index key; top-`sparse_topk_blocks=16`
  key-blocks of `sparse_block_size=128` by `sparse_score_type=max`, always keep
  `sparse_init_block=0` sinks + `sparse_local_block=1` recent). **NEW** (DeepSeek-V3.2-DSA-style).
  KEY LEVER: at T ≤ ~2048 tokens top-16 = all blocks ⇒ **sparse == dense at short ctx**, so build
  dense-correct first; the indexer is the long-context milestone (the InternLM2.5
  `quanta.modeling.xattention` execution substrate transfers).
- **MoE**: 128 experts, top-4, **+1 shared expert** (`shared_experts.{gate,up,down}_proj`),
  **sigmoid noaux_tc** routing (`gate` [128,6144] f32 + `e_score_correction_bias` [128] f32),
  `routed_scaling_factor=2.0`, expert width 3072 (w1=gate, w3=up, w2=down). ≈ nemotron/dsv4 routing.
- **Clamped SwiGLU-OAI** activation (`hidden_act=swigluoai`, `swiglu_alpha=1.702`,
  `swiglu_limit=7.0`) for dense FFN AND experts — **NEW** (gpt-oss-style clamp).
- **MTP declared `num_mtp_modules=7` but ZERO `mtp.*` weights on disk** → refined to 0 (Nex
  pattern). Native-MTP spec-decode **N/A**; an EAGLE drafter is the only B=1 latency path if wanted.
- **1M native context** (`max_position_embeddings=1048576`).
- **Tokens**: bos 200019, eos 200020 (the chat template ends a turn on the lone eos `[e~[` 200020 —
  no second turn-ender to derive, unlike Nex). vocab 200064, untied lm_head. Reasoning markers
  `<mm:think>`/`</mm:think>` (200059/200060); tool calls namespaced nested-XML
  (`]<]minimax[>[<tool_call>` … `to_xml` macro). PreTrainedTokenizerFast (tokenizer.json + merges.txt
  + vocab.json). image_token 200025 / video 200026.

**Vision tower** (full-VL build, per user): CLIP-style ViT 32L (hidden 1280, patch 14, image 2016,
3D-RoPE) + `multi_modal_projector` (→6144) + `patch_merge_mlp`; dynamic-res tiling
(`image_grid_pinpoints`). 523 tensors, ~1.6 GiB bf16.

## Decisions (user, 2026-06-13)

- **Build full VL now** (vision tower + projector + image processor + a multimodal input path in the
  oMLX shim — the shim has no image pathway today).
- **Quant at int6-g64 for margin** (skip int4). Fit (M0): **int6-g64 = 329.6 GiB resident** (experts
  312.6 int6 + int8 10.2 + dense/bf16 6.8 incl. the bf16 vision tower + the bf16 trained indexer),
  **160.8 GiB headroom** under 490.4. (int4 reference projection 233.4 GiB.)

## Quant policy (`quanta.minimax.quant_policy_m3`)

- `expert_int` (int6 g64) — `block_sparse_moe.experts.<e>.{w1,w2,w3}` (21,888 tensors, the footprint).
- `int8` — GQA q/k/v/o, the dense-FFN `mlp.{gate,up,down}_proj` (layers 0–2), and the shared expert
  `block_sparse_moe.shared_experts.*` (420 tensors).
- `dense` (bf16/f32 verbatim) — all RMSNorms (incl. per-head q/k + Gemma `(1+w)`), router `gate` +
  `e_score_correction_bias` (f32), the **trained sparse indexer** (`index_{q,k}_proj/norm`, kept bf16
  to protect block selection), embed/lm_head, and the **whole vision tower** (1108 tensors).
- Coverage proven exact vs the real index (rule 6): 23,416 = 1108 dense + 420 int8 + 21,888 expert.

## Milestones

- **M0 ✅ (this commit) — groundwork, model-free / header-only (no 809 GB load).**
  `config_m3.MiniMaxM3Config` (nested text+vision parse, eos `(200020,)`, MTP refine 7→0 by index
  presence, per-layer dense/MoE + sparse-attn typing, validated schedules) + `quant_policy_m3`
  (key→scheme + resident projection). Gates: `parity/minimax_m3_fit_test.py` (real-path SOLO,
  headers only, **13 checks** — int6 329.6 GiB fits, coverage exact, header acct <1%) +
  `parity/minimax_m3_config_test.py` (model-free, **24 checks** — synthetic checkpoints). Manifest
  101 model_free / 53 real_weight.
- **M1a ✅ — module + model-free layer parity** (`src/quanta/minimax/model_m3.py` +
  `parity/minimax_m3_layer_test.py`, **12 checks**). The checkpoint ships **no modeling file** (only
  `configuration_minimax_m3_vl.py`; `auto_map` has `AutoConfig` only) and the comment says it mirrors
  **sglang** (not installed) — so there is NO `transformers`/`trust_remote_code` M3 forward. Instead
  the risky formulas are pinned to AUTHORITATIVE transformers SIBLINGS in isolated single-call checks,
  and the full block is cross-checked against a pure-numpy-fp64 reference:
  - **clamped SwiGLU-OAI** == `transformers` `GptOssExperts._apply_gate` (`gate=clamp(g,max=limit);
    up=clamp(u,±limit); (up+1)·(g·σ(α·g))`; w1=gate/swish, w3=up, w2=down; M3's α=1.702/limit=7.0
    ARE gpt-oss's). **NOT** a registered `swigluoai` HF activation — gpt-oss is the formula.
  - **sigmoid-noaux router** == `transformers` `MiniMaxM2TopKRouter` (sigmoid; bias for SELECTION
    only; weights gathered from the PURE sigmoid; renorm) **+ the M3 `routed_scaling_factor` 2.0**
    (M2 has none). **No DeepSeek group machinery** (M3 config has no `n_group`/`topk_group`) ⇒ the
    nemotron/dsv4 in-tree router path, not deepseek_v3's.
  - **partial rotate-half RoPE** == `minimax_m2.apply_rotary_pos_emb` (rotary_dim 64, θ=5e6, NO YaRN).
  - **Gemma `(1+w)`** confirmed empirically: ALL non-gated RMSNorms apply `(1+w)` (one `use_gemma_norm`
    flag) — the per-head q/k + index norms are tightly 0-centered (~+0.15 ⇒ eff ~1.15), the
    input/post/final norms are varied learned scales; the `(1+w)` fold is at LOAD time. [PINNED; the
    decisive arbiter is M2 ppl — a wrong fold degrades it uniformly.]
  - **Shared expert has NO scalar gate** (the checkpoint ships no `shared_expert_gate`; rule-6
    coverage is exact without one) — unlike Qwen2-MoE / qwen35.
  Also gated (rule 4): MoE dense oracle == sparse `gather_mm`; `use_fast` (mx.fast rope+SDPA) ==
  naive attention. Model-free (synthetic dims), runs in the sweep under the `reference` extra
  (peak RSS 0.32 GiB). Manifest 102 model_free / 53 real_weight.
- **M1b — layer parity @ scale (real weights), SOLO / layer-streamed (rule 8).** Load real layer-0
  (dense) + layer-3 (first MoE, carries the indexer) from the 796 GB checkpoint, run the
  `model_m3` block in fp32, diff vs a pure-numpy-fp64 reference on the SAME dequantized weights +
  input — validates loading + real-shape/dtype wiring at scale (64q/4kv, 128 experts, dense_inter
  12288). Reference is numpy (the formulas are already pinned to transformers in M1a).
- **M2 — bake int6-g64** (self-contained artifact: relative refs only, declares the 1M window,
  synthesizes `generation_config.json` eos [200020], copies tokenizer; data-free RTN; one layer
  resident at a time, rule 8). Teacher-forced **ppl arbiter** (the e2e arbiter, methodology #4) on
  held-out prose: int6 vs bf16 → ship.
- **M3 — serving.** Resident + batched re-gate (packed-int6 `gather_qmm` experts + int8 mixer);
  **oMLX shim** (`QuantaOmlxEngine` route + chat template + the `<mm:think>` reasoning parser + the
  MiniMax nested-XML tool parser + a **multimodal image input path** — the VL-specific work);
  **paged-KV + prefix caching** (GQA 4 kv heads ⇒ cheap KV; int8 KV); **trained block-sparse
  attention** as the long-context lever (parity: sparse==dense at short ctx; xattention substrate);
  chunked prefill; multi-stream batched decode (B≈32 operating point).
- **Vision track** (folded into M1/M3 since full-VL): CLIP ViT forward + 3D-RoPE + patch-merge
  compression + projector parity; image processor (dynamic-res tiling); multimodal prefill (splice
  image embeddings at `image_token_index` 200025).

## Reusable quanta assets

- `quanta.qwen35` — Gemma `(1+w)` norm fold (`runtime._one_plus`), partial RoPE, per-head QK-norm,
  GQA full-attn, packed-int4 `gather_qmm` experts, chunked prefill, batched/paged serving.
- `quanta.nemotron` / `quanta.dsv4` — sigmoid noaux_tc routing + bias + shared expert +
  routed_scaling, MoE packing, paged-KV loop-kill, EAGLE/spec scaffolding.
- `quanta.modeling.xattention` — block-sparse gather execution (for the trained indexer's selection).
- `quanta.shim.omlx` / `quanta.shim.tool_parsers` — serving integration + parser protocols.

## Open questions for later milestones

- Exact clamped-SwiGLU-OAI formula (alpha/limit application order) — pin vs reference in M1.
- The trained indexer's exact score path (`index_q · index_k` per block, `score_type=max` reduction,
  the init/local always-keep) — pin vs reference when the long-context lever lands.
- Whether to PTQ the vision tower (kept bf16 in M0) once VL ppl is measurable.
- M2.7 module retirement (dead, no weights) — a doc/cleanup decision with the user.
