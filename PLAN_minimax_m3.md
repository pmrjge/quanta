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
- **M1b ✅ — layer parity @ scale (real weights), SOLO / layer-streamed (rule 8).** New
  `src/quanta/minimax/loader_m3.py` (`MiniMaxM3SourceCheckpoint`): a lazy single-shard-mmap reader
  for the TEXT decoder — `embed`/`final_norm`/`lm_head` + per-layer `block_norms`/`attention`/
  `sparse_index`/`dense_mlp`/`moe`. Routed experts ship **per-expert**
  (`block_sparse_moe.experts.{e}.{w1,w2,w3}`), so `moe()` **pre-stacks** them at load time (bounded
  loop) into `experts_gate_up` `[E,2*inter,h]` (w1 over w3) + `experts_down` `[E,h,inter]` (w2) — the
  `gather_mm`-ready layout `MiniMaxM3MoE.set_experts` wants. Text-only (refuses a `vision_tower.*`
  key — the ViT is a separate VL track, not dropped). Gate `parity/minimax_m3_layer_parity.py`
  (real-weight SOLO, non-`_test.py` ⇒ excluded from the sweep; loads only L0+L3, streamed+released):
  the `model_m3` block in fp32 vs a self-contained **numpy-fp64** oracle (same formulas M1a pinned to
  transformers, now on real weights — torch-free, runs on base deps) on IDENTICAL dequantized weights
  + input. **Measured (machine-precision):** dense L0 block Δ **7.8e-7**, MoE L3 block Δ **8.6e-7**,
  fast==naive 5e-8/1.6e-7, sparse `gather_mm`==dense oracle 4e-7, router (real F32 gate/bias)
  set-match + weights Δ 6e-8, trained-indexer tensors stream with the expected shapes (rule-6
  coverage). 8 checks. Confirms the loader + real-shape/dtype wiring (hidden 6144, GQA 64q/4kv hd128,
  128 experts top-4 + shared, dense_inter 12288) + the per-expert→stacked pack. Peak ~29 GiB
  (one MoE block's fp32 expert stacks) « 490.
- **M2a ✅ — int6-g64 bake + artifact reader (the artifact-producing path, proven on a real-weight
  smoke).** New `src/quanta/minimax/bake_m3.py` (`bake_minimax_m3`): streamed one text layer resident
  at a time (rule 8) over `loader_m3`, writing a self-contained int6/int8/bf16 bundle via the shared
  `ArtifactWriter` — routed experts (pre-stacked `experts.{gate_up,down}_proj`) → **int6 affine g64**
  (3-D one-shot, `gather_qmm`-ready, rule 3); GQA q/k/v/o + dense-FFN (L0–2) + shared expert → int8;
  norms / router `gate`+`e_score_correction_bias` (**f32**) / trained indexer (bf16) / embed / head →
  dense verbatim; **full VL** = the whole vision tower + projector + patch-merge (523 tensors) copied
  dense bf16 (a shard-grouped pass — the text loader stays text-only). M3 is **natively 1M** (no
  YaRN) → `_assert_native_1m_context` asserts + stamps a `quanta_long_context` marker; the source
  `generation_config.json` (eos 200020) + tokenizer + the VL `preprocessor_config.json` are copied;
  `_audit_self_contained` fails loud unless standalone. New `src/quanta/minimax/artifact_m3.py`
  (`MiniMaxM3Artifact`): a dequant-on-read reader **duck-typing `loader_m3`** (same
  `embed`/`block_norms`/`attention`/`sparse_index`/`dense_mlp`/`moe`/`moe_packed` surface + dicts), so
  one forward serves both the bf16 source and the int6 artifact — **the router gate/bias are returned
  at native F32 via `get()` (NOT bf16-downcast — a downcast could flip a top-k tie ⇒ a different
  expert; confirmed only gate+bias are F32, the rest bf16).** Gates: model-free
  `parity/minimax_m3_bake_test.py` (**12 checks**, in the sweep — a tiny synthetic M3 checkpoint
  through the real `bake_minimax_m3` then back through BOTH readers: every quantized dequant ==
  source-RTN bit-exact, F32 router preserved bit-exact, dense/vision verbatim, manifest schemes,
  raw/refusals, native-1M + self-contained) + the real bake `parity/run_bake_minimax_m3_int6g64.py`
  (SOLO; `--smoke` slice ran on real weights in 2.7s → a 6.1 GiB self-contained artifact, readback of
  L0 int8 / L3 int6 stacks [128,6144,6144]·[…,3072] / F32 gate / bf16 indexer / packed-int6 triplet
  all correct). Manifest 103 model_free / 53 real_weight.
- **M2b ✅ — full int6 bake + teacher-forced ppl arbiter (this milestone).** The real
  `run_bake_minimax_m3_int6g64` ran SOLO and produced `~/models/MiniMax-M3-quanta_int6g64` in **3.9 min**
  (RTN is data-free / fast — no GPTQ): **329.6 GiB** on disk (exactly the M0 projection, < the 490.4
  ceiling), self-contained (0 symlinks, sidecars present, no leaks, 30 shards / 2710 weight-map
  entries), **full VL** (523 vision tensors dense bf16), native 1M; counts `int8 420 / expert_int 114
  (57 MoE × 2 ✓) / dense 1108`. New SOLO arbiter `parity/minimax_m3_ppl.py` (non-`_test.py` ⇒ excluded
  from the sweep; `# parity-gate: real-weight`): two streamed `MiniMaxM3Block` forwards (one layer
  resident, rule 8) — bf16 source vs int6 artifact — over held-out prose, teacher-forced ppl + top-1
  agreement. The tokenizer is built directly from `MiniMaxM3Config` (it duck-types bos/eos/eos_token_ids;
  the BPE reads only `tokenizer.json`), `add_bos_token` absent ⇒ raw encode (the Nex precedent).
  **Verdict (637 tok, all 60 layers):** **bf16 ppl 4.96 / acc 0.591** ⇒ the pinned Gemma `(1+w)` fold
  is **CONFIRMED e2e** (the decisive check — there is no HF/sglang M3 forward to diff against; a wrong
  fold degrades ppl uniformly into the hundreds, 4.96 is exactly a healthy 397B value); **int6-g64 ppl
  5.00 / Δppl +0.82% / acc 0.591 (identical) / top-1 agree 0.943** ⇒ ~lossless, the user's int6
  margin choice (over int4) is validated e2e. **SHIP int6-g64.** PARITY-CHECKS: 4 (bf16+int6 finite,
  bf16 ppl < 30 ceiling, int6 Δppl < 5% & agree > 0.90). Smoke (`8 160`) validated the code path first.
- **M3 — serving (decomposed into sub-milestones, Nex-style).**
  - **M3-1 ✅ — resident single-stream serving runtime (this milestone).** `model_m3` gains the
    **packed-int6 `gather_qmm`** routed path (`_routed_sparse_packed` + `MiniMaxM3MoE.set_experts_packed`;
    `__call__` auto-detects a triplet dict ⇒ `gather_qmm` vs a bf16 stack ⇒ `gather_mm`, and refuses
    `sparse=False` on packed — rule 6). New `runtime_m3.MiniMaxM3ResidentModel` loads the int6 artifact
    **one text layer resident at a time** (rule 8): routed experts held **packed int6** (`artifact_m3.moe_packed`
    → `set_experts_packed`, the ~300 GiB resident lever) over the **int8 mixer dequantized to bf16**
    (q/k/v/o + dense-FFN + shared; the proven M1/M2 forward — a packed-int8 mixer saving ~10 GiB is a
    later memory milestone, far under the 160 GiB headroom), router gate/bias native **F32**; prefill
    (`caches=None`) == the streamed reference, decode threads a per-layer GQA `KVCache` (`make_caches`),
    plus a greedy `generate`. Gates: model-free `parity/minimax_m3_runtime_test.py` (9 checks, in the
    sweep — packed==bf16 top-1-exact, cached==prefill **bit-exact**, incremental-decode==full-prefill
    **bit-exact**, rule-4 dense==sparse, rule-6 refusal, `generate` smoke) + SOLO
    `parity/minimax_m3_runtime_real.py` (non-`_test.py`, excluded; the **397B resident re-gate** — all 60
    layers RAM-resident in **33 s load**, packed `gather_qmm` vs the streamed `gather_mm` reference on the
    real int6 codes: **ppl 5.870 / Δppl +0.171% / top-1 agree 0.969**; ships the M2b int6 quality — the
    resident path actually dequantizes int6 at *higher* precision than the bf16-rounded reference, the few
    top-1 flips are bf16 near-ties). Manifest 104 model_free / 53 real_weight.
  - **M3-2 ✅ — batched serving (Design A) + the packed-int8 mixer (this milestone).** The int8 mixer
    (GQA q/k/v/o on all 60 layers + the dense-FFN gate/up/down on layers 0–2) is held **packed
    `nn.QuantizedLinear`** (`mx.quantized_matmul`) via `runtime_m3._packed_linear` + a new
    `MiniMaxM3ResidentModel(packed=…)` flag (default `False` = the bf16-mixer parity reference; the
    serving runtime sets `True`) — the ~6 GiB memory lever + the batch-M bit-exact substrate,
    greedy-exact on the SAME int8 codes; the **shared expert stays bf16** (the qwen35 convention).
    New `batched_runtime_m3.MiniMaxM3BatchedResidentModel` (default `packed=True` + `packed_experts=True`)
    wraps the resident model — **Design A**: per-stream GQA `KVCache` lists, a bounded per-stream
    attention step (M=1 ⇒ bit-exact), then ONE batched FFN over the stacked `[B,1,hidden]` (the
    routed-expert `gather_qmm` reads each touched expert tile once for all B that route to it — the
    bandwidth win); `step_batch` / `prefill` / `make_batch_caches`, ragged offsets, rule-6
    desync/over-batch refusals. Gates: model-free `parity/minimax_m3_batched_test.py` (19 checks, in the
    sweep — packed-mixer==bf16-mixer greedy-exact, batched==single-stream **bit-exact** on the synthetic
    incl. ragged offsets + B=1, rule-6) + SOLO `parity/minimax_m3_batched_real.py` (non-`_test.py`,
    excluded; the **397B re-gate** off ONE 325 GiB resident load — packed mixer+experts vs the streamed
    bf16 reference **ppl 5.879 / Δppl +0.316% / agree 0.953**; batched B=8 ragged == single-stream
    greedy-token-equivalent — at scale the F32 router GEMM at M=B flips a routing near-tie on 1/8 streams,
    the documented batched boundary; decode **2.32× aggregate @ B=8**, climbing with B). Manifest 105
    model_free / 53 real_weight.
  - **M3-3+ (next).** The **GQA loop-kill** (ONE batched attention across streams via a ragged-offset
    padded SDPA, on the packed-mixer substrate — the bigger B>1 lever now that the MoE read is already
    amortized); **paged-KV + prefix caching** (GQA 4 kv heads ⇒ cheap KV; int8 KV); chunked prefill; the
    **trained block-sparse attention** long-context lever (parity: sparse==dense at short ctx; xattention
    substrate); the **oMLX shim** (`QuantaOmlxEngine` route + chat template + `<mm:think>` reasoning
    parser + MiniMax nested-XML tool parser + a **multimodal image input path**); multi-stream.
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
