# PLAN — MiniMax-M3-VL serving runtime + int4-g64 bake + vision track (full VL)

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
- **Quant at int4-g64 — the only width going forward** (2026-06-14; "only 4bit from now on";
  supersedes the original int6-margin pick — int6 retired). Fit: **int4-g64 = 233.4 GiB resident**
  (96 GiB under int6's 329.6, **257 GiB headroom** under 490.4). Measured e2e: int4 WEIGHTS lossless
  (arbiter Δppl −0.24% vs bf16); served via packed `gather_qmm` +2.86% vs bf16 (the fused low-bit kernel
  gap, healthy). **Consequence:** loop-kill (M3-3) + paged-batched (M3-4) AUTO-OFF at int4 (rule 4 — the
  batched-cross-stream SDPA reorder is amplified by the coarse MoE to 0.875 token-agree, not bit-exact),
  so int4 serves per-stream attention + batched MoE + paged KV + chunked prefill. (`--bits 6` still bakes
  the retired int6 arm; 329.6 GiB.)

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
  - **M3-3 ✅ — the GQA loop-kill (this milestone).** ONE batched attention across all B streams (the
    bigger B>1 lever now the MoE expert-read is amortized). `model_m3.MiniMaxM3Attention` gains
    `decode_step_batched` (batched **chunked** q/k/v/o projections + a per-stream RoPE *kernel* loop —
    only the absolute offset differs, M3 has no YaRN — + the shared fused padded SDPA
    `quanta.modeling.batched_attention.batched_decode_attention_kv`, the #153 primitive InternLM2 /
    Nemotron / qwen35 use) + `_project_chunked` (≤`chunk` row-slices keep each packed
    `mx.quantized_matmul` in the M=1-equivalent gemv regime ⇒ **bit-exact projections**; #153 option B).
    `batched_runtime_m3` gets a `loopkill` flag (**graduated ON** — `MINIMAX_M3_BATCHED_LOOPKILL_DEFAULT`,
    chunk `MINIMAX_M3_LOOPKILL_CHUNK=8`): the `if loopkill` branch in `batched_decode_step` runs the
    batched attention; the M3-2 per-stream loop stays the rule-4 reference (pinned `loopkill=False` in the
    M3-2 gate). `_check_loopkill_requires_packed` enforces **loop-kill ⇒ packed** at construction AND every
    `step_batch` (a dense-bf16 projection reorders across batch-M — rule 4/6). Output-equivalent: only the
    fused padded-SDPA softmax reorders ⇒ greedy-token-equivalent. Gates: model-free
    `parity/minimax_m3_loopkill_test.py` (24 checks, in the sweep — **§M0** chunked-8 int8
    `quantized_matmul` bit-exact vs the M=1 loop at B∈{1,4,8,32} / full-batch reorders @ B≥12 [the qwen35
    #153 finding re-proven for int8], loop-kill == per-stream loop **and** == single-stream on the
    synthetic incl. ragged + B=1, loop-kill⇒packed refused at construction AND at step) + SOLO
    `parity/minimax_m3_loopkill_real.py` (non-`_test.py`, excluded; the **397B re-gate** off ONE resident
    load — loop-kill == the per-stream loop **BIT-EXACT** (top-1 1.0000 / rel 0, 64/64 over 8 decode steps
    × B=8: same batched MoE, bit-exact chunked projections, bit-identical RoPE, SDPA reorder ~0 at these
    lengths) and == single-stream (top-1 1.000, no near-tie flip); ships the M2b int6 quality (ppl 5.879 /
    Δppl +0.316% / agree 0.953); decode **1.19× over the per-stream loop @ B=8 / 2.83× aggregate
    B=1→B=8** — the mixer-read bandwidth win on top of M3-2's batched MoE). Manifest 106 model_free / 53
    real_weight.
  - **M3-4 ✅ — paged-KV + prefix caching (int8 KV).** M3 is the clean dense-GQA paged case (all 60
    layers attention, NO recurrent state — like InternLM2.5), so it exposes the #152 paged contract the
    shared `quanta.shim.omlx._BaseBatchedSession` drives. `model_m3.KVCache` gains int8 modes
    (`quantized`/`group_size`/`bits`, mirroring `quanta.internlm2` + `cache_quant`; default bf16 = the
    M1/M2 reference, **int8-g64** the serving lever — quant groups on `head_dim` orthogonal to the
    seq-axis blocks ⇒ a paged gather is **bit-identical** to the discrete cache). `decode_step_batched`
    gains a `paged_batched` flag → the shared `batched_decode_attention_kv` does ONE `write_batched` +
    ONE `gather_batched` over paged views (the paged KV loop-kill) vs the per-stream `.update()` loop —
    bit-identical. `batched_runtime_m3` exposes `has_recurrent_state=False` + `paged_kv_spec` +
    `make_paged_state` + `prefill_paged` (dense ⇒ no boundary payloads; `recurrent_in` must be None) +
    `_paged_kv_batched` (`MINIMAX_M3_PAGED_KV_BATCHED_DEFAULT`, graduated ON); `step_batch` auto-detects
    paged views; KV is int8-g64 on `__init__` (serving), bf16 on `from_inner` (model-free gates). Gates:
    model-free `parity/minimax_m3_paged_test.py` (19 checks, in the sweep — paged prefix-reuse + suffix
    == discrete continue-from-prefix **BIT-EXACT** for int8-g32 + bf16 KV, prefix blocks dedup, paged
    loop-kill == per-stream paged loop **bit-exact**, dense emits no boundary payloads, rule-6) + SOLO
    `parity/minimax_m3_paged_real.py` (the **397B re-gate**: paged == discrete **BIT-EXACT** (|Δ| 0), the
    paged KV loop-kill == the per-stream paged loop **BIT-EXACT** (|Δ| 0 @ B=8 ragged), **int8 KV
    near-lossless** (bf16 ppl 5.879 → int8-KV 5.927, **Δppl +0.823%** / top-1 agree 0.949), paged decode
    == single-stream (top-1 1.000), reuse-after-free **bit-exact**). **Finding:** paged prefix reuse is
    bit-exact when the committing prefill SHAPE matches; a re-admit committed at a *different* shape is
    greedy-token-equivalent (the #153 batch-M tiling effect, now in prefill: same tokens prefilled in
    different-length batches give quant-ULP-different KV codes, compounding over 60 layers). Manifest 107
    model_free / 53 real_weight.
  - **M3-5 ✅ — long-context chunked prefill (this milestone).** The single-shot prefill holds the whole
    `[1,T,hidden]` window resident; chunked prefill consumes the prompt in seq blocks, each chunk
    extending every layer's GQA KV with a bottom-right causal mask (`mx.fast.scaled_dot_product_attention`
    `mask="causal"` is bottom-right aligned — the M3-1 cached forward / qwen35 shipped-chunked path), so
    the per-chunk transient is **O(chunk)** (the fused flash-attn kernel never materializes the
    `[chunk, kv_len]` scores) and a 1M-token prompt admits in O(chunk) memory + the int8 KV. M3 is all
    dense GQA (no GDN, no YaRN) so — unlike `quanta.qwen35.runtime.chunked_prefill` — there is no
    per-request RoPE factor to pin and no recurrent continuation; each chunk reads its absolute position
    from the cache offset. New `runtime_m3.chunked_prefill` (shared driver, one bounded `MiniMaxM3Block`
    forward per chunk, per-chunk `mx.eval`+`mx.clear_cache`, rule 8) + `MiniMaxM3ResidentModel.prefill_chunked`;
    `batched_runtime_m3` adds `MINIMAX_M3_PREFILL_CHUNK_TOKENS`=4096 + `MINIMAX_M3_CHUNKED_PREFILL_FROM`
    (=chunk+1) and routes `prefill` (and thus the paged `prefill_paged` admit) through `prefill_chunked`
    above the threshold; below it the bit-exact single-shot path is kept (M3-1/2/3/4 chat-length gates
    untouched). Works over discrete `KVCache` OR `PagedKVCacheView` lists (the manager allows sub-range
    writes from the open cursor). Bit-exact to single-shot on the bf16 mixer; greedy-token-equivalent on
    the packed serving mixer (the projections run at batch-M=chunk vs M=T — the #153 batch-M ULP). Gates:
    model-free `parity/minimax_m3_prefill_chunked_test.py` (41 checks, in the sweep — bf16 chunked ==
    single-shot **BIT-EXACT** across chunk sizes incl. ragged + per-token; int8-KV bit-exact for chunk≥2,
    ct=1 hits the decode `mask=None` kernel ⇒ greedy-equiv [the documented int8 decode-vs-prefill
    boundary]; continue-from-non-empty-cache; chunked-over-paged == discrete == single-shot for both KV
    modes; rule-6; threshold routing) + SOLO `parity/minimax_m3_prefill_chunked_real.py` (the **397B
    re-gate**: chunked == single-shot **greedy-token-equivalent** [top-1 ==, rel 3.36e-2], chunked-over-
    paged == discrete-chunked **BIT-EXACT** [|Δ| 0 — the M3-4 orthogonal-axes foundation holds under
    chunked writes], chunked-seeded 6-step decode == single-shot-seeded **1.000**). Manifest 108
    model_free / 53 real_weight.
  - **int4-g64 switch ✅ (this milestone).** Served width → int4-g64 (int6 retired; "only 4bit from now
    on"). Re-baked via `run_bake_minimax_m3_int4g64` (233.4 GiB, full VL, native 1M, 3.3 min); arbiter +
    5 serving `_real` gates + the model-free bake gate (sweeps int4+int6) + the M0 fit gate all repointed.
    **int4 weights lossless** (arbiter Δppl −0.24% vs bf16); **served `gather_qmm` +2.86% vs bf16** (the
    fused low-bit kernel gap — healthy, the intrinsic int4 serving cost, anchored by the lossless arbiter).
    Serving re-gate: **paged ✅ + chunked ✅** unchanged; **runtime/batched** `DPPL_CEILING` 1.0→4.0 (int4
    `gather_qmm` vs dequant gap +2.1%/+1.7% — intrinsic, not a regression); **loop-kill AUTO-OFF at int4**
    (`_resolve_loopkill_default`: ON int6+/bf16, OFF int4 — 0.875 token-agree, rule 4) ⇒ paged-batched
    cascades off ⇒ **int4 serves per-stream attention + batched MoE + paged KV + chunked prefill**.
    **int6 artifact freed 2026-06-14** (user; 330 GiB reclaimed; `--bits 6` reproduces it).
  - **M3-6a / vision V1 ✅ (this milestone).** The CLIP-ViT **vision tower forward** — the user picked
    the vision track first (full-VL requirement; lowest reference-risk; cleanly parity-gateable). New
    additive `src/quanta/minimax/model_vision_m3.py`: **Conv3d-as-linear patch embed** (on-disk
    `[1280,3,2,14,14]` conv → `[1280,1176]`, `1176=3·2·14·14` in the shipped `image_processor.py`'s
    `[channel,temporal,h,w]` patch-flatten order ⇒ a `[1176→1280]` linear; Qwen2-VL style), `pre_layrnorm`,
    32 **pre-norm CLIP encoder layers** (biased q/k/v/out, full bidirectional, exact-erf GELU, LayerNorm;
    no learned pos-embed / CLS / post-norm — none ship), **3-D vision RoPE**, then the **project→merge**
    head whose ORDER is forced by the on-disk input dims (`multi_modal_projector.linear_1` input 1280 ⇒
    per-patch `1280→6144→6144` FIRST; `patch_merge_mlp.linear_1` input `24576=4·6144` ⇒ consecutive-4 = one
    2×2 spatial block `24576→6144→6144` SECOND). One image `grid=(t,h,w)` → `t·h·w` ViT tokens →
    `(t·h·w)/4` LLM tokens (== the processor's `num_tokens = grid.prod()//merge²` at `image_token_index`
    200025). The **3-D RoPE** is the one piece with **no shipped reference** (transformers CLIP = learned
    pos-embeds; even Qwen3-VL *vision* is 2-D h/w) — built on the **Qwen2.5-VL M-RoPE convention** (one
    shared `inv_freq` ladder, freq pairs *sectioned* across t/h/w), which **degenerates exactly to the 2-D
    (h,w) rope for an image** (`grid_t=1` ⇒ t-pos 0 ⇒ identity on the t-section). [PINNED-pending-e2e: the
    exact `rope_section` split (default `(8,16,16)`, h==w) is the lone knob no artifact fixes; vision V2
    settles it.] Gate: model-free `parity/minimax_m3_vision_test.py` (20 checks, in the sweep — CLIP
    encoder layer (RoPE off) == REAL `transformers.CLIPEncoderLayer`; patch-embed / 3-D rope / projector /
    merge / (t,h,w) position-ids == a numpy-fp64 oracle; rule-4 fast==naive; rule-6 section-sum +
    indivisible-merge refusals; the 2-D-degenerate property; per-image attention isolation). Manifest
    **109 model_free / 53 real_weight**.
  - **M3-6b / vision V2 ✅ (this milestone).** Native image processor + real-weight standalone ViT forward.
    NEW `image_m3.py` (numpy-only, no torch/torchvision/PIL — rule 5): the shipped `image_processor.py`
    reproduced (smart-resize geometry verbatim → bicubic → rescale + CLIP-normalize → temporal-dup →
    patchify) → `pixel_values [N,1176]` + `grid_thw`; resize interpolation is best-effort (no torchvision
    in-env), but a **factor-aligned in-bounds image makes resize the identity** ⇒ exactly pinnable (the
    gate/e2e path). NEW `artifact_m3.vision_state()` (523 dense ViT tensors → bf16, loaded as a unit — a
    1.6 GiB rule-8 exception) + `model_vision_m3.load_vision_model()` (Conv3d→linear reshape; 1:1 suffix
    map; two-way coverage assertions, rule 6). Gates: model-free `parity/minimax_m3_image_test.py` (23
    checks, in the sweep — smart_resize/normalize/patchify == oracles, factor-aligned identity, bicubic
    well-formed, num_tokens rule, rule-6 refusals) + SOLO `parity/minimax_m3_vision_real.py` (non-`_test.py`,
    excluded, ~1.6 GiB — real 56×56 image → 4 tokens finite/sane; rule-4 fast==naive layer-0 op rel 5.9e-3;
    2-D-degenerate t-section inert on real q; per-image isolation **bit-exact**). No numeric ViT reference
    exists; V2 validates **mechanics + invariants**, the `rope_section (8,16,16)` arbiter is the V3 e2e.
    Manifest **110 model_free / 53 real_weight** (+`minimax_m3_image_test`).
  - **M3-6c / vision V3a ✅ (this milestone).** The **multimodal prefill splice** + the runtime
    `inputs_embeds` path. NEW `model_vision_m3.splice_image_embeddings(...)` replaces the `image_token_index`
    200025 placeholder rows with the merged ViT tokens **in sequence order** (image-0's first; the shipped
    `processing_minimax.py` rule — `]<]image[>[` → start(200029) + `num_tokens` placeholders(200025) +
    end(200030)) via a vectorized cumsum-scatter (rule 3); fails loud on a count/hidden mismatch (rule 6),
    bit-exact passthrough with no placeholders. `runtime_m3` gains the `inputs_embeds` path
    (`__call__`/`chunked_prefill`/`prefill_chunked`, exactly one of token_ids/inputs_embeds; the token-id
    path stays byte-for-byte) + `embed_tokens()` + `multimodal_prefill()` (embed → splice → forward). Gates:
    model-free `parity/minimax_m3_splice_test.py` (19 checks, in the sweep — splice == nested-loop oracle
    incl. multi-image order, rule-6 refusals, text-only passthrough, **inputs_embeds == token_ids BIT-EXACT**,
    `multimodal_prefill` == manual, chunked-embeds == single-shot) + SOLO `parity/minimax_m3_multimodal_real.py`
    (non-`_test.py`, excluded, **~235 GiB** — int4 text decoder (60L in 10s) + dense ViT; real 56×56 image →
    4 merged tokens spliced into a real prompt: **image is SEEN** — prefix BIT-EXACT |Δ| 0, suffix max|Δ| 21.6;
    finite/sane; inputs_embeds == token_ids BIT-EXACT @ scale). Validates the splice + ViT→text wiring e2e at
    397B; does **NOT** settle `rope_section` (every section shows "image seen"). Manifest **111 model_free / 53
    real_weight** (+`minimax_m3_splice_test`).
  - **M3-6d / vision V3b-prep ✅ (this milestone).** The `rope_section` **arbiter machinery** (the verdict
    needs a user image — the one input not in-env). User decision: **add `pillow` + give an image** ⇒
    `pillow>=11.0.0` under the **`reference`** extra (offline-only, rule 5; runtime `image_m3` stays
    PIL-free). New offline `parity/_image_decode.py` (file/bytes → `[H,W,3]` uint8 RGB; L/RGBA→RGB; in
    `parity/`, never imported by `src/`); `model_vision_m3.candidate_rope_sections(head_dim)` (symmetric
    h==w splits, temporal share `st` 0→half//2; 11 for head_dim 80, default included). The heavy SOLO
    arbiter `parity/minimax_m3_rope_section_real.py` (~235 GiB, excluded; sentinel): image → ViT(section)
    → splice → **teacher-forced ppl of the caption span** across all candidates (mutate `vis.rope_section`
    in place, weights reused) + a text-only baseline; min-ppl section = the trained one. Smoke-validated
    (`--layers 4`, synthetic). Gates: model-free `parity/minimax_m3_rope_section_test.py` (12 — knob is
    live / candidate set valid / rule-6; no PIL) + skip-eligible `parity/minimax_m3_image_decode_test.py`
    (14 — lossless round-trip; `optional_deps` now maps `pillow`→`PIL`). Manifest **113 model_free / 53
    real_weight** (+2).
  - **vision V3b (next) — RUN the verdict.** One command once the user supplies a real **natural** image +
    a true caption: `uv run python -m parity.minimax_m3_rope_section_real /path/to/photo.jpg "caption"`
    (28-multiple image ⇒ exact identity-resize; off-grid ⇒ best-effort/unpinned bicubic). The section
    minimizing caption ppl wins — judged downstream by the LLM, the only arbiter.
  - **M3-7a / oMLX shim — output parsers ✅ (this milestone).** The MiniMax-M3 reasoning + tool-call
    output parsers (qwen35-N3-2 analog; pure-text, model-free, **independent of the V3b verdict** so it
    lands now). M3 markup: reasoning `<mm:think>…</mm:think>` (ids 200059/200060, not bare `<think>`
    200050/1) + a *namespace-prefixed recursive nested XML* tool call — `]<]minimax[>[<tool_call>` …
    `]<]minimax[>[</tool_call>` (ns_token 200058), `ns<invoke name="N">`…`ns</invoke>`, args from the
    template's recursive `to_xml` (mapping `ns<k>…ns</k>`, list `ns<item>…ns</item>`, bool tojson, scalar
    raw; keys are real names, not M2.7 `<parameter name=>`). Every tag ns-prefixed ⇒ split on ns_token →
    one segment/tag → a small recursive descent inverts it to typed args. New in
    `quanta.shim.tool_parsers`: `parse_minimax_m3_tool_calls`, `MiniMaxM3ReasoningParser`,
    `MiniMaxM3ToolParser` (`format_tool_response`→`<response>…</response>`), registered in `_PARSERS`
    (disjoint from M2.7/GLM/Hermes/Qwen3-Coder). Gate: model-free `parity/minimax_m3_tools_test.py` (53 —
    a **reference renderer** re-implements the jinja `to_xml` and the parser is asserted to invert it for
    flat/typed/nested/list-of-dicts/None-skip/multi-section [empty-container→`""` collapse documented],
    `<mm:think>` shapes, `<response>`, Protocol conformance, disjointness + dispatcher routing). Manifest
    **114 model_free / 53 real_weight** (+1).
  - **M3-7b (next) — oMLX shim engine route.** The `_MiniMaxM3BatchedSession` (load-runtime + decode-stepper
    + batched-session dispatch on `model_type` `minimax_m3*`, currently swallowed by the M2.7
    `mt.startswith("minimax")` route) + the chat-template / `apply_chat_template` rendering path + the
    multimodal image input path (wire the V3a splice through `batched_runtime_m3`) + multi-stream.
  - **After the shim.** The **trained block-sparse attention** long-context compute lever (deferred — no
    M3 forward exists; only sparse==dense-at-short-ctx is bit-gateable; it is a speed optimization on the
    already-correct dense path).

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
