---
name: project-internlm2-eagle
description: InternLM2.5 speed levers — EAGLE-3 spec-decode (lossless) + MInference sparse prefill (lossy). InternLM2.5 is the under-optimized keeper. EAGLE M0 9582072, M1 357a507, M2 09306da, M3 (real-model bench + DRAFTER QUANTIZATION) ec0f6f3: quantizing the drafter (body + k×/round vocab head) flips k=2 from 0.91× (bf16 = net SLOWDOWN) to 1.42× (int4 PTQ) lossless; k=2 optimal, target verify is the decode floor. EAGLE spec-decode on InternLM2.5 = DONE/shippable.
metadata:
  node_type: memory
  type: project
  originSessionId: 78ef7db6-f4ec-4c53-857e-3bd77cc65962
---

**Why:** InternLM2.5 is the **under-optimized keeper** — being dense it received none of the
MoE-era decode levers (sorted gather_qmm, packed experts, batched-tree spec), and it still lacks
**both** spec-decode **and** sparse prefill. It's the cheap always-on small-dense-1M serve target,
so every request pays its decode latency and long prompts pay its full **O(T²)** dense prefill (the
only keeper with that — DSV4 is native-sparse, Nemotron-Mamba/qwen35-GDN are linear). A June-2026
research pass (user: "what levers for speed of decode and prefill are still out there?") ranked the
two highest-value remaining levers as both landing on InternLM2.5:
1. **EAGLE-3 spec-decode** (lossless; ~1.5–2.5× decode) — STARTED, see below.
2. **MInference dynamic sparse prefill** (training-free, lossy → quality-gated, up to ~10×@1M) — a
   separate track ([[prefill-optimization-landscape]]), now its own milestone series at **M6 ✅**, M7
   next: [[project-internlm2-minference]]. MInference = offline per-head pattern (A-shape /
   vertical-slash / block-sparse) + sparse SDPA.

**Settled (do not re-litigate):** spec-decode is NOT diluted on the MoE keepers — the Kimi
"top-8 routing → expert-union tax" finding ([[project-eagle3-speculative]]) was OVERTURNED by
**batched-tree verify**: DSV4 `spec_generate_tree(W=2,D=2,batched=True)` = **1.37 tok/s vs chained
k=2's 0.36 — 3.77×** (`dsv4/spec.py:367`, #157, default-on), because B=W^D candidate paths share ONE
forward and amortize the routed-expert reads. 2026 lit (MoESD) confirms MoEs benefit MORE from spec
than dense at moderate batch. So DSV4/Nemotron/qwen35 already ship spec; InternLM2.5 is the gap.

**EAGLE is already MODEL-AGNOSTIC.** `quanta.eagle.spec_core.spec_generate` does lossless
draft/verify/accept/rollback over ANY runtime exposing
`(ids, caches, offset, capture_layers) -> (logits, {layer: hidden})` + a `truncate(length)` hook;
`EagleDrafter` is fully parameterizable (Kimi dims are just defaults). `minimax/eagle.py` (~72 lines)
is the precedent adapter for a non-Kimi, no-native-MTP target. `InternLM2Cache.truncate` (+ `replicate`
for batched tree verify) already exist and are lossless. So InternLM2.5 spec = the per-model adapter +
a drafter sized to InternLM2.5 + the one missing **capture** primitive.

**How to apply — roadmap (one milestone/commit, then STOP to compact; one model at a time):**
- **M0 ✅ `9582072`** — `capture_layers` on the **bf16 reference** `InternLM2Model.__call__`
  (`internlm2/model.py`): returns `(logits, {layer: post-layer residual [T,H]})`, mirrors
  `MiniMaxResidentModel.__call__` exactly; default None => logits only, byte-unchanged shipped path.
  bf16-ONLY (packed runtime untouched, zero prod risk). Gate
  `parity/internlm2_eagle_capture_test.py` (model-free, tiny random model): capture logits Δ=0,
  recon norm+head(caps[last])==logits Δ=0, truncate forward-N→trunc-M→continue == prefill-M Δ=0.
- **M1 ✅ `357a507`** — `internlm2/eagle.py` adapter (mirror minimax/eagle.py): `spec_generate(model,
  drafter, embed, head, prompt_ids, …)` builds `model.new_cache()` + wires InternLM2 forward/truncate
  into `spec_core`; `INTERNLM2_DRAFTER_CFG`(hidden=4096, 32×128 heads, 8192 SwiGLU, rope θ=5e7,
  eps=1e-5, n_feature_layers=3), `DEFAULT_CAPTURE_LAYERS=(8,16,24)/32`. runtime.py: `capture_layers`
  extended to PACKED `_PackedModel.__call__` (post-layer residual [T,H]; byte-unchanged when None) +
  threaded thru `InternLM2ResidentModel.__call__` to both inner paths; NEW `embed_head()` accessor
  (frozen [V,H] embed+head, resolves packed/bf16 + tied/untied). Gate
  `parity/internlm2_eagle_spec_test.py` (model-free, UNTRAINED drafter, `object.__new__` stand-ins +
  tiny `mx.quantize`d `_PackedModel`): **A bf16 spec==greedy** 12-tok mean_accept=1.00 (random drafter
  accepts NOTHING yet output bit-identical — the EAGLE guarantee), **B packed capture** logits Δ=0 +
  recon Δ=0 (true pre-norm residual), **C packed spec==greedy** (full `mx.quantized_matmul` fwd thru
  spec loop). M0 capture gate + batched-attention runtime gate still green (edits additive).
- **M2 ✅ `09306da`** — InternLM2-specific drafter-TRAINING pipeline around the already
  model-agnostic `eagle/capture.py` + `eagle/train.py` core (mirror the MiniMax pair). Two deferred
  GPU-job scripts: `parity/eagle_capture_internlm2.py` (load resident int8g64 7B bake, tokenize the
  raw corpus in InternLM2's **own** SentencePiece space — the shared `corpus_mix` is a foreign vocab,
  wrong ids poison both features AND target-argmax labels — capture (8,16,24) post-layer residuals →
  shards via `internlm2_capture_forward`+`capture_features_to_shards_fn`, one BOS/doc, `MAX_TOKENS`
  bound) → `parity/eagle_train_internlm2.py` (after capture frees base, train `EagleDrafter`
  (`INTERNLM2_DRAFTER_CFG`, steps=4) on shards; frozen embed/head pulled from `InternLM2Artifact.embed`
  /`.lm_head` — resolves packed/bf16+tie/untie, holds ONLY those 2 tensors — the InternLM2 analogue of
  `eagle.train.load_frozen_embed_head`). Real run = solo GPU, capture THEN train, one model resident.
  Gate `parity/internlm2_eagle_train_test.py` (model-free, tiny random model, reuses M1 stand-ins):
  **A** capture→shards→reload feat3 `[256, 3H]`; **B** drafter LEARNS loss 3.19→0.09, holdout accept
  0.036→1.000 (Adam + CE + feat-regression wired, not just shape-correct); **C** train+reload stays
  lossless — `save_drafter`→`load_drafter`→`il2_spec_generate` == greedy, and the TRAINED reloaded
  drafter shows **mean_accept 1.57** (>1.0, accepts drafts) bit-identical. Loss-decrease is the
  asserted learn signal (random-init holdout accept can already sit at the modal token). M0/M1 gates +
  ruff/compileall/lock/pytest green (additive). Reused Kimi findings
  ([[project-eagle3-speculative]]): normalized-feature self-feed, CE + smooth-L1 feature regression,
  LayerScale init ~0 — already baked into `eagle/train.py`/`eagle/drafter.py`, not re-derived.
- **M3 ✅ `ec0f6f3`** — real-model bench on the int8-g64 7B bake (solo GPU) + the **drafter-quantization
  win**. Ran capture (1.08M tok → 9 shards) → train → a **staged-anneal** of the drafter (new
  `parity/eagle_refine_internlm2.py`, SGDR-style: warm-restart from prior best, lower cosine LR +
  grad-clip per stage; lifted holdout accept **0.409 base → 0.448 → 0.460**, sidecars in
  `~/models/internlm2_eagle/`: `drafter_int8g64{,_base}`=0.409, `{_refined,_refined_s2}`=0.448,
  `_refined2`=0.460) → bench (new `parity/internlm2_eagle_spec_bench.py`). **KEY FINDING (generalizable):
  a bf16 EAGLE drafter is a NET SLOWDOWN on a fast quantized target** — k=2 **0.91×** (worse with k)
  despite lossless, because `mean_accept` saturates ~1.81 (deep-step accept caps the chain) AND each
  draft step is dominated by the frozen **92k-vocab head projection run k×/round in bf16**. The lever is
  **quantize the drafter, NOT more training** (training plateaued ~+0.01/stage, ~2h/stage): k=2 bf16
  **0.91× → int8 1.34× → int4 1.42×**, accept HELD (1.76→1.79), **spec==greedy every k**. **k=2 is
  optimal** (accept saturates; higher k only adds draft cost); the **~18ms target verify is the decode
  floor** (free-drafter ceiling ~1.77×), so **shrink+retrain was FORGONE** (user call: bounded gain +
  regression risk if a smaller drafter loses accept). Plumbing (rule 4, **lossless-safe** — the passed
  `head` drives only DRAFT selection; target verifies with its OWN head): `spec_core.spec_generate`
  opt-in `head_bits`/`head_group_size` (quantizes head once, draft loop `mx.quantized_matmul`; **default
  None = byte-unchanged for Kimi/MiniMax/all callers**) → threaded thru `internlm2/eagle.spec_generate`;
  bench `IL2_QUANT_BITS` env = PTQ (`nn.quantize` body + `head_bits`). Repro:
  `IL2_QUANT_BITS=4 uv run python -m parity.internlm2_eagle_spec_bench <…/drafter_int8g64_refined2.safetensors>`.
  Gates: `pytest tests/` + capture/spec/train model-free gates green (additive); ruff/compile/lock/diff
  clean. **EAGLE spec-decode on InternLM2.5 is DONE** (1.42× lossless @ k=2); the MInference prefill
  track (lossy, separate) is now its own milestone series — **M6 ✅, M7 next** ([[project-internlm2-minference]]).

See also [[project-model-targets]] (keeper set), [[project-batched-operating-point]] (B=32 serving),
[[project-internlm2-minference]] (the 2nd InternLM2.5 lever — sparse prefill).
