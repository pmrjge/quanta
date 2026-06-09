# PLAN — Nex-N2-Pro serving runtime + int4/int6 bake (quanta)

**Model.** `nex-agi/Nex-N2-Pro` — a 397B-param MoE (A17B active) agentic-reasoning model, **post-trained
on Qwen3.5-397B-A17B** (`model_type=qwen3_5_moe`, `Qwen3_5MoeForConditionalGeneration`). Apache-2.0.
Local checkpoint `~/models/Nex-N2-Pro` — **739 GiB / 122 shards, bf16**. Recommended sampling temp 0.7
/ top-p 0.95 / top-k 40; served upstream with `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`.

**The reuse win.** This is the *exact* architecture the in-tree **`quanta.qwen35`** module already
targets (its `config.py` header literally reads "Qwen3.5-397B-A17B"). The module is generic, validated
+ baked at 35B scale (the **Qwen3.6-35B-A3B** fleet keeper); **Nex is the 397B sibling — same code,
bigger checkpoint, re-gate at scale** (the Nemotron Super→Ultra pattern). So this is *validate + bake
an existing runtime on a new post-trained checkpoint*, not build-from-scratch.

**Architecture** (confirmed from `config.json` + the on-disk index): 60 layers, **3:1 hybrid**
(`full_attention_interval=4`) = **45 Gated-DeltaNet linear-attention + 15 gated-GQA full-attention**;
hidden 4096; full attn 32 heads / **2 KV** (GQA), head_dim 256, **partial mRoPE 0.25** (rotary 64),
`attn_output_gate`, `rope_theta 1e7`, mrope_section [11,11,10]; linear attn 16 key / 64 value heads,
conv kernel 4, fp32 SSM state; **MoE on all 60 layers**: 512 experts **top-10** + **1 shared** (width
1024); vocab 248320, untied lm_head; **256K native** (`max_position_embeddings 262144`). A ViT
(`model.visual.*`, 333 tensors) — **text-only serving ignores it** (the loader is language-model-only).

## Goal / requirements (user)

1. **Runtime first, then quantization.** Parity-gated at every step (project methodology).
2. **Package int6-RTN or int4-RTN**, chosen by measured quality-vs-speed/VRAM (the e2e-ppl arbiter).
3. **Extendable to 1M context automatically** (dynamic YaRN — already coded in `qwen35`). **The
   generated artifact's `config.json` MUST declare the 1M context** (first-class, not a separate
   runtime flag). ✅ done in N0.
4. **All the optimizations** we apply to every keeper: packed-int4 experts + `gather_qmm`,
   batched/paged decode (#153 loop-kill), **sparse-prefill (MInference) on the 15 full-attn layers**,
   **prefix caching** (paged COW), fused decode-step kernels, multi-stream decode.

## Phased plan

- **N0 — groundwork (model-free / header-only, no 739 GB load). ✅ COMPLETE (this commit).**
- **N1 — layer parity @ 397B (SOLO).** The `qwen35` runtime vs an independent `transformers`
  `Qwen3_5Moe` reference, **layer-streamed** (one real layer resident, rule 8): Gated DeltaNet
  (recurrence/chunk/step), gated-GQA + partial mRoPE + per-head QK-norm, MoE top-10 softmax routing +
  shared expert. This is "runtime first." Confirm `norm_topk_prob`/`scoring_func` against the oracle
  (read with defaults today). Use Nex itself as the 397B vehicle (base Qwen3.5-397B is gone from disk).
- **N2 — bake + bits decision.** RTN is data-free/cheap → **bake int4-g64 AND int6-g64**, teacher-force
  ppl on held-out prose, **pick by the quality-vs-VRAM rule** (int4-RTN was ~lossless +0.3% on
  bf16-source Nemotron-Ultra → strong default; int6 is the safety net). The dynamic-YaRN 1M policy is
  baked into `config.json` (N0). `include_mtp=False` (Nex ships no MTP weights — see N0 finding).
  Artifact `~/models/Nex-N2-Pro-quanta_int4rtn_g64` (and/or int6). Expected resident **int4 ≈ 214 GiB /
  int6 ≈ 304 GiB** (N0 projection).
- **N3 — serving + optimizations.** Resident e2e ppl gate (dequant-ref parity); the **`qwen3_coder`
  tool parser** (XML `<tool_call><function=…><parameter=…>` — NEW, the shim's JSON parser doesn't fit)
  + `qwen3` reasoning parser (account for the template's pre-opened `<think>`); the **1M long-doc /
  needle gate** (the YaRN arbiter); packed-int4 `gather_qmm` experts; **paged-KV + prefix caching**
  (only the 15 full-attn layers hold a KV cache — the 45 linear layers are O(1) recurrent state, so 1M
  KV is ~4× cheaper than a dense model); **MInference sparse-prefill** on the full-attn layers
  (InternLM2.5 M0–M10 substrate transfers); **fused/batched Gated-DeltaNet decode step** (the Nemotron
  `BATCHED_FUSED_SSD_STEP` win, +36% @ B=32, applied to `gdn_step`); multi-stream batched decode.
  **Native-MTP spec-decode is N/A for Nex** (no MTP weights) — an EAGLE-style external drafter is the
  only B=1 latency path if wanted later.

## N0 — what landed (this commit)

Three model-specific fixes (all in `quanta.qwen35`, additive — the 35B keeper path is unchanged) +
two gates. The N0 fit-test **caught two real divergences** between Nex and the 35B-verified contract:

1. **EOS stop-set (rule 6).** Nex ships **no `generation_config.json`**, and `config.json` lists only
   `eos_token_id: 248044` (`<|endoftext|>`, a doc separator — never ends a chat turn). `from_pretrained`
   now derives the real ChatML stop set **{248046 `<|im_end|>`, 248044 `<|endoftext|>`}** from the
   tokenizer's `added_tokens` when no `generation_config` is present (canonical fallback if even that
   is unresolvable; the 35B path with a `generation_config` is byte-unchanged). The **bake synthesizes**
   a correct `generation_config.json` into the artifact when the source lacks one (was: hard-refuse).
2. **1M in the artifact config (the explicit requirement).** `_bake_long_context` now writes **standard
   HF YaRN** (`rope_type=yarn` / `factor=4` / `original_max_position_embeddings=262144`) on
   `rope_parameters` + a mirrored `rope_scaling`, **raises `max_position_embeddings` to 1,010,000**, and
   keeps the `quanta_long_context` block. `from_pretrained` reads `yarn_original_max` from
   `rope.original_max_position_embeddings` — **DECOUPLED** from `max_position_embeddings` — so the served
   window declares 1M while the **dynamic-YaRN baseline stays 262144** (`eff@8k=1.0`, `eff@1M=3.85`).
   The artifact is a first-class 1M model for *any* loader, mRoPE preserved alongside YaRN.
3. **MTP absent.** Nex **declares `mtp_num_hidden_layers=1` but ships ZERO `mtp.*` weights** (post-train
   dropped the head). `from_pretrained` refines `num_mtp_modules → 0` by index presence (rule 6: trust
   the weights). ⇒ **native-MTP spec-decode is unavailable for Nex**; the bake runs `include_mtp=False`.

**New:** `quanta/qwen35/quant_policy.py` — key→scheme map (`dense`/`int8`/`expert_int4`) built from the
**bake's actual suffix partition + the loader's enumeration** (single source of truth, can't drift) +
an analytic resident projection. **Coverage (rule 6): 1038 text tensors → 453 dense / 465 int8 / 120
expert_int4** (+333 vision excluded); the expected keymap EXACTLY equals Nex's on-disk index (loader
contract holds at 397B).

**Fit (real header shapes, 490.4 GiB ceiling):** **int4-g64 = 214.1 GiB** (experts 202.5 + int8 7.6 +
dense 4.0; 276 GiB headroom) / **int6-g64 = 304.1 GiB** (186 GiB headroom) — **both fit**. Source text
bf16 738.3 GiB (739.1 on-disk, header-accounted within 1%).

**Gates.** `parity/nex_n2_pro_fit_test.py` (real-path, SOLO/excluded from the sweep; reads index +
safetensors headers only, no tensor materialized; PARITY-CHECKS 11). `parity/qwen35_config_eos_yarn_test.py`
(model-free, in the sweep; 21 checks over synthetic temp-dir checkpoints — eos derivation, the 1M bake
round-trip, MTP refine). Manifest regenerated (99 model_free / 51 real_weight).

**Cadence:** single thread, NO subagents, commit each milestone, then STOP for the user to compact.
