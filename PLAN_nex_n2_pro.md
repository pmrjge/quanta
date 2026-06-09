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

- **N0 — groundwork (model-free / header-only, no 739 GB load). ✅ COMPLETE.**
- **N1 — layer parity @ 397B (SOLO). ✅ COMPLETE.** The `qwen35` runtime vs an independent
  `transformers` `Qwen3_5Moe` reference (transformers 5.9.0 ships `qwen3_5_moe`), **layer-streamed**
  (one real layer resident, rule 8), `parity/nex_n2_pro_layer_parity.py`: **deltanet** our
  `GatedDeltaNet` prefill vs `Qwen3_5MoeGatedDeltaNet` (pure-torch `torch_chunk_gated_delta_rule`
  fallback — no FLA) **Δ 1.95e-06** + prefill==decode 1.44e-06; **attn** our `Qwen35Attention` vs
  `Qwen3_5MoeAttention` (eager + partial-mRoPE rope + doubled-`q_proj` sigmoid output gate + per-head
  `(1+w)` q/k norm) **Δ 2.10e-06** + fast==naive 7.5e-08 + prefill==decode 4.8e-07; **moe** router
  top-10 **set-exact** (softmax + `norm_topk_prob` renorm — confirmed against the oracle, NOT
  DeepSeek sigmoid/noaux_tc) w Δ 4.9e-07 + experts/sigmoid-shared vs inline-dense 1.55e-03 + chunk Δ
  **0.0**; **block** our full `Qwen35Block` vs `Qwen3_5MoeDecoderLayer` (the end-to-end gate that
  exercises the `Qwen3_5MoeRMSNorm` **`(1+w)`** input/post norms + residual wiring + mixer dispatch)
  **linear L0 Δ 1.50e-06 / full L3 Δ 1.90e-06**. All fp32 cross-impl at machine precision — the
  whole forward path is correct at 397B (no forward bug surfaced; the qwen35 code was already correct
  from the 35B keeper, so N1 is the at-scale re-gate — the Super→Ultra pattern). The `(1+w)` fold
  lives in `runtime.py:_one_plus` (layer/q/k/final norms, NOT the gated-DeltaNet norm).
- **N2 — bake + bits decision. ✅ COMPLETE → SHIP int4-g64.** RTN is data-free/cheap; both arms baked,
  the e2e-ppl arbiter decided.
  - **int4-g64** → `~/models/Nex-N2-Pro-quanta_int4g64` (`parity/run_bake_nex_n2_pro_int4g64.py`,
    2.7 min): **214.1 GiB / 25 shards**, 60 layers / 512 experts, counts {int8 465, expert_int4 120,
    dense 453} (== N0 quant-policy projection exactly), MTP excluded (`include_mtp=False`).
  - **int6-g64** → `~/models/Nex-N2-Pro-quanta_int6g64` (`parity/run_bake_nex_n2_pro_int6g64.py`,
    3.2 min, `expert_bits=6`): **304.1 GiB / 31 shards**, SAME counts {int8 465, expert_int4 120
    (now int6), dense 453} (== the N0 int6 projection 304.1 GiB exactly). The bake gained an
    `expert_bits` knob (`bake.py`, default 4; threaded through `_bake_moe_block`/`_bake_mtp`/
    `_write_expert_stack`) — the int4 path is byte-identical, int6 is the same recipe at a wider grid
    (MLX affine {2,3,4,6,8}; `Qwen35Artifact`/`gather_qmm` decode at the manifest width, never a
    hardcoded 4).
  - **Both artifacts are self-contained (the user's rule, now ENFORCED as code).** `bake_qwen35` ends
    with `_audit_self_contained` (rule 6, fail-loud): no symlinks, required sidecars present
    (config/manifest/index/synthesized `generation_config.json` eos `[248046,248044]`/tokenizer), no
    path leak in any json metadata, relative weight_map, all shards present. Both **config declare the
    1M window** (`max_position_embeddings 1,010,000` + standard HF YaRN, dynamic-YaRN baseline 262144).
    Family-consistent names `_int4g64`/`_int6g64` (the Qwen3.6-35B keeper convention; `rtn` was
    Nemotron-only to disambiguate its AWQ artifact — qwen35 has no AWQ path).
  - **ppl arbiter** (`parity/nex_n2_pro_ppl.py`, SOLO; 3 sequential streamed forwards over the SAME
    645-tok held-out prose via the proven `_load_block(packed=False)` reference path — bf16 source /
    int4 dequant / int6 dequant, one block resident at a time, rule 8): **bf16 ppl 5.0386 / acc
    0.5590** (low-single-digit on real prose — the forward is e2e-coherent at 397B, the project
    thesis), **int4 5.0729 / acc 0.5559 / Δ +0.68% / agree 0.9472**, **int6 5.0237 / acc 0.5590 / Δ
    −0.30% / agree 0.9550**. **int4-RTN is ~lossless** (+0.68% ppl, −0.3% acc) — the Nemotron-Ultra
    finding (int4-RTN +0.3% on a bf16 source) reproduces; int6 (−0.30%, within noise) recovers <1pp
    for +90 GiB. teacher-forced ppl is THE arbiter (methodology #4); top-1 agreement ~0.95 is the
    *secondary* signal (noisy on prose — bf16-ULP near-tie flips at low-confidence positions, a settled
    finding), a >0.90 sanity floor not a tight gate. **Decision: SHIP int4-g64** (214 GiB, ~lossless,
    90 GiB lighter than int6).
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
