# PLAN — Nemotron-3-Ultra-550B (main agent) + Mellum2-12B (orchestrator)

A two-model agentic stack on one M3 Ultra (≤ 490.4 GiB), quantized through `quanta`'s
parity-first pipeline.

- **Main agent:** `NVIDIA-Nemotron-3-Ultra-550B-A55B` — hybrid Mamba2 + attention + MoE
  (`model_type: nemotron_h`) → **int4-RTN g64 experts + int8 dense + bf16 core** (U3 ✅ — RTN beat AWQ e2e).
- **Orchestrator:** `JetBrains/Mellum2-12B-A2.5B-Thinking` — sliding-window + full-attn MoE
  (`model_type: mellum`) → **int8 (AWQ)**.

## Decisions (user, this session)

1. Nemotron experts **int4-AWQ g64** — user pivot this session. NB the earlier "int4-GPTQ, already
   baked on Super" premise was **wrong**: `bake_nemotron` only implements AWQ/RTN (no GPTQ path is
   wired into the Nemotron bake), and Super actually shipped plain int4 **RTN** (manifest: `awq_packed`,
   s=1). Finding #38 had flagged AWQ as +75% e2e on the relu² down-proj, but the U2 slice de-risk
   (`parity/nemotron_ultra_awq_slice_test.py`) shows that collapse does **not** reproduce at Ultra scale
   (AWQ helps up-proj 0.806 / ties down-proj 0.984; the α-grid rejects the degenerate scales). **U3
   RESOLVED → int4-RTN ships:** at the 1024-token teacher-forced arbiter AWQ regressed **+24.3%** (recon
   mispredicted — recon ≠ e2e) while RTN held **+0.3%**, so finding #38 reproduced e2e and the RTN
   fallback is the shipped expert method (AWQ retired).
2. Mellum **int8 (AWQ)** — user choice. int8-from-bf16 is near-lossless, so AWQ here is
   belt-and-suspenders (harmless; `bake/awq.py` exists).
3. **One model resident at a time** — honors the OOM-safety rule; the agentic loop swaps
   main/orchestrator. No concurrent-resident budget needed now.
4. **Drive Nemotron-Ultra to completion first**, then Mellum.

## Key facts (authoritative — from on-disk `config.json`)

**Nemotron-Ultra** (`nemotron_h`): **108 layers = 48 mamba / 48 moe / 12 attention**; hidden 8192;
GQA 64 Q / 2 KV, head_dim 128; **512 routed experts, top-22**, 1 shared; relu² **latent**-MoE
(latent 2048, inter 5120, shared-inter 10240), routed_scaling 5.0; Mamba2 (256 heads, head_dim 64,
state 128, conv 4, n_groups 8, chunk 128, expand 2); RoPE θ=1e4, partial_rotary 1.0; **native MTP
head** (`num_nextn_predict_layers=1`) for spec-decode; ctx 262144; vocab 131072; stop set **{2, 11}**
(from `generation_config.json`). Ships the *newer* config schema: an explicit `layers_block_type`
list, **no** `hybrid_override_pattern` / `num_hidden_layers`.

> Already supported: `src/quanta/nemotron/` implements the whole family (`mamba_ssd`/`mamba_mixer`,
> `attention`, latent `moe`, `mtp`, `calibrate`, `routing_capture`, `bake`, `batched_runtime`,
> `spec`). The **120B-Super sibling is already baked int4** at
> `~/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64`. Ultra is a config-driven scale-up.

**Mellum2** (`mellum`): 28 layers, hidden 2304; GQA 32 Q / 4 KV, head_dim 128; **64 experts top-8**
(SwiGLU, moe_inter 896); **sliding-window (1024) on 3 of every 4 layers + full attention every 4th**;
**dual RoPE** — full-attn layers YaRN (θ=5e5, factor 16, orig 8192, β_fast 32 / β_slow 1, attn_factor
1.2772588722), sliding-attn layers default RoPE (θ=5e5); ctx 131072; vocab 98304; thinking model
(`<think>…</think>`, qwen3 reasoning-parser, hermes tool-call). **No module yet** — a genuine new
port; closest template `src/quanta/qwen35/`.

## Memory (one-at-a-time)

- Ultra **int4-RTN g64** mix **306 GiB resident** (U3-shipped; `du` of the baked artifact: int4 routed
  + int8 dense + bf16 core — **30 GiB under the retired AWQ 336**, since RTN stores bf16 vs AWQ's fp32
  expert scales). Headroom **184 GiB** for KV + activations. (NB the U0 fit projection of 289.7 GiB
  under-counted — it tracked the routed int4-g64 portion only; reconcile `nemotron_ultra_fit_test.py`,
  non-blocking since 306 ≤ 490.4 fits.) Only **12 / 108** layers carry growing KV — the 48 Mamba layers
  have **O(1)** state (a real long-context win at 256K).
- Mellum int8 ≈ 11.5 GiB.

## Roadmap

### Nemotron-Ultra
- **U0 ✅ — config adapter + fit-check.** `NemotronHConfig.from_pretrained` now normalises both
  checkpoint schemas via `_hybrid_pattern` (compact letter string **or** explicit
  `layers_block_type` list). Gate `parity/nemotron_ultra_fit_test.py`: Ultra parses, derived split
  reproduces the explicit list bit-for-bit, **quant policy covers all 51,023 tensors** (rule #6),
  and the mix **fits 289.7 GiB ≤ 490.4** (200.7 GiB headroom). Super (old schema) backward-compat
  green. Files: `src/quanta/nemotron/config.py`, `parity/nemotron_ultra_fit_test.py`.
- **U1 ✅ — per-layer numeric parity vs an independent transformers `NemotronH*` reference**, at full
  Ultra scale, layer-streamed (rule 8: one real layer resident; the moe's ~21.5 GiB bf16 expert stacks
  the peak — the 1023 GiB whole model is never loaded, and the transformers MoE's 512 experts stay on
  the `meta` device for a router-only cross-check). `parity/nemotron_ultra_layer_parity.py`:
    - **mamba** our `MambaMixer` prefill vs `NemotronHMamba2Mixer` (naive CPU path), fp32 — **Δ 3.1e-04**;
    - **attn** our `NemotronAttention` (naive) vs transformers' own `apply_rotary_pos_emb` +
      `eager_attention_forward` + o_proj (rope θ=10000, GQA 64/2), fp32 — **Δ 4.5e-06**;
    - **moe** router top-22 **set + weights** vs `route_tokens_to_experts` — **set-exact, w Δ 1.2e-07**
      (our `noaux_tc` sigmoid+bias routing is provably exact); experts/latent/shared vs an inline dense
      per-token/per-expert reference — **Δ 7e-04**; token-chunk invariant (Δ 0). transformers/torch are
      reference-only (offline, rule #5).
    - **BUG CAUGHT (the parity-first payoff):** the Mamba-2 **gated RMSNorm is group-wise** — variance
      over `d_inner // n_groups` channels (`Zamba2RMSNormGated`, `group_size = intermediate_size //
      n_groups`), **not** the full `d_inner`. Our mixer used a full-width `nn.RMSNorm`: *self-consistent*
      (prefill==decode 1.2e-06) so the old self-consistency-only `nemotron_layers_test` never caught it,
      but **42% off** the transformers reference. Fixed with a new `MambaRMSNormGated` (group-wise, fused
      `mx.fast.rms_norm` per group, weight after) in `src/quanta/nemotron/mamba_mixer.py`. **Forward-only**
      — the bf16 `norm.weight` is unchanged, so it also corrects the **already-baked Super-120B** with no
      re-bake (Super ppl/quality should be re-measured under the fix; it was previously measured buggy).
  Files: `parity/nemotron_ultra_layer_parity.py`, `src/quanta/nemotron/mamba_mixer.py`.
  > Note: `nemotron_layers_test.py`'s *attention* prefill==decode assertion (2e-3) is pre-existing-stale
  > vs the int8 `KVCache` default (#133) — ~5.3e-3, unrelated to U1; flagged for a separate cleanup.
- **U2 de-risk ✅ — AWQ slice diagnostic.** `parity/nemotron_ultra_awq_slice_test.py` streams Ultra
  layers 0–1 (layer 1 = first MoE; NO 21.5 GiB expert stack materialized — gate+fc1 only) and runs, per
  warm expert, a **held-out** activation-weighted recon test (fit the AWQ scale on 70% of the expert's
  routed rows, measure error on the held-out 30%) for AWQ vs RTN. Result: finding #38's relu² down-proj
  AWQ collapse does **not** reproduce at Ultra — AWQ *helps* up-proj (ratio 0.806) and *ties* down-proj
  (0.984, 23/24 experts AWQ≤RTN); relu² channel sparsity 99.74% (the #38 precondition) is present but
  AWQ's α-grid rejects the degenerate scales (range ≈1, not ≈1e6). Caveat: L1-only + activation-weighted
  recon (not e2e ppl). **AWQ cleared.**
- **U2 ✅ — full int4-AWQ g64 + int8 bake.** `parity/run_bake_nemotron_ultra_int4awq_g64.py` drove
  `bake_nemotron(..., expert_method="awq", group_size=64, scale_dtype=bf16)` layer-streamed (rule 8) over
  ~4K agentic-corpus calib tokens (capture per-MoE latent+routing → α-grid each expert's up/down),
  **0.48h solo** → `~/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-quanta_int4awq_g64`. Stats: 108 layers /
  48 moe / 512 experts-per-layer, **warm_experts 24,235 / 24,576 (98.6%)** got real AWQ scales; the 341
  cold experts → plain int4 RTN (s=1, one runtime path). **Artifact audited self-contained + fully
  covered**: no symlinks, zero external refs in index/manifest/config, all weight_map relative, **42/42
  shards**, tokenizer in-artifact, manifest `format=quanta` 49,983 tensors; coverage = all 108 layers,
  512 up + 512 down experts/moe-layer, embeddings/lm_head/norm_f present. **Resident 336 GiB** (≤490.4,
  154 GiB headroom). RTN (`expert_method="rtn"`) the known-good fallback if U3 ppl regresses.
- **U3 ✅ — teacher-forced ppl + top-1, the AWQ-vs-RTN e2e arbiter → SHIP int4-RTN.**
  `parity/nemotron_ultra_ppl.py` ran three sequential rule-8 streamed forwards (bf16 → int4-AWQ →
  int4-RTN, each freed before the next so one is resident) over a held-out **1024-token** prose corpus
  (≈10× the noisy 109-tok pilot; original expository text, held out from the agentic calib set):
  **bf16 ppl 3.835 / acc 0.651**; **int4-AWQ ppl 4.766 / acc 0.604 / Δ +24.3% / agree 0.811**;
  **int4-RTN ppl 3.845 / acc 0.644 / Δ +0.3% / agree 0.964**. RTN is ~lossless; **AWQ regresses hard** —
  **finding #38 reproduced e2e** (the relu² down-proj AWQ tax got *worse* with more tokens, +11.2%→
  +24.3%; the U2 slice de-risk's "AWQ ties/helps" was recon-only + L1-only, and recon does NOT predict
  e2e — settled finding). **Shipping int4-RTN** (`expert_method="rtn"`): clears the gate (Δ +0.3% < 5%,
  agree 0.964 > 0.90), same 4-bit footprint, **306 GiB resident (30 GiB < the AWQ 336** — RTN stores
  bf16 expert scales vs AWQ's fp32; `awq_quantize` doesn't forward `scale_dtype`). AWQ retired for
  Nemotron experts. Bake `parity/run_bake_nemotron_ultra_int4rtn_g64.py` (data-free experts, **0.10h
  solo**, warm_experts 0 = the RTN signature; audited inventory-identical to AWQ — 198,111 index keys,
  49,983 manifest tensors, format=quanta, 39 shards, tokenizer in-artifact). Files:
  `parity/run_bake_nemotron_ultra_int4rtn_g64.py`, `parity/nemotron_ultra_ppl.py`.
- **U4 — optimizations**, each behind a flag and ppl-equivalent: native **MTP spec-decode**
  (`spec.py`/`mtp.py`), **paged-KV** on the 12 attn layers (port #153 loop-kill), **packed int4
  experts + `gather_qmm`** (the resident decode path — already coded in `moe.py`/`runtime.py`,
  built+gated for Super-120B, now validated at Ultra scale on the RTN artifact), batched decode +
  Mamba-state batching (`batched_runtime.py`). MInference sparse-prefill only if long-ctx attn-layer
  prefill proves a bottleneck (just 12 layers). **Stream chosen first (user): packed int4 + gather_qmm.**
  - **U4/M1 ✅ — resident-MoE numeric parity @ Ultra.** `parity/nemotron_ultra_qmoe_test.py`:
    `NemotronQuantizedMoE` (gather_qmm over packed int4-g64 stacks, built by the *real* runtime
    constructor `build_resident_block(art, cfg, 1).mixer`) vs `NemotronLatentMoE` (gather_mm on the
    artifact's dequantized weights), real Ultra L1 (512 experts, latent 2048, inter 5120), rule-8
    (~5.4 GiB packed + ~21.5 GiB bf16 ref). **rel err 0.0282% « 2% gate** — the packed-int4 decode
    path is output-equivalent to dequant (RTN ⇒ s=1, no AWQ rescale; gather_qmm decodes the same
    grid). Mirrors the Super `nemotron_qmoe_test` gate at Ultra scale + the shipped RTN artifact.
  - **U4/M2 ✅ — full-resident e2e ppl @ Ultra.** `parity/nemotron_ultra_resident_ppl.py`: load
    `NemotronResidentModel` over the **306 GiB RTN artifact** RAM-resident (solo, 400 GiB wired —
    load 1.9 min, peaks ~306 GiB, freed clean) and teacher-force the **same** U3 1024-tok `LONG_PROSE`
    corpus (metric `_ppl_acc` imported verbatim, so directly comparable). **ppl 3.839 / acc 0.646**
    vs the U3 streamed-dequant RTN reference **3.845 / 0.644** — **Δ −0.1% « 2% gate, PASS** (the
    −0.006 is the resident bf16-head vs streamed fp32-head difference, within noise; forward 11.3s).
    Closes the packed-int4 + gather_qmm stream **end-to-end**: M1 gated the MoE at one layer; M2 runs
    the whole 108-layer resident model, so it also covers the **dense mamba/attn int8
    `QuantizedLinear` wiring** end-to-end. The resident gather_qmm / int8-QuantizedLinear forward is
    output-equivalent e2e to the dequant reference at full Ultra scale.
  - **U4 / MTP spec-decode — native MTP self-speculation** (user-picked next stream; #40). `mtp.py` /
    `spec.py` (draft head + lossless k≥1 / chained / tree / batched verify) and the model-free
    `nemotron_mtp_spec_test` were already built (for Super), but the head was never baked/loaded.
    - **MTP-M0 ✅ — native MTP draft-head bf16 numeric parity @ Ultra.**
      `parity/nemotron_ultra_mtp_parity.py`: build `NemotronMTPModule` (fuse
      `eh_proj(concat([enorm(embed), hnorm(prev_hidden)]))` → attn sub-block `mtp.layers.0` →
      512-expert relu² latent-moe `mtp.layers.1` → final_layernorm → shared head), fill from the
      source's **1040 `mtp.*` tensors** (rule-6 coverage 1040/1040), diff vs an independent inline
      reference (raw-mx fusion/pre-norms/residuals/readout + U1-gated standalone `NemotronAttention` /
      `NemotronLatentMoE`): **logits Δ 0.0 / new_hidden Δ 0.0 (bit-identical)**. Rule-8 streamed (the
      512-expert ~21.5 GiB bf16 stack the peak, solo). Gates the head's *structural assembly*; the
      *functional* accept-rate is the separate MTP-M2 gate (losslessness holds for any head quality —
      the main model verifies every draft, rule 4).
    - **MTP-M1 ✅ — bake the head into an int4-RTN sidecar + recon gate.** New `bake_nemotron_mtp`
      (`bake.py`) bakes the head as a self-contained **sidecar** bundle
      `…-quanta_int4rtn_g64_mtp` (driver `parity/run_bake_nemotron_ultra_mtp_int4rtn_g64.py`) — same
      policy as the backbone (int4-RTN experts + int8 dense + bf16 core; `quant_policy` already
      classifies `mtp.*`), its own bundle so the immutable backbone artifact is untouched (M2's loader
      pairs the two). Streamed one expert resident (rule 8, **no 21.5 GiB stack**; 0.08 min, data-free
      RTN warm 0) → **1040/1040** tensors, single 6.56 GiB shard, audited self-contained (zero path
      leaks, relative refs, manifest **9 int8 / 7 bf16 / 1024 int4**). Gated solo
      `parity/nemotron_ultra_mtp_bake_parity.py` (two 21.5 GiB heads loaded **sequentially**, peak one):
      (1) coverage+format exact vs `classify` (1040/1040), (2) **bit-exact faithfulness** — an
      independent in-script RTN `quantize_affine` reproduces the baked packed/scale/bias **bit-for-bit**
      (eh_proj int8 + experts 0/256/511 int4; awq_scale==ones ⇒ s=1), (3) **recon forward** baked-dequant
      vs bf16 head through the *identical* M0-gated `NemotronMTPModule` (bf16 router ⇒ routing identical ⇒
      delta is pure quant): **logits Δ 7.0% / new_hidden Δ 7.8% < 10%, top-1 agree 0.875** (the inherent
      int4-g64 expert recon — the bit-exact gate is the tight proof; recon is bounded, and a *drafter*
      moves only accept-rate, never correctness).
    - **MTP-M2 next** — loader (`NemotronMTP` filled from the baked sidecar `mtp.*`, a
      `build_resident_mtp` mirroring `build_resident_block`) + resident spec-contract adapter on
      `NemotronResidentModel` (`offset` / `make_caches` / `truncate`); real lossless accept-rate gate:
      `spec_generate(_k)` output == greedy on real prose, report mean_accept + decode speedup for
      k ∈ {1, 2, 3}.
  - **U4 remaining streams** (each behind a flag, ppl-equivalent, not started): paged-KV on the 12 attn
    layers, batched decode + Mamba-state batching.

### Mellum2 (after Ultra)
- **M0** — new `src/quanta/mellum/`: config + reference forward (dual-RoPE per `layer_types` +
  sliding-window mask). Template `qwen35`.
- **M1** — layer-by-layer numeric parity vs `transformers` `MellumForCausalLM`.
- **M2** — int8 (AWQ) bake → `~/models/Mellum2-12B-A2.5B-Thinking-quanta_int8g64`.
- **M3** — teacher-forced ppl + top-1 vs bf16.
- **M4** — orchestrator integration: `<think>` parsing, hermes tool-calls, stop on eos=0.

### Stack
- Two-model agentic loop, one-at-a-time residency (swap main↔orchestrator); measure swap latency.
  If swapping is too slow for the loop, revisit concurrent-resident (~301 GiB; needs a measured
  `mx.set_wired_limit` budget — a deviation from the one-at-a-time rule).

## Cadence (standing)
Single thread, **no subagents**; implement → gate green → commit named files (trailer
`Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`) → **STOP** for the user to
compact. One model resident at a time. Keep `~/models/Kimi-K2.6`. The InternLM2.5 MInference **M7 is
paused** (handover preserved in `PLAN_minference.md`), not abandoned.
