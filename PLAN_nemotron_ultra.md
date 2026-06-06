# PLAN — Nemotron-3-Ultra-550B serving runtime

A single-model serving/runtime track on one M3 Ultra (≤ 490.4 GiB), quantized through `quanta`'s
parity-first pipeline. The eventual second agentic-stack model is **deferred to MiniMax-M3 when it
ships** — **Mellum2 was dropped (its context length is too short)**; the `minimax` module is already
substantially ported in-tree.

- **Main model:** `NVIDIA-Nemotron-3-Ultra-550B-A55B` — hybrid Mamba2 + attention + MoE
  (`model_type: nemotron_h`) → **int4-RTN g64 experts + int8 dense + bf16 core** (U3 ✅ — RTN beat AWQ e2e).
- **Second model (deferred, when available):** MiniMax-M3 as the orchestrator — **not** Mellum2.

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
2. **Second model deferred to MiniMax-M3** (when released) — **Mellum2 dropped** (its context length is
   too short for the orchestrator role). The `minimax` module is already substantially ported in-tree.
3. **One model resident at a time** — honors the OOM-safety rule; the agentic loop swaps
   main/orchestrator. No concurrent-resident budget needed now.
4. **Drive Nemotron-Ultra to completion first**; the second model follows once MiniMax-M3 ships.

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

**Second model — MiniMax-M3 (deferred).** To be integrated as the orchestrator when released; the
`src/quanta/minimax/` module is already substantially ported in-tree. **Mellum2 was dropped** — its
context length is too short for the orchestrator role.

## Memory (one-at-a-time)

- Ultra **int4-RTN g64** mix **306 GiB resident** (U3-shipped; `du` of the baked artifact: int4 routed
  + int8 dense + bf16 core — **30 GiB under the retired AWQ 336**, since RTN stores bf16 vs AWQ's fp32
  expert scales). Headroom **184 GiB** for KV + activations. (NB the U0 fit projection of 289.7 GiB
  under-counted — it tracked the routed int4-g64 portion only; reconcile `nemotron_ultra_fit_test.py`,
  non-blocking since 306 ≤ 490.4 fits.) Only **12 / 108** layers carry growing KV — the 48 Mamba layers
  have **O(1)** state (a real long-context win at 256K).
- Second-model (MiniMax-M3) footprint TBD once weights ship.

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
    - **MTP-M2 ✅ — native MTP spec-decode wired into the resident loop + real lossless gate.**
      (1) Loader `build_resident_mtp` (`runtime.py`) fills `NemotronMTP` from the sidecar `mtp.*` —
      packed-int4 experts via `gather_qmm` + int8 dense `QuantizedLinear` + bf16 core, mirroring
      `build_resident_block`. (2) Resident spec adapter on `NemotronResidentModel`: `make_caches` (the
      `(caches, ssm, conv)` triple, `max_rollback=8`), `truncate` (KV only; the Mamba `(ssm,conv)`
      summary can't be sliced — the spec loop owns it), `offset` (accepted-and-ignored; the KV cache
      tracks position). (3) **The gate caught a real k=1 hybrid bug:** `spec_generate` (k=1) never rolled
      the un-sliceable Mamba recurrence back on a *rejected* draft (only `spec_generate_k` k≥2 did) — so
      k=1 corrupted `(ssm,conv)` on the hybrid. Fixed: on reject, snapshot/restore `(ssm,conv)` + re-run
      `[cur]` (gated on `ssm is not None`; the stub/pure-attention path is byte-identical). Gated
      **bit-exact model-free** (`nemotron_mtp_spec_test` gate 7: a Mamba-carrying stub whose argmax
      depends on a running recurrent state, so a non-rolled-back rejected draft would diverge — spec ==
      greedy with the rollback branch firing). **Real gate**
      (`parity/nemotron_ultra_mtp_resident_spec.py`, solo, 306 GiB backbone + 6.56 GiB sidecar resident,
      eager, 64-tok prose → 48 tok): **k=2/k=3 EXACT (bit-identical 48/48)**; **k=1 bit-identical 24/48
      then a confirmed bf16 ULP near-tie** (spec's token is greedy's **rank-2** runner-up, **margin
      0.125 ≈ 1 ULP** on greedy's own step path) after which the two *valid* greedy trajectories
      chaos-diverge. mean_accept **1.52/2, 1.81/3, 1.81/4**. **Settled finding:** on a bf16 Mamba hybrid
      the spec VERIFY forward (T>1) and a T=1 decode differ by ~1 bf16 ULP (`path_ulp`=0.1875 — attention
      `mask=None`-vs-`causal` + the recurrence), so **"spec == T=1 greedy" is the wrong real-weight
      criterion** (a single near-tie flip cascades chaotically) — the gate verifies the logic is bit-exact
      (gate 7) + the **first** divergence (the only valid-prefix position) is a near-tie; a
      large-margin/low-rank first divergence FAILS as a logic bug.
    - **MTP-M3 ✅ — perf: wall-clock spec-vs-greedy on the real head.** Re-pointed
      `parity/nemotron_mtp_k_bench.py` to `build_resident_mtp` + the Ultra backbone (solo, ~313 GiB
      wired); loads the model + baked sidecar once and sweeps both runtime speed levers —
      `draft_topk ∈ {2,4,8,full(22)}` × `k ∈ {1,2,3}` — against the production **compiled** greedy
      baseline, with economics probes (`t_main` / `t_verify` / `t_draft`) printed so a sub-1× result is
      actionable. **Result:** single-stream B=1 lossless spec tops out at **0.79× greedy**
      (`draft_topk=8 k=1`, 8.9 vs 11.2 tok/s, mean_accept 1.60/2); full sweep **0.44–0.79×**, k=1 best at
      every topk. Lands in the pre-stated 0.5–0.8× band — but the probes **refute the assumed cause**: the
      512-expert draft is *not* the dominator (`t_draft ≈ 5 ms` flat across draft_topk « `t_main` 88.9 ms
      ⇒ `draft_topk` is near-inert as a *speed* lever; it only moves accept quality 1.45→1.60 at k=1). The
      tax is the **compiled-decode asymmetry** — greedy runs the compiled T=1 fused mamba/moe graph
      (88.9 ms/tok) but spec's T=k+1 verify falls to **eager** (`t_verify` 1.54/1.94/2.33× t_main at
      T=2/3/4) — plus the hybrid partial-reject 2nd main forward (the un-sliceable Mamba `(ssm,conv)`
      re-run, ≈0.4×t_main/round); together they outweigh the 1.60-tok/round amortization (a closed-form
      `(t_verify + t_draft + reject·t_main)/mean_accept` predicts the measured sweep to ~1%). Reproduces
      M2 exactly (full-topk k=1 first-diverges at 24/48 — the bf16 ULP near-tie — else 48/48; the bench
      reports `match` as INFO, never asserts: M2 owns the losslessness proof). **>1× at B=1 needs a
      compiled T>1 verify graph; serving throughput needs the already-built batched (B>1) tree-verify**
      (`spec_generate_tree` / `batch_verify` / `NemotronBatchedResidentModel`) — the MTP-M3-perf
      follow-ups.
    - **MTP-M3-perf (B) ✅ — bf16-drafter quality-ceiling counterfactual.**
      `parity/nemotron_mtp_bf16_drafter_bench.py` (solo, ~330 GiB: the int4-RTN backbone **unchanged** +
      the **un-quantized bf16 source `mtp.*` head**, built via M0's `_mtp_tensors`/`_fill_module` into a
      default bf16 `NemotronMTP` — *not* a dequantized int4 head; dequantizing the sidecar would only
      return the lossy int4 values, so we load the real bf16 source weights). Re-runs the IDENTICAL M3
      economics + `draft_topk × k` sweep with the int4 numbers printed side-by-side (Δaccept). **Result:**
      the perfect-quality drafter tops out at **0.79× greedy** (8.8 tok/s, `draft_topk=8 k=1`) — *tied*
      with the int4 head's 0.79× and **below** the predicted 0.88–1.26× band. Δaccept(bf16−int4) ≈ **0**:
      +0.00 at `draft_topk ≥ 4` (bit-identical accept 1.50 / 1.60 / 1.81), only +0.10 at the degenerate
      `draft_topk=2`. So the int4 quantization tax on accept-rate is **negligible** — the int4-RTN drafter
      already drafts as well as the bf16 ceiling for this workload (M1's 12.5% top-1 logit disagreement
      lands on low-confidence positions that don't dominate accepted-token mass; `t_draft(bf16)`
      5.5–6.6 ms is even slightly *higher* than int4's ~5 ms, still « `t_main` 88.8 ms). Together with M3's
      *lighter*-drafter direction (worse via accept), this **brackets the drafter as near-inert at B=1**
      from both sides and confirms the **compiled T>1 verify graph (part A)** as the sole B=1 lever left to
      test. Losslessness unaffected (M2 — the int4-RTN main model verifies every draft; `match`/divergence
      reported as INFO, never asserted).
    - **MTP-M3-perf (A) ✅ — compiled the T>1 spec-VERIFY graph.** New
      `NemotronResidentModel.compile_verify` (default off → eager, byte-unchanged, rule 4) routes the T>1
      verify *continuation* through the SAME compiled fused mamba/moe mixers as the T==1 decode; the guard
      fires only when a Mamba `conv` is populated (a continuation) and never on fresh or chunked-suffix
      prefill, so prefill stays eager/byte-identical and `mx.compile` auto-keys a fused trace per T (=k+1).
      Gated **output-equivalent** (`parity/nemotron_ultra_compiled_verify_parity.py`, solo ~306 GiB:
      compiled T>1 verify == eager on {logits, last-hidden, ssm, conv, follow-on T==1 token} for k∈{1,2,3},
      **worst Δ 0.00e+00 — bit-identical**; `mx.compile` is pure fuse on the branch-3 per-token-step graph,
      the only numeric risk). Bench (`parity/nemotron_mtp_compiled_verify_bench.py`, solo ~313 GiB,
      eager-then-compiled in ONE process so the comparison isn't cross-run): the compiled verify is only
      **1.08–1.10× faster than eager** — t_verify T=2/3/4 134→124 / 169→156 / 203→184 ms, still 1.42–2.11×
      `t_main`=87.4 (vs eager's 1.54–2.32×). The eager T>1 verify was already a single, mostly
      launch-amortized forward (NOT the per-token T==1 decode loop greedy runs), so kernel-launch fusion has
      little to remove. **Result: best B=1 spec 0.79× → 0.84× greedy** (9.0 → 9.6 tok/s, `draft_topk=8 k=1`,
      accept 1.60) — a real lift but **still <1×**. Reproduces M2 exactly (`acc==` every config; full-topk
      k=1 first-diverges at 24/48 = the bf16 ULP near-tie, else 48/48; `match` is INFO, M2 owns the
      losslessness proof). **So plain `mx.compile` is NOT the >1× B=1 lever** — crossing 1× at B=1 needs a
      **fused multi-token verify kernel** (one kernel for the whole T-step mamba+moe, deeper than
      `mx.compile` auto-fusion); serving throughput goes to the already-built **batched (B>1) tree-verify**
      (`spec_generate_tree` / `batch_verify` / `NemotronBatchedResidentModel`).
    - **MTP-M4 ✅ — batched tree-verify on the REAL baked MTP head (first trained-head measurement).** The
      prior tree-verify real gate (`parity/nemotron_batched_tree_verify_real.py`) ran on Super-120B with a
      RANDOM-init MTP (0 `mtp.*` keys), so every `W^D` draft path sat at the ~1/W accept floor — no fan-out
      to amortize. MTP-M1 baked the real Ultra sidecar (1040 `mtp.*`), so `parity/nemotron_ultra_tree_spec.py`
      (solo ~313 GiB) is the FIRST tree-verify with a trained head: `spec_generate_tree(batched=True)` over
      the int4-RTN backbone + int4 sidecar, no runtime change (`batched=True` already exists, model-free-
      gated). **PARITY (rule 4) PASS** — `batched=True == batched=False` **BIT-IDENTICAL** (W=2 D=2, 32 tok;
      the bf16 batched-moe-reorder ≥0.99 tolerance wasn't even needed), so the `B=W^D`
      one-`gather_qmm`-over-all-paths verify is output-equivalent to the naive per-path verify (rule 4/6).
      Losslessness reproduces M2 exactly (both tree paths 30/32 vs greedy, first-div pos 24 = the same bf16
      ULP near-tie M2 owns). **The trained head fans out as designed** — tree mean_accept **1.96/3 (W2D2) →
      2.35/3 (W4D2)** vs the k-chain's 1.81 (the random-head gate could never show fan-out). **BUT the `W^D`
      weight-amortization thesis FAILS at B=1:** `bat/seq` **0.89–0.99×** (batched is NOT faster than
      sequential — marginally slower) and the whole tree is **0.07–0.19× greedy** (W2D2 0.19× / W2D3 0.08× /
      W4D2 0.07×) — the WORST B=1 path measured, far below the M3 k-chain (0.79×) and the (A) compiled-verify
      (0.84×). Root cause: `batch_step` amortizes only the MoE (one `gather_qmm` over `[B,1,hidden]`) but
      **48/108 layers are MoE — the other 60 (Mamba-2 + attn) run per-stream in a bounded B-loop**, so B paths
      cost ~B× the *dominant* SSM/attn work and cancel the MoE win (the settled M3 finding: on this hybrid the
      experts are NOT the single dominator). **So tree-verify is a serving-throughput (multi-stream B>1)
      lever, not a single-stream B=1 latency lever** — and even for throughput the per-stream Mamba loop caps
      amortization on a hybrid. The B=1 latency lever stays a **fused multi-token verify kernel**; the B>1 win
      wants genuine **multi-stream decode** (independent requests, not path-replication) + Mamba-state
      batching. Logic was already model-free-gated (`nemotron_batched_tree_verify_test.py`: batched==
      sequential incl. replica fidelity/divergence; `nemotron_tree_spec_test.py`: tree==greedy for any MTP
      quality) — MTP-M4 adds the WEIGHT-level parity + the real-head economics the headless artifact made
      impossible.
  - **U4/paged-KV ✅ — real-Ultra paged==discrete gate (the deferred #152 rule-4 ship gate).**
    `parity/nemotron_ultra_paged_real_test.py` (solo ~306 GiB, no MTP sidecar). The paged contract
    (`make_paged_state` / `prefill_paged` / `step_batch(paged_batched=…)` + the #153 batched-KV
    loop-kill) was already built on `NemotronBatchedResidentModel`, model-free-green
    (`paged_engine_equiv_test.py`) and Super-120B-green (#174); only the real-Ultra-artifact gate was
    "deferred, one model at a time" (`batched_runtime.py:613`). Drives the real serving path
    (`_BaseBatchedSession` paged mode): seq A stores a 32-tok/2-block prefix's int8 KV across all **12**
    attention layers + the Mamba recurrent boundary; seq B reuses them + suffix-only prefill; 10 greedy
    steps. **paged == discrete top-1 10/10 (bit-exact)**, 12-attn-layer coverage (derived from the
    artifact, not the Super's 8), prefix reused 32 tok/2 blocks, recurrent boundary restored 1/1, engine
    `get_cache_stats()` agrees; logit max-abs 0.875 INFO (chunked-Mamba-prefill + int8-KV reblock,
    bf16-ULP class — top-1, the rule-4 arbiter, never flips over 10 steps). The #153 loop-kill's
    *throughput* win (B>1 decode tok/s) folds into the multi-stream stream (MTP-M4: Ultra's per-stream
    Mamba loop, 48/108 layers, caps the attention-KV-only amortization on a hybrid).
  - **U4/decode-economics ✅ — combined measure-first run for the two remaining streams.**
    `parity/nemotron_ultra_decode_economics.py` (solo ~306 GiB). Scoping discovery: MTP-M4's "60
    Mamba/attn layers per-stream" was the **T>1 verify** path; the **T=1 multi-stream decode** path
    (`batched_decode_step_fused`/`_native`) already batches Mamba (one `[B,…]` mixer call; `ssd_step_fused`
    is `grid=(p,h,bn)`) + attn (fused SDPA) + MoE (stacked).
    - **Stream A (multi-stream B>1 decode) — characterization win, MTP-M4 pessimism overturned for
      decode.** Real-Ultra native (form-2 persistent `BatchedMambaState`) aggregate decode scales
      **10.3→28.4→40.0→47.5 tok/s @ B=1/8/16/32 (4.61× @ B=32)**, loop-kill **1.77× @ B=8**, parity bit-exact
      (native==fused==loop, Δ=0), peak 367 GiB @ B=32 (room under 490). No kernel needed; the sublinear
      per-stream drop (10.3→1.48) is the batched-SSD-step-compute ceiling (optional follow-on: push B>32 /
      a batched-SSD-step tune).
    - **Stream B (fused multi-token verify kernel) — GO.** B=1 T>1-verify breakdown (T∈{1..4}): the
      T-growth is **59% Mamba per-token step loop** (`mamba_mixer.py:148`, +77.8ms — launch-bound, the part
      `mx.compile` can't fuse across the sequential T-loop) + **40% MoE** `gather_qmm` (+52.3ms — weight
      bandwidth, NOT fusable) + ~0% attn. Build a fused multi-token SSD scan kernel (extend the one-token
      `_ssd_step_kernel` in `mamba_ssd.py:97` to loop T internally, carrying state) — targets the majority
      grower; MoE 40% caps the speedup. Next: gate output-equivalent to eager (rule 4), bench vs the 0.84×
      B=1 ceiling.
  - **U4/Stream-A (decode batch-scaling) ✅ — pushed the multi-stream decode sweep past B=32 to the memory
    ceiling.** `parity/nemotron_ultra_decode_scale.py` (solo ~306 GiB, exit 0), native form-2 serving path,
    adaptive per-stream-memory guard (never launch a B that could OOM). **Aggregate decode throughput
    PLATEAUS at ~48 tok/s from B=32 on** — B=32 48.03 / B=48 47.78 / B=64 47.93 / B=80 48.32 tok/s (all ~48
    ± run-noise; per-stream 1.50→1.00→0.75→0.60, agg 4.75–4.78× B=1). So **B≈32 is the throughput knee**
    (367 GiB, 123 GiB headroom); B>32 buys zero aggregate, only per-user latency + memory (flat ~1.92
    GiB/stream). Guard skipped B=96 (projected 494.7 > 465 safe); measured to B=80 @ 459.5 GiB ⇒ extrapolated
    max B ~83 @ 465 safe / ~94 @ 490 hard (B>32 = an admission/concurrency policy choice, not a correctness
    limit). Parity self-check green (B=1 fused==loop |Δ|=0, B=4 native==fused |Δ|=0; overlap rows reproduce
    `0de52a9`). **Confirms the batched-SSD-step ceiling** — the per-stream Mamba recurrence (NOT memory/MoE)
    caps the amortization, so the only lever past ~48 tok/s is the batched-SSD-step tune, the same
    `mamba_ssd.py` SSD-step surface as Stream B. **Serving default pinned: B=32** — already the Nemotron
    `BEST_BATCH` operating point + uniform `SERVING_BATCH_CAP=32` (`shim/omlx.py`); the Ultra measurement now
    backs it (was Super-120B-only). Stream A recommendation **settled**.
  - **U4/Stream-B (fused multi-token SSD-scan verify kernel) ✅ — built + gated + benched.**
    `ssd_scan_fused` (`mamba_ssd.py`) extends the one-token `_ssd_step_kernel` to loop T internally,
    carrying the N=128 state through the `new_state` buffer (register-carry spills at N=128) — the whole
    T-token verify Mamba recurrence is ONE Metal launch per layer (at T=1 == `ssd_step_fused`). Wired into
    `MambaMixer`'s T>1 continuation behind `FUSED_SSD_SCAN` (default off, rule 4) with a **bit-identical**
    batched conv (per-token `causal_conv1d_step` window over T, same `mx.sum`-over-K, same `conv_state`);
    off/T=1 byte-unchanged. **Gates**: model-free `parity/nemotron_ssd_scan_kernel_test.py` (scan ==
    per-token loop 2.2e-7, T=1 bit-exact to `ssd_step_fused`, batched conv bit-identical); real-weight
    per-block `parity/nemotron_ultra_fused_scan_parity.py` (rule-8 streamed all 48 mamba blocks — fused
    output == eager ≤1 bf16 ULP, 83/144 bit-identical, conv bit-identical, ssm ≤7.6e-6). **The gate caught
    the bf16 cascade**: a first full-model gate FAILED intermediate-state (hiddenΔ 106, convΔ 2.00) despite
    perfect top-1; per-layer trace = a clean power-of-two cascade from a single bf16 ULP flip at the 2nd
    mamba layer (2⁻⁶→…→10²) — the M2/M3 settled finding (a ULP near-tie cascades chaotically), so the
    honest criterion is **per-block equivalence + top-1**, NOT intermediate magnitude (gate redesigned to
    per-block). **Bench** `parity/nemotron_ultra_fused_scan_bench.py` (solo ~313 GiB; E/F/FC ordered so the
    only compiled-verify traces are flag-True): t_main 88.8 ms; t_verify **F 1.07–1.12× / FC 1.10–1.15× vs
    eager** (launch-overhead only); **best B=1 spec 0.80× eager → 0.90× F → 0.92× FC** (draft_topk=8 k=1,
    accept 1.66/2) — **best B=1 single-stream spec measured, +15% over the eager ceiling, beats the 0.84×
    compiled ceiling, but STILL <1×**: the fused verify is 1.43–2.01× t_main, the residual is the **unfused
    MoE `gather_qmm`** (40%, weight-bandwidth, NOT launch-bound — a scan kernel can't touch it). `acc==`
    shows `!!` at bf16 near-ties (F/FC ≤1 ULP, not bit-identical; `match` mostly 48/48); losslessness owned
    by M2 (the int4 main model verifies every draft). **Crossing 1× at B=1 now needs the MoE verify
    bandwidth reduced (a harder lever, not a scan kernel); the throughput lever stays the
    already-characterized B>1 tree-verify (MTP-M4).**
  - **U4/MoE gather_qmm batch-scaling ✅ — measured the MoE "bandwidth lever" before building.**
    `parity/nemotron_ultra_moe_qmm_bench.py` (one real MoE layer, rule 8) times the routed
    `gather_qmm` across N=B·T. The MoE is **already fused** (sorted dispatch, the DSV4
    `_swiglu_stack_packed` / qwen35 pattern) and amortizes hard at batch: **per-token cost 1195 µs @
    B=1 → 209 µs @ B=32 (5.58× cheaper/token) → 182 µs @ B=128 (6.6×)**; sorted dispatch **0.93× @ B=1**
    (overhead, no overlap) **→ 1.81× @ B=32** (each touched expert's int4 weights read once for all
    tokens routed to it). So the MoE "bandwidth lever" is **not a missing fusion** — it's a **B=1-vs-B=32
    regime**: a single stream can't amortize (why Stream B's B=1 spec stayed <1×), B=32 serving gets it
    for free. **Reconciles Stream A**: at B=32 the MoE is cheap (5.6× amortized), so the per-stream
    **Mamba** recurrence is the decode ceiling — not the MoE. So the only serving lever past ~48 tok/s is
    the **batched Mamba SSD-step** (same `mamba_ssd.py` surface as Stream B); the MoE needs nothing more.

  - **U4/decode-step breakdown ✅ — localized the B=32 serving ceiling + CONFIRMED the lever (measure-first,
    no kernel built yet).** `parity/nemotron_ultra_decode_step_breakdown.py` (solo ~306 GiB) decomposes a
    real native (form-2) **T=1 decode** step by layer-kind + a mamba sub-breakdown + an **e2e fused-step
    A/B**. At B=32: **MoE 47% + mamba 40% + attn 12%** (real total 642 ms ⇒ 49.8 tok/s, reproduces the
    Stream-A knee); **every** kind amortizes per-token (moe 0.26× / mamba 0.21× — dense GEMMs share the
    weight read across B). The lone non-amortizer is the SSD recurrence, and the sub-breakdown nails the
    real cost: the **composed `ssd_step` is 64% of the mamba block at B=32** (4.6 ms/block vs projections'
    ~2 ms) because the **eager** batched path materializes several `[B,H,N,P]` fp32 temporaries (~268 MiB
    each) — the already-built **`ssd_step_fused` kernel is 3.86× faster** (in-kernel state carry). So the
    lever is **NOT a new kernel** — it's **graduating `FUSED_SSD_STEP`** (shelved as a "no-win", but that
    was B=1-*compiled*-only). The **e2e A/B confirms it greedy-exact** (argmax_match; |Δlogit| 2.12 = the
    bf16-ULP reorder class): composed→fused **1.04× / 1.15× / 1.26× / 1.36× @ B=1/8/16/32** — **+36% agg
    decode @ B=32 (49.4 → 67.0 tok/s)**, output-equivalent. **Corrects Stream A**: mamba IS co-dominant,
    but the cost is the composed-op blowup, not the recurrence FLOPs. **Next:** graduate `FUSED_SSD_STEP`
    into the batched steppers (`batched_decode_step_fused`/`_native`) + re-gate `native==fused` (bit-exact
    → greedy-exact); then the residual ceiling is the **MoE+mamba co-dominant weight bandwidth** (B>32 =
    admission policy, not a kernel).

### Second model — MiniMax-M3 (when available)
- **Mellum2 dropped** — its context length is too short for the orchestrator role. Replaced by
  **MiniMax-M3 once it ships**; the `src/quanta/minimax/` module is already substantially ported in-tree
  (config/attention/bake/calibrate/decode/eagle/generate), so integration is mostly bring-up + parity
  + bake when weights are released. Until then this is a **single-model (Nemotron-Ultra) track**.

### Stack (deferred until the second model lands)
- Two-model agentic loop, one-at-a-time residency (swap main↔orchestrator); measure swap latency.
  If swapping is too slow for the loop, revisit concurrent-resident (needs a measured
  `mx.set_wired_limit` budget — a deviation from the one-at-a-time rule).

## Cadence (standing)
Single thread, **no subagents**; implement → gate green → commit named files (trailer
`Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`) → **STOP** for the user to
compact. One model resident at a time. Keep `~/models/Kimi-K2.6`. The InternLM2.5 MInference **M7 is
paused** (handover preserved in `PLAN_minference.md`), not abandoned.
