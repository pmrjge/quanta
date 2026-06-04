# PLAN — Nemotron-3-Ultra-550B (main agent) + Mellum2-12B (orchestrator)

A two-model agentic stack on one M3 Ultra (≤ 490.4 GiB), quantized through `quanta`'s
parity-first pipeline.

- **Main agent:** `NVIDIA-Nemotron-3-Ultra-550B-A55B` — hybrid Mamba2 + attention + MoE
  (`model_type: nemotron_h`) → **int4-GPTQ experts + int8 dense + bf16 core**.
- **Orchestrator:** `JetBrains/Mellum2-12B-A2.5B-Thinking` — sliding-window + full-attn MoE
  (`model_type: mellum`) → **int8 (AWQ)**.

## Decisions (user, this session)

1. Nemotron experts **int4-GPTQ** — keep the proven `nemotron/quant_policy.py`, **not** AWQ.
   GPTQ error-feedback is the stronger lever on a *bf16* source (the "AWQ≈no-help" finding was an
   int4-source artifact); it is already built and baked on the 120B-Super sibling.
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

- Ultra int4 mix **289.7 GiB resident** (U0-measured: int4-GPTQ 255.0 + int8 30.3 + bf16 4.4 GiB;
  + ~1–2 GiB MTP head). Headroom **200.7 GiB** for KV + activations. Only **12 / 108** layers carry
  growing KV — the 48 Mamba layers have **O(1)** state (a real long-context win at 256K).
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
- **U2 — full int4-GPTQ + int8 bake**, layer-streamed (rule 8), via `nemotron/bake.py`
  (cf. `parity/run_bake_nemotron_int4g64.py`); calibration + routing-capture over 512 experts top-22
  (Woodbury for under-covered experts). Self-contained artifact (config + manifest + tokenizer +
  relative shards) → `~/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-quanta_int4g64`. Gate: loads +
  manifest in-artifact only. **Hours; run solo (OOM hazard).**
- **U3 — teacher-forced ppl + top-1** vs bf16 reference on real prose (stop set {2, 11}). Gate: sane.
- **U4 — optimizations**, each behind a flag and ppl-equivalent: native **MTP spec-decode**
  (`spec.py`/`mtp.py`), **paged-KV** on the 12 attn layers (port #153 loop-kill), packed int4 experts
  + `gather_qmm` (already in `moe.py`), batched decode + Mamba-state batching (`batched_runtime.py`).
  MInference sparse-prefill only if long-ctx attn-layer prefill proves a bottleneck (just 12 layers).

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
