---
name: batched-serving-operating-point
description: "Production runs the batched-decode runtime at B=32 — judge/optimize the batched path at that operating point, not low B"
metadata:
  node_type: memory
  type: project
  originSessionId: 6be05c92-80da-434c-a665-9af2923665b3
---

Production will run the batched-decode runtime at **B=32** (user, 2026-05-29) — the real
concurrent-serving operating point for the Nemotron keeper (and the other batched runtimes).

**B=32 is now a UNIFORM HARD CEILING for every throughput worker (user, 2026-05-30, `bf242c2`;
Nemotron-only since `40a9ac5`).** `QuantaOmlxEngine._resolve_capacity` + `_hard_batch_cap` CLAMP an
explicit `batch_size>32` DOWN to 32 (warns, rule 6), AND `_default_capacity` clamps the no-`batch_size`
default too — for Nemotron, InternLM2.5 **and DSV4**. Decoupled from the measured `BEST_BATCH` knee via a
new `SERVING_BATCH_CAP=32` constant, so DSV4's benched knee (48; it scales there per #18 M5) stays HONEST
in `BEST_BATCH` while serving is held at 32 (rule 6: never overwrite a benched knee with a policy cap);
Nemotron/InternLM2 regress past 32 anyway. The **Qwen3.5 ORCHESTRATOR is EXEMPT** (`_hard_batch_cap`
returns None for it) — its B=4 is a sweepable latency PIN, never a cap. Benches that must sweep B>32
drive the batched SESSION directly (`DSV4BatchedResidentModel.step_batch`), bypassing this engine-layer
policy (`dsv4_batched_bench.py` / `dsv4_b48_noise.py` unaffected). Gate
`parity/omlx_best_batch_test.py::_run_hard_batch_cap` (nemotron/internlm2/dsv4 clamp >32→32; qwen3_5 honors explicit B).

**Why:** stated deployment target. Throughput/memory tradeoffs and "is this optimization worth
it" calls should be evaluated at B=32, not at B=1–8 where the per-stream overhead is small.

**How to apply:** weight B=32 behavior heavily when assessing batched-decode work.
- The attention-only fused path (Approach-1) measured ~1.0–1.08× — negligible — because it only
  touched 8 of 88 layers. The real lever at B=32 is batching the **40 Mamba layers**: form-1
  (concat per-stream `(ssm,conv)` each step) and form-2 (persistent `BatchedMambaState`, no
  per-step concat). **form-2 is the production decode path** — `step_batch_native` /
  `make_batched_state` in `quanta.nemotron.batched_runtime` — since it drops form-1's per-step
  state-concat IO (decode here is launch/IO-bound, not FLOP-bound).
- **Wired into serving (default-on):** `quanta.shim.omlx._BaseBatchedSession` holds a persistent
  `BatchedMambaState` per alive-slot set (flush/scatter/rebuild on admit/release; paged recurrent
  snapshot reads `BatchedMambaState.recurrent_row`), gated `native_decode=True` for capable runtimes
  (only Nemotron exposes `step_batch_native`; DSV4/InternLM2/Qwen35 fall back to form-1). So B=32
  production serves form-2 automatically. Gated model-free by `parity/nemotron_native_serving_test.py`
  (unpaged + paged native==form-1).
- Confirm the corrected peak footprint at B=32 fits the ~490 GiB ceiling: weights ~68 GiB
  (int4-g64, gather_qmm) + per-stream KV (~negligible, int8) + fp32 Mamba SSM state (~5 GiB at
  B=32) + transient. The earlier "200 GiB" was a bench cache-leak artifact, not the real cost.
- **DSV4 decode attention is now batched too (2026-05-29):** `decode_step_dense_batched` /
  `decode_step_compressed_batched` in `quanta.dsv4.decode` collapse the Design-A per-stream MLA loop
  (every layer, all 3 regimes — dense sliding-window, ratio-128 compressed, ratio-4 compressed +
  Lightning-indexer) into ONE batched projection + ONE windowed-sink SDPA across streams. Per-stream
  work is only the bounded cache append + the data-dependent window-closing compressor pool
  (`_maybe_pool`) + ring push. Wired **default-on** (`DSV4BatchedResidentModel._fused=True`), the
  per-stream Design-A loop kept as the multi-token-tail fallback + parity reference. Gated model-free
  in `parity/dsv4_batched_attention_test.py` (B=1 bit-exact, ragged B≥2 ~1e-7) + the existing
  `dsv4_batched_test` / `dsv4_omlx_engine_test` (`per_stream_eq=True`) still green. This is the DSV4
  analog of the Nemotron Mamba win — DSV4 was the HARD case (compressor state machine + indexer +
  ragged ckv), not the quick one. **Throughput unmeasured** (the real DSV4 bench `dsv4_batched_bench`
  is a ~180 GiB solo GPU job — run alone).
- The remaining per-stream loop everywhere is the attention KV-update `.update()` / `append_kv` itself
  (ragged quantized stores). Batching it needs a paged/batched KV store, and the production default is
  PAGED (`PAGED_KV_DEFAULT=True`) ⇒ a custom varlen/paged Metal kernel (#153-class) + drops int8 KV
  quant. Lower priority than the per-layer attention/Mamba wins already landed. See [[model-targets]].

**Measured (2026-05-29, real int4-g64, commit ed2288d):** native form-2 vs per-stream loop, aggregate
tok/s — B=2 1.14×, B=4 1.41×, B=8 1.77×, B=16 2.08×, **B=32 2.32×** (127.1 vs 54.8). form-2 beats
form-1 by +8.9% at B=32 (the per-step state-concat IO). Active floor **67.9 GiB** (confirms the int4
weight floor); B=32 peak **159.6 GiB** — but that is the 32-stream *seed/prefill* transient (per-B,
isolated by clear_cache), not steady-state; steady decode sits near the active floor + KV/Mamba state.
Caveat: the bench uses a uniform 1024-tok prompt replicated across streams ⇒ identical routing + no
ragged SDPA padding (best case for MoE amortization / attention); the **speedup ratio** is robust
(Mamba batching is routing/padding-independent) but absolute tok/s is optimistic vs ragged production.
