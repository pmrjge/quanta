---
name: feedback-batched-rope-bf16
description: "Hand-rolled batched/vectorized reimplementations of mx.fast kernels (esp. RoPE) are fp32- and random-bf16-correct yet drift at bf16 on REAL large-magnitude values and compound across layers — loop the same mx.fast kernel per stream; gate B=1 bit-exact + B>=2 greedy-exact."
metadata:
  node_type: memory
  type: feedback
  originSessionId: e88fadb0-5f2c-48da-82a1-e14175af6551
---

**THE BUG (2026-05-28, InternLM2.5 batched decode).** A hand-rolled fp32 rotate-half batched RoPE
(`batched_rope_explicit`) matched `mx.fast.rope` to ~1e-6 in fp32 AND was bit-exact (0.00) on random
bf16 micro-tests — but on the REAL model its bf16 ULP rounding diverged on large-magnitude q (up to
7.8e-3 at one layer) and **compounded over 32 layers** to flip greedy tokens (|Δlogit|≈61,
greedy_match=False). It passed every model-free gate and every random-tensor micro-test; only the real
int8-g64 bench (`parity/internlm2_batched_bench.py`) caught it. Bisection that found it: per-layer
(B=1==B=2, exact at layers 0-2 then compounds) → per-op (`rope_q` isolated as the diverging op;
attention bit-exact given identical inputs).

**THE FIX (`0d531c3`).** `batched_rope_fast` loops the runtime's OWN `mx.fast.rope` per stream
(`x[s:s+1]`, per-stream base+offset) and concatenates — a bounded B-loop (rule-3 OK, off the per-token
hot core: one call per stream, not per token/dim) that calls the SAME kernel the single-stream decode
reference uses → **bit-identical at every dtype** (B=1 = 0.00e0).

**DURABLE LESSONS — do not re-learn:**
1. A "vectorized" reimplementation of an `mx.fast.*` kernel can be fp32-correct AND random-bf16-correct
   yet wrong on real activations. bf16 parity is **input-dependent**: random small values hide it, real
   large-magnitude q exposes it. Always bit-check batched/fused paths on the REAL model across ALL
   layers — never trust a 2-layer tiny config + random tensors alone.
2. Prefer **looping the existing `mx.fast.*` kernel per stream** (bounded, rule-3-allowed) over
   hand-rolling the math — bit-exact by construction, and the loop is off the per-token hot path.
3. The ACCEPTED batched equivalence class is **greedy-token agreement**, not bitwise logits. Once
   batched `quantized_matmul`/padded SDPA engage (B>=2), per-row-independent bf16 reduction-order ULP
   gives |Δlogit|~0.5 (input-dependent: random micro-tests show 0.00, real values ~0.5) — argmax-stable.
   So the real-model gate asserts **B=1 bit-exact (0.00 — proves a faithful port; ANY Δ is a real bug)
   + B>=2 greedy-exact**. Do NOT gate B>=2 on a logit tolerance calibrated on a tiny config: the
   32-layer path's ULP is larger and a 5e-3 tol mis-fired at the real ~0.5 (the fix was a correct gate,
   not a loosened tolerance).

**Confirmed contained:** no sibling batched runtime (DSV4 / Nemotron / Qwen3.5) hand-rolls batched RoPE
— they all delegate to per-stream `mx.fast.rope`. Same caution applies to any future batched rewrite of
an `mx.fast` primitive (SDPA, rms_norm, rope).
