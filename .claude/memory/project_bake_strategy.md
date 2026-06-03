---
name: project-bake-strategy
description: "The bake plan — mixed int3/int4 GPTQ experts (DP-allocated on activation-weighted loss), int8 non-experts, Woodbury inverse"
metadata:
  node_type: memory
  type: project
  originSessionId: e88fadb0-5f2c-48da-82a1-e14175af6551
---

Bake = quantize the source checkpoint into a self-contained resident artifact. Policy
(user-directed, 2026-05-23):

- **Routed experts → mixed int3 / int4 GPTQ.** int3 g128 default; int4 g128 **fallback**
  for sensitive projections. A DP / knapsack allocates int3↔int4 per projection
  ("back and forward": promote the highest-sensitivity int3 projections to int4, demote
  if over budget) to drive **total error below 8%** subject to bytes ≤ 490 GiB.
- **DP error metric = activation-weighted GPTQ loss** `‖WX−ŴX‖²/‖WX‖²` per projection on
  the captured calibration X — NOT raw Frobenius recon. Chosen because the settled finding
  ([[project-forward-bug-resolved]] era) is that per-expert recon does **not** predict e2e
  ppl; the activation-weighted loss is what GPTQ minimizes and is far more e2e-predictive.
  **Final gate is e2e teacher-forced ppl** (re-allocate if it misses). This is the
  int3-floor measurement.
- **Non-experts (attention q/kv/o, dense L0 MLP, lm_head) → affine int8 g128** via
  `mx.quantize` (MLX packed → `mx.quantized_matmul`). Validated near-lossless: 0.78% recon,
  0.79% vs bf16. (int3 RTN is 20.7% → experts must use GPTQ error-feedback, not RTN.)
- **Shared expert + norms + router control tensors → bf16/fp32** (always-on path, ~free).

**Inverse = Woodbury (primary), not Cholesky.** Under top-8 routing over 384 experts most
experts see `n ≪ in` calibration rows, so `H=XᵀX` is effectively low-rank. Invert the small
`[n,n]` Gram `(I + XXᵀ/λ)` (`O(n³)`), never form/Cholesky the `[in,in]` Hessian; GPTQ's
ordered update reads coefficients from a Cholesky of that small Gram + the Woodbury identity.
Direct `[in,in]` Cholesky is the fallback only for the rare well-covered expert (`n ≳ in`).
`λI` damping keeps both PD.

**Budget:** all-int3 experts ≈ 412 GB; non-experts int8 + shared bf16 ≈ 15 GB → ~427 GB,
~63 GB headroom under 490 GiB ⇒ the DP can promote **~50% of experts to int4**.

**Pipeline:** calibration capture (per-MoE-layer post-norm acts `ln2` + routing `idx`;
~8192 tokens; store unique-token acts, not topk-duplicated rows; ~7 GB) → per-projection
sensitivity (activation-weighted loss int3 vs int4) → DP allocation → mixed GPTQ (Woodbury)
→ self-contained artifact (config.json + manifest + safetensors, relative refs) → resident
`gather_qmm` runtime → e2e ppl. One layer resident at a time throughout; never unbounded
([[feedback-memory-safety]]).

**Shipped (2026-05-23/24):** full **int2-g64** and **int4-g64** Kimi bakes ran with an e2e ppl gate
+ 512-tok gen (#45–46). The runtime reads **bits + group_size per-tensor from the manifest** (#42),
so it isn't locked to one split; bf16 scales are an opt-in (#43–44) and **MLACache int8 is now the
default** (#47). The int3/int4-**g128** split above is the original plan; the artifacts that actually
shipped went more aggressive (int2/int4-**g64**) — EAGLE training loads the int2-g64 artifact. Tasks
#23–28, #42–47 (Kimi); Nemotron's parallel bake is int4-g64 experts / int8 dense / bf16 SSM (#37, #53).
