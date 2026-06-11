---
name: project-nemotron-ultra
description: "Nemotron-3-Ultra-550B (nemotron_h hybrid Mamba2+attn+MoE) — int4-RTN SHIPPED (306 GiB). U0–U4 + native-MTP spec + paged + fused-SSD-step graduation all done. U1 caught a REAL forward bug (group-wise gated RMSNorm); U3 retired AWQ→RTN e2e (recon≠e2e). Paused for the Nex-N2-Pro active task. Handover PLAN_nemotron_ultra.md."
metadata:
  node_type: memory
  type: project
  originSessionId: be1e7097-a051-4573-af5f-0995c6587155
---

**What:** Nemotron-3-Ultra-550B-A55B — hybrid **Mamba2 + attn + MoE** (`nemotron_h`; already supported,
the 120B-Super sibling baked int4 earlier). 108 layers / 48 MoE / 512 experts top-22 `noaux_tc` sigmoid.
Source `~/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` → **`…-quanta_int4rtn_g64` (306 GiB)** via
`parity/run_bake_nemotron_ultra_int4rtn_g64.py`, + a native-MTP sidecar `…-quanta_int4rtn_g64_mtp` via
`parity/run_bake_nemotron_ultra_mtp_int4rtn_g64.py`. **SHIPPED int4-RTN.** One of the served keepers
([[project-model-targets]]); paused in favor of [[project-nex-n2-pro]] (active). Handover
`PLAN_nemotron_ultra.md`.

**U1 layer parity CAUGHT A REAL FORWARD BUG (the CLAUDE.md thesis):** the Mamba-2 **gated RMSNorm is
GROUP-WISE** (variance over `d_inner//n_groups`, NOT full `d_inner` — `Zamba2RMSNormGated`); ours was a
full-width `nn.RMSNorm` — **self-consistent** (prefill==decode) so the old self-consistency-only test
never caught it, but **42% off** the reference. Fixed via a group-wise `MambaRMSNormGated` (forward-only;
**corrects the already-baked Super-120B too**, no re-bake — bf16 norm.weight unchanged). Super ppl
re-measured under the fix: bf16 **5.981→3.379**, served int4g64 resident **3.305** (≈lossless). The
pre-fix 5.981 was measuring the degraded buggy-norm forward (the residual skip kept it coherent-ish).

**U3 e2e ppl arbiter RETIRED AWQ → int4-RTN ships (recon ≠ e2e, settled):** U2's activation-weighted
*recon* de-risk said AWQ was fine on relu² down-proj, but the **e2e** arbiter (1024-tok prose): bf16
**3.835** / int4-**AWQ +24.3%** / int4-**RTN +0.3%**. The relu² down-proj AWQ tax got **worse** with more
tokens (finding #38 reproduced e2e — the U2 recon could not see it). RTN ~lossless, data-free,
`expert_method="rtn"`. 306 GiB resident (30 GiB < AWQ's 336 — RTN stores bf16 vs AWQ's fp32 expert
scales). AWQ artifact retired.

**U4 packed-int4 `gather_qmm` stream:** M1 (one MoE layer) + **M2 full-resident e2e ppl** == streamed
dequant RTN ref (Δ −0.1%) — the whole 108-layer resident model, covering dense mamba/attn int8
`QuantizedLinear` wiring end-to-end too.

**Native MTP spec-decode (MTP-M0–M4):** baked a self-contained int4 sidecar (1040 `mtp.*` tensors),
wired `build_resident_mtp` + a resident spec adapter, real-gated. **SETTLED (do not re-discover):** on a
bf16 **Mamba hybrid** the spec VERIFY forward (T>1) differs from a T=1 decode by ~**1 bf16 ULP** which
**cascades chaotically**, so **"spec == T=1 greedy" is the WRONG real-weight criterion** — the honest
gate is **per-block equivalence + top-1** (what spec consumes). B=1 single-stream spec tops out **<1×**
(best **0.92×** with the fused SSD-scan verify; eager 0.79×, plain `mx.compile` 0.84×) — the residual
T-growth is the **unfused MoE `gather_qmm`** (40%, weight-bandwidth, NOT launch-bound). The drafter is
**near-inert at B=1** (a bf16-quality drafter ties int4 at 0.79×). **Throughput is multi-stream decode,
not tree-spec path-replication.**

**U4 decode economics + GRADUATIONS:** the ~48→67 tok/s knee is **MoE+mamba co-dominant weight
bandwidth**, not memory/MoE-fusion. **Fused SSD step graduated** — `BATCHED_FUSED_SSD_STEP` (default ON,
threaded as a `fused_step` kwarg, no global state) = **+36% aggregate decode @ B=32 (49.4 → 67.0 tok/s)**,
greedy-exact; the per-stream-loop ref + tree-spec `batch_step` stay composed, the compiled single-stream
path unchanged. Native form-2 `BatchedMambaState` decode scales to a **B≈32 throughput knee (~65 tok/s,
~78 streams to the 490 GiB ceiling)** — B>32 is admission policy, not correctness.
([[batched-serving-operating-point]]) MoE `gather_qmm` amortizes **5.58×@B32** (it's a B=1-vs-B=32 regime,
already fused — not a missing kernel).

**Paged + tree-spec-over-paged:** U4 paged-KV real-Ultra green (the paged manager covers exactly the
**12 attn layers** the artifact declares; backbone-only, no MTP). #158-160 **tree-spec-over-paged M0–M3
COMPLETE** — DSV4 (the keeper where tree-spec is a real B=1 lever), qwen35 (N/A — unpaged, pinned),
**Nemotron** (the hard paged triple `(caches, ssm, conv)`: paged==discrete bit-identical, `release`
bounds the pool). ([[project-paged-batched-153]])

**Rode the same arc — InternLM2.5 MInference M0–M10 COMPLETE** ([[project-internlm2-minference]]): the
sparse-prefill track finished alongside Ultra.

**Status:** SHIPPED int4-RTN, all U-streams done. Real-weight gates SOLO ([[feedback-memory-safety]]).
Cadence: single thread, no subagents, commit per milestone then STOP.
