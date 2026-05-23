# Initial prompt — paste this to start a fresh session in `finally_quanta`

> Copy everything in the fenced block below into the first message of a new
> Claude Code session opened in `~/Environment/quant/finally_quanta`. It frames
> the exploration: what is already settled (don't redo it), the plan to follow,
> and the fallbacks if a branch fails. Read `CLAUDE.md` first for the permanent
> rules and the GPTQ/inverse details.

```
We are building `quanta`: a parity-first, MLX-native quantization + sparse-MoE
inference runtime for Kimi-K2.6 (then GLM-5.1, DeepSeek-V4-Pro), to run RAM-
resident on one M3 Ultra (≤490 GiB working set). This is a clean restart of a
prior effort that failed because the runtime was built and refactored WITHOUT a
numeric parity gate, so a localized forward-pass bug went undiagnosed and was
mis-attributed to int3 expert quantization. Read CLAUDE.md before doing anything.

Hard rules (also in CLAUDE.md, do not violate): build layers as mlx.nn modules;
use mlx.fast primitives maximally; NO Python loops on compute paths (vectorize /
gather_qmm / compile); parity-first (match a reference before optimizing); no
mlx-lm runtime dep (transformers/torch offline only); no silent failures; sparse
MoE; one text layer resident at a time during bake/parity.

=== ALREADY EXPLORED — treat as settled, do NOT redo ===
- Bit budget: int8-everything ≈0.78% recon but ≈975 GiB (won't fit). int4-all
  ≈12% recon AND ≈517 GiB (> 490 ceiling). The fit-able routed split is ~gate/up
  int3 g128 + down int4 g128 (≈438 GiB). 3–5% compounded expert error is
  infeasible by bit allocation under the ceiling.
- On an int4 source, DWQ ≈ AWQ ≈ no help (no scale headroom). GPTQ error-feedback
  is the only expert-coding lever that lowers reconstruction (~5% busy-expert vs
  ~21%) — but it does NOT improve end-to-end perplexity (GPTQ ≈ DWQ e2e).
- Therefore per-expert / compounded reconstruction error does NOT predict e2e
  quality. The arbiter is teacher-forced perplexity through a CORRECT runtime.
- The failure mode is uniform (flat per-position, flat across depth, wrecks even
  literal repetition and counting) and expert-coding-independent ⇒ a localized
  bug in the SHARED forward path, not the experts.
- Already eliminated as the cause: RoPE factor (correct = 64) and the YaRN
  mscale² attention scale (correctly applied: (128+64)^-0.5 · mscale², mscale =
  0.1·ln(64)+1 ≈ 1.416).
- Reasoning models loop under greedy decode regardless of quant — never diagnose
  with generation; use perplexity / numeric parity.
- GPTQ inverse is solved (Cholesky-of-inverse + MLX CPU Cholesky + Woodbury small-
  Gram for n≪in + damping + batched trailing matmul). See CLAUDE.md.

=== THE PLAN — explore in this order ===
1. PARITY HARNESS FIRST. Build a dead-simple reference forward in plain mlx.core
   from dequantized Kimi source weights (or HF/transformers offline) for ONE real
   layer (L0 dense, then L1 MoE). Then build the mlx.nn runtime layer and diff the
   residual stream numerically on identical token ids. Find the first divergence.
   Bisect within a layer: RMSNorm placement → MLA q/k/v projections → RoPE freqs →
   softmax scale → matrix-absorb path → top-k routing → expert dispatch → shared
   expert → KV/latent cache across positions. THIS is the whole point; do it before
   anything else.
2. Once bf16/int8 parity is green end-to-end, re-measure int3 routed-expert quality
   THROUGH the now-correct runtime (teacher-forced ppl + top-1 vs bf16). This finally
   answers the int3-floor question that the broken runtime made unanswerable.
3. Build the bake pipeline (affine packing → GPTQ with the inverse handling from
   CLAUDE.md → optional QuaRot rotations), gating each stage on parity.
4. Resident loader + wired limit + decode-speed levers (matrix-absorb, fused M=1
   MoE GEMV, sorted dispatch, EAGLE-3) — each added behind a flag and proven
   output-equivalent to the naive path via parity before being turned on.

=== IF THINGS GO SIDEWAYS — fallbacks ===
- Parity diverges and you can't localize by layer-diff: drop to op-level — diff a
  single MLA forward and a single MoE forward in isolation against the reference
  on random input; binary-search the ops.
- The bug is in an OPTIMIZATION (matrix-absorb, fused MoE, sorted dispatch): keep
  the naive expand / gather_qmm path as the default; the optimization stays off
  until it matches.
- Runtime is correct but int3 routed is genuinely insufficient (int3-floor
  confirmed): mixed int3/int4 by per-layer sensitivity to claw back quality under
  the ceiling; or byte-budget offload of the coldest experts to fit int4 on the
  rest; or accept a smaller context/target. (Do NOT pre-optimize for this until
  measured through a correct runtime.)
- HF/transformers reference won't fit in memory for a full forward: reference one
  layer at a time from dequantized source weights (layer-by-layer rule), comparing
  residual-in → residual-out per layer rather than a whole-model forward.

=== FIRST CONCRETE ACTION ===
Start the parity harness for Kimi layer 0 (dense): load source weights, dequantize,
build the reference forward and the mlx.nn runtime forward, run the same ~16 token
ids through both, and report the per-op max abs / rel error. Do not optimize or
quantize anything yet.
```
