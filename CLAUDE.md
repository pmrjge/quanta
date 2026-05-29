# CLAUDE.md — quanta

`quanta` is a **parity-first**, MLX-native quantization + sparse-MoE inference
runtime. It is a clean restart of the prior `quantification` effort, keeping the
hard-won findings (below) but rebuilding the runtime so that **every component is
gated against a numeric reference before it is optimized or its quantization is
judged.** The immediate target is **Kimi-K2.6** (then GLM-5.1, DeepSeek-V4-Pro).

The previous build reached a runtime that produced incoherent output (teacher-
forced perplexity ~165 with BOS, ~5–9× fuzzy on trivial tasks) and the failure
was wrongly attributed to int3 expert quantization. It is **not** the experts
(see Settled Findings). It is a localized bug in the forward pass that was never
caught because the runtime was built and refactored **without a parity gate**.
That is the mistake this project exists to not repeat.

---

## Active task (transient — full handover in PLAN.md)

**#18 — kill the per-stream KV-update IO loop** in DSV4 batched decode: replace the
`B` ragged per-stream `_LayerCache` streams (per-stream `append_kv` loop + per-step
`_pad_stack`) with a persistent `max_batch`-sized **batched KV arena** — ONE scatter
write + ONE gather read. Staged M0–M5, flag-guarded (`kv_arena`, **default ON since
M4**). **M0 ✅ `41a4d0f`, M1 ✅ `6f33cc1`, M2 ✅ `05d1171`, M3 ✅ `bf7af6b`, M4 ✅
`e08888d`** (flipped default ON + `_CompArena`→`_KVArenaSet` + `_ArenaCacheHandle` +
`make_cache`/`prefill`-seed/`step_batch`/`free_cache` + session `release` + full
regression; serving leases an arena row per stream, a discrete `DSV4Cache` still takes
the per-stream loop — dispatch keys off the cache type). **Only M5 remains — the
real-model B-sweep bench (`parity/dsv4_batched_bench.py`), DEFERRED (solo GPU, not a
correctness blocker).** So #18 is effectively done. Full context, design, file/line
anchors, gates, and the M5 note are in **`PLAN.md`** (repo root). Cadence (standing user
instruction): single thread, NO subagents, commit each milestone, then STOP for the user
to compact before the next.

---

## Permanent engineering rules (do not violate)

These are non-negotiable and apply to every line of runtime/bake code:

1. **Build layers as `mlx.nn` modules.** Subclass `mlx.nn.Module`; compose with
   `nn.Linear`/`nn.RMSNorm`/`nn.QuantizedLinear`/etc. where they fit. Do not
   hand-roll parameter plumbing that `mlx.nn` already gives you. Simplicity of
   the layer definition is a feature.
2. **Prefer `mlx.fast` primitives, maximally.** Use `mx.fast.rms_norm`,
   `mx.fast.scaled_dot_product_attention`, `mx.fast.rope`, and any other
   `mx.fast.*` fused kernel instead of an equivalent hand-written sequence of
   ops. If a needed primitive is missing, wrap the closest `mx.fast` op and note
   why; do not silently reimplement it slowly.
3. **No Python loops on compute/hot paths.** Vectorize. Use batched ops,
   `mx.gather_qmm`/`mx.quantized_matmul`, `mx.compile` for stable shapes,
   broadcasting, `vmap`, and segment/gather primitives. The ONLY loops allowed
   are coarse, bounded, non-hot ones: iterating layers at load/bake time (one
   text layer resident at a time), IO/accounting boundaries, and the bounded
   `group_size` inner loop inside the GPTQ block solver. A loop over tokens,
   over experts per token, or over hidden dims is a bug.
4. **Parity-first.** No component is "done" until it matches a reference forward
   numerically (see Methodology). Optimizations (matrix-absorb, fused kernels,
   sorted dispatch, speculative decode) must be **output-equivalent** to the
   naive path and are kept behind a flag that defaults to the naive path until
   parity is proven.
5. **No `mlx-lm` as a runtime dependency.** `transformers`/`torch` are allowed
   **offline only** (parity references, tokenizers, source-checkpoint loading)
   under the `reference` extra — never on the inference hot path.
6. **No silent failures.** Code must work correctly or fail loudly. Never drop a
   baked tensor, never dequantize at the wrong bits by falling back to a default,
   never emit wrong output silently. Refuse to load a layer that bakes a tensor
   with no runtime consumer.
7. **Keep MoE routing sparse.** Never materialize a dense `tokens × experts ×
   hidden` intermediate. Route top-k, gather, dispatch.
8. **Layer-by-layer memory discipline.** Bake/calibration/parity must not hold
   more than one text layer's source weights resident at a time unless a measured
   exception is justified in the commit.

---

## Hardware / deployment target

- One **M3 Ultra**, 512 GB unified memory. Usable working-set ceiling
  **≈ 490.4 GiB** (`mx.metal.device_info()` recommended max working set). The
  whole quantized model is held **RAM-resident** (no offload/streaming); all
  current targets must fit under that ceiling.
- MLX is the runtime. `mx.set_wired_limit` pins the resident weight set.

---

## Methodology: parity-first (the core discipline)

Before optimizing or quantizing anything, establish a **reference** and diff
against it. Order of operations for any new model or layer:

1. **Reference forward.** Build a dead-simple, obviously-correct forward in
   plain `mlx.core` from the *dequantized* source weights (or a HF/transformers
   reference, offline). No fused kernels, no absorb, no rotations.
2. **Numeric parity, layer by layer.** Run identical token ids through both the
   reference and the runtime; capture the residual stream after each decoder
   layer and diff. The **first** layer/op that diverges beyond fp tolerance is
   the bug. Bisect within a layer across: RMSNorm placement, MLA attention
   (q/k/v projections, RoPE freqs, softmax scale incl. YaRN `mscale`, the
   matrix-absorb path), top-k routing, expert dispatch, shared expert.
3. **Only then** turn on an optimization or tighten quantization, re-running
   parity each time. A green parity gate is the definition of "done".
4. **End-to-end arbiter = teacher-forced perplexity** on real prose (with the
   correct BOS), plus top-1 next-token agreement vs the bf16 reference — not
   greedy generation (reasoning models loop under greedy regardless of quant;
   test behavior before blaming quant) and not per-expert reconstruction error
   (it does not predict e2e quality — see Settled Findings).

---

## Model facts — Kimi-K2.6

- DeepSeek-V3-style architecture. 61 decoder layers: **L0 dense**, **L1–L60 MoE**.
- MoE: **384 routed experts + 1 shared**, top-8, `noaux_tc` sigmoid routing with
  `e_score_correction_bias`. hidden=7168, moe_intermediate=2048.
- Attention: **MLA** (multi-head latent attention) with compressed KV latent;
  `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`,
  `kv_lora_rank`/`q_lora_rank` low-rank projections.
- RoPE: **YaRN**, `factor=64`, `rope_theta=50000`, `original_max=4096`,
  `beta_fast=32`, `beta_slow=1`, `mscale=1.0`, `mscale_all_dim=1.0`. The YaRN
  attention scale is `softmax_scale = (128+64)^-0.5 · mscale²` where
  `mscale = 0.1·ln(64)+1 ≈ 1.4159` (so `mscale² ≈ 2.005`). **factor is 64, not
  96** — a wrong factor uniformly degrades every token.
- Tokens: `bos=163584`. **Two distinct eos**: the tokenizer's nominal `[EOS]=163585`
  vs the model's *generation* eos `<|im_end|>=163586` (config.json / generation_config.json
  `eos_token_id`); plus end-of-turn `[EOT]=163593`. Generation/serving must stop on the set
  `{163585, 163586, 163593}` (`<|im_end|>` is the one the model actually emits to end a turn).
- Source checkpoint ships **int4** routed experts. Param split: routed gate+up
  ≈ 676.5B, routed down ≈ 338.2B (gate/up dominate ~2:1).

Keep `~/models/Kimi-K2.6` (the int4 source / reference teacher) — **never delete
it**. Baked artifacts and their `<artifact>_offload` siblings live under
`~/models`, outside this repo.

---

## Quantization policy

- **Routed experts (gate/up/down):** affine integer, group-128, per-projection
  bits chosen by the byte budget. int8-everything is ~lossless (~0.78% recon) but
  ≈975 GiB — does not fit. The split that fits ≤490 GiB is roughly **gate/up
  int3 g128 + down int4 g128** (≈438 GiB). Affine carries the zero-point bias
  that `mx.gather_qmm` needs. Whether int3 routed is *sufficient for coherence*
  is an OPEN question to be answered **only through a parity-correct runtime**
  (the int3-floor question).
- **Shared expert (gate/up/down):** **bf16, never quantized.** It runs on every
  token and is a single expert per layer, so full precision on the always-on path
  is ~free. Computed as `routed(x) + shared(x)`.
- **Attention + other matmul weights:** int8 (affine) or mxfp8.
- **norms, biases, router control tensors, positional/control tensors,
  tokenizer/data metadata:** bf16/fp32.
- Effective bits (affine) = `bits + 32/group_size`: int3 g128 = 3.25, int4 g128 =
  4.25, int8 g128 = 8.25 bpp.

---

## GPTQ — and how the matrix inverse is overcome

GPTQ minimizes the layer-wise quadratic `‖WX − ŴX‖²` over the quantized weights
`Ŵ`. Because that loss is *exactly* quadratic in `W`, the curvature is the **exact
Hessian** `H = XᵀX` (`X` = calibration activations, `[n_rows, in]`). There is
nothing to Taylor-approximate in forming `H` — GPTQ *is* the second-order
(Gauss-Newton) method. The cost is the inverse `H⁻¹` (an `[in, in]` solve;
`in = 7168` for gate/up, `2048` for down), recomputed per expert × 384 experts ×
61 layers. We overcome it on five fronts:

1. **Cholesky-of-the-inverse, not a per-weight inverse.** The Optimal-Brain-
   Surgeon update for quantizing column `j` and compensating the remaining columns
   needs only the rows of the upper-triangular factor `R` with `Rᵀ R = H⁻¹`.
   Compute `R` **once** and read every update coefficient off it (`R[j,j]` and
   `R[j, j+1:]`). No rank-1 re-inversion per weight. This is the classic GPTQ
   reformulation: one `O(in³/3)` factorization replaces `O(in³)` of repeated
   inverse downdates with bad locality.

2. **Damping for positive-definiteness.** `H ← H + λ·mean(diag H)·I` (λ≈0.01) so
   the Cholesky never fails on a rank-deficient `H` (which happens whenever an
   expert saw too few calibration rows).

3. **MLX CPU Cholesky (~32× over numpy).** Use `mx.linalg.cholesky` /
   `mx.linalg.cholesky_inv` on the **CPU stream** (MLX 0.31 has no GPU Cholesky —
   it errors "pass a cpu stream"). Measured: MLX CPU Cholesky 0.077 s vs numpy
   `inv`+`chol` 2.5 s. The "inverse" is thus a fast triangular factorization.

4. **Low-rank + diagonal Woodbury for under-covered experts (the Kimi win).**
   Under sparse top-8 routing over 384 experts with an ~8192-token calibration
   set, most experts see `n ≪ in` rows. Inverting `[in, in]` is wasteful when the
   data has rank ≤ `n`. Use the identity (exact, not an approximation):

   ```
   (λI + XᵀX)⁻¹  =  (1/λ)I − (1/λ²) Xᵀ (I + (1/λ) X Xᵀ)⁻¹ X
   ```

   which replaces the `[in, in]` inverse with the much smaller `[n, n]` Gram
   inverse `(I + (1/λ) X Xᵀ)⁻¹`. Trigger it when `n < woodbury_ratio · in`
   (≈0.5). The `λI` damping keeps both forms PD.

5. **Block + batched trailing update; shared-Hessian tail.** Quantize columns in
   `group_size` (128) blocks. Within a block, a *bounded* sequential loop over its
   ≤128 columns applies the `R`-coefficient compensation (the only sequential
   work). Between blocks, **one batched GPU matmul** propagates accumulated quant
   error to all trailing columns across every expert in the chunk at once
   (`[E,in,in] @ [E,out,in]`), so ~all FLOPs stay in dense GEMMs. Experts with
   `n < min_calib_rows` (128) reuse a pooled per-layer "shared-H" factor instead
   of a degenerate per-expert one, so cold experts are still well-conditioned.

> Status note: GPTQ produced ~4× lower per-expert reconstruction error than DWQ
> but **identical end-to-end perplexity** — proof that the int3 *coding method* is
> not the e2e lever. GPTQ stays in the toolbox; it is only worth re-running once
> the runtime is parity-correct and the int3-floor question is actually
> measurable. **Do not chase expert-quant quality before the runtime is correct.**

---

## Settled findings — DO NOT re-explore (see memory + INITIAL_PROMPT.md)

- int4 source ⇒ DWQ ≈ AWQ ≈ ~no help (scale-only methods have no headroom once
  the int4 grid already discarded the info). GPTQ error-feedback is the only
  expert-coding lever that moves recon — but not e2e.
- 3–5% *compounded* expert error is infeasible by bit allocation under 490 GiB
  (int4-all ≈ 12% recon AND ≈517 GiB > ceiling; only int8 is <1% but ≈975 GiB).
- Per-expert / compounded reconstruction error does **not** predict e2e
  perplexity. The only arbiter is teacher-forced ppl through a correct runtime.
- The e2e degeneration is **uniform** (flat per-position, flat across depth,
  wrecks even literal repetition/counting) and **expert-coding-independent**
  (GPTQ ≈ DWQ) ⇒ a localized bug in the shared forward path, NOT the experts.
- Already eliminated as the cause: RoPE `factor` (correct, 64) and the YaRN
  `mscale²` attention scale (correctly applied). Remaining suspects: int8
  attention quant, MLA matrix-absorb decode, RoPE freq construction, R2/R3
  rotations, top-k routing, KV/latent cache across positions.
- Reasoning models loop under greedy decoding regardless of quant — diagnose with
  perplexity/parity, not generation.

---

## MLX gotchas (0.31.x, this machine)

- `mx.fast.hadamard_transform` is orthonormal for `n = m·2^k`, `m ∈ {1,12,20,28}`,
  `k ≥ 1` (7168 = 28·256 ✓). **18432 = 9·2048 has NO valid factorization and
  silently returns a wrong result** — guard it; the dense FFN R4 uses 9 blocks of
  2048. Bare 12/20/28 fail to JIT.
- No GPU Cholesky (`mx.linalg.cholesky` needs a CPU stream). `mx.linalg.expm` is
  **absent** (a learned-rotation/SpinQuant path must use Cayley/QR).
- nvfp4 = group-16, mxfp8 = group-32. Affine packing is a contiguous LSB-first
  bitstream (validated == `mx.quantize` for bits 3/4/8).
- MLX slice-assignment works; `mx.async_eval` overlaps decode; one `mx.eval` per
  token (not per layer) lets MLX overlap the whole layer graph.

---

## Verification commands

Run targeted first, then broad, before committing:

```bash
uv run --with pytest pytest tests/ -q
uv run --with ruff ruff check src tests
uv run python -m compileall -q src tests
uv lock --check
git diff --check
```

---

## Memory

Permanent cross-session memory for this project lives in the auto-memory dir and
is loaded via `MEMORY.md`. The permanent engineering rules above are mirrored
there as a feedback memory so they are never dropped. Settled findings and the
user profile are seeded so a fresh session starts informed, not from zero.

## Git / collaboration rules

- Do **not** commit unless explicitly asked. Add files by name (never blind
  `git add -A`). Never push unless asked. Never skip hooks.
- Commit trailer: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- Baked artifacts are immutable bundles; manifest references are relative,
  in-artifact only (no absolute/source/symlink/cache paths). Runtime offload
  state lives in the sibling `<artifact>_offload`, never inside the artifact, and
  `manifest.json` is never mutated at runtime.
