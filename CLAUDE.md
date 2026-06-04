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

## Active task (transient — full handover in PLAN_nemotron_ultra.md)

**In flight (PIVOT, this session): Nemotron-3-Ultra-550B (main agent) + Mellum2-12B (orchestrator)
agentic stack.** Handover **`PLAN_nemotron_ultra.md`**. Quantize Nemotron-Ultra (hybrid
Mamba2+attn+MoE, `nemotron_h` — already supported; the 120B-Super sibling is already baked int4) as
**int4-GPTQ experts + int8 dense + bf16 core** (keep the proven `nemotron/quant_policy.py`, NOT AWQ —
GPTQ is the stronger lever on a bf16 source), then Mellum2 (`mellum`, a new port) as **int8**; **one
model resident at a time**; drive Ultra first. **U0 ✅** — config adapter (`_hybrid_pattern`
normalises the newer explicit-`layers_block_type` schema, which omits `num_hidden_layers`) +
fit-check: Ultra parses, the derived split reproduces the explicit list bit-for-bit, the quant policy
covers all 51,023 tensors (rule #6), and the mix is resident at **289.7 GiB ≤ 490.4** (200.7 GiB
headroom) — `parity/nemotron_ultra_fit_test.py`. **U1 ✅** — per-layer numeric parity vs an
**independent transformers `NemotronH*` reference** at full Ultra scale, layer-streamed (rule 8: one
real layer resident, the moe's ~21.5 GiB expert stacks the peak), `parity/nemotron_ultra_layer_parity.py`:
**mamba** our `MambaMixer` vs `NemotronHMamba2Mixer` (fp32, Δ 3.1e-04), **attn** ours vs transformers'
`apply_rotary_pos_emb`+`eager_attention_forward` (Δ 4.5e-06), **moe** router top-22 set+weights vs
`route_tokens_to_experts` (set-exact, w Δ 1.2e-07 — our `noaux_tc` sigmoid routing is provably exact)
+ experts/latent/shared vs inline-dense (Δ 7e-04) + chunk-invariant. **The gate caught a real
forward-path bug** (the kind CLAUDE.md's thesis warns about): the Mamba-2 **gated RMSNorm is
group-wise** (variance over `d_inner//n_groups`, NOT full `d_inner` — `Zamba2RMSNormGated`); ours was
a full-width `nn.RMSNorm` — *self-consistent* (prefill==decode) so the old self-consistency-only test
never caught it, but **42% off** the reference. Fixed via a new group-wise `MambaRMSNormGated`
(`mamba_mixer.py`, forward-only — corrects the **already-baked Super-120B** too; bf16 `norm.weight`
unchanged, no re-bake; Super ppl should be re-measured under the fix). **U2 next** = full int4-GPTQ +
int8 bake (layer-streamed, hours, solo) → `…-quanta_int4g64`. The InternLM2.5 MInference track below
is **paused at M6 ✅ (M7 deferred)**, not abandoned.

**Paused: InternLM2.5 sparse-prefill (MInference family) — M6 ✅, M7 next.** Handover
**`PLAN_minference.md`**. Reuse the validated block-sparse substrate (`quanta.modeling.xattention`,
`gather_sparse_attention`/`sparse_prefill_mask`, `threshold=1.0`==dense); M0 wired a `self.sparse`
hook into `InternLM2Attention` (default None = dense byte-unchanged). M1 measured XAttention's lossy
lever on the int8-g64 bake (`parity/internlm2_ppl_sparse.py`, solo GPU): prefill @ threshold 0.9 costs
**+0.31% ppl** (knee t=0.80 +2.39%) — "free"; gather speed-path == mask quality-path. **M2 added
MInference's A-shape selector** (sink block 0 + `local`-block window) onto the SAME execution via a
`selector` discriminant on `XAttnConfig` (`"xattn"` default byte-unchanged; `"ashape"` new) +
`select_keep` dispatch (`xattn` path byte-for-byte preserved) + model-free gate
`parity/internlm2_ashape_test.py`: A-shape keep-all **== dense EXACTLY**, gather==mask, measured cost
**L=4 (512-tok) +0.58% / L=2 (256-tok) +3.76%** (cheaper-but-lossier than XAttention, per MInference).
**M3 added MInference's vertical-slash selector** (online last-query-block probe → ONE global pattern:
top-`vert` vertical key-blocks ∪ top-`slash` slash block-offset bands, MInference §3) onto the SAME
execution via a `"vslash"` `select_keep` branch + precomputed global `index` threaded into every gather
chunk (so gather==mask); model-free `parity/internlm2_vslash_test.py` (causal/anchor/twin) + real-model
gate: vslash keep-all **== dense EXACTLY**, gather==mask @v3s3, measured cost **v3s3 +3.01% / v2s2
+7.29%** (lossiest of the three at this 7-block doc — vertical-slash is a long-context, per-head-assigned
pattern; integration green is the point, not winning at 7 blocks). **M4 made the selector per-head**: a
`head_selectors` tuple on `XAttnConfig` (None = uniform, byte-unchanged) routing each query head to its
own kind via `_select_keep_per_head` (bounded loop over the ≤3 KINDS, not heads → one `take_along_axis`;
each head's keep == the uniform keep for its kind — pure routing); offline policy `assign_head_selectors`
(cheapest candidate within `tol`, else accurate fallback); a parity-preserving `InternLM2Attention.
_attn_heads` extraction so the ppl harness measures per-head error vs dense. Model-free
`parity/internlm2_perhead_test.py` (policy + routing-exactness + mixed-keep-all==dense + gather==mask +
validation) + real-model gate: **perhead mixed keep-all == dense EXACTLY**, gather==mask (8.88e-3),
measured **+0.40% ppl** with the offline router assigning **86% xattn / 14% A-shape / 0% vslash** (Σ
32×32 heads) — buys back A-shape-L2's +3.76% → +0.40% (≈ best uniform xattn +0.31%) while running 14% of
heads on the cheaper static kernel; vslash 0% at 7 blocks (long-context pattern, per M3). **M5 made the
selector per-head *params*** (not just kind): a frozen `HeadSpec(kind, threshold, local, vert, slash)` +
`head_specs` tuple on `XAttnConfig` (None = M4/uniform, byte-unchanged; precedence over `head_selectors`,
both-set rejected) routing each head to its own (kind, params) via `_select_keep_per_head_specs` (bounded
loop over DISTINCT specs, not heads → one `take_along_axis`; vslash params shared via the threaded global
index, fail-loud pin; ashape/xattn params freely per-head); offline policy `assign_head_specs` = the dual
of M4's (most-accurate candidate within a kernel-aware FLOP `budget`, else cheapest); a parity-preserving
`_attn_qkv` extraction shared by `_attn_heads` + the new offline `_attn_keep_counts` (per-candidate cost =
mean kept blocks). Model-free `parity/internlm2_perhead_params_test.py` (budget policy + routing-exactness
incl. same-kind-different-params + mixed-keep-all==dense + gather==mask + validation) + real-model gate:
**perhd-p mixed keep-all == dense EXACTLY**, gather==mask (3.29e-3), measured **+0.15% ppl** — **beats M4's
per-head-kind +0.40% AND best uniform xattn +0.31%** — with the FLOP-budgeted search (budget=4 blocks)
assigning **75% ashape:L4 / 23% xattn:t0.9 / 1% vslash / 1% xattn:t0.95** (Σ 32×32 heads); per-head params
let 75% of heads run the cheap static kernel while each still gets its most-accurate-affordable approx, so
the aggregate beats any uniform — the MInference thesis. **M6 made per-head *vslash params* vary** (lifted
M5's vslash-pin): `vertical_slash_index` now returns **param-independent** masses `(key_mass, slash_mass)`
and the top-`vert`/`slash` cut moved into `select_keep`, so two heads read the ONE global probe yet cut
DIFFERENT vert/slash from the shared masses (`__post_init__` pin removed; M3/M4/M5 vslash *selections*
byte-identical — same masses + same top-k, relocated). Model-free `parity/internlm2_vslash_perhead_test.py`
(two vslash heads at different vert/slash each == its uniform spec & keep different blocks; config-vert/slash
irrelevance; mixed keep-all==dense; gather==mask) + real-model gate (ppl harness search grid gains a 2nd
vslash param v2s2+v3s3): **perhd-p keep-all == dense EXACTLY**, gather==mask (7.45e-4), measured **+0.04% ppl
— beats M5's +0.15%** with the FLOP-budgeted search assigning **73% ashape:L4 / 22% xattn:t0.9 / 4%
vslash:v3s3 / 1% xattn:t0.95** (4% of heads now run the WIDER vslash, vs M5's 1% — per-head vslash params pay
off even at 7 blocks; M1–M5 reproduced bit-identically). M7 next = **key-chunk the long-context probe**
(accumulate the param-independent masses over key chunks so it scales to 100K+; single-shot stays the
short-doc default) + a wall-clock **gather-path prefill bench**, ppl-gated vs M6.

Prior InternLM2.5 **EAGLE spec-decode** track is **COMPLETE** (M0–M3, `ec0f6f3`; **1.42× lossless @
k=2** via drafter quantization — memory `project_internlm2_eagle.md`). The earlier batched-decode /
paged-KV / expert-footprint sweep across the serving keepers (DSV4, Nemotron, InternLM2.5, Qwen3.6)
is fully landed:

- **#18** — kill the per-stream KV-update IO loop in DSV4 batched decode via a persistent
  `max_batch` **batched KV arena** (ONE scatter + ONE gather; flag `kv_arena`, default ON):
  **COMPLETE M0–M5** (`41a4d0f`/`6f33cc1`/`05d1171`/`bf7af6b`/`e08888d`/`f4935b5`; M5 real-model
  bench arena **greedy-exact** vs the per-stream loop AND **+37% decode tok/s @ B=32**).
  Handover **`PLAN.md`**.
- **#152** — block-paged KV with copy-on-write prefix sharing: **CLOSED**; `PAGED_KV_DEFAULT`
  ON; all keepers real-paged-green.
- **#153** — bring the #18 loop-kill to the PROD **paged** path (ONE block-table scatter + ONE
  gather): **COMPLETE across all keepers + Qwen3.6** — DSV4 M0–M4
  (`62609ba`/`c442c31`/`35dcd78`/`d19a254`/`cb2476b`, +13% @ B=32/48), Nemotron (+18% @ B=48),
  InternLM2.5 (**3.20× @ B=32**), Qwen3.6 option-B (1.63× @ B=32) — each graduated ON behind its
  own scoped flag. Handover **`PLAN_153.md`**.
- **qwen35 routed-expert packing** — keep int4 experts packed + `mx.gather_qmm` instead of
  dequant-to-bf16: **COMPLETE** (`a6b3b49`/`d17882e`/`f720fda`, marked complete `b62596e`;
  resident **63→20 GiB**, greedy-exact, ppl unchanged). Handover **`PLAN_qwen35_experts.md`**.

Optional, non-blocking: extend the #18 bench to B=48/64 on a free solo GPU (largely subsumed —
#153 M4 already benched DSV4 at B=48). Cadence (standing user instruction): single thread, NO
subagents, commit each milestone, then STOP for the user to compact.

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
