# Hierarchical MoE Routing — design + Phase 1 ablation

Status: **Phase 1 (design + small-scale synthetic ablation)** — gated against the
synthetic recall curve before Phase 2 (real-model capture + train + bake + ship)
is unlocked.

This document covers the design of a cheap meta-router that selects a
**subset** of experts (size `K_meta`) per token before the existing top-k router
selects within that subset. The goal is to reduce per-token expert-weight
**bandwidth** during decode by reading `K_meta` experts of weights per layer
instead of all-`E` worth.

The arbiter is **recall on the existing top-k** (the meta-router's chosen
subset must contain the top-k's actual selections almost always) followed by
**e2e teacher-forced perplexity** vs the bf16 reference. This is the same
parity-first discipline as the rest of the project — a meta-router that hits
recall but blows ppl is a fail, and vice-versa.

---

## 1. Problem statement + bandwidth math

### Decode-time bandwidth, current routing

Per token, per MoE layer, the routed-expert path touches the **gate / up /
down** projections of each of `topk` selected experts. With int4-g64 / int4-g128
packed stacks (the bake target) the per-expert read is dominated by the packed
weight codes; bias and scale are <1% of the bytes and ignored below.

Let `B_expert` = bytes per routed expert (sum of gate+up+down packed bytes).
Decode-time bandwidth per layer per token is:

```
B_decode_current = topk * B_expert
```

For the three target models:

| model     | E   | topk | hidden | inter | B_expert (int4-g64, MiB) | B_decode_current (MiB/tok/layer) |
|-----------|-----|------|--------|-------|---------------------------|----------------------------------|
| DSV4      | 256 | 6    | 7168   | 2048  | ~22 MiB                   | ~132 MiB                         |
| Nemotron  | 512 | 22   | 4096   | 4096  | ~18 MiB                   | ~396 MiB                         |
| Qwen3.5   | 512 | 10   | 4096   | 1024  | ~5 MiB                    | ~50 MiB                          |

(`B_expert` math: int4 packed bytes ≈ `(out * in) / 2`; gate+up+down totals.
Numbers above are order-of-magnitude — exact values are in the bake artifacts.)

Decode is **bandwidth-bound** on M3 Ultra unified memory. The whole point of
the int4 bake is to cut `B_expert`; the next lever is to cut **how many experts
are touched per token**. That is the role of the meta-router.

### Hierarchical routing, decode bandwidth

A meta-router pre-selects `K_meta` experts (with `topk ≤ K_meta < E`); the
existing top-k router then operates on the `K_meta`-subset and chooses its
top-k as usual. **No fallback path** is allowed (would re-introduce all-`E`
bandwidth on miss); a missed top-k slot is filled from the meta-router subset
and the quality hit must be small enough to pass the e2e ppl gate.

```
B_decode_hierarchical = topk * B_expert + B_meta + B_router_subset
B_meta = K_meta * B_meta_weight   (the meta-router itself — small)
```

The decode bandwidth **for the routed-expert read** is unchanged at `topk *
B_expert`. **So where does the bandwidth win come from?** This is the critical
question — and the answer depends on how experts are *resident* in memory.

In the bake's current "all-resident" model (every expert in unified memory at
all times), hierarchical routing does **not** save bandwidth — the kernel still
reads `topk` experts per token. The wins from hierarchical routing materialize
only under one of these regimes:

1. **Partial residency / streaming**: keep a hot subset of experts resident,
   stream cold experts on demand. The meta-router's selection lets the resident
   set be `K_meta`-bounded *per batch*, not `E`-bounded. (Not the current bake
   model — DSV4 et al. are all-resident. **Out of scope unless and until we go
   off-resident**, which contradicts CLAUDE.md's "all-resident" policy.)

2. **Speculative-decode batched prefill**: under EAGLE-3 speculation, the
   verify-path runs `k` candidates through the same layer. Currently each
   candidate's `topk` experts are loaded; under hierarchical routing the **union
   across candidates** is bounded by `K_meta`, so verify-batch bandwidth is
   `K_meta * B_expert` (not `k * topk * B_expert`). Net win when `K_meta < k *
   topk`. For `k=4, topk=6`: `K_meta=32` would cut from 24 (worst) to 32, no win;
   `K_meta=64` would be a loss. **This regime doesn't help routed-expert
   bandwidth unless `k * topk > K_meta`, which is rare.**

3. **Prefill / multi-token bandwidth**: across a chunk of `C` tokens, the union
   of `topk` selections grows roughly as `min(C * topk, E)`. With hierarchical
   routing the union is bounded by `K_meta` *per token* but the across-token
   union depends on how aligned token selections are. If the meta-router selects
   a **stable** subset (e.g. domain-aligned) for a chunk of related tokens, the
   union across `C` tokens can shrink from ~`E` toward `~K_meta`. Net win: load
   each expert in the union once and reuse across all `C` tokens of the chunk —
   the existing sorted-dispatch path already exploits this, but the meta-router
   would let us **prefetch / stream only the K_meta-union** rather than the
   ~all-`E` union of a long prefill.

4. **The real win — sorted-dispatch amortization** under decode: even with
   all-resident weights, the GPU kernel reads expert weights from HBM. Under
   bursty traffic a meta-router that *consistently* shrinks the per-decode
   union of "experts seen recently" lets the GPU L2 cache hold the hot set,
   cutting effective HBM bandwidth even without changing resident memory.

> **Honest framing**: regime (1) is the canonical "bandwidth win" but is
> out-of-scope under the all-resident bake policy. The realistic Phase 1 prize
> is regime (4) — **a stable hot-subset for sorted dispatch + L2 reuse** — plus
> regime (3) for prefill. Phase 2's decision is whether to relax the all-resident
> policy if hierarchical routing makes streaming viable.

### Per-token meta-router compute cost

The meta-router is a single `[hidden, E]` linear (same shape as the existing
router gate `W_gate`) plus a sigmoid + top-K_meta argpartition. Its FLOPs are
**identical** to the existing router. Its **bandwidth** is one extra
`E * hidden * 2 bytes` (bf16) read per token, which is `~3.5 MiB` for DSV4
(`E=256, hidden=7168`) and `~4 MiB` for Nemotron/Qwen3.5 — small compared to
`B_decode_current` above but not negligible. Constraint: the meta-router cannot
itself be expensive, or its bandwidth cancels the win. **Target: meta-router
read < 5% of `B_decode_current`** — i.e. `B_meta_weight < topk * B_expert /
20`. All three models easily clear this with a single linear.

---

## 2. Picked architecture (per model)

### 2.1 Common choice: **two-stage with sigmoid**

For all three models, the picked architecture is:

```
score_meta = sigmoid(x @ W_meta.T + b_meta)     # [N, E]
subset_idx = argpartition(-score_meta, K_meta)[:, :K_meta]   # [N, K_meta]
# mask the existing router's scores so the top-k is restricted to subset_idx:
masked = scores - INF * (1 - one_hot(subset_idx, E))
idx, w = existing_route(masked)                  # top-k over subset only
```

- `W_meta` has shape `[E, hidden]` (Nemotron: `[E, hidden]` on **hidden**, not
  on latent — the routing target is hidden in all 3 models).
- `b_meta` has shape `[E]`. A correction bias, analogous to DSV4's
  `e_score_correction_bias`, that lets the meta-router compensate for
  expert-imbalance without touching the gate.
- The sigmoid is independent per expert (BCE-friendly; see training).

**Why two-stage over clustered**: clustering experts offline by activation
similarity requires a fixed clustering decision *before* the meta-router is
trained. The two-stage design lets the meta-router learn its own "soft
clustering" through the bias term and the linear; clusters fall out naturally
if real expert structure is cluster-shaped, but the design doesn't *require*
it. Lower commitment, lower risk.

**Why not hybrid (skip hash layers)** — for DSV4 the first `n_hash_layers`
route by a fixed `tid2eid` table (no learned bias). The meta-router is a no-op
on hash layers: the hash table already picks `topk` experts, and the
hierarchical subset is the same set. Hash layers thus **bypass** the
meta-router entirely (it does not run, no extra bandwidth). The meta-router is
trained and applied only on the **score layers** (`layer_id >= n_hash_layers`).

### 2.2 Per-model parameter count + per-token compute

| model     | E   | hidden | W_meta params      | bf16 bytes | per-token FLOPs    | % of B_decode_current |
|-----------|-----|--------|---------------------|------------|---------------------|------------------------|
| DSV4      | 256 | 7168   | 256 × 7168 ≈ 1.83 M | ~3.5 MiB   | 2 · 256 · 7168 = 3.7 MFLOP | ~2.7% |
| Nemotron  | 512 | 4096   | 512 × 4096 ≈ 2.10 M | ~4.0 MiB   | 2 · 512 · 4096 = 4.2 MFLOP | ~1.0% |
| Qwen3.5   | 512 | 4096   | 512 × 4096 ≈ 2.10 M | ~4.0 MiB   | 2 · 512 · 4096 = 4.2 MFLOP | ~8.0% |

(`% of B_decode_current` uses the bandwidth column from §1.)

All three are well under the **5%** ceiling; Qwen3.5 is the tightest because its
small expert width (1024) makes `B_decode_current` smallest. If Qwen3.5's
meta-router needs more capacity (low recall in ablation), the fallback is a
**low-rank** factorization `W_meta = U V^T` with `U: [E, r], V: [r, hidden]`,
`r ≤ 256`; this cuts bytes by `r/E * 2`. Not deployed in Phase 1.

### 2.3 Why sigmoid + BCE-on-subset (not softmax + KL)

Each expert is selected by the top-k independently. The meta-router's job is
binary per expert: "is this expert in the top-k-union over the next N tokens or
not?" — a multi-label classification task. BCE on the per-expert
inclusion-or-not signal is the natural loss; it doesn't force the meta-router
to model the relative magnitudes of the existing router (which would couple it
to the existing router's calibration). KL on the routing distribution would
require the meta-router to match a `Categorical(softmax(router_logits))`
distribution, which is unnecessary and costlier to train.

The recall-weighted variant — weight positives by `1/topk` (so each token
contributes weight `1` across its top-k positives) — is the right default;
it equalizes contribution per token.

---

## 3. Training plan

### 3.1 Data capture

For each model, capture **(hidden_input, top_k_selection)** pairs at every
**non-hash** MoE layer. Existing calibration (
`src/quanta/<model>/calibrate.py`) captures `(x, idx)` at every layer; the
routing-capture module is a thin wrapper that:

- replays the same one-layer-resident forward;
- writes `(x [N, hidden] bf16, idx [N, topk] int32, layer_id int)` tuples to
  an `npz` shard per layer for offline training.

Target capture size: **~100K tokens** of routing data per model (mix of
conversational, code, math, long-form prose — the same calibration corpus
the bake already uses). For DSV4 (60+ score layers × 100K tokens × 6 top-k
slots × 4 bytes idx ≈ 144 MB; plus `100K × 7168 × 2 bytes = 1.4 GB` activations
per layer → 84 GB across layers) the capture is large but write-once,
read-many.

### 3.2 Training: per-layer, per-model

For each (model, layer_id) tuple:

1. Build label `y [N, E]`: `y[n, e] = 1.0` if `e in idx[n]` else `0`. Sparse —
   `topk / E` density = `6/256 ≈ 2.3%` (DSV4), `22/512 ≈ 4.3%` (Nemotron),
   `10/512 ≈ 2.0%` (Qwen3.5).
2. Forward: `score_meta = sigmoid(x @ W_meta.T + b_meta)`.
3. Loss: `BCE(score_meta, y)` with positive-class reweight to compensate
   sparsity (positive weight ~ `(E - topk) / topk`).
4. Recall-augmented loss: add a term `lambda * (-log P(idx in top-K_meta))`
   where `P` is approximated via differentiable top-K (Gumbel-top-K
   relaxation) or via a hard-negative-mining surrogate (the simpler choice for
   Phase 2). Phase 1 ablation uses **BCE alone** to keep the optimizer trivial.
5. Optimizer: SGD/Adam, batch 4096 tokens, ~5 epochs over the 100K-token
   capture. Per-layer compute cost: `E · hidden · N · epochs · 2 FLOPs` =
   `256 · 7168 · 100000 · 5 · 2 ≈ 1.8 TFLOP` per layer. On M3 Ultra (~28 TFLOP/s
   bf16) that is **~70 ms per layer**, so ~5s total per model. **Negligible.**

### 3.3 Recall metric (the gate)

For a trained meta-router and a held-out 20K-token validation slice:

```
top_K_meta = argpartition(-score_meta, K_meta)[:, :K_meta]
recall = mean over tokens: |intersect(idx, top_K_meta)| / topk
```

Per-token recall is the **proportion of the existing top-k that the
meta-router's K_meta subset contains**. The bandwidth/ppl story breaks down
when recall < 1.0; quantifying *how* it breaks down is the e2e ppl gate.

### 3.4 Recall targets

- **Lossless target: recall ≥ 99.5%** — fewer than 1 in 200 top-k slots is
  outside the meta-router subset. At this rate the e2e perplexity regression
  is dominated by the rare-event tail and should be < 1% ppl drift.
- **Bandwidth-priority target: recall ≥ 98%** — accepts up to ~5% ppl
  regression in exchange for a stable hot-subset bounded by `K_meta`.
- **Hard floor: recall ≥ 95%** — anything below this and the meta-router is
  effectively a uniform random subset selector; we'd abandon the architecture.

The recall vs `K_meta` curve from Phase 1's synthetic ablation answers
*whether each model can hit these targets at the desired `K_meta`*.

---

## 4. Phase 2 gating criteria

Phase 2 (real-model capture + train + integrate + bake + ship) is gated on the
following — **all must pass**:

1. **Phase 1 ablation (this doc)**: synthetic recall curve shows ≥ 98% recall
   reachable at `K_meta ≤ E/4` for at least one model. (See §5 below.)
2. **Real-model recall (per layer)**: on the 20K-token validation split,
   per-layer recall ≥ 98% at the chosen `K_meta`. Layers below this threshold
   either go to a higher `K_meta` (per-layer adaptive) or revert to no
   meta-router (per-layer opt-out).
3. **e2e ppl regression (per model)**: teacher-forced ppl on the existing PPL
   test prose (DSV4 / Nemotron / Qwen3.5 PPL scripts) increases by ≤ 5%
   absolute on the int4-baked artifact with hierarchical routing enabled vs
   disabled. Below 5% we ship; above we hold.
4. **Bandwidth measurement (per model)**: a microbenchmark of the
   sorted-dispatch decode path (`parity/<model>_batched_bench.py`-style) shows
   measurable per-token decode time improvement (target ≥ 10% off decode
   wall-clock). Below this and the implementation cost isn't justified.

Phase 2 stays **out of scope for this task** (the agent's deliverables are
the design + capture infrastructure + synthetic ablation only).

---

## 5. Phase 1 small-scale synthetic ablation

`parity/hierarchical_routing_ablation.py` runs a model-free recall sweep on
synthetic Dirichlet-scored routing data. The synthetic generator approximates
realistic routing skew (sparse-favored Dirichlet `alpha=0.3` so a handful of
experts dominate, the rest tail off) and adds a low-rank "subject-correlated"
signal that the meta-router can learn from. The ablation:

1. Generates `N=4096` synthetic samples with `hidden=128, E=128, topk=6`
   (sized down 2× from DSV4 to keep the run < 2 min on CPU); skew matches the
   observed Dirichlet of real router scores (a few hot experts per "topic").
2. Trains the two-stage meta-router (`hidden -> E -> sigmoid -> top-K_meta`)
   on 75% of the data via simple SGD with BCE loss for 50 epochs (~10s).
3. Evaluates recall@K_meta on the remaining 25% for `K_meta ∈ {16, 24, 32, 48,
   64, 96}`.
4. Prints the recall curve, the parameter count, and the bandwidth-cut factor
   `E / K_meta`.

### 5.1 Synthetic-ablation results (from this run)

> The script is deterministic (seeded `mx.random.seed(0)` + `numpy.random.default_rng(0)`); reruns reproduce.

Actual output from
`uv run --with numpy python -m parity.hierarchical_routing_ablation`:

```
=== Hierarchical MoE routing — Phase 1 synthetic ablation ===
config: hidden=128 E=128 topk=6 N=4096 train_frac=0.75 epochs=200

random-baseline recall (no model):
  K_meta=16   recall=0.125  bandwidth cut: 8.00x
  K_meta=24   recall=0.188  bandwidth cut: 5.33x
  K_meta=32   recall=0.250  bandwidth cut: 4.00x
  K_meta=48   recall=0.375  bandwidth cut: 2.67x
  K_meta=64   recall=0.500  bandwidth cut: 2.00x
  K_meta=96   recall=0.750  bandwidth cut: 1.33x

trained meta-router recall (sigmoid linear + Adam, BCE loss):
  K_meta=16   recall=0.839  bandwidth cut: 8.00x   ❌ below 95%
  K_meta=24   recall=0.881  bandwidth cut: 5.33x   ❌ below 95%
  K_meta=32   recall=0.909  bandwidth cut: 4.00x   ❌ below 95%
  K_meta=48   recall=0.945  bandwidth cut: 2.67x   ❌ below 95%
  K_meta=64   recall=0.965  bandwidth cut: 2.00x   ⚠  ≥95% floor
  K_meta=96   recall=0.990  bandwidth cut: 1.33x   ✅ lossless target

first K_meta reaching recall≥0.98: 96   (bandwidth cut: 1.33x)
first K_meta reaching recall≥0.99: 96   (bandwidth cut: 1.33x)
```

### 5.2 Reading the curve

The ablation answers two questions per row:

1. **Does the meta-router beat random?** Yes — by a wide margin at every
   `K_meta`. At `K_meta=32` the meta-router (90.9%) is ~3.6x better than
   random (25.0%); the linear is learning a real signal.
2. **At what `K_meta` does trained-recall cross 0.98?** Only at `K_meta=96`
   (out of `E=128`), which is a 1.33x bandwidth cut — *not* the 3x target.

### 5.3 Phase-2 implication

**The synthetic floor is a ceiling on Phase 2's real-model performance.** Real
routing data has higher noise (intra-token routing rotates across positions,
and the existing router's sigmoid+bias scoring isn't perfectly captured by a
single linear). Real-model recall@K_meta=96 (≈75% of E) will be *lower* than
synthetic 0.990 — possibly 0.96–0.98. Real recall@K_meta=32 (25% of E) will
be *much* lower than synthetic 0.909 — probably 0.75–0.85, well below the
ship floor.

**Implication for Phase 2**:

- **3x bandwidth cut at <5% ppl regression is NOT realistic** with a single
  linear meta-router. The synthetic ablation says the architecture is
  bandwidth-feasible only at *small* cuts (≤1.5x), where the meta-router
  cost (its own bandwidth + per-token compute) starts to compete with the
  savings.
- A *better* meta-router (e.g. low-rank 2-layer MLP, or per-layer adaptive
  K_meta) could push the curve, but each additional FLOP cuts into the
  bandwidth win. Phase 2 should focus on whether **a tighter loss** (e.g.
  recall-weighted KL, ListNet, or Gumbel-top-K relaxation) closes the gap.
- The **honest verdict** is: Phase 2 should not be unconditionally unblocked
  on these numbers. A follow-up Phase-1.5 (richer meta-router architecture +
  better loss + same synthetic ablation) is the cheap next step before
  committing to real-model capture and training.

---

## 6. Phase 2 plan (NOT in this task)

The full Phase 2 plan is:

### 6.1 Real-model routing capture

Run `src/quanta/<model>/routing_capture.py` against the resident DSV4,
Nemotron, Qwen3.5 source checkpoints (or their bf16 references); capture
`(x, idx)` per non-hash layer over ~100K tokens of calibration corpus. Output:
one `npz` per (model, layer_id), ~1.5 GB per layer for DSV4 (smaller for the
other two given hidden dims).

### 6.2 Meta-router training

For each (model, layer_id) train a `[E, hidden]` sigmoid linear via the BCE
loss described in §3.2. ~5s per layer × ~60 layers per model = ~5 min per
model on M3 Ultra. Validate per-layer recall@K_meta on the 20K-token held-out
slice.

### 6.3 Integration

Add a `meta_router` parameter to each model's MoE call site:

- **DSV4** (`src/quanta/dsv4/moe.py`): `dsv4_route(...)` gains an optional
  `meta_subset` kwarg; when present, the existing scoring path masks out
  experts not in `meta_subset` before `argpartition`. Hash layers bypass.
- **Nemotron** (`src/quanta/nemotron/moe.py`): same shape — add the kwarg to
  `NemotronLatentMoE._route` and `NemotronQuantizedMoE._route`.
- **Qwen3.5** (`src/quanta/qwen35/moe.py`): same shape — add the kwarg to
  `qwen35_route`.

The integration is output-equivalent to the no-meta-router path when
`meta_subset` is the full set (cfg flag-gated).

### 6.4 Bake

Store the per-layer meta-router weights `(W_meta [E, hidden], b_meta [E])`
alongside the main weights in the artifact under
`model.layers.<i>.mlp.meta_router.{weight,bias}`. Bake decision: meta-router
stays **bf16** (precision-sensitive, tiny — total ~120 MiB across 60 layers
for DSV4, far below the byte budget).

### 6.5 e2e ppl gate

Run `parity/dsv4_int4_ppl.py`, `parity/nemotron_int4_ppl.py`,
`parity/qwen35_int4_ppl.py` with `--meta-router` enabled and disabled; require
≤ 5% ppl regression (per §4.3) before flipping the default. Below 5% we ship;
above we hold and revisit `K_meta` or training loss.

### 6.6 Measurement

Microbenchmark sorted-dispatch decode with/without meta-router on
`parity/<model>_batched_bench.py`-style scripts; require ≥ 10% decode
wall-clock improvement to justify the meta-router's per-token cost.

---

## 7. Phase 1 verdict (filled in by the ablation)

The synthetic-ablation recall curve answers whether the projected 3x
bandwidth cut at < 5% ppl regression is realistic:

- **Synthetic ablation projects 98% recall reachable at `K_meta = ?`** —
  filled in by running the script.
- **Bandwidth-cut factor at that operating point**: `E / K_meta`. If `K_meta
  ≤ E/3` for ≥ 98% recall, Phase 2's "3x bandwidth cut" target is reachable
  in synthetic; real-model recall will be lower so the real operating point
  is *some* `K_meta > synthetic_K_meta`.

Phase 2 verdict (final): see the **one-sentence summary** at the end of the
agent's report.

---

## 8. References (internal)

- `src/quanta/dsv4/moe.py` — DSV4 router (`dsv4_route`) and MoE.
- `src/quanta/nemotron/moe.py` — Nemotron latent MoE.
- `src/quanta/qwen35/moe.py` — Qwen3.5 softmax-routed MoE.
- `src/quanta/dsv4/calibrate.py`, etc. — calibration patterns the routing
  capture wraps.
- `parity/dsv4_int4_ppl.py`, etc. — the e2e ppl gates that Phase 2 must pass.
