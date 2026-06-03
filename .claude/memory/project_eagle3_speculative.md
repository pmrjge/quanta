---
name: project-eagle3-speculative
description: "EAGLE-3 spec-decode for Kimi is BUILT but FORGONE — benchmark confirmed it's a no-op (MoE top-8 dilutes the verify win). Drafter/train findings kept for any future dense target."
metadata:
  node_type: memory
  type: project
  originSessionId: e88fadb0-5f2c-48da-82a1-e14175af6551
---

**DECISION (2026-05-28): EAGLE-3 on Kimi-K2.6 is FORGONE — a no-op per our own benchmark.** The
drafter is built and correct (lossless vs greedy), and the trained version reaches healthy accept
rates, but the wall-clock speedup does not materialize on Kimi: under 384-expert **top-8** routing,
verifying K drafts activates the *union* of their experts (~30–60, not 8), so each accepted token
still pays ~the full routed-expert load cost — high acceptance never converts to throughput. We are
**not** pursuing #54 (embed-into-artifact) or any further Kimi retrain/benchmark. Do **not**
re-explore EAGLE-on-Kimi. The drafter/training findings below are retained because they are reusable
on a future **dense / low-active-param** target (no MoE union-of-experts tax → spec-decode can pay
off there; e.g. a dense 7B like InternLM2.5/Qwen2.5 with no native MTP).

EAGLE-3 speculative decode for Kimi-K2.6 was **implemented**: drafter (`quanta/eagle/drafter.py`),
feature capture from the quantized target, training-time-test loop (`quanta/eagle/train.py`),
lossless spec-decode + verify (`quanta/eagle/spec.py`). Tasks #48–52 done; #54 (embed the drafter
into the artifact) is **dropped**, not pending.

**Architecture (matches train.py + spec.py).** The drafter fuses the target's **low/mid/high**
hidden features (capture layers **10, 30, 50** → `LAYERS`), reduces them to one feature, and at each
step consumes the *previous* reduced feature `f_{p-1}` + the *next* token embedding `e_p` to predict
`token_{p+1}` through the target's **frozen** embed+head (predicts in the target's logit space).
Verify = one target forward over `[cur, d1..dk]`; accept the longest matching prefix +1 bonus; roll
back the MLA cache (`truncate`) on rejects ⇒ **bit-identical to greedy** (the drafter changes only
speed, never output). It self-feeds its own **normalized** output as the recurrent feature.

**The training-correctness finding (2026-05-24) — the architecture WAS right, the loss was wrong.**
The first trained drafter gave **~0.39× / ~4.5% accept** (slower than no speculation); a linear probe
on the same features beat it ⇒ a **training bug, not the data/architecture**. Two coupled causes,
both fixed in `train.py`/`spec.py`:
1. The old loss paired `(f_p,e_p)→token_{p+1}` with **no feature-regression term**, so the recurrent
   feature was never trained → steps-2+ accept ≈ 0 under self-feed.
2. The fix regresses/self-feeds the **normalized** feature `normed = out_norm(x)` (unit scale,
   matches the reduced target `red`). Regressing/self-feeding the **raw residual `x`** (large
   magnitude) explodes smooth-L1 and starves CE.
Canonical loss now = next-token **CE** + `feat_w`·**smooth-L1** feature regression (stop-grad
target), self-feeding `normed` across `steps` for the multi-step training-time-test rollout.

**Durable caveats (still true).** Accelerates **decode**, not prefill (long *prompts* are
FLOP-bound — [[prefill-optimization-landscape]]). **MoE top-8 sparsity dilutes the win** — verifying
K drafts activates the *union* of their experts (~30–60), not 8 (shared+attention amortize fully,
routed only partially). It's **lossless** vs greedy, so it obeys the parity-first rule.

**How to apply.** Retrain on a larger capture (204K tokens) with fewer epochs and confirm per-step
holdout top-1 climbs off the old ~10%/~0; then benchmark `drafter_ttt.safetensors` (389 GB resident,
run **alone** — one big process at a time, [[feedback-memory-safety]]) for real speedup before #54.
