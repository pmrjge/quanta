"""#124 EAGLE stabilized retrain on the int3g128 1M-mix features (task #124).

The unconstrained Adam run (``parity/eagle_train_mix.py``, lr=2e-4, no clip, no schedule) reached
per-step accept `[0.436, 0.331, 0.275, 0.241]` (mean 0.321) at epoch 24, then diverged: loss
2.07 → 6.89 → 7.33 by epoch 27, early-stopped after patience=8. The eagle_spec_sweep landed at
**1.16x walltime / mean_accept 2.11** with the best-epoch (24) drafter at k=2 — across the
break-even line by 16%, but the drafter caps `mean_accept` at ~2.1 across k values, suggesting
capacity or training quality is the ceiling rather than the spec config.

Four stabilizations, mirroring the standard recipe for Adam late-stage blowup:

* **lr 2e-4 → 1e-4** — halve the Adam step; the blowup was a single bad batch sending one weight
  off-cliff, and a smaller step makes that recoverable.
* **grad clip max-norm 1.0** — bounds the worst-case Adam-state corruption per step (a single
  exploded grad coordinate can no longer push the update past the basin).
* **warmup + cosine decay (100 warmup steps, then cosine to 0.1·lr)** — warmup lets Adam's moments
  populate against a small lr; cosine decay shrinks the late-epoch step where the prior run blew up,
  so the same trajectory at epoch 24 doesn't have the energy to diverge by epoch 27.
* **fresh seed=42** — the original run's chunk-permutation order may have hit a pathological batch
  sequence; a different seed orders the same data differently so we don't reproduce that path.

**Warm-start from the epoch-24 checkpoint** of the unconstrained run
(``drafter_int3g128.safetensors``, the currently-shipping drafter giving 1.16x at k=2). Adam state
restarts from zero — the warmup_steps ramp re-applies, so the same stabilization recipe runs but
starting AT the prior peak instead of from random init. This isolates the stabilization effect to
exactly the divergence-risk region (epoch 24+) and skips the ~12 epochs of re-climbing that
fresh-init wastes. Epochs 40 → 15: that's the budget needed to push past epoch 24's mean 0.321
without the explosion that hit at epoch 27 originally.

Everything else matches ``eagle_train_mix.py``: same features, same drafter geometry,
``batch=8, steps=4, holdout=4, feat_w=1.0``. The output goes to a NEW path so the current drafter
(which still gives 1.16x at k=2) stays available as a fallback. If the stabilized run beats
mean_accept 2.11, we ship it; if it doesn't, we keep the current drafter at k=2 and either accept
the 1.16x or move on.

NOT a big-resident job (no main model — embed/head ≈9 GiB + features ≈43 GiB only), but does use
the GPU heavily for ~6h. Run only when no big-resident job is alive.

    uv run python -m parity.eagle_train_int3g128_stable
"""

from __future__ import annotations

import time

from quanta.eagle.capture import load_feature_shards
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.train import load_frozen_embed_head, train_drafter

ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int3g128"
SHARDS = "/Users/pmrj/models/kimi_eagle/features_int3g128"
WARM_START = "/Users/pmrj/models/kimi_eagle/drafter_int3g128.safetensors"  # epoch-24 unstabilized best
OUT = "/Users/pmrj/models/kimi_eagle/drafter_int3g128_stable.safetensors"

# Stabilization recipe — keep these as named constants so the commit message and any future
# comparison run can read them out of one place.
LR = 1e-4              # half the unconstrained 2e-4
GRAD_CLIP = 1.0        # max-norm
LR_SCHEDULE = "warmup_cosine"
WARMUP_STEPS = 100     # ~3 epochs of warmup at batch=8 over ~1500 chunks
SEED = 42              # fresh; original run used the default (unseeded) order


def run() -> None:
    t0 = time.perf_counter()
    g = load_feature_shards(SHARDS)
    embed, head = load_frozen_embed_head(ART)
    H = embed.shape[1]
    print(f"features {tuple(g['feat3'].shape)} | base {ART}", flush=True)
    print(f"warm-start {WARM_START}", flush=True)
    print(f"stabilizations: lr={LR}  grad_clip={GRAD_CLIP}  schedule={LR_SCHEDULE}({WARMUP_STEPS})  "
          f"seed={SEED}", flush=True)
    d = EagleDrafter(hidden=H, n_heads=56, head_dim=128, intermediate=14336, eps=1e-6, rope_base=50000.0)
    d.load_weights(WARM_START)
    res = train_drafter(d, g["feat3"], g["in_tokens"], g["targets"], embed, head,
                        epochs=15, lr=LR, batch=8, steps=4, holdout=4, feat_w=1.0, save_path=OUT,
                        seed=SEED, grad_clip=GRAD_CLIP, lr_schedule=LR_SCHEDULE,
                        warmup_steps=WARMUP_STEPS)
    accs = " ".join(f"{a:.3f}" for a in res["final_holdout"])
    print(f"DONE in {(time.perf_counter() - t0) / 60:.1f} min | best epoch {res['best_epoch']} | "
          f"per-step accept [{accs}] | saved {OUT}", flush=True)


if __name__ == "__main__":
    run()
