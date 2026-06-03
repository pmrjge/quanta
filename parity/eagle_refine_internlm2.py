"""Refine (anneal) the InternLM2.5-7B EAGLE drafter from the flat-LR base checkpoint (deferred — GPU job).

The base run (:mod:`parity.eagle_train_internlm2`, flat ``lr=2e-4``) climbs fast but its per-epoch
holdout-accept gains taper into a shallow plateau (the high constant LR can no longer resolve the
basin). This is the standard second phase: **warm-start from the best base checkpoint** and anneal —
a lower peak LR, **cosine** decay to ``0.1×``, and **gradient clipping** — so the optimizer settles
into a better minimum instead of bouncing around it. Same warm-start pattern as
:mod:`parity.eagle_finetune` stage-2, but on the *same* general corpus (a refinement, not a
distribution shift), so the only changes vs the base run are the LR schedule + clip.

Run AFTER ``parity.eagle_train_internlm2`` (which leaves ``drafter_int8g64.safetensors`` = the base
best) and ALONE (one model resident — the 25 GB feature set + frozen embed/head, no target model).
Writes the annealed drafter to a SEPARATE sidecar so the warm-start source is never clobbered; the
real-model bench (:mod:`parity.internlm2_eagle_spec_bench`) is the arbiter of which one to keep.

The default args reproduce **stage-2** (warm-start the flat-LR base at ``lr=1e-4``). When stage-2's
own gains taper, **chain another stage** (SGDR-style warm restart): warm-start from the prior refined
checkpoint at a still-lower LR with a fresh cosine, so each stage settles deeper into the basin. The
script is a generic driver — ``[base_in] [refined_out] [lr] [epochs]``:

    uv run python -m parity.eagle_refine_internlm2                            # stage-2 (defaults)
    uv run python -m parity.eagle_refine_internlm2 \\
        .../drafter_int8g64_refined.safetensors \\
        .../drafter_int8g64_refined2.safetensors 5e-5 24                      # stage-3 (from stage-2 best)
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

from quanta.eagle.capture import load_feature_shards
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.train import load_drafter, train_drafter
from quanta.internlm2.eagle import INTERNLM2_DRAFTER_CFG

# Reuse the base run's artifact paths + memory-light embed/head loader (cross-import is idiomatic here).
from parity.eagle_train_internlm2 import ART, SHARDS, load_frozen_embed_head_internlm2

BASE = "/Users/pmrj/models/internlm2_eagle/drafter_int8g64.safetensors"        # warm-start = base best
REFINED = "/Users/pmrj/models/internlm2_eagle/drafter_int8g64_refined.safetensors"

# Anneal hyperparameters — lower peak LR + cosine decay + grad clip vs the flat-2e-4 base run.
LR = 1.0e-4            # half the base 2e-4; cosine decays this to 0.1× = 1e-5 over the run
EPOCHS = 30           # cap; early-stop (patience) + best-epoch restore decide the real length
GRAD_CLIP = 1.0       # global-L2 clip — bounds the fresh-Adam restart + the late-stage blowup risk
PATIENCE = 8
SEED = 0


def run() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else BASE                          # warm-start source
    refined = sys.argv[2] if len(sys.argv) > 2 else REFINED                    # separate output sidecar
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else LR                       # lower each stage
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else EPOCHS
    assert Path(base).exists(), f"warm-start checkpoint missing: {base}"
    assert Path(refined) != Path(base), "refined output must differ from the warm-start source"
    t0 = time.perf_counter()
    g = load_feature_shards(SHARDS)
    embed, head = load_frozen_embed_head_internlm2(ART)
    cfg = INTERNLM2_DRAFTER_CFG
    assert cfg.hidden == embed.shape[1], f"drafter hidden={cfg.hidden} != artifact H={embed.shape[1]}"
    d = load_drafter(base, EagleDrafter(**asdict(cfg)))                        # warm-start from prior best
    print(f"refine: features {tuple(g['feat3'].shape)} | warm-start {Path(base).name} | "
          f"lr={lr} cosine grad_clip={GRAD_CLIP} epochs={epochs} -> {Path(refined).name}", flush=True)
    res = train_drafter(d, g["feat3"], g["in_tokens"], g["targets"], embed, head,
                        epochs=epochs, lr=lr, batch=8, steps=4, holdout=4, feat_w=1.0,
                        patience=PATIENCE, grad_clip=GRAD_CLIP, lr_schedule="cosine", seed=SEED,
                        save_path=refined)
    base_mean = sum(res["base_holdout"]) / len(res["base_holdout"])            # warm-start, pre-anneal
    best_mean = res["best_mean_accept"]
    accs = " ".join(f"{a:.3f}" for a in res["final_holdout"])
    verdict = "IMPROVED" if best_mean > base_mean + 1e-4 else "no gain (keep warm-start)"
    print(f"DONE in {(time.perf_counter() - t0) / 60:.1f} min | holdout mean {base_mean:.3f} "
          f"(warm-start) -> {best_mean:.3f} (refined) [{verdict}] | best epoch {res['best_epoch']} | "
          f"per-step accept [{accs}] | saved {refined}", flush=True)


if __name__ == "__main__":
    run()
