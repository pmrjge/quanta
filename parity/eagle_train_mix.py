"""Train the FIXED EagleDrafter (raw feat3 front-end + LayerScale block) on the int3g128 1M-mix
features and report the per-step accept vector. Run AFTER eagle_capture_mix. The target model is NOT
resident here (only the frozen embed/head ~9 GiB + the ~43 GiB feature set), so it is safe to run once
capture has freed the 398 GiB base. Mini-batched (batch=8); steps=4 trains the self-fed multi-step
rollout that spec-decode runs in, so final_holdout is the 4-token accept profile.

    uv run python -m parity.eagle_train_mix
"""

from __future__ import annotations

import time

from quanta.eagle.capture import load_feature_shards
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.train import load_frozen_embed_head, train_drafter

ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int3g128"
SHARDS = "/Users/pmrj/models/kimi_eagle/features_int3g128"
OUT = "/Users/pmrj/models/kimi_eagle/drafter_int3g128.safetensors"


def run() -> None:
    t0 = time.perf_counter()
    g = load_feature_shards(SHARDS)
    embed, head = load_frozen_embed_head(ART)
    H = embed.shape[1]
    print(f"features {tuple(g['feat3'].shape)} | base {ART}", flush=True)
    d = EagleDrafter(hidden=H, n_heads=56, head_dim=128, intermediate=14336, eps=1e-6, rope_base=50000.0)
    res = train_drafter(d, g["feat3"], g["in_tokens"], g["targets"], embed, head,
                        epochs=40, lr=2e-4, batch=8, steps=4, holdout=4, feat_w=1.0, save_path=OUT)
    accs = " ".join(f"{a:.3f}" for a in res["final_holdout"])
    print(f"DONE in {(time.perf_counter() - t0) / 60:.1f} min | best epoch {res['best_epoch']} | "
          f"per-step accept [{accs}] | saved {OUT}", flush=True)


if __name__ == "__main__":
    run()
