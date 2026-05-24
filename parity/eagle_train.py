"""Train the EAGLE-3 drafter on captured features — light (no resident model; frozen embed/head only).

    uv run python -m parity.eagle_train
"""

from __future__ import annotations

import sys
import time

from quanta.eagle.capture import load_features
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.train import load_frozen_embed_head, save_drafter, train_drafter

ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int2g64"
FEATURES = "/Users/pmrj/models/kimi_eagle/features.safetensors"
DRAFTER_OUT = "/Users/pmrj/models/kimi_eagle/drafter.safetensors"


def run() -> None:
    features = sys.argv[1] if len(sys.argv) > 1 else FEATURES
    out = sys.argv[2] if len(sys.argv) > 2 else DRAFTER_OUT
    d = load_features(features)
    feat3, ins, tgts = d["feat3"], d["in_tokens"], d["targets"]
    print(f"features: feat3 {tuple(feat3.shape)} | tokens {tuple(ins.shape)}", flush=True)
    embed, head = load_frozen_embed_head(ART)
    drafter = EagleDrafter(hidden=embed.shape[1], n_heads=56, head_dim=128, intermediate=14336,
                           eps=1e-6, rope_base=50000.0)
    t0 = time.perf_counter()
    stats = train_drafter(drafter, feat3, ins, tgts, embed, head,
                          chunk=2048, batch=2, epochs=60, lr=2e-4, holdout=2)
    save_drafter(out, drafter)
    print(f"\ntrained in {(time.perf_counter() - t0) / 60:.1f} min | holdout top1 "
          f"{stats['base_holdout_top1']:.3f} -> {stats['final_holdout_top1']:.3f} | saved {out}",
          flush=True)


if __name__ == "__main__":
    run()
