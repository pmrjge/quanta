"""Two-stage EAGLE drafter training: pretrain on the large general capture, then warm-start
fine-tune on the small on-policy (agentic) capture.

The on-policy set alone (~10.8k tokens) is far too small to train the drafter from scratch (holdout
top1 stalled ~0.05). Pretraining on the large general capture first, then fine-tuning from that
checkpoint at a lower LR, adapts to the on-policy distribution without starting cold. Stage-2 ``base``
holdout = the pretrained drafter measured on the agentic holdout *before* fine-tuning (transfer);
``final`` = after fine-tuning.

    uv run python -m parity.eagle_finetune
"""

from __future__ import annotations

import time

from quanta.eagle.capture import load_features
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.train import load_drafter, load_frozen_embed_head, save_drafter, train_drafter

ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int2g64"
GENERAL = "/Users/pmrj/models/kimi_eagle/features.safetensors"
AGENTIC = "/Users/pmrj/models/kimi_eagle/features_agentic.safetensors"
GEN_OUT = "/Users/pmrj/models/kimi_eagle/drafter_general.safetensors"
FT_OUT = "/Users/pmrj/models/kimi_eagle/drafter_agentic.safetensors"


def _mk(embed):
    return EagleDrafter(hidden=embed.shape[1], n_heads=56, head_dim=128, intermediate=14336,
                        eps=1e-6, rope_base=50000.0)


def run() -> None:
    embed, head = load_frozen_embed_head(ART)

    # stage 1: pretrain on the large general capture
    g = load_features(GENERAL)
    print(f"[pretrain] general feat3 {tuple(g['feat3'].shape)}", flush=True)
    d = _mk(embed)
    t0 = time.perf_counter()
    s1 = train_drafter(d, g["feat3"], g["in_tokens"], g["targets"], embed, head,
                       chunk=2048, batch=2, epochs=60, lr=2e-4, holdout=2)
    save_drafter(GEN_OUT, d)
    print(f"[pretrain] {(time.perf_counter() - t0) / 60:.1f} min | holdout top1 "
          f"{s1['base_holdout_top1']:.3f} -> {s1['final_holdout_top1']:.3f} | saved {GEN_OUT}", flush=True)

    # stage 2: warm-start fine-tune on the on-policy agentic capture
    a = load_features(AGENTIC)
    print(f"[finetune] agentic feat3 {tuple(a['feat3'].shape)}", flush=True)
    d2 = load_drafter(GEN_OUT, _mk(embed))
    t1 = time.perf_counter()
    s2 = train_drafter(d2, a["feat3"], a["in_tokens"], a["targets"], embed, head,
                       chunk=2048, batch=2, epochs=40, lr=5e-5, holdout=2)
    save_drafter(FT_OUT, d2)
    print(f"[finetune] {(time.perf_counter() - t1) / 60:.1f} min | holdout top1 "
          f"{s2['base_holdout_top1']:.3f} (general-transfer) -> {s2['final_holdout_top1']:.3f} | saved {FT_OUT}",
          flush=True)


if __name__ == "__main__":
    run()
