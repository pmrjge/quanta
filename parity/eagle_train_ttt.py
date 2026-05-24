"""EAGLE training-time-test: multi-step self-fed drafter training (fix for spec-decode accept).

Single-step training left steps-2+ accept ~0 (the drafter's ``x`` was never trained as a recurrent
feature), so spec-decode accepted 0 drafts (0.39x). This trains with ``steps=S`` self-fed rollout —
step 1 on the real target feature, steps 2..S on the drafter's own ``x``. The signal to watch is the
per-step accept vector: step-2+ rising from ~0 means the recurrence is now learned. Pretrain on
general, then warm-start fine-tune on on-policy. Light (no resident target; frozen embed/head only).

    uv run python -m parity.eagle_train_ttt [steps]
"""

from __future__ import annotations

import sys

from quanta.eagle.capture import load_features
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.train import load_drafter, load_frozen_embed_head, save_drafter, train_drafter

ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int2g64"
GENERAL = "/Users/pmrj/models/kimi_eagle/features.safetensors"
AGENTIC = "/Users/pmrj/models/kimi_eagle/features_agentic.safetensors"
GEN_OUT = "/Users/pmrj/models/kimi_eagle/drafter_ttt.safetensors"
FT_OUT = "/Users/pmrj/models/kimi_eagle/drafter_ttt_agentic.safetensors"


def _mk(embed):
    return EagleDrafter(hidden=embed.shape[1], n_heads=56, head_dim=128, intermediate=14336,
                        eps=1e-6, rope_base=50000.0)


def run() -> None:
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    embed, head = load_frozen_embed_head(ART)

    g = load_features(GENERAL)
    print(f"[ttt pretrain] steps={s} epochs={epochs} general feat3 {tuple(g['feat3'].shape)}", flush=True)
    d = _mk(embed)
    s1 = train_drafter(d, g["feat3"], g["in_tokens"], g["targets"], embed, head,
                       chunk=2048, batch=2, epochs=epochs, lr=2e-4, holdout=2, steps=s)
    save_drafter(GEN_OUT, d)
    print(f"[ttt pretrain] per-step accept {tuple(round(a, 3) for a in s1['base_holdout'])} -> "
          f"{tuple(round(a, 3) for a in s1['final_holdout'])} | saved {GEN_OUT}", flush=True)

    a = load_features(AGENTIC)
    print(f"[ttt finetune] agentic feat3 {tuple(a['feat3'].shape)}", flush=True)
    d2 = load_drafter(GEN_OUT, _mk(embed))
    s2 = train_drafter(d2, a["feat3"], a["in_tokens"], a["targets"], embed, head,
                       chunk=2048, batch=2, epochs=epochs, lr=5e-5, holdout=2, steps=s)
    save_drafter(FT_OUT, d2)
    print(f"[ttt finetune] per-step accept {tuple(round(a2, 3) for a2 in s2['base_holdout'])} -> "
          f"{tuple(round(a2, 3) for a2 in s2['final_holdout'])} | saved {FT_OUT}", flush=True)


if __name__ == "__main__":
    run()
