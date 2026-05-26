"""Train the EAGLE drafter for MiniMax-M2.7 on captured features (deferred — GPU job).

Mirrors :mod:`parity.eagle_train_mix`. Run AFTER ``parity.eagle_capture_minimax``: the resident
target is released after capture; only the embed/head + feature set are resident here, so it is safe
to run once the capture has freed the large base.

The embed/head dequant uses :class:`quanta.minimax.artifact.MiniMaxArtifact` (which baked the
embedding + LM head as affine-packed int8); the dequant follows the same pattern as
:func:`quanta.eagle.train.load_frozen_embed_head` but is keyed off the MiniMax manifest. Mini-batched
(batch=8); steps=4 trains the self-fed multi-step rollout that spec-decode runs in, so
``final_holdout`` is the 4-token accept profile.

    uv run python -m parity.eagle_train_minimax
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

from quanta.eagle.capture import load_feature_shards
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.train import train_drafter
from quanta.minimax.artifact import MiniMaxArtifact
from quanta.minimax.eagle import MINIMAX_DRAFTER_CFG

ART = "/Users/pmrj/models/MiniMax-M2.7-quanta_int6g64"
SHARDS = "/Users/pmrj/models/minimax_eagle/features_int6g64"
OUT = "/Users/pmrj/models/minimax_eagle/drafter_int6g64.safetensors"

# Keys baked into the MiniMax artifact for the embedding + LM head; verify against bake.py before
# the first real run if the bake layout changes.
EMBED_KEY = "model.embed_tokens"
LM_HEAD_KEY = "lm_head"


def load_frozen_embed_head_minimax(art_dir: str | Path) -> tuple[mx.array, mx.array]:
    """Dequantize the artifact's int8 embedding + LM head to bf16 ``[V, H]`` (frozen, shared)."""
    art = MiniMaxArtifact(art_dir)

    def deq(key: str) -> mx.array:
        m = art.manifest[key]
        return mx.dequantize(art.get(f"{key}.weight_packed"), art.get(f"{key}.weight_scale"),
                             art.get(f"{key}.weight_bias"), group_size=m["group_size"],
                             bits=m["bits"])

    embed = deq(EMBED_KEY).astype(mx.bfloat16)
    head = deq(LM_HEAD_KEY).astype(mx.bfloat16)
    mx.eval(embed, head)
    art.release()
    return embed, head


def run() -> None:
    t0 = time.perf_counter()
    g = load_feature_shards(SHARDS)
    embed, head = load_frozen_embed_head_minimax(ART)
    H = embed.shape[1]
    cfg = MINIMAX_DRAFTER_CFG
    assert cfg.hidden == H, f"drafter cfg hidden={cfg.hidden} != artifact H={H}"
    print(f"features {tuple(g['feat3'].shape)} | base {ART}", flush=True)
    d = EagleDrafter(**asdict(cfg))
    res = train_drafter(d, g["feat3"], g["in_tokens"], g["targets"], embed, head,
                        epochs=40, lr=2e-4, batch=8, steps=4, holdout=4, feat_w=1.0, save_path=OUT)
    accs = " ".join(f"{a:.3f}" for a in res["final_holdout"])
    print(f"DONE in {(time.perf_counter() - t0) / 60:.1f} min | best epoch {res['best_epoch']} | "
          f"per-step accept [{accs}] | saved {OUT}", flush=True)


if __name__ == "__main__":
    run()
