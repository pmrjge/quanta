"""Train the EAGLE drafter for InternLM2.5-7B on captured features (deferred — GPU job).

Mirrors :mod:`parity.eagle_train_minimax`. Run AFTER ``parity.eagle_capture_internlm2``: the resident
target is released after capture, so only the frozen embed/head + the feature set are resident here.
The embed/head are read straight from the artifact (``InternLM2Artifact.embed`` / ``.lm_head``, which
already resolve packed-vs-bf16 and tied-vs-untied) — the same role
:func:`quanta.eagle.train.load_frozen_embed_head` plays for Kimi, but holding only those two tensors,
not the whole model. Mini-batched (batch=8); ``steps=4`` trains the self-fed multi-step rollout that
spec-decode runs in, so ``final_holdout`` is the 4-token accept profile.

    uv run python -m parity.eagle_train_internlm2
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

from quanta.eagle.capture import load_feature_shards
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.train import train_drafter
from quanta.internlm2.artifact import InternLM2Artifact
from quanta.internlm2.eagle import INTERNLM2_DRAFTER_CFG

ART = "/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64"
SHARDS = "/Users/pmrj/models/internlm2_eagle/features_int8g64"
OUT = "/Users/pmrj/models/internlm2_eagle/drafter_int8g64.safetensors"


def load_frozen_embed_head_internlm2(art_dir: str | Path) -> tuple[mx.array, mx.array]:
    """Frozen ``[V, H]`` token-embedding + LM head as bf16 (tied-vs-untied resolved by the artifact),
    holding only those two tensors resident — not the full model. Same role as
    :func:`quanta.eagle.train.load_frozen_embed_head` for Kimi, keyed off the InternLM2 artifact (its
    ``embed`` / ``lm_head`` accessors mirror the runtime's :meth:`InternLM2ResidentModel.embed_head`)."""
    art = InternLM2Artifact(art_dir)
    embed = art.embed().astype(mx.bfloat16)
    head = art.lm_head().astype(mx.bfloat16)
    mx.eval(embed, head)
    art.release()
    return embed, head


def run() -> None:
    t0 = time.perf_counter()
    g = load_feature_shards(SHARDS)
    embed, head = load_frozen_embed_head_internlm2(ART)
    h = embed.shape[1]
    cfg = INTERNLM2_DRAFTER_CFG
    assert cfg.hidden == h, f"drafter cfg hidden={cfg.hidden} != artifact H={h}"
    print(f"features {tuple(g['feat3'].shape)} | base {Path(ART).name}", flush=True)
    d = EagleDrafter(**asdict(cfg))
    res = train_drafter(d, g["feat3"], g["in_tokens"], g["targets"], embed, head,
                        epochs=40, lr=2e-4, batch=8, steps=4, holdout=4, feat_w=1.0, save_path=OUT)
    accs = " ".join(f"{a:.3f}" for a in res["final_holdout"])
    print(f"DONE in {(time.perf_counter() - t0) / 60:.1f} min | best epoch {res['best_epoch']} | "
          f"per-step accept [{accs}] | saved {OUT}", flush=True)


if __name__ == "__main__":
    run()
