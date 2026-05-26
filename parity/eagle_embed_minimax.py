"""Embed the trained MiniMax-M2.7 EAGLE drafter into the artifact as a self-contained sidecar.

Deferred — runs **after** :mod:`parity.eagle_train_minimax` validates with a good per-step accept
vector. Same template as :mod:`parity.eagle_embed_int3g128`: copies
``~/models/minimax_eagle/drafter_int6g64.safetensors`` into
``MiniMax-M2.7-quanta_int6g64/eagle/`` and writes ``eagle.json`` with the drafter config + capture
layers + training provenance — all relative refs, so the artifact stays portable.

After this lands, the inference-time consumer becomes::

    from quanta.eagle.artifact import load_eagle
    from quanta.minimax.eagle import spec_generate
    from parity.eagle_train_minimax import load_frozen_embed_head_minimax

    drafter, layers = load_eagle(ART)
    embed, head = load_frozen_embed_head_minimax(ART)
    tokens, stats = spec_generate(model, drafter, embed, head, prompt_ids,
                                  max_new=64, layers=layers)

— the artifact root is the only input the EAGLE spec-decode path needs.

    uv run python -m parity.eagle_embed_minimax
"""

from __future__ import annotations

from quanta.eagle.artifact import embed_eagle
from quanta.minimax.eagle import DEFAULT_CAPTURE_LAYERS, MINIMAX_DRAFTER_CFG

ART = "/Users/pmrj/models/MiniMax-M2.7-quanta_int6g64"
WEIGHTS = "/Users/pmrj/models/minimax_eagle/drafter_int6g64.safetensors"
CAPTURE_LAYERS = DEFAULT_CAPTURE_LAYERS

# Training provenance — fill ``best_epoch``, ``best_mean_accept``, ``final_holdout`` from the last
# line of ``eagle_train_minimax.py``'s stdout before invoking this script.
TRAINING_META: dict = {
    "base_artifact": "MiniMax-M2.7-quanta_int6g64",
    "corpus": "minimax_eagle/features_int6g64 (~1M-token mix)",
    "capture_layers": list(CAPTURE_LAYERS),
    "training_script": "parity.eagle_train_minimax",
    "training_steps": 4,
    "epochs_cap": 40,
    "batch": 8,
    "lr": 2e-4,
    "feat_w": 1.0,
    "holdout_chunks": 4,
    # filled in at run time:
    # "feature_shards": <int>, "corpus_tokens": <int>,
    # "best_epoch": <int>, "best_mean_accept": <float>, "final_holdout": [<float> x 4],
}


def run() -> None:
    out = embed_eagle(
        ART, WEIGHTS,
        capture_layers=CAPTURE_LAYERS,
        drafter_cfg=MINIMAX_DRAFTER_CFG,
        training_meta=TRAINING_META,
    )
    print(f"embedded EAGLE -> {out}")


if __name__ == "__main__":
    run()
