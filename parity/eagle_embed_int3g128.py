"""Embed the int3g128-trained EAGLE-3 drafter into the Kimi-K2.6 artifact as a self-contained
sidecar (task #54). Run AFTER ``parity.eagle_train_mix`` validates with a good per-step accept
vector.

Copies ``~/models/kimi_eagle/drafter_int3g128.safetensors`` into
``Kimi-K2.6-quanta_int3g128/eagle/`` and writes ``eagle.json`` with the drafter config + capture
layers + training provenance — all relative refs, so the artifact stays portable (move the dir,
EAGLE moves with it). The artifact's main ``manifest.json`` is **not** modified.

After this lands, the inference-time consumer becomes::

    from quanta.eagle.artifact import load_eagle
    from quanta.eagle.train import load_frozen_embed_head
    from quanta.eagle.spec import spec_generate

    drafter, layers = load_eagle(ART)
    embed, head = load_frozen_embed_head(ART)
    tokens, stats = spec_generate(model, drafter, embed, head, prompt_ids,
                                  max_new=64, layers=layers)

— the artifact root is now the only input the EAGLE spec-decode path needs.

    uv run python -m parity.eagle_embed_int3g128
"""

from __future__ import annotations

from quanta.eagle.artifact import DrafterConfig, embed_eagle

ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int3g128"
WEIGHTS = "/Users/pmrj/models/kimi_eagle/drafter_int3g128.safetensors"
CAPTURE_LAYERS = (10, 30, 50)  # must match the layers used by parity.eagle_capture_mix

# Training provenance — the runner fills ``best_epoch``, ``best_mean_accept``, ``final_holdout``
# from the last line of ``eagle_train_mix.py``'s stdout before invoking this script.
TRAINING_META: dict = {
    "base_artifact": "Kimi-K2.6-quanta_int3g128",
    "corpus": "kimi_eagle/features_int3g128 (1.02M-token mix)",
    "corpus_tokens": 1_020_000,
    "feature_shards": 8,
    "capture_layers": list(CAPTURE_LAYERS),
    "training_script": "parity.eagle_train_mix",
    "training_steps": 4,
    "epochs_cap": 40,
    "batch": 8,
    "lr": 2e-4,
    "feat_w": 1.0,
    "holdout_chunks": 4,
    # filled in at run time:
    # "best_epoch": <int>,
    # "best_mean_accept": <float>,
    # "final_holdout": [<float>, <float>, <float>, <float>],
}


def run() -> None:
    out = embed_eagle(
        ART, WEIGHTS,
        capture_layers=CAPTURE_LAYERS,
        drafter_cfg=DrafterConfig(),  # defaults match Kimi-K2.6 (H=7168, 56×128 heads, 14336 SwiGLU)
        training_meta=TRAINING_META,
    )
    print(f"embedded EAGLE -> {out}")


if __name__ == "__main__":
    run()
