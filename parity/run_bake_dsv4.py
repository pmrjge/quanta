"""DSV4-Flash bake → ~/models/DeepSeek-V4-Flash-quanta_int4g64 (background, ~hours).

int4 AWQ experts (g64, bf16 scales), int8 affine non-experts (g64, bf16 scales), bf16 norms /
router / hyper-connection control tensors / embedding / head / MTP. Self-contained: the
DeepSeek-V4 ``tokenizer.json``/``tokenizer_config.json`` are copied into the artifact.

Calibration: 8192 tokens sliced from ``corpus/corpus_mix.safetensors`` (the agentic-domain mix
that drove Kimi/EAGLE/Nemotron capture). The bake's ``capture_calibration`` streams a bf16
reference forward across all MoE layers (one block resident at a time, rule-8) and records
per-layer post-norm activations + routing for AWQ; each expert's w1/w3 are calibrated on its
routed rows of ``x`` and w2 on the SwiGLU intermediate of those rows (see
:mod:`quanta.dsv4.bake`).

    uv run python -m parity.run_bake_dsv4
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import mlx.core as mx

from quanta.dsv4.bake import bake_dsv4

MODEL = "/Users/pmrj/models/DeepSeek-V4-Flash"
OUT = "/Users/pmrj/models/DeepSeek-V4-Flash-quanta_int4g64"
CORPUS = "/Users/pmrj/models/corpus/corpus_mix.safetensors"
GROUP_SIZE = 64
CALIB_TOKENS = 8192
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                    "generation_config.json")


def _calibration_ids() -> mx.array:
    """8192 calibration ids from the agentic-domain ``corpus_mix.safetensors``."""
    ids = mx.load(CORPUS)["ids"][:CALIB_TOKENS].astype(mx.int32)
    return ids


def run() -> None:
    mx.set_cache_limit(32 * 1024**3)  # cap the MLX buffer cache so the resident set can't balloon
    ids = _calibration_ids()
    print(f"calibration tokens: {ids.shape[0]} | int4 AWQ g{GROUP_SIZE} experts, int8 dense g{GROUP_SIZE}, "
          f"bf16 norms/router/HC/embed/head/MTP, bf16 scales", flush=True)
    t0 = time.perf_counter()
    stats = bake_dsv4(MODEL, OUT, ids, include_head=True, group_size=GROUP_SIZE,
                      expert_method="awq", scale_dtype=mx.bfloat16)
    for fn in _TOKENIZER_FILES:
        src = Path(MODEL) / fn
        if src.exists():
            shutil.copy(src, Path(OUT) / fn)
    print(f"DSV4 INT4-g64 BAKE DONE in {(time.perf_counter() - t0) / 3600:.2f}h\n{stats}", flush=True)


if __name__ == "__main__":
    run()
