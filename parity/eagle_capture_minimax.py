"""Capture EAGLE-3 features from the resident MiniMax-M2.7 int6g64 bake (deferred — GPU job).

Mirrors :mod:`parity.eagle_capture_mix` for the MiniMax architecture. Run AFTER ``bake_minimax``
finishes, and ALONE (the int6g64 artifact is ~250 GB resident — never alongside another large job).
The same per-2048-block teacher-forced capture as the Kimi version, but the forward callable wires
MiniMax's signature (no ``sparse`` / ``absorbed`` kwargs).

    uv run --with tiktoken python -m parity.eagle_capture_minimax
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx

from quanta.eagle.capture import capture_features_to_shards_fn
from quanta.minimax.eagle import DEFAULT_CAPTURE_LAYERS, minimax_capture_forward
from quanta.minimax.runtime import MiniMaxResidentModel

ART = "/Users/pmrj/models/MiniMax-M2.7-quanta_int6g64"
CORPUS = "/Users/pmrj/models/corpus/corpus_mix.safetensors"
OUT = "/Users/pmrj/models/minimax_eagle/features_int6g64"
LAYERS = DEFAULT_CAPTURE_LAYERS                     # low / mid / high of 62 (10, 30, 50)


def run() -> None:
    mx.set_wired_limit(int(490 * 1024**3))
    mx.set_cache_limit(8 * 1024**3)
    t0 = time.perf_counter()
    rm = MiniMaxResidentModel(ART)
    ids = mx.load(CORPUS)["ids"]
    fwd = minimax_capture_forward(rm)
    print(f"corpus: {ids.shape[0]} tokens | base {Path(ART).name} | capture layers {LAYERS}",
          flush=True)
    info = capture_features_to_shards_fn(fwd, ids, LAYERS, OUT, chunk=2048, shard_tokens=131072)
    print(f"captured {info['total_tokens']} tok -> {len(info['shards'])} shards in "
          f"{(time.perf_counter() - t0) / 60:.1f} min | {OUT}", flush=True)


if __name__ == "__main__":
    run()
