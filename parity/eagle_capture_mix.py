"""Capture EAGLE-3 features from the resident int3g128 Kimi over the 1M math/coding/research mix.

Replaces the old int2g64 / 204K-token capture: better base (int3 gate/up beats int2's low next-token
ceiling) + a larger, domain-matched corpus (corpus_mix.safetensors, ~1.02M tok). OOM-safe — shards
features to disk incrementally (one ~128K-token shard resident at a time, not the whole 1M-token set,
on top of the ~398 GiB model). Run AFTER run_bake_int3g128 finishes, and ALONE (398 GiB resident).

    uv run --with tiktoken python -m parity.eagle_capture_mix
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx

from quanta.eagle.capture import capture_features_to_shards
from quanta.runtime import ResidentModel

ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int3g128"
CORPUS = "/Users/pmrj/models/kimi_eagle/corpus_mix.safetensors"
OUT = "/Users/pmrj/models/kimi_eagle/features_int3g128"
LAYERS = (10, 30, 50)  # low / mid / high of 61 decoder layers (EAGLE-3 fuses three)


def run() -> None:
    mx.set_wired_limit(int(490 * 1024**3))
    mx.set_cache_limit(8 * 1024**3)  # 398 GiB weights leave ~92 GiB; cap the buffer cache so it can't balloon
    t0 = time.perf_counter()
    rm = ResidentModel(ART)
    ids = mx.load(CORPUS)["ids"]
    print(f"corpus: {ids.shape[0]} tokens | base {Path(ART).name} | capture layers {LAYERS}", flush=True)
    info = capture_features_to_shards(rm, ids, LAYERS, OUT, chunk=2048, shard_tokens=131072)
    print(f"captured {info['total_tokens']} tok -> {len(info['shards'])} shards in "
          f"{(time.perf_counter() - t0) / 60:.1f} min | {OUT}", flush=True)


if __name__ == "__main__":
    run()
