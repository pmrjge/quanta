"""Prefix-cache (#1) validation: reuse correctness + disk round-trip.

1. continue_from_cache(prefix-cache, suffix) must equal one-shot prefill of
   prefix+suffix on the suffix positions.
2. save → load → continue must equal the in-memory continue (bit-identical),
   proving disk persistence is lossless.

    uv run python -m parity.prefix_cache 4
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import mlx.core as mx

from quanta.cache import load_caches, save_caches
from quanta.config import KimiTextConfig
from quanta.loader import SourceCheckpoint
from quanta.model import KimiModel

MODEL = "/Users/pmrj/models/Kimi-K2.6"
TOKEN_IDS = [163584, 100, 500, 1024, 2048, 4096, 8192, 16000, 32000, 64000, 100000, 120000, 150000, 42, 7, 9001]
SPLIT = 8


def _max_abs(a: mx.array, b: mx.array) -> float:
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()


def run(n_layers: int | None = None) -> None:
    cfg = KimiTextConfig.from_pretrained(MODEL)
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    model = KimiModel(cfg, SourceCheckpoint(MODEL), mx.bfloat16)
    ids = mx.array(TOKEN_IDS)
    prefix, suffix = ids[:SPLIT], ids[SPLIT:]

    full = model(ids, n_layers=n)  # [1, 16, V]

    caches = model.build_prefix_cache(prefix, n_layers=n)  # offset == SPLIT
    path = Path(tempfile.gettempdir()) / "quanta_prefix_cache.safetensors"
    save_caches(path, caches)  # persist the PREFIX (before continue mutates it)
    caches_disk = load_caches(path)

    cont_mem = model.continue_from_cache(suffix, caches)  # [1, 8, V]; mutates caches
    cont_disk = model.continue_from_cache(suffix, caches_disk)  # [1, 8, V]; mutates copy
    mx.eval(full, cont_mem, cont_disk)

    print(f"\n=== Prefix-cache parity (layers={n}, prefix={SPLIT}) ===")
    print(f"reuse vs one-shot (suffix logits) : max_abs {_max_abs(full[:, SPLIT:], cont_mem):.3e}")
    print(f"disk round-trip (load vs memory)  : max_abs {_max_abs(cont_mem, cont_disk):.3e}")
    print(f"saved prefix cache to {path}")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else None)
