"""KV-cache parity: chunked prefill (two cached passes) must equal one-shot prefill.

Validates the MLA cache machinery — latent append, position-offset RoPE, and the
rectangular lower-right causal mask — for both the naive and the fast (mx.fast.sdpa
mask="causal") paths. Output-equivalence here is what later makes prefix caching
(#1) and decode correct.

    uv run python -m parity.cache_parity        # all 61 layers
    uv run python -m parity.cache_parity 4      # first 4 layers (fast)
"""

from __future__ import annotations

import sys

import mlx.core as mx

from quanta.cache import MLACache
from quanta.config import KimiTextConfig
from quanta.loader import SourceCheckpoint
from quanta.model import KimiModel

MODEL = "/Users/pmrj/models/Kimi-K2.6"
TOKEN_IDS = [163584, 100, 500, 1024, 2048, 4096, 8192, 16000, 32000, 64000, 100000, 120000, 150000, 42, 7, 9001]
SPLIT = 8


def _diff(a: mx.array, b: mx.array) -> tuple[float, float]:
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    abs_err = mx.max(mx.abs(a - b)).item()
    denom = mx.maximum(mx.abs(a), mx.abs(b))
    rel = mx.max(mx.abs(a - b) / mx.where(denom > 0, denom, mx.array(1.0))).item()
    return abs_err, rel


def run(n_layers: int | None = None) -> None:
    cfg = KimiTextConfig.from_pretrained(MODEL)
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    model = KimiModel(cfg, SourceCheckpoint(MODEL), mx.bfloat16)
    ids = mx.array(TOKEN_IDS)

    print(f"\n=== KV-cache parity (layers={n}, split={SPLIT}) — chunked vs one-shot prefill ===")
    for use_fast in (False, True):
        full = model(ids, n_layers=n, use_fast=use_fast)
        caches = [MLACache() for _ in range(n)]
        a = model(ids[:SPLIT], n_layers=n, caches=caches, offset=0, use_fast=use_fast)
        b = model(ids[SPLIT:], n_layers=n, caches=caches, offset=SPLIT, use_fast=use_fast)
        chunked = mx.concatenate([a, b], axis=1)
        mx.eval(full, chunked)
        abs_err, rel = _diff(full, chunked)
        top1 = (mx.argmax(full, -1) == mx.argmax(chunked, -1)).astype(mx.float32).mean().item()
        tag = "fast(sdpa)" if use_fast else "naive"
        print(f"{tag:<11} logits max_abs {abs_err:.3e}  max_rel {rel:.3e}  top1 {top1:.4f}")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else None)
