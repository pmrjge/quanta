"""#52 EAGLE spec-decode benchmark: real accept-rate + tok/s with the trained drafter (full Kimi).

Loads the full RAM-resident Kimi int2-g64 target + the trained EAGLE drafter (layers 10/30/50),
prompts with real corpus tokens, and compares lossless spec-decode vs plain greedy decode:
mean accepted tokens per target forward (1 = no speedup, k+1 = perfect) and wall-clock tok/s.

    uv run python -m parity.eagle_spec_bench [drafter.safetensors] [k]

NOTE: loads the ~389 GB resident target — run only with the memory free.
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

from quanta.eagle.capture import load_features
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.spec import LAYERS, spec_generate
from quanta.eagle.train import load_drafter, load_frozen_embed_head
from quanta.generate import generate
from quanta.runtime import ResidentModel

ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int2g64"
DRAFTER = "/Users/pmrj/models/kimi_eagle/drafter_general.safetensors"
FEATURES = "/Users/pmrj/models/kimi_eagle/features.safetensors"
MAXN = 128


def run() -> None:
    drafter_path = sys.argv[1] if len(sys.argv) > 1 else DRAFTER
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    mx.set_wired_limit(int(480 * 1024**3))
    model = ResidentModel(ART)
    embed, head = load_frozen_embed_head(ART)
    drafter = load_drafter(drafter_path, EagleDrafter(
        hidden=embed.shape[1], n_heads=56, head_dim=128, intermediate=14336, rope_base=50000.0))
    mx.eval(drafter.parameters())
    prompt = [int(x) for x in load_features(FEATURES)["in_tokens"][:32].tolist()]

    # spec-decode (warm once, then timed)
    spec_generate(model, drafter, embed, head, prompt, max_new=8, k=k, layers=LAYERS)
    t0 = time.perf_counter()
    spec, stats = spec_generate(model, drafter, embed, head, prompt, max_new=MAXN, k=k, layers=LAYERS)
    spec_dt = time.perf_counter() - t0

    # plain greedy baseline
    generate(model, prompt, max_new_tokens=8, temperature=0.0)
    t0 = time.perf_counter()
    base = generate(model, prompt, max_new_tokens=MAXN, temperature=0.0)
    base_dt = time.perf_counter() - t0

    s_tps, b_tps = len(spec) / spec_dt, len(base) / base_dt
    print(f"\n=== EAGLE spec-decode benchmark (drafter={drafter_path.split('/')[-1]}, k={k}) ===")
    print(f"spec : {len(spec):3d} tok  {spec_dt:5.1f}s  {s_tps:5.2f} tok/s  | mean_accept={stats['mean_accept']:.2f}/{k + 1} "
          f"max={stats['max_accept']} rounds={stats['rounds']}")
    print(f"base : {len(base):3d} tok  {base_dt:5.1f}s  {b_tps:5.2f} tok/s")
    print(f"speedup: {s_tps / b_tps:.2f}x  (accept-bound ceiling {stats['mean_accept']:.2f}x)")


if __name__ == "__main__":
    run()
