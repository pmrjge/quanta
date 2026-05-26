"""Quick sweep of EAGLE spec-decode configs against the trained int3g128 drafter — finds the
cheapest path past breakeven (>1.0x walltime vs plain greedy) without retraining the drafter.

The k=4 absorbed=False bench landed at 0.83x walltime (spec slower than greedy) because the
verify cost dominates the savings. Two free knobs we haven't swept yet:

* **k** — shrinking the draft length cuts the verify-batch size (k+1 tokens) which is the main
  spec-decode overhead. mean_accept drops by the deepest steps, but per-round verify cost drops
  faster. k=2 means a 3-token verify (vs 5); k=3 a 4-token verify.
* **absorbed** — the absorbed MLA fast path computes attention through a quantized_matmul absorb
  rather than materializing K=V (cheaper SDPA). Same argmax as the non-absorbed path → lossless
  preserved (the drafter's accept rate may shift slightly because the captured features come
  from the verify forward).

Loads the resident model + drafter + embed/head ONCE (~10–15 min for the 398 GB base) and runs
5 configs + greedy back-to-back so we're paying the resident-load cost just once. Each spec run
is ~30s of actual compute on top.

    uv run python -m parity.eagle_spec_sweep

NOTE: ~410 GB resident — run only with the memory free, never alongside another big-resident job.
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.eagle.capture import load_features
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.spec import LAYERS, spec_generate
from quanta.eagle.train import load_drafter, load_frozen_embed_head
from quanta.generate import generate
from quanta.runtime import ResidentModel

ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int3g128"
DRAFTER = "/Users/pmrj/models/kimi_eagle/drafter_int3g128.safetensors"
FEATURES = "/Users/pmrj/models/kimi_eagle/features_int3g128/feat_0000.safetensors"
MAXN = 128

# (k, absorbed) — ordered cheapest first so an early break-even saves the rest
CONFIGS: list[tuple[int, bool]] = [
    (4, False),  # baseline (matches eagle_spec_bench — known 0.83x)
    (3, False),  # smaller verify batch (4 tokens instead of 5)
    (2, False),  # smallest verify batch (3 tokens)
    (4, True),   # absorbed MLA fast path, original k
    (2, True),   # combined: smallest verify + absorbed
]


def run() -> None:
    mx.set_wired_limit(int(480 * 1024**3))
    t_load = time.perf_counter()
    model = ResidentModel(ART)
    embed, head = load_frozen_embed_head(ART)
    drafter = load_drafter(DRAFTER, EagleDrafter(
        hidden=embed.shape[1], n_heads=56, head_dim=128, intermediate=14336, rope_base=50000.0))
    mx.eval(drafter.parameters())
    prompt = [int(x) for x in load_features(FEATURES)["in_tokens"][:32].tolist()]
    print(f"loaded resident + drafter in {(time.perf_counter() - t_load) / 60:.1f} min", flush=True)

    # greedy baseline (warm once, then time)
    generate(model, prompt, max_new_tokens=8, temperature=0.0)
    t0 = time.perf_counter()
    base = generate(model, prompt, max_new_tokens=MAXN, temperature=0.0)
    base_dt = time.perf_counter() - t0
    b_tps = len(base) / base_dt
    print("\n=== greedy baseline ===")
    print(f"base : {len(base):3d} tok  {base_dt:5.1f}s  {b_tps:5.2f} tok/s\n")

    print("=== EAGLE spec-decode sweep ===")
    print(f"{'k':>2} {'absorbed':>8} {'tok':>4} {'time':>6} {'tok/s':>6} {'mean_accept':>11} {'speedup':>8}")
    for k, absorbed in CONFIGS:
        # warm once (k+1 tokens drafted/verified)
        spec_generate(model, drafter, embed, head, prompt, max_new=k + 1, k=k,
                     layers=LAYERS, absorbed=absorbed)
        t0 = time.perf_counter()
        spec, stats = spec_generate(model, drafter, embed, head, prompt, max_new=MAXN, k=k,
                                    layers=LAYERS, absorbed=absorbed)
        dt = time.perf_counter() - t0
        tps = len(spec) / dt
        print(f"{k:>2} {str(absorbed):>8} {len(spec):>4} {dt:>6.1f} {tps:>6.2f} "
              f"{stats['mean_accept']:>11.2f} {tps / b_tps:>7.2f}x", flush=True)


if __name__ == "__main__":
    run()
