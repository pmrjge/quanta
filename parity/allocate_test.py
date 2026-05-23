"""Validate the int3/int4 DP allocator (synthetic, instant).

Checks: all-int3 baseline exceeds target; allocation drives weighted error under 8% when the
budget allows; only the highest-Δloss projections get promoted; the byte budget is respected;
and a too-small budget yields best-effort (error above target, bytes within budget).

    uv run python -m parity.allocate_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.bake.allocate import BPP, Projection, allocate_bits


def _make(n: int) -> list[Projection]:
    mx.random.seed(0)
    # int3 losses spanning ~3%..20% so all-int3 mean > 8%; int4 ~0.45x of int3
    l3 = (0.03 + 0.17 * mx.random.uniform(shape=(n,))).tolist()
    return [Projection(f"p{i}", params=1_000_000 + i, loss3=l3[i], loss4=0.45 * l3[i]) for i in range(n)]


def _werr(projs, bits):
    tp = sum(p.params for p in projs)
    return sum(p.params * (p.loss4 if bits[p.key] == 4 else p.loss3) for p in projs) / tp


def run() -> None:
    projs = _make(200)
    total_p = sum(p.params for p in projs)
    base_err = sum(p.params * p.loss3 for p in projs) / total_p
    all_int4_bytes = total_p * BPP[4] / 8.0
    print("\n=== int3/int4 allocator ===")
    print(f"all-int3 weighted error: {base_err:.4%}  (target 8%)")

    # generous budget: should reach target
    bits, err, used = allocate_bits(projs, byte_budget=all_int4_bytes, target=0.08)
    n4 = sum(v == 4 for v in bits.values())
    print(f"budget=all-int4: err {err:.4%}  int4 {n4}/{len(projs)}  bytes {used / 1e6:.1f}M")
    assert err <= 0.08 + 1e-9 and abs(err - _werr(projs, bits)) < 1e-9

    # promotions must be the highest-Δloss projections
    promoted = {k for k, v in bits.items() if v == 4}
    by_drop = sorted(projs, key=lambda p: p.loss3 - p.loss4, reverse=True)
    assert promoted == {p.key for p in by_drop[:n4]}, "must promote highest-Δloss first"

    # tight budget: best-effort, never exceeds budget
    tight = total_p * BPP[3] / 8.0 + 10 * (1_000_000 * (BPP[4] - BPP[3]) / 8.0)  # ~10 promotions
    bits_t, err_t, used_t = allocate_bits(projs, byte_budget=tight, target=0.08)
    n4_t = sum(v == 4 for v in bits_t.values())
    print(f"tight budget   : err {err_t:.4%}  int4 {n4_t}/{len(projs)}  bytes {used_t / 1e6:.1f}M (<= {tight / 1e6:.1f}M)")
    assert used_t <= tight + 1e-6 and n4_t <= 12

    print("allocator: hits target when affordable, respects budget, promotes by sensitivity")


if __name__ == "__main__":
    run()
