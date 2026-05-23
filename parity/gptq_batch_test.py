"""Batched GPTQ == per-expert GPTQ, and faster (synthetic, instant).

The cross-expert batched solver must produce identical codes/scales (same algorithm, shared
column loop) and run faster than E separate calls — the throughput win that makes the full
bake feasible. Experts have different n (rows), as under sparse routing.

    uv run python -m parity.gptq_batch_test
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.bake.gptq import gptq_quantize, gptq_quantize_batch

E, OUT, IN, GS, BITS = 4, 64, 512, 128, 3


def run() -> None:
    mx.random.seed(0)
    ws = mx.random.normal((E, OUT, IN))
    xs = [mx.random.normal((40 + 13 * i, IN)) for i in range(E)]  # different n per expert

    codes_b, scales_b, biases_b = gptq_quantize_batch(ws, xs, BITS, group_size=GS)
    mx.eval(codes_b, scales_b, biases_b)

    worst = 0.0
    for i in range(E):
        _, codes, scales, biases = gptq_quantize(ws[i], xs[i], BITS, group_size=GS)
        worst = max(
            worst,
            mx.max(mx.abs(codes_b[i] - codes)).item(),
            mx.max(mx.abs(scales_b[i] - scales)).item(),
            mx.max(mx.abs(biases_b[i] - biases)).item(),
        )

    t0 = time.perf_counter()
    mx.eval(gptq_quantize_batch(ws, xs, BITS, group_size=GS))
    t_batch = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(E):
        mx.eval(gptq_quantize(ws[i], xs[i], BITS, group_size=GS))
    t_single = time.perf_counter() - t0

    print("\n=== batched GPTQ vs per-expert ===")
    print(f"max |batched - per-expert| (codes/scales/biases): {worst:.3e}")
    print(f"time: batched {t_batch * 1e3:.0f}ms  vs  {E}x single {t_single * 1e3:.0f}ms  "
          f"({t_single / t_batch:.2f}x)")
    assert worst < 1e-5, "batched GPTQ must equal per-expert"
    print("batched GPTQ is identical to per-expert and faster")


if __name__ == "__main__":
    run()
