"""Parity: fused ssd_step_fused (one Metal kernel) == ssd_step (composed ops).

The fused decode kernel must be output-equivalent to the composed-op recurrence (rule-4) before
it goes on the hot path. Random inputs at Nemotron's mamba dims (H=128, P=64, N=128, G=8),
batch 1 and 2. Both run the fp32 scan; the kernel just fuses ~8 ops into one launch.

    uv run python -m parity.nemotron_ssd_kernel_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.nemotron.mamba_ssd import ssd_step, ssd_step_fused

H, P, N, G = 128, 64, 128, 8


def _rel(a: mx.array, b: mx.array) -> float:
    return (mx.linalg.norm((a - b).astype(mx.float32)) / mx.linalg.norm(b.astype(mx.float32)) + 1e-30).item()


def run() -> None:
    mx.random.seed(0)
    ok = True
    for bn in (1, 2):
        x = mx.random.normal([bn, H, P])
        dt = mx.abs(mx.random.normal([bn, H])) * 0.1          # dt > 0
        A = -mx.abs(mx.random.normal([H]))                    # A < 0
        B = mx.random.normal([bn, G, N])
        C = mx.random.normal([bn, G, N])
        D = mx.random.normal([H])
        state = mx.random.normal([bn, H, N, P])
        y_ref, s_ref = ssd_step(x, dt, A, B, C, D, state)
        y_fus, s_fus = ssd_step_fused(x, dt, A, B, C, D, state)
        ry, rs = _rel(y_fus, y_ref), _rel(s_fus, s_ref)
        ok = ok and ry < 1e-4 and rs < 1e-4
        print(f"bn={bn}: y rel {ry:.2e} | state rel {rs:.2e}")
    print("PASS" if ok else "FAIL (fused != composed)")


if __name__ == "__main__":
    run()
