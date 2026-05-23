"""Parity-gate the Mamba-2 SSD kernel on small synthetic tensors (no model load).

Proves the optimized paths are output-equivalent to the naive sequential recurrence:
  * chunked SSD prefill  == sequential (y and carried state)
  * decode step-loop     == sequential
  * block-split prefill with carried state == sequential over the full sequence
  * causal conv prefill  == stepwise conv
All in fp32 so the tolerance is tight. Safe to run alongside the bake (tiny tensors).

    uv run python -m parity.mamba_ssd_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.nemotron.mamba_ssd import (
    causal_conv1d,
    causal_conv1d_step,
    ssd_chunked,
    ssd_sequential,
    ssd_step,
)


def _maxdiff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)))


def run() -> None:
    mx.random.seed(0)
    bn, length, h, p, g, n, q = 1, 12, 8, 16, 2, 8, 4  # H=8, G=2 (4 heads/group), nc=3

    x = mx.random.normal((bn, length, h, p))
    B = mx.random.normal((bn, length, g, n))
    C = mx.random.normal((bn, length, g, n))
    dt = mx.random.uniform(0.001, 0.1, (bn, length, h))  # post-softplus timestep (>0)
    A = -mx.exp(mx.random.normal((h,)))                  # A < 0
    D = mx.random.normal((h,))

    y_seq, s_seq = ssd_sequential(x, dt, A, B, C, D)
    y_ch, s_ch = ssd_chunked(x, dt, A, B, C, D, q)
    chunk_ok = _maxdiff(y_seq, y_ch) < 1e-4 and _maxdiff(s_seq, s_ch) < 1e-4

    # decode: applying ssd_step token-by-token reproduces the sequential output
    s = mx.zeros((bn, h, n, p))
    ys = []
    for t in range(length):
        y_t, s = ssd_step(x[:, t], dt[:, t], A, B[:, t], C[:, t], D, s)
        ys.append(y_t)
    y_step = mx.stack(ys, axis=1)
    step_ok = _maxdiff(y_seq, y_step) < 1e-4 and _maxdiff(s_seq, s) < 1e-4

    # bounded-memory prefill: process in two blocks carrying the state, == full sequence
    cut = 8  # both blocks divisible by q=4
    y1, s1 = ssd_chunked(x[:, :cut], dt[:, :cut], A, B[:, :cut], C[:, :cut], D, q)
    y2, s2 = ssd_chunked(x[:, cut:], dt[:, cut:], A, B[:, cut:], C[:, cut:], D, q, state_in=s1)
    carry_ok = _maxdiff(mx.concatenate([y1, y2], axis=1), y_seq) < 1e-4 and _maxdiff(s2, s_seq) < 1e-4

    # causal depthwise conv: prefill == stepwise
    cch, k, lc = 6, 4, 10
    u = mx.random.normal((bn, lc, cch))
    w = mx.random.normal((cch, k))
    cb = mx.random.normal((cch,))
    y_pf = causal_conv1d(u, w, cb)
    cstate = mx.zeros((bn, k - 1, cch))
    cys = []
    for t in range(lc):
        y_t, cstate = causal_conv1d_step(u[:, t], w, cstate, cb)
        cys.append(y_t)
    conv_ok = _maxdiff(y_pf, mx.stack(cys, axis=1)) < 1e-4

    print("\n=== Mamba-2 SSD parity (synthetic) ===")
    print(f"chunked prefill == sequential        : {chunk_ok}  maxdiff_y={_maxdiff(y_seq, y_ch):.2e}")
    print(f"decode step-loop == sequential       : {step_ok}")
    print(f"block-split + carried state == full  : {carry_ok}")
    print(f"causal conv prefill == stepwise      : {conv_ok}")
    assert all([chunk_ok, step_ok, carry_ok, conv_ok])
    print("Mamba-2 SSD OK (chunked == sequential == decode; conv prefill == step)")


if __name__ == "__main__":
    run()
