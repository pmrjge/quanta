"""Parity (model-free): fused multi-token ``ssd_scan_fused`` == per-token ``ssd_step`` loop, and the
mixer's batched conv == the per-token ``causal_conv1d_step`` loop (bit-identical).

Stream B's fused multi-token SSD scan kernel (one Metal launch for a whole ``T``-token spec-VERIFY
continuation) must be output-equivalent to the per-token step loop it replaces (rule 4) before it goes
behind ``FUSED_SSD_SCAN`` on the real backbone. Two checks at Nemotron-Ultra's mamba dims
(H=256, P=64, N=128, G=8):

1. **scan kernel** — ``ssd_scan_fused(x[:, :T], …)`` vs a reference loop of ``ssd_step`` over the T
   tokens carrying state. The kernel runs the SAME fp32 recurrence; the only numeric difference is the
   ``C·s`` reduction (kernel's sequential ``acc`` vs ``mx.sum``), the same source as the one-token
   ``ssd_step_fused`` ~2e-7 (``nemotron_ssd_kernel_test.py``), here compounded over ``T`` steps. Also
   asserts ``T == 1`` reduces **bit-exactly** to ``ssd_step_fused`` (the prologue copies state then runs
   one step).
2. **batched conv** — the mixer's fused branch builds the conv output for all T at once (the rolling
   ``causal_conv1d_step`` window materialised over T, reduced by the SAME ``mx.sum`` over the K axis).
   It must be **bit-identical** to the per-token loop (max abs diff 0) — the conv carries no fp risk, so
   any divergence is a window/ordering bug.

    uv run python -m parity.nemotron_ssd_scan_kernel_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.nemotron.mamba_ssd import (
    causal_conv1d_step,
    ssd_scan_fused,
    ssd_step,
    ssd_step_fused,
)

H, P, N, G = 256, 64, 128, 8          # Nemotron-Ultra mamba dims
T_VALUES = (1, 2, 3, 4)               # verify widths k+1 for k in {0..3}
CONV_DIM, K = 257, 4                  # conv bit-identity is dim-independent; odd C catches indexing


def _rel(a: mx.array, b: mx.array) -> float:
    return (mx.linalg.norm((a - b).astype(mx.float32))
            / (mx.linalg.norm(b.astype(mx.float32)) + 1e-30)).item()


def _maxabs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _scan_ref(x, dt, A, B, C, D, state):
    """Per-token ``ssd_step`` loop carrying state — the path the fused kernel replaces (mamba_mixer)."""
    ys = []
    for ti in range(x.shape[1]):
        y_t, state = ssd_step(x[:, ti], dt[:, ti], A, B[:, ti], C[:, ti], D, state)
        ys.append(y_t[:, None])
    return mx.concatenate(ys, axis=1), state


def _conv_ref(xbc, weight, conv_state, bias):
    """Per-token ``causal_conv1d_step`` loop (the eager mixer conv path)."""
    ys = []
    for ti in range(xbc.shape[1]):
        y_t, conv_state = causal_conv1d_step(xbc[:, ti], weight, conv_state, bias)
        ys.append(y_t[:, None])
    return mx.concatenate(ys, axis=1), conv_state


def _conv_batched(xbc, weight, conv_state, bias, k):
    """The mixer's fused-branch batched conv (must be bit-identical to ``_conv_ref``)."""
    t = xbc.shape[1]
    xbc_ext = mx.concatenate([conv_state, xbc], axis=1)
    wk = mx.swapaxes(weight, 0, 1)
    windows = mx.stack([xbc_ext[:, j:j + t] for j in range(k)], axis=2)
    return mx.sum(windows * wk[None, None], axis=2) + bias, xbc_ext[:, -(k - 1):]


def run() -> None:
    mx.random.seed(0)
    ok = True

    print(f"=== ssd_scan_fused vs per-token ssd_step loop (H={H} P={P} N={N} G={G}) ===")
    for bn in (1, 2):
        x = mx.random.normal([bn, max(T_VALUES), H, P])
        dt = mx.abs(mx.random.normal([bn, max(T_VALUES), H])) * 0.1        # dt > 0
        A = -mx.abs(mx.random.normal([H]))                                 # A < 0
        B = mx.random.normal([bn, max(T_VALUES), G, N])
        C = mx.random.normal([bn, max(T_VALUES), G, N])
        D = mx.random.normal([H])
        state0 = mx.random.normal([bn, H, N, P])
        for t in T_VALUES:
            y_ref, s_ref = _scan_ref(x[:, :t], dt[:, :t], A, B[:, :t], C[:, :t], D, state0)
            y_fus, s_fus = ssd_scan_fused(x[:, :t], dt[:, :t], A, B[:, :t], C[:, :t], D, state0)
            ry, rs = _rel(y_fus, y_ref), _rel(s_fus, s_ref)
            ok = ok and ry < 1e-4 and rs < 1e-4
            print(f"  bn={bn} T={t}: y rel {ry:.2e} | state rel {rs:.2e}")

    # T==1 must reduce bit-exactly to the one-token fused kernel
    print("\n=== ssd_scan_fused(T=1) == ssd_step_fused (bit-exact) ===")
    bn = 2
    x1 = mx.random.normal([bn, 1, H, P])
    dt1 = mx.abs(mx.random.normal([bn, 1, H])) * 0.1
    A1 = -mx.abs(mx.random.normal([H]))
    B1 = mx.random.normal([bn, 1, G, N])
    C1 = mx.random.normal([bn, 1, G, N])
    D1 = mx.random.normal([H])
    st1 = mx.random.normal([bn, H, N, P])
    ys, ss = ssd_scan_fused(x1, dt1, A1, B1, C1, D1, st1)
    yf, sf = ssd_step_fused(x1[:, 0], dt1[:, 0], A1, B1[:, 0], C1[:, 0], D1, st1)
    dy, ds = _maxabs(ys[:, 0], yf), _maxabs(ss, sf)
    ok = ok and dy == 0.0 and ds == 0.0
    print(f"  y maxabs {dy:.2e} | state maxabs {ds:.2e}")

    print("\n=== batched conv == per-token causal_conv1d_step loop (bit-exact) ===")
    for bn in (1, 2):
        for t in (2, 3, 4):
            xbc = mx.random.normal([bn, t, CONV_DIM])
            weight = mx.random.normal([CONV_DIM, K])
            bias = mx.random.normal([CONV_DIM])
            cstate = mx.random.normal([bn, K - 1, CONV_DIM])
            yr, cr = _conv_ref(xbc, weight, cstate, bias)
            yb, cb = _conv_batched(xbc, weight, cstate, bias, K)
            dy, dc = _maxabs(yb, yr), _maxabs(cb, cr)
            ok = ok and dy == 0.0 and dc == 0.0
            print(f"  bn={bn} T={t}: conv maxabs {dy:.2e} | conv_state maxabs {dc:.2e}")

    print("\nPASS" if ok else "\nFAIL (fused scan / batched conv != per-token reference)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
