"""AWQ beats RTN on a bf16-source weight with salient channels (the Nemotron case).

Synthetic linear with a few high-activation input channels. AWQ's per-channel scaling must
(a) be exactly invariant in the continuous limit — ``(x/s)(W·diag(s))ᵀ == x Wᵀ`` — and
(b) yield lower activation-weighted int4 reconstruction error than plain RTN. A no-salient
control should make AWQ ≈ RTN (nothing to protect).

    uv run python -m parity.awq_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.bake.awq import awq_quantize, awq_scale
from quanta.bake.quant import quantize_affine

OUT, IN, N, BITS, GS = 128, 256, 96, 4, 128


def _err(w_recon: mx.array, x_scaled: mx.array, y_ref: mx.array) -> float:
    return float(mx.mean((x_scaled @ w_recon.T - y_ref) ** 2).item())


def _rtn_err(w: mx.array, x: mx.array, y_ref: mx.array) -> float:
    p, s, b = quantize_affine(w, BITS, GS)
    return _err(mx.dequantize(p, s, b, group_size=GS, bits=BITS), x, y_ref)


def _awq_err(w: mx.array, x: mx.array, y_ref: mx.array) -> tuple[float, mx.array]:
    s, p, sc, b = awq_quantize(w, x, BITS, GS)
    wq = mx.dequantize(p, sc, b, group_size=GS, bits=BITS)
    return _err(wq, x / s[None, :], y_ref), s


def run() -> None:
    mx.random.seed(0)
    w = mx.random.normal((OUT, IN))

    # salient activations: a handful of input channels with ~8x magnitude
    cols = mx.arange(IN)
    sal = mx.array([3, 50, 120, 200])
    salmask = mx.any(cols[:, None] == sal[None, :], axis=1)
    x = mx.random.normal((N, IN)) * mx.where(salmask, 8.0, 1.0)[None, :]
    y = x.astype(mx.float32) @ w.astype(mx.float32).T

    # invariance: with s applied, the continuous (unquantized) product is unchanged
    s = awq_scale(w, x, BITS, GS)
    inv = float(mx.max(mx.abs((x / s[None, :]) @ (w * s[None, :]).T - y)).item())
    invariant_ok = inv < 1e-2

    rtn = _rtn_err(w, x, y)
    awq, _ = _awq_err(w, x, y)
    beats_ok = awq < rtn  # AWQ must reduce activation-weighted int4 error on salient data

    # control: no salient channels → AWQ has nothing to protect → ≈ RTN
    xc = mx.random.normal((N, IN))
    yc = xc.astype(mx.float32) @ w.astype(mx.float32).T
    rtn_c, awq_c = _rtn_err(w, xc, yc), _awq_err(w, xc, yc)[0]
    control_ok = awq_c <= rtn_c * 1.05  # no worse than RTN

    print("\n=== AWQ (per-channel scaling) vs RTN ===")
    print(f"continuous invariance (x/s)(Ws)ᵀ==xWᵀ : {invariant_ok}  max|Δ|={inv:.2e}")
    print(f"salient int4: AWQ {awq:.4f} < RTN {rtn:.4f}  : {beats_ok}  ({rtn / max(awq, 1e-9):.2f}x lower)")
    print(f"no-salient control: AWQ ≈ RTN          : {control_ok}  (AWQ {awq_c:.4f} / RTN {rtn_c:.4f})")
    assert all([invariant_ok, beats_ok, control_ok])
    print("AWQ OK (invariant; beats RTN on salient channels; harmless without)")


if __name__ == "__main__":
    run()
