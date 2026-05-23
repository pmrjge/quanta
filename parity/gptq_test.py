"""GPTQ solver validation (synthetic, instant, CPU): Woodbury exactness + beats RTN.

1. woodbury_inverse must match the direct (δI + XᵀX)⁻¹.
2. GPTQ's activation-weighted error ‖(Ŵ−W)Xᵀ‖/‖WXᵀ‖ must be below round-to-nearest at the
   same bits/group — the whole point of error feedback. n < in (the under-covered regime).

    uv run python -m parity.gptq_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.bake.gptq import gptq_quantize, woodbury_inverse
from quanta.bake.quant import dequantize_affine, quantize_affine

OUT, IN, N = 256, 512, 96


def _awl(w: mx.array, wq: mx.array, x: mx.array) -> float:
    wf, xt = w.astype(mx.float32), x.astype(mx.float32).T
    return (mx.linalg.norm((wq.astype(mx.float32) - wf) @ xt) / (mx.linalg.norm(wf @ xt) + 1e-9)).item()


def run() -> None:
    mx.random.seed(0)
    w = mx.random.normal((OUT, IN))
    x = mx.random.normal((N, IN))

    delta = 0.01 * (mx.sum(x.astype(mx.float32) ** 2) / IN).item()
    h_inv_w = woodbury_inverse(x, delta)
    h = x.astype(mx.float32).T @ x.astype(mx.float32) + delta * mx.eye(IN)
    with mx.stream(mx.cpu):
        h_inv_d = mx.linalg.cholesky_inv(mx.linalg.cholesky(h))
    wood = (mx.linalg.norm(h_inv_w - h_inv_d) / mx.linalg.norm(h_inv_d)).item()

    print("\n=== GPTQ solver checks (out=256, in=512, n=96) ===")
    print(f"Woodbury vs direct inverse : rel {wood:.3e}   (expect ~0)")
    print("activation-weighted loss, GPTQ vs RTN:")
    ok = wood < 1e-3
    for bits in (3, 4):
        w_rtn = dequantize_affine(*quantize_affine(w, bits, 128), bits, 128)
        w_g, _, _, _ = gptq_quantize(w, x, bits, group_size=128)
        lr, lg = _awl(w, w_rtn, x), _awl(w, w_g, x)
        print(f"  int{bits}: RTN {lr:.4%}   GPTQ {lg:.4%}   ({lr / lg:.2f}x lower)")
        ok = ok and lg < lr
    assert ok, "GPTQ must beat RTN and Woodbury must match direct"
    print("GPTQ beats RTN; Woodbury inverse exact")


if __name__ == "__main__":
    run()
