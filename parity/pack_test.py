"""Validate the affine packer matches MLX and round-trips GPTQ codes (synthetic, instant).

1. pack/unpack must reproduce ``mx.quantize``'s packed layout **bit-for-bit** (bits 3/4/8).
2. GPTQ codes packed then ``mx.dequantize``'d must equal the GPTQ Ŵ — i.e. the runtime
   (gather_qmm / quantized_matmul) will see exactly what GPTQ produced.

    uv run python -m parity.pack_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.bake.gptq import gptq_quantize
from quanta.bake.quant import pack_affine, quantize_affine, unpack_affine

OUT, IN, GS = 64, 256, 128


def run() -> None:
    mx.random.seed(0)
    w = mx.random.normal((OUT, IN))

    print("\n=== affine packer == mx.quantize layout ===")
    ok = True
    for bits in (3, 4, 8):
        wq_ref, _, _ = quantize_affine(w, bits, GS)  # mx.quantize packed words
        repacked = pack_affine(unpack_affine(wq_ref, IN, bits), bits)
        match = bool(mx.all(repacked == wq_ref).item())
        ok = ok and match
        print(f"  int{bits}: wq {wq_ref.dtype} {tuple(wq_ref.shape)}  repack==mx.quantize: {match}")

    print("\n=== GPTQ codes -> pack -> mx.dequantize == Ŵ ===")
    x = mx.random.normal((96, IN))
    for bits in (3, 4):
        w_hat, codes, scales, biases = gptq_quantize(w, x, bits, group_size=GS)
        packed = pack_affine(codes.astype(mx.uint32), bits)
        recon = mx.dequantize(packed, scales, biases, group_size=GS, bits=bits)
        err = mx.max(mx.abs(recon - w_hat)).item()
        ok = ok and err < 1e-3
        print(f"  int{bits}: max|dequant(pack(codes)) - w_hat| = {err:.3e}")

    assert ok, "packer must match mx.quantize and round-trip GPTQ codes"
    print("packer matches mx.quantize and round-trips GPTQ codes")


if __name__ == "__main__":
    run()
