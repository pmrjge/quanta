"""bf16-scale + group_size=64 round-trip — the two novel bits of the int2-g64 bake, proven on
small synthetic weights with no model load (run before the multi-hour bake).

(a) ``quantize_affine(scale_dtype=bf16)`` emits **bf16** scales/biases (halves the per-group
    overhead — int2-g64 drops 3.0→2.5 bpp) and reconstructs within RTN tolerance.
(b) those stored bf16 scales decode correctly through ``mx.quantized_matmul`` (the non-expert path).
(c) ... and through ``mx.gather_qmm`` (the routed-expert path) at bits 2 and 4, group_size 64.

The kernel-correctness bound (qmm/gather vs an explicit dequant@x using the *same* bf16 scales)
is the real gate: a kernel ignoring/mis-typing the scales would be O(1) wrong, not ~bf16-eps.

    uv run python -m parity.bf16_scale_g64_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.bake.quant import quantize_affine

OUT, IN, GS = 256, 512, 64


def _relmax(a: mx.array, b: mx.array) -> float:
    return _md(a, b) / (float(mx.max(mx.abs(b.astype(mx.float32)))) + 1e-6)


def _md(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def run() -> None:
    mx.random.seed(0)
    w = mx.random.normal((OUT, IN)).astype(mx.float32)
    x = mx.random.normal((3, IN)).astype(mx.bfloat16)

    ok = True
    for bits, recon_tol in ((2, 0.50), (4, 0.12)):  # loose sanity on recon; kernel bound is the gate
        packed, s, b = quantize_affine(w, bits, GS, scale_dtype=mx.bfloat16)
        dtype_ok = s.dtype == mx.bfloat16 and b.dtype == mx.bfloat16
        wd = mx.dequantize(packed, s, b, group_size=GS, bits=bits)
        rel = float(mx.linalg.norm((w - wd).astype(mx.float32))) / (float(mx.linalg.norm(w)) + 1e-12)
        y_ref = x.astype(mx.float32) @ wd.astype(mx.float32).T  # explicit dequant @ x, same bf16 scales

        # (b) non-expert path
        y_qmm = mx.quantized_matmul(x, packed, s, b, transpose=True, group_size=GS, bits=bits)
        qmm_err = _relmax(y_qmm, y_ref)

        # (c) routed-expert path: gather_qmm over a 2-expert stack, all rows routed to slot 1
        ps, ss, bs = mx.stack([packed, packed]), mx.stack([s, s]), mx.stack([b, b])
        lhs = mx.arange(3, dtype=mx.int32)
        rhs = mx.ones(3, dtype=mx.int32)
        y_g = mx.gather_qmm(x[:, None, :], ps, ss, bs, lhs_indices=lhs, rhs_indices=rhs,
                            transpose=True, group_size=GS, bits=bits)[:, 0, :]
        gather_err = _relmax(y_g, y_ref)

        finite = bool(mx.all(mx.isfinite(y_qmm)).item() and mx.all(mx.isfinite(y_g)).item())
        good = dtype_ok and rel < recon_tol and qmm_err < 2e-2 and gather_err < 2e-2 and finite
        ok = ok and good
        print(f"int{bits}-g{GS}: bf16-scales {dtype_ok} | recon {rel:.3f}<{recon_tol} | "
              f"qmm {qmm_err:.2e} | gather {gather_err:.2e} | finite {finite} -> {good}")

    assert ok
    print("bf16-scale + g64 OK (round-trip; quantized_matmul & gather_qmm decode bf16 scales)")


if __name__ == "__main__":
    run()
