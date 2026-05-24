"""AWQ (activation-aware weight quantization) — per-input-channel scaling before affine int.

A linear ``y = x Wᵀ`` is invariant under ``W → W·diag(s)``, ``x → x·diag(1/s)``. AWQ picks a
per-input-channel scale ``s`` that *amplifies the salient channels* (the ones the calibration
activations weight most) so the subsequent int rounding spends its grid where it matters,
shrinking the activation-weighted error vs plain RTN. We grid-search the exponent ``α`` over
``s = mean|x|^α`` and keep the ``s`` with the lowest reconstruction error.

This only helps when there is sub-grid information to protect — i.e. a **bf16 source** (Nemotron),
not an already-int4 source (Kimi: AWQ ≈ RTN, settled). Output layout matches ``mx.quantize`` /
``mx.gather_qmm``; the runtime applies ``1/s`` to the layer input (folded per expert).
"""

from __future__ import annotations

import mlx.core as mx

from quanta.bake.quant import quantize_affine


def _recon(w: mx.array, bits: int, group_size: int) -> mx.array:
    packed, scales, biases = quantize_affine(w, bits, group_size)
    return mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits)


def awq_scale(w: mx.array, x: mx.array, bits: int, group_size: int, n_grid: int = 20,
              eps: float = 1e-6) -> mx.array:
    """Per-input-channel scale ``s`` ``[in]`` minimizing the activation-weighted int-recon error.

    ``w`` ``[out, in]``; ``x`` ``[n, in]`` calibration activations into this linear. ``α=0`` is
    plain RTN (``s=1``); ``α=1`` scales fully by channel activation magnitude.
    """
    wf, xf = w.astype(mx.float32), x.astype(mx.float32)
    a = mx.mean(mx.abs(xf), axis=0) + eps  # [in] per-channel activation magnitude
    a = a / mx.max(a)                      # normalize so α just reshapes the profile
    y = xf @ wf.T                          # [n, out] continuous reference
    best_s, best_err = mx.ones_like(a), float("inf")
    for i in range(n_grid):
        alpha = i / (n_grid - 1)
        s = mx.maximum(a**alpha, eps)                  # [in]
        yq = (xf / s[None, :]) @ _recon(wf * s[None, :], bits, group_size).T
        err = float(mx.mean((yq - y) ** 2).item())
        if err < best_err:
            best_err, best_s = err, s
    return best_s


def awq_quantize(w: mx.array, x: mx.array, bits: int, group_size: int, n_grid: int = 20):
    """AWQ-quantize ``w`` ``[out, in]`` on calibration ``x``. Returns ``(s, packed, scales, biases)``
    where ``(packed,scales,biases)`` is the affine int of ``W·diag(s)`` and the runtime applies
    ``x·diag(1/s)``; together they reproduce ``x Wᵀ`` up to the (reduced) int error."""
    s = awq_scale(w, x, bits, group_size, n_grid)
    packed, scales, biases = quantize_affine(w.astype(mx.float32) * s[None, :], bits, group_size)
    return s, packed, scales, biases
