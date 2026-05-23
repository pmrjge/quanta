"""Affine quantization primitives for the bake (MLX-native, ``mx.quantized_matmul``-ready).

Non-expert weights (attention q/kv/o, dense L0 MLP, lm_head) are RTN **affine int8,
group-128** — near-lossless (~0.8% recon) and cheap vs the experts. ``quantize_affine``
returns MLX's packed ``(w_q, scales, biases)`` so the resident runtime calls
``mx.quantized_matmul`` directly; no custom unpacking. (Routed experts use int3 **GPTQ**
with error feedback — a separate module — whose integer codes are packed into this same
MLX layout, so the runtime path is uniform.)
"""

from __future__ import annotations

import mlx.core as mx

GROUP_SIZE = 128


def quantize_affine(
    w: mx.array, bits: int, group_size: int = GROUP_SIZE
) -> tuple[mx.array, mx.array, mx.array]:
    """RTN affine-quantize a 2-D weight ``[out, in]`` → ``(w_q, scales, biases)`` (MLX packed)."""
    return mx.quantize(w, group_size=group_size, bits=bits)


def dequantize_affine(
    w_q: mx.array, scales: mx.array, biases: mx.array, bits: int, group_size: int = GROUP_SIZE
) -> mx.array:
    """Inverse of :func:`quantize_affine` → dense ``[out, in]``."""
    return mx.dequantize(w_q, scales, biases, group_size=group_size, bits=bits)


def recon_error(w: mx.array, bits: int, group_size: int = GROUP_SIZE) -> float:
    """Relative reconstruction error ``‖w − dequant(quant(w))‖ / ‖w‖`` (bake QC gauge)."""
    wq, s, b = quantize_affine(w, bits, group_size)
    wd = dequantize_affine(wq, s, b, bits, group_size)
    num = mx.linalg.norm((w - wd).astype(mx.float32)).item()
    den = mx.linalg.norm(w.astype(mx.float32)).item() + 1e-12
    return num / den
