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
    w: mx.array, bits: int, group_size: int = GROUP_SIZE, *, scale_dtype: mx.Dtype | None = None
) -> tuple[mx.array, mx.array, mx.array]:
    """RTN affine-quantize a 2-D weight ``[out, in]`` → ``(w_q, scales, biases)`` (MLX packed).

    ``scale_dtype`` (e.g. ``mx.bfloat16``) downcasts the weight before quantizing so MLX returns
    scales/biases in that dtype — halving the per-group overhead vs fp32 scales (e.g. int2-g64
    drops 3.0→2.5 bpp), and the integer codes stay consistent with the stored bf16 scales. ``None``
    keeps the input dtype (fp32 here)."""
    if scale_dtype is not None:
        w = w.astype(scale_dtype)
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


def pack_affine(codes: mx.array, bits: int) -> mx.array:
    """Pack integer codes ``[out, in]`` (0..2^bits−1) into MLX's uint32 layout for GPTQ output.

    LSB-first contiguous bitstream — code ``i`` occupies bit positions ``[i·bits, i·bits+bits)``
    — matching ``mx.quantize`` so the result feeds ``mx.dequantize`` / ``gather_qmm`` directly.
    Needed because GPTQ rounds with error feedback (not RTN), so its codes can't come from
    ``mx.quantize``; we pack them ourselves. Requires ``in·bits`` divisible by 32 (holds for
    group-128 weights). Vectorized — no per-code loop.
    """
    out, in_ = codes.shape
    c = codes.astype(mx.uint32)
    bit_pos = mx.arange(bits, dtype=mx.uint32)
    expanded = (c[..., None] >> bit_pos) & 1  # [out, in, bits], LSB-first per code
    stream = expanded.reshape(out, (in_ * bits) // 32, 32)  # 32-bit words
    word_pos = mx.arange(32, dtype=mx.uint32)
    return mx.sum((stream.astype(mx.uint32) << word_pos), axis=-1).astype(mx.uint32)


def unpack_affine(words: mx.array, in_features: int, bits: int) -> mx.array:
    """Inverse of :func:`pack_affine`: uint32 words ``[out, n_words]`` → codes ``[out, in]``."""
    out = words.shape[0]
    word_pos = mx.arange(32, dtype=mx.uint32)
    stream = (words[..., None] >> word_pos) & 1  # [out, n_words, 32]
    grouped = stream.reshape(out, in_features, bits)  # [out, in, bits]
    bit_pos = mx.arange(bits, dtype=mx.uint32)
    return mx.sum((grouped.astype(mx.uint32) << bit_pos), axis=-1).astype(mx.uint32)
