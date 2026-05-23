"""Dequantize the source checkpoint's int4 routed experts (MLX).

The Kimi-K2.6 source ships routed experts as group-quantized int4, ``group_size=32``
along the input dim. config.json carries **no** ``quantization_config``; the format
is read off the tensors themselves. Each projection is three tensors:

* ``weight_packed`` — int32, ``[out, in // 8]`` (8 int4 codes per int32, LSB-first)
* ``weight_scale``  — bf16,  ``[out, in // 32]`` (one scale per 32-wide input group)
* ``weight_shape``  — int32, the original ``[out, in]``

The 4-bit codes are **offset-binary** (a.k.a. excess-8), not two's-complement: a code
``c`` in ``0..15`` represents ``c - 8`` in ``-8..7``, so the implicit zero-point is 8
(no zero-point tensor; the encoding is symmetric, centered at code 8). Verified
empirically — the unsigned-code histogram is a bell curve centered exactly at 8, and
``w = (c - 8) * scale`` reconstructs zero-mean weights matching the bf16 shared expert
(sign-extension instead injects a ~-0.0067 DC bias that detonates the residual).
Offline tool (bake/parity loading); pure MLX, no torch / compressed_tensors dep.
"""

from __future__ import annotations

import mlx.core as mx

PACK_FACTOR = 8  # int4 values per int32


def dequantize_packed_int4(
    packed: mx.array,
    scale: mx.array,
    out_features: int,
    in_features: int,
    group_size: int = 32,
    dtype: mx.Dtype = mx.bfloat16,
) -> mx.array:
    """Unpack + dequantize one int4 projection → dense ``[out, in]`` weight."""
    shifts = mx.arange(0, 32, 4, dtype=mx.int32)  # [8] LSB-first nibble offsets
    nibbles = (packed[..., None] >> shifts) & 0xF  # [out, in//8, 8], codes 0..15
    q = nibbles.reshape(out_features, in_features).astype(mx.float32) - 8.0  # offset-binary → [-8, 7]

    s = mx.repeat(scale.astype(mx.float32), group_size, axis=1)  # [out, in]
    return (q * s).astype(dtype)
