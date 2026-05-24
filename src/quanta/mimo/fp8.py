"""Blockwise FP8 (e4m3) dequant for MiMo-V2.5 source weights — the source-quant parity gate.

MiMo ships DeepSeek-V3-style **block-fp8**: each quantized weight ``W`` is ``F8_E4M3`` ``[out,in]``
(MLX has no fp8 dtype, so :func:`mlx.core.load` returns it as ``uint8`` raw bytes) paired with a
sibling ``weight_scale_inv`` (f32) of shape ``[ceil(pad512(out)/128), ceil(in/128)]``. Dequant is

    W_bf16 = e4m3_to_float(W_u8) * block_broadcast(scale_inv)[:out, :in]

i.e. each ``[128,128]`` block of the weight is multiplied by one scale entry. ``out`` is padded up
to a multiple of **512** before the scale grid is built, so when ``out`` is not a multiple of 512
the scale has *trailing padding rows*. The only such tensor in MiMo-V2.5 is the **full-attention
fused qkv** (``out=13568`` → padded ``13824`` → 108 scale-rows vs 106 real ``128``-blocks); SWA qkv
(``14848 = 29*512``) and every dense/expert tensor are already 512-aligned. The ``[:out,:in]`` slice
after broadcasting drops those padding rows automatically — so the same code is correct for all
tensors. This was the historical "loads fine, emits garbage" trap (cf. Kimi's offset-binary int4):
get the decode + scale mapping exactly right *before* anything downstream is trusted.

e4m3**fn** (the e4m3 used here): 1 sign, 4 exponent (bias 7), 3 mantissa; **no infinities**; the
only NaN encodings are ``0x7f`` / ``0xff`` (S.1111.111); finite max = 448. Decode is exact via a
256-entry lookup table (built once below); gated bit-exact vs torch ``float8_e4m3fn`` in
``parity/mimo_fp8_dequant_test.py``.
"""

from __future__ import annotations

import mlx.core as mx

E4M3_MAX = 448.0
BLOCK = 128


def _build_e4m3fn_table() -> list[float]:
    """All 256 e4m3fn byte values -> Python float. Bounded 256-iter table build (not a hot path)."""
    out: list[float] = []
    for b in range(256):
        sign = (b >> 7) & 0x1
        exp = (b >> 3) & 0xF
        man = b & 0x7
        if exp == 0xF and man == 0x7:        # the two NaN bytes (no inf in e4m3fn)
            v = float("nan")
        elif exp == 0:                       # subnormal: man/8 * 2^(1-bias)
            v = (man / 8.0) * (2.0 ** -6)
        else:                                # normal: (1 + man/8) * 2^(exp-bias)
            v = (1.0 + man / 8.0) * (2.0 ** (exp - 7))
        out.append(-v if sign else v)
    return out


_E4M3_LUT = mx.array(_build_e4m3fn_table(), dtype=mx.float32)  # [256]


def e4m3_to_float(w_u8: mx.array) -> mx.array:
    """Decode e4m3fn bytes (uint8, any shape) -> float32. Vectorized gather through the LUT."""
    return _E4M3_LUT[w_u8.astype(mx.uint32)]


def dequant_block_fp8(w_u8: mx.array, scale_inv: mx.array, block: int = BLOCK,
                      dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """Blockwise-fp8 dequant. ``w_u8``: uint8 ``[out,in]`` (e4m3 bytes); ``scale_inv``: f32 grid.

    Broadcasts each scale entry over its ``[block,block]`` tile, slices to ``[out,in]`` (dropping
    any trailing padding scale-rows/cols), and casts. Pure MLX — usable at bake time without torch.
    """
    if w_u8.ndim != 2:
        raise ValueError(f"dequant_block_fp8 expects a 2-D weight, got shape {w_u8.shape}")
    out, inn = w_u8.shape
    nbo, nbi = (out + block - 1) // block, (inn + block - 1) // block
    if scale_inv.shape[0] < nbo or scale_inv.shape[1] < nbi:
        raise ValueError(f"scale grid {tuple(scale_inv.shape)} too small for weight "
                         f"{(out, inn)} at block {block} (need >= {(nbo, nbi)})")
    wf = e4m3_to_float(w_u8)
    sf = mx.repeat(mx.repeat(scale_inv.astype(mx.float32), block, axis=0), block, axis=1)
    return (wf * sf[:out, :inn]).astype(dtype)
