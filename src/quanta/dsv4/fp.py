"""DeepSeek-V4 source-quant dequant gate — fp8-e4m3, packed fp4-e2m1, and e8m0/MX scales.

This is the "bug we faced before" gate, in DSV4's *new* formats. The checkpoint ships two quant
schemes, **both with OCP-Microscaling e8m0 (power-of-two) scales** (not fp32 like Kimi):

* **non-experts** — weight ``F8_E4M3`` ``[out,in]`` + scale ``F8_E8M0`` ``[ceil(out/128),ceil(in/128)]``
  (one scale per ``[128,128]`` block). :func:`dequant_block_fp8`.
* **routed experts** — weight packed ``FP4`` (e2m1, 2 values/byte along ``in``) stored as ``I8``
  ``[out,in/2]`` + scale ``F8_E8M0`` ``[out,in/32]`` (one scale per 32 elements along ``in``).
  :func:`dequant_group_fp4`.

**MLX gotcha (0.31.x):** ``mx.load`` raises ``unsupported dtype F8_E8M0`` and aborts the *whole*
shard, so we cannot use it here — we read raw safetensors bytes ourselves (:func:`decode_buffer`,
:func:`read_raw_tensor`) and decode in MLX. ``BF16`` is read as ``uint16`` then bit-viewed; the fp8
families are read as raw ``uint8``.

Decode is exact via precomputed lookup tables (gated bit-exact in ``parity/dsv4_dequant_test.py``):

* **e4m3fn**: 1 sign / 4 exp (bias 7) / 3 mantissa; no inf; NaN only at ``0x7f``/``0xff``; max 448.
* **e2m1 (fp4)**: 16-entry signed grid ``{0,.5,1,1.5,2,3,4,6}`` (low nibble first, then high).
* **e8m0**: 8-bit exponent only; value ``2^(byte-127)``; ``0xFF`` = NaN; no sign/mantissa/zero.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

E4M3_MAX = 448.0
BLOCK = 128          # fp8 block (square, both dims)
FP4_GROUP = 32       # fp4 group (along the in / reduce dim)


# --- bit-exact decode tables -------------------------------------------------
def _build_e4m3fn_table() -> list[float]:
    """All 256 e4m3fn bytes -> Python float (bounded 256-iter table build; not a hot path)."""
    out: list[float] = []
    for b in range(256):
        sign, exp, man = (b >> 7) & 0x1, (b >> 3) & 0xF, b & 0x7
        if exp == 0xF and man == 0x7:          # the two NaN bytes (no inf in e4m3fn)
            v = float("nan")
        elif exp == 0:                          # subnormal: man/8 * 2^(1-bias)
            v = (man / 8.0) * (2.0 ** -6)
        else:                                   # normal: (1 + man/8) * 2^(exp-bias)
            v = (1.0 + man / 8.0) * (2.0 ** (exp - 7))
        out.append(-v if sign else v)
    return out


def _build_e8m0_table() -> list[float]:
    """All 256 e8m0 bytes -> float. value = 2^(b-127); 0xFF = NaN. 2^n is exact in f64->f32."""
    return [float("nan") if b == 0xFF else 2.0 ** (b - 127) for b in range(256)]


# e2m1 (fp4) grid: index = 4-bit nibble. 0..7 positive, 8..15 negative (sign bit = bit 3).
_FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

_E4M3_LUT = mx.array(_build_e4m3fn_table(), dtype=mx.float32)  # [256]
_E8M0_LUT = mx.array(_build_e8m0_table(), dtype=mx.float32)    # [256]
_FP4_LUT = mx.array(_FP4_VALUES, dtype=mx.float32)             # [16]


def e4m3_to_float(w_u8: mx.array) -> mx.array:
    """Decode e4m3fn bytes (uint8, any shape) -> float32 via the 256-entry LUT."""
    return _E4M3_LUT[w_u8.astype(mx.uint32)]


def e8m0_to_float(s_u8: mx.array) -> mx.array:
    """Decode e8m0 scale bytes (uint8, any shape) -> float32 (= 2^(byte-127))."""
    return _E8M0_LUT[s_u8.astype(mx.uint32)]


def unpack_fp4(w_u8: mx.array) -> mx.array:
    """Unpack packed e2m1 ``[..., in/2]`` (uint8, 2 nibbles/byte, low first) -> float32 ``[..., in]``."""
    if w_u8.dtype != mx.uint8:
        w_u8 = w_u8.view(mx.uint8)            # bitcast int8 storage -> raw bytes
    idx = w_u8.astype(mx.uint32)
    lo = _FP4_LUT[idx & 0xF]
    hi = _FP4_LUT[(idx >> 4) & 0xF]
    return mx.stack([lo, hi], axis=-1).reshape(*w_u8.shape[:-1], w_u8.shape[-1] * 2)


# --- weight dequant ----------------------------------------------------------
def dequant_block_fp8(w_u8: mx.array, scale_u8: mx.array, block: int = BLOCK,
                      dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """Block-fp8 dequant. ``w_u8``: e4m3 bytes ``[out,in]``; ``scale_u8``: e8m0 grid
    ``[ceil(out/block),ceil(in/block)]``. Each ``[block,block]`` tile shares one scale."""
    if w_u8.ndim != 2:
        raise ValueError(f"dequant_block_fp8 expects a 2-D weight, got shape {tuple(w_u8.shape)}")
    out, inn = w_u8.shape
    nbo, nbi = (out + block - 1) // block, (inn + block - 1) // block
    if scale_u8.shape[0] < nbo or scale_u8.shape[1] < nbi:
        raise ValueError(f"fp8 scale grid {tuple(scale_u8.shape)} too small for weight "
                         f"{(out, inn)} at block {block} (need >= {(nbo, nbi)})")
    wf = e4m3_to_float(w_u8)
    sf = e8m0_to_float(scale_u8)
    sf = mx.repeat(mx.repeat(sf, block, axis=0), block, axis=1)[:out, :inn]
    return (wf * sf).astype(dtype)


def dequant_group_fp4(w_u8: mx.array, scale_u8: mx.array, group: int = FP4_GROUP,
                      dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """Packed-fp4 dequant. ``w_u8``: packed e2m1 ``[out,in/2]``; ``scale_u8``: e8m0 ``[out,in/group]``
    (one scale per ``group`` elements along ``in``). Returns ``[out,in]``."""
    if w_u8.ndim != 2:
        raise ValueError(f"dequant_group_fp4 expects a 2-D weight, got shape {tuple(w_u8.shape)}")
    wf = unpack_fp4(w_u8)                       # [out, in]
    out, inn = wf.shape
    nbi = (inn + group - 1) // group
    if scale_u8.shape[0] != out or scale_u8.shape[1] < nbi:
        raise ValueError(f"fp4 scale grid {tuple(scale_u8.shape)} mismatched for weight {(out, inn)} "
                         f"at group {group} (need [{out}, >= {nbi}])")
    sf = mx.repeat(e8m0_to_float(scale_u8), group, axis=1)[:, :inn]
    return (wf * sf).astype(dtype)


# --- raw safetensors reading (mx.load can't read F8_E8M0) --------------------
_ST_NUMPY = {"F32": np.float32, "F16": np.float16, "F64": np.float64,
             "I8": np.int8, "I16": np.int16, "I32": np.int32, "I64": np.int64,
             "U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "BOOL": np.bool_}
_ST_RAW_U8 = {"F8_E4M3", "F8_E5M2", "F8_E8M0", "F4", "F4_E2M1", "FP8_E4M3", "FP8_E8M0"}


def decode_buffer(dtype: str, shape, buf) -> mx.array:
    """Decode a raw safetensors byte range into an MLX array in its natural dtype.

    ``BF16`` -> ``bfloat16`` (uint16 bit-view); fp8/fp4 families -> raw ``uint8``; everything else
    maps through numpy. No dequant here — that is the caller's job via the functions above.
    """
    if dtype == "BF16":
        a = mx.array(np.frombuffer(buf, dtype=np.uint16)).view(mx.bfloat16)
    elif dtype in _ST_RAW_U8:
        a = mx.array(np.frombuffer(buf, dtype=np.uint8))
    elif dtype in _ST_NUMPY:
        a = mx.array(np.frombuffer(buf, dtype=_ST_NUMPY[dtype]))
    else:
        raise ValueError(f"unsupported safetensors dtype {dtype!r} (refusing to guess)")
    return a.reshape(tuple(shape)) if shape else a


def read_safetensors_header(path: str | Path) -> tuple[dict, int]:
    """Parse a safetensors header -> ``(metadata_dict, data_base_offset)`` (cheap; no tensor data)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def read_raw_tensor(path: str | Path, key: str) -> mx.array:
    """Read a single tensor's raw bytes from a safetensors file and decode to its natural dtype.

    Standalone (re-parses the header per call) — for tests/inspection. The streamed loader keeps
    shards mmapped instead (see :mod:`quanta.dsv4.loader`)."""
    hdr, base = read_safetensors_header(path)
    if key not in hdr:
        raise KeyError(f"{key} not in {path}")
    m = hdr[key]
    b, e = m["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + b)
        buf = f.read(e - b)
    return decode_buffer(m["dtype"], m["shape"], buf)
