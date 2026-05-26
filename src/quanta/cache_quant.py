"""Shared int8 affine KV-cache quantization helpers — used by every model whose decode cache opts
in to ``quantized=True``.

The win is **steady-state memory** at long context (cache storage is int8 codes + per-group scales +
biases ≈ 8.25 bpp vs bf16's 16 bpp). Decode compute is unchanged: ``update`` returns a dequantized
bf16 tensor for the existing SDPA path (no quantized-matmul absorbed path here — that's a perf
follow-up). The Kimi MLA cache (:class:`quanta.cache.MLACache`) has used this exact scheme on its
``c_kv`` latent since #47 and is the e2e arbiter; this module just extracts the reshape + call
boilerplate so the same proven scheme drives every model's GQA / MLA / latent KV caches.

Both ``mx.quantize`` and ``mx.dequantize`` operate on 2D inputs; we wrap them to take an N-D tensor
and quantize **along the last axis** (the feature / head_dim / lora dim), preserving the leading
shape so the caller can concat the (codes, scales, biases) trio along the same seq axis as the bf16
path.
"""

from __future__ import annotations

import mlx.core as mx

BITS = 8


def quantize_last_axis(t: mx.array, group_size: int) -> tuple[mx.array, mx.array, mx.array]:
    """Affine int8 quantize ``t`` along its last axis, preserving every leading axis.

    Returns ``(codes, scales, biases)`` each sharing ``t.shape[:-1]`` as a prefix; only the last
    axis differs (codes: ``t.shape[-1] / pack_factor``, scales/biases: ``t.shape[-1] / group_size``).
    The caller concatenates the trio along whatever **non-last** axis represents seq / ncomp."""
    shape = t.shape
    flat = t.reshape(-1, shape[-1])
    q, s, b = mx.quantize(flat, group_size=group_size, bits=BITS)
    q = q.reshape(*shape[:-1], q.shape[-1])
    s = s.reshape(*shape[:-1], s.shape[-1])
    b = b.reshape(*shape[:-1], b.shape[-1])
    return q, s, b


def dequantize_last_axis(q: mx.array, s: mx.array, b: mx.array, group_size: int,
                        dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """Affine int8 dequantize ``(q, s, b)`` back to ``[..., orig_last_axis]`` in ``dtype``.

    Inverse of :func:`quantize_last_axis`; assumes the trio was concatenated along the same non-last
    axis (so they still share the leading prefix)."""
    shape = q.shape
    qf = q.reshape(-1, shape[-1])
    sf = s.reshape(-1, s.shape[-1])
    bf = b.reshape(-1, b.shape[-1])
    out = mx.dequantize(qf, sf, bf, group_size=group_size, bits=BITS)
    return out.reshape(*shape[:-1], out.shape[-1]).astype(dtype)
