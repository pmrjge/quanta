"""Shared affine KV-cache quantization helpers — used by every model whose decode cache opts in
to ``quantized=True``.

The win is **steady-state memory** at long context (cache storage is int<bits> codes + per-group
scales + biases vs bf16's 16 bpp). Per-bit-width effective bpp (with fp16 scales+biases at g64):

* ``bits=8 g64``  ≈ 8.5 bpp  — the historical default; near-lossless (Kimi MLA latent since #47).
* ``bits=4 g64``  ≈ 4.5 bpp  — aggressive but viable for long-context decode (Qwen2.5-1M default).
* ``bits=4 g32``  ≈ 5.0 bpp  — slightly tighter groups, more quality headroom at int4.

Decode compute is unchanged: ``update`` returns a dequantized bf16 tensor for the existing SDPA
path (no quantized-matmul absorbed path here — that's a perf follow-up). The Kimi MLA cache
(:class:`quanta.cache.MLACache`) has used this exact scheme on its ``c_kv`` latent since #47 and
is the e2e arbiter; this module just extracts the reshape + call boilerplate so the same proven
scheme drives every model's GQA / MLA / latent KV caches.

``bits`` defaults to ``8`` so every existing caller (Nemotron / DSV4 / GLM / MiniMax / Qwen3.5)
keeps its int8-g64 behavior; Qwen2.5-1M opts in to ``bits=4`` for the 1M-context steady state.

Both ``mx.quantize`` and ``mx.dequantize`` operate on 2D inputs; we wrap them to take an N-D tensor
and quantize **along the last axis** (the feature / head_dim / lora dim), preserving the leading
shape so the caller can concat the (codes, scales, biases) trio along the same seq axis as the bf16
path.
"""

from __future__ import annotations

import mlx.core as mx

BITS = 8  # default for callers that don't pass an explicit ``bits`` kwarg


def quantize_last_axis(t: mx.array, group_size: int,
                       bits: int = BITS) -> tuple[mx.array, mx.array, mx.array]:
    """Affine int-``bits`` quantize ``t`` along its last axis, preserving every leading axis.

    Returns ``(codes, scales, biases)`` each sharing ``t.shape[:-1]`` as a prefix; only the last
    axis differs (codes: ``t.shape[-1] / pack_factor``, scales/biases: ``t.shape[-1] / group_size``).
    The caller concatenates the trio along whatever **non-last** axis represents seq / ncomp."""
    shape = t.shape
    flat = t.reshape(-1, shape[-1])
    q, s, b = mx.quantize(flat, group_size=group_size, bits=bits)
    q = q.reshape(*shape[:-1], q.shape[-1])
    s = s.reshape(*shape[:-1], s.shape[-1])
    b = b.reshape(*shape[:-1], b.shape[-1])
    return q, s, b


def dequantize_last_axis(q: mx.array, s: mx.array, b: mx.array, group_size: int,
                        dtype: mx.Dtype = mx.bfloat16,
                        bits: int = BITS) -> mx.array:
    """Affine int-``bits`` dequantize ``(q, s, b)`` back to ``[..., orig_last_axis]`` in ``dtype``.

    Inverse of :func:`quantize_last_axis`; assumes the trio was concatenated along the same non-last
    axis (so they still share the leading prefix) and was packed at the same ``bits`` width."""
    shape = q.shape
    qf = q.reshape(-1, shape[-1])
    sf = s.reshape(-1, s.shape[-1])
    bf = b.reshape(-1, b.shape[-1])
    out = mx.dequantize(qf, sf, bf, group_size=group_size, bits=bits)
    return out.reshape(*shape[:-1], out.shape[-1]).astype(dtype)
