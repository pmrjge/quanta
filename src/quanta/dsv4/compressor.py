"""DeepSeek-V4 KV Compressor (prefill) — learned gated pooling that builds the compressed KV cache.

On compressed layers (ratio 4 or 128), attention attends to a sliding window of recent per-token KV
**plus** a set of *compressed* KV tokens. Each compressed token is a softmax-gated weighted pool over
``ratio`` consecutive positions (with a learned per-slot positional bias ``ape``), then RMSNorm +
partial RoPE (at the window-start position). For ``ratio==4`` the windows **overlap** (``coff=2``):
each compressed token pools over its own ``ratio`` positions (second projection half) plus the
previous window's ``ratio`` positions (first projection half) — 8 positions total, the first window's
"previous" slots masked out.

This is the prefill path (``start_pos==0``): it compresses the first ``cutoff = seqlen - seqlen%ratio``
tokens into ``cutoff//ratio`` compressed tokens (the ``remainder`` tokens are held for the decode
state machine, task #77 — they don't affect the prefill output). Faithful MLX port of the reference
``Compressor.forward``; the indexer reuses it with a Hadamard rotate (handled in
:mod:`quanta.dsv4.indexer`). Gated vs the authors' real ``Compressor`` in
``parity/dsv4_compressor_test.py``.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.dsv4.attention import _rms_w, rope_partial


def _overlap_transform(t: mx.array, value: float, ratio: int, d: int) -> mx.array:
    """``[b,nc,ratio,2d] -> [b,nc,2*ratio,d]``: first ``ratio`` slots = previous window's first-half
    projection (shifted; window 0 filled with ``value``), last ``ratio`` slots = current window's
    second-half projection."""
    b = t.shape[0]
    prev, cur = t[..., :d], t[..., d:]                                   # [b,nc,ratio,d] each
    pad = mx.full((b, 1, ratio, d), value, dtype=t.dtype)
    prev_shift = mx.concatenate([pad, prev[:, :-1]], axis=1)            # compressed token c uses c-1
    return mx.concatenate([prev_shift, cur], axis=2)


def compressor_prefill(x: mx.array, ape: mx.array, norm_w: mx.array, wkv: mx.array, wgate: mx.array,
                       *, ratio: int, head_dim: int, rope_head_dim: int, eps: float,
                       cos: mx.array, sin: mx.array) -> mx.array | None:
    """Compress ``x`` ``[B,S,dim]`` -> compressed KV ``[B, S//ratio, head_dim]`` (or ``None`` if
    ``S < ratio``). ``cos``/``sin`` are the layer's RoPE tables ``[S, rope_head_dim/2]`` (the
    compressed token at index ``c`` is RoPE'd at absolute position ``c*ratio``)."""
    b, seqlen, _ = x.shape
    if seqlen < ratio:
        return None
    overlap = ratio == 4
    coff = 2 if overlap else 1
    xf = x.astype(mx.float32)
    kv = (xf @ wkv.T)                                                    # [B,S,coff*head_dim]
    score = (xf @ wgate.T)
    cutoff = seqlen - seqlen % ratio
    ncomp = cutoff // ratio
    kv = kv[:, :cutoff].reshape(b, ncomp, ratio, coff * head_dim)
    score = score[:, :cutoff].reshape(b, ncomp, ratio, coff * head_dim) + ape   # ape [ratio, coff*hd]
    if overlap:
        kv = _overlap_transform(kv, 0.0, ratio, head_dim)
        score = _overlap_transform(score, float("-inf"), ratio, head_dim)
    kv = mx.sum(kv * mx.softmax(score, axis=2), axis=2)                  # [B, ncomp, head_dim]
    kv = _rms_w(kv, norm_w, eps)
    return rope_partial(kv, cos[0:cutoff:ratio], sin[0:cutoff:ratio], rope_head_dim)
