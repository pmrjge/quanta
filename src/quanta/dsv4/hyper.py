"""Hyper-Connections (HC) for DeepSeek-V4 — the Sinkhorn-mixed multi-copy residual stream.

DSV4 replaces the usual single residual with ``hc_mult`` (=4) **parallel copies**. Each sub-block
(attention, FFN) is wrapped by a learned mix:

* :func:`hc_pre` reduces the ``hc`` copies ``[B,T,hc,d] -> [B,T,d]`` with per-token weights ``pre``,
  and also emits ``post`` (broadcast weights) and ``comb`` (a Sinkhorn-normalized ``[hc,hc]`` mixing
  matrix) for the recombination;
* the sub-block runs on the reduced ``[B,T,d]`` stream;
* :func:`hc_post` expands back to ``[B,T,hc,d]``: ``out[m] = post[m]*sublayer + Σ_j comb[j,m]*residual[j]``.

``pre``/``post``/``comb`` come from a single ``mixes = (rmsnorm-scaled x) @ hc_fn.T`` projection split
into three parts (``hc`` + ``hc`` + ``hc*hc`` = ``mix_hc``), passed through :func:`hc_split_sinkhorn`
(sigmoid for pre, ``2*sigmoid`` for post, softmax + ``sinkhorn_iters`` row/col normalizations for the
doubly-stochastic-ish ``comb``). The final logit head uses a simpler :func:`hc_head` reduction.

Faithful MLX port of the reference ``model.py`` (``Block.hc_pre/hc_post``, ``ParallelHead.hc_head``)
and ``kernel.py`` (``hc_split_sinkhorn_kernel``). All maths in float32 (as the reference does); the
``sinkhorn_iters`` loop is a bounded per-call iteration over ``[B,T,hc,hc]`` (vectorized across
tokens — not a token/expert/dim loop). Gated MLX-vs-numpy in ``parity/dsv4_hyper_test.py``.
"""

from __future__ import annotations

import mlx.core as mx


def hc_expand(h: mx.array, hc_mult: int) -> mx.array:
    """Embed output ``[B,T,d]`` -> initial HC residual ``[B,T,hc,d]`` (broadcast copy)."""
    return mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], hc_mult, h.shape[-1]))


def hc_split_sinkhorn(mixes: mx.array, hc_scale: mx.array, hc_base: mx.array,
                      hc_mult: int, iters: int, eps: float) -> tuple[mx.array, mx.array, mx.array]:
    """Split the ``[...,mix_hc]`` mix vector into ``pre`` ``[...,hc]``, ``post`` ``[...,hc]`` and the
    Sinkhorn-normalized combine ``comb`` ``[...,hc,hc]`` (``comb[...,j,k]``)."""
    hc = hc_mult
    pre = mx.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * mx.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
    comb = comb.reshape(*mixes.shape[:-1], hc, hc)
    # comb = softmax over rows (-1) + eps, then one column normalization (-2)
    comb = mx.softmax(comb, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):                         # bounded Sinkhorn iterations (vectorized)
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _mixes(x_flat: mx.array, hc_fn: mx.array, norm_eps: float) -> mx.array:
    """``(rmsnorm-scale of flattened HC copies) applied to a linear projection`` -> ``[...,mix_hc]``."""
    rsqrt = mx.rsqrt(mx.mean(x_flat * x_flat, axis=-1, keepdims=True) + norm_eps)
    return (x_flat @ hc_fn.T) * rsqrt


def hc_pre(x: mx.array, hc_fn: mx.array, hc_scale: mx.array, hc_base: mx.array,
           hc_mult: int, iters: int, norm_eps: float, hc_eps: float
           ) -> tuple[mx.array, mx.array, mx.array]:
    """Reduce HC copies ``[B,T,hc,d] -> [B,T,d]`` (float32) and return ``(reduced, post, comb)``."""
    shape = x.shape
    xf = x.reshape(*shape[:2], -1).astype(mx.float32)
    mixes = _mixes(xf, hc_fn.astype(mx.float32), norm_eps)
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale.astype(mx.float32),
                                        hc_base.astype(mx.float32), hc_mult, iters, hc_eps)
    reduced = mx.sum(pre[..., None] * xf.reshape(shape), axis=2)
    return reduced, post, comb


def hc_post(sublayer: mx.array, residual: mx.array, post: mx.array, comb: mx.array) -> mx.array:
    """Expand ``[B,T,d]`` sublayer output back to ``[B,T,hc,d]``:
    ``out[...,m,:] = post[...,m]*sublayer + Σ_j comb[...,j,m]*residual[...,j,:]``."""
    term1 = post[..., None] * sublayer[..., None, :]                    # [B,T,hc,d]
    term2 = mx.matmul(comb.swapaxes(-2, -1), residual.astype(comb.dtype))  # Σ_j comb[j,m]·res[j]
    return term1 + term2


def hc_head(x: mx.array, hc_fn: mx.array, hc_scale: mx.array, hc_base: mx.array,
            hc_mult: int, norm_eps: float, hc_eps: float) -> mx.array:
    """Final/MTP HC reduction ``[B,T,hc,d] -> [B,T,d]`` (sigmoid weights, no Sinkhorn)."""
    shape = x.shape
    xf = x.reshape(*shape[:2], -1).astype(mx.float32)
    mixes = _mixes(xf, hc_fn.astype(mx.float32), norm_eps)
    pre = mx.sigmoid(mixes * hc_scale.astype(mx.float32) + hc_base.astype(mx.float32)) + hc_eps
    return mx.sum(pre[..., None] * xf.reshape(shape), axis=2)
