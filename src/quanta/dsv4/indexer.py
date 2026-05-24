"""DeepSeek-V4 Lightning Indexer (DSA) + compressed-layer attention — MLX port.

Compressed layers attend to a causal **sliding window** of per-token KV **plus** a set of *compressed*
KV tokens (built by :mod:`quanta.dsv4.compressor`). Which compressed tokens depends on the layer:

* ratio 128 (no indexer): every causally-valid compressed token (token ``c`` for query ``i`` iff
  ``c < (i+1)//ratio``).
* ratio 4 (Lightning Indexer): the **top-``index_topk``** compressed tokens by a learned score
  ``Σ_h relu(q_idx[h]·kv_idx) · weight[h]`` (own low-rank q + own rotated compressor + per-head
  weights). When ``ncomp <= index_topk`` this is just "all causal" — the top-k only bites at long
  context.

The Hadamard rotate + fp4 fake-quant of the reference's indexer cancel for selection (orthonormal H,
argsort-invariant scale) and are skipped here (and in the oracle), matching the model's selection.

:func:`attention_compressed` computes the full compressed-layer attention as a dense masked softmax
with per-head sink over ``[window per-token KV | selected compressed KV]`` — output-equivalent to the
reference's gathered ``sparse_attn``. (The gather-based sparse runtime for decode is task #77.) Gated
vs the authors' real ``Attention`` in ``parity/dsv4_indexer_test.py``.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.dsv4.attention import output_proj, project_qkv, rope_partial, rope_tables
from quanta.dsv4.compressor import compressor_prefill
from quanta.dsv4.config import DeepSeekV4Config

_NEG = -1e30


def indexer_index_score(x: mx.array, qr: mx.array, idx_p: dict, cfg: DeepSeekV4Config,
                        cos: mx.array, sin: mx.array) -> tuple[mx.array, int]:
    """Per-query score over compressed tokens, causally masked: ``[B,S,ncomp]`` (and ``ncomp``).
    ``score[i,c] = Σ_h relu(q_idx[i,h]·kv_idx[c]) · weight[i,h]`` for causal ``c``, else ``-inf``."""
    b, s, _ = x.shape
    inh, ihd, rd, eps = cfg.index_n_heads, cfg.index_head_dim, cfg.rope_head_dim, cfg.norm_eps
    qb = (qr @ idx_p["wq_b"].T).reshape(b, s, inh, ihd)
    qb = rope_partial(qb, cos, sin, rd).astype(mx.float32)                 # rope last rd of ihd
    icp = idx_p["compressor"]
    ikv = compressor_prefill(x, icp["ape"], icp["norm"], icp["wkv"], icp["wgate"],
                             ratio=4, head_dim=ihd, rope_head_dim=rd, eps=eps, cos=cos, sin=sin)
    ncomp = ikv.shape[1]
    weights = (x @ idx_p["weights_proj"].T).astype(mx.float32) * (ihd ** -0.5 * inh ** -0.5)  # [B,S,inh]
    score = mx.einsum("bshd,btd->bsht", qb, ikv.astype(mx.float32))        # [B,S,inh,ncomp]
    score = (mx.maximum(score, 0.0) * weights[..., None]).sum(axis=2)      # [B,S,ncomp]
    i = mx.arange(s)[:, None]
    c = mx.arange(ncomp)[None, :]
    causal = c < ((i + 1) // 4)                                           # [S,ncomp]
    return mx.where(causal[None], score, _NEG), ncomp


def indexer_select(x: mx.array, qr: mx.array, idx_p: dict, cfg: DeepSeekV4Config,
                   cos: mx.array, sin: mx.array) -> tuple[mx.array, int]:
    """Boolean ``[B,S,ncomp]`` mask of attended compressed tokens (causal ∧ top-``index_topk``)."""
    score, ncomp = indexer_index_score(x, qr, idx_p, cfg, cos, sin)
    k = min(cfg.index_topk, ncomp)
    causal = score > _NEG
    if k >= ncomp:
        return causal, ncomp
    thr = mx.sort(score, axis=-1)[..., ncomp - k][..., None]              # k-th largest per query
    return (score >= thr) & causal, ncomp


def attention_compressed(x: mx.array, p: dict, cfg: DeepSeekV4Config, layer_id: int,
                         offset: int = 0) -> mx.array:
    """Compressed-layer attention (ratio 4/128): window per-token KV + selected compressed KV, with a
    per-head sink. Dense-mask form (output-equivalent to the reference gathered sparse_attn)."""
    b, s, _ = x.shape
    ratio = cfg.compress_ratio(layer_id)
    cos, sin = rope_tables(cfg, layer_id, s, offset)
    qr, q, kv = project_qkv(x, p, cfg, cos, sin)
    cp = p["compressor"]
    ckv = compressor_prefill(x, cp["ape"], cp["norm"], cp["wkv"], cp["wgate"], ratio=ratio,
                             head_dim=cfg.head_dim, rope_head_dim=cfg.rope_head_dim,
                             eps=cfg.norm_eps, cos=cos, sin=sin)
    qf, kvf = q.astype(mx.float32), kv.astype(mx.float32)
    sink, scale = p["attn_sink"].astype(mx.float32), cfg.attn_scale

    sc = mx.einsum("bshd,btd->bsht", qf, kvf) * scale                     # window scores [B,S,H,S]
    qi = (mx.arange(s) + offset)[:, None]
    ki = mx.arange(s)[None, :]
    win = (ki <= qi) & (ki > qi - cfg.sliding_window)
    sc = sc + mx.where(win, 0.0, _NEG)[None, :, None, :]
    kv_all = kvf

    if ckv is not None:
        ncomp = ckv.shape[1]
        if cfg.has_indexer(layer_id):
            sel, _ = indexer_select(x, qr, p["indexer"], cfg, cos, sin)
        else:
            c = mx.arange(ncomp)[None, :]
            sel = (c < ((mx.arange(s)[:, None] + 1) // ratio))[None]      # [1,S,ncomp] causal
        sc_c = mx.einsum("bshd,btd->bsht", qf, ckv.astype(mx.float32)) * scale
        sc_c = sc_c + mx.where(sel, 0.0, _NEG)[:, :, None, :]
        sc = mx.concatenate([sc, sc_c], axis=-1)                          # [B,S,H,S+ncomp]
        kv_all = mx.concatenate([kvf, ckv.astype(mx.float32)], axis=1)

    m = mx.max(sc, axis=-1, keepdims=True)
    ex = mx.exp(sc - m)
    denom = mx.sum(ex, axis=-1) + mx.exp(sink[None, None, :] - m[..., 0])
    o = mx.einsum("bsht,btd->bshd", ex, kv_all) / denom[..., None]
    return output_proj(o, p, cfg, cos, sin)
