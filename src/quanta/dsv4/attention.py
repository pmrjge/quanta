"""DeepSeek-V4 attention core (dense / pure sliding-window path) — MLX port of the reference.

This implements the ratio-0 attention regime (layers 0,1 and the MTP block): an MLA-style block with
a single shared latent KV head. The compressed-KV + Lightning-Indexer regimes (ratio 4/128) extend
this and live in :mod:`quanta.dsv4.compressor` / :mod:`quanta.dsv4.indexer` (tasks #71/#72).

Flow (faithful to ``model.py`` ``Attention.forward``):

* **q**: ``wq_a -> q_norm (RMSNorm, weighted, over q_lora_rank) -> wq_b`` reshaped to
  ``[B,T,n_heads,head_dim]``, then an **unweighted per-head RMSNorm** over ``head_dim``, then partial
  RoPE on the last ``rope_head_dim`` dims.
* **kv**: ``wkv -> kv_norm (RMSNorm, weighted, over head_dim)``, then partial RoPE on the last
  ``rope_head_dim`` dims. A *single* latent KV vector per position is shared by all query heads (MQA).
* **attention**: scaled dot-product over a **causal sliding window** (``sliding_window``) with a
  learned **per-head sink** added to the softmax denominator only (an always-present logit with a
  zero value). Then an **inverse RoPE** is applied to the attention output's rotated dims.
* **output**: grouped low-rank projection — reshape to ``o_groups`` groups, per-group
  ``[.,.,g,d] x wo_a[g] -> [.,.,g,o_lora_rank]``, flatten, then ``wo_b``.

RoPE uses the reference's interleaved-complex form with YaRN-corrected frequencies (YaRN is active
only on compressed layers; pure-SW layers pass ``original_seq_len=0`` -> base theta, no scaling).
The QAT activation fake-quant of the reference (``act_quant`` on the non-rope KV dims) is **omitted**
— the dequantized-weight bf16/f32 forward is the cleaner oracle (per project methodology). Gated
MLX-vs-numpy in ``parity/dsv4_attention_test.py``.
"""

from __future__ import annotations

import math

import mlx.core as mx

from quanta.dsv4.config import DeepSeekV4Config


# --- RoPE (interleaved complex, YaRN-corrected) ------------------------------
def rope_cos_sin(dim: int, seqlen: int, original_seq_len: int, base: float,
                 factor: float, beta_fast: float, beta_slow: float) -> tuple[mx.array, mx.array]:
    """Precompute ``(cos, sin)`` of shape ``[seqlen, dim/2]`` for partial RoPE (matches the reference
    ``precompute_freqs_cis``). YaRN frequency interpolation is applied iff ``original_seq_len > 0``."""
    freqs = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))   # [dim/2]
    if original_seq_len > 0:
        def corr_dim(num_rot):
            return dim * math.log(original_seq_len / (num_rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corr_dim(beta_fast)), 0)
        high = min(math.ceil(corr_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = mx.clip((mx.arange(dim // 2, dtype=mx.float32) - low) / (high - low), 0.0, 1.0)
        smooth = 1.0 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = mx.arange(seqlen, dtype=mx.float32)
    ang = t[:, None] * freqs[None, :]                                       # [seqlen, dim/2]
    return mx.cos(ang), mx.sin(ang)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array, inverse: bool) -> mx.array:
    """Rotate the last dim of ``x`` (size ``rd``) as ``rd/2`` interleaved complex pairs. ``x`` is
    ``[B,T,rd]`` or ``[B,T,H,rd]``; ``cos``/``sin`` are ``[T,rd/2]``."""
    *lead, rd = x.shape
    xr = x.reshape(*lead, rd // 2, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    if x.ndim == 4:                                # [B,T,H,rd] -> broadcast over heads
        c, s = cos[None, :, None, :], sin[None, :, None, :]
    else:                                          # [B,T,rd]
        c, s = cos[None, :, :], sin[None, :, :]
    if inverse:
        s = -s
    o0 = x0 * c - x1 * s
    o1 = x0 * s + x1 * c
    return mx.stack([o0, o1], axis=-1).reshape(*lead, rd)


def rope_partial(x: mx.array, cos: mx.array, sin: mx.array, rd: int, inverse: bool = False) -> mx.array:
    """Apply RoPE to only the last ``rd`` dims of ``x`` (the first ``head_dim-rd`` pass through)."""
    return mx.concatenate([x[..., :-rd], _apply_rope(x[..., -rd:], cos, sin, inverse)], axis=-1)


def _rms(x: mx.array, eps: float) -> mx.array:
    """Unweighted RMS normalization over the last dim (the reference's per-head q normalization)."""
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)


def _rms_w(x: mx.array, w: mx.array, eps: float) -> mx.array:
    """Weighted RMSNorm over the last dim (computed in float32, like the reference RMSNorm)."""
    xf = x.astype(mx.float32)
    xf = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (w.astype(mx.float32) * xf).astype(x.dtype)


# --- dense windowed attention with per-head sink -----------------------------
def sdpa_window_sink(q: mx.array, kv: mx.array, sink: mx.array, scale: float,
                     window: int, offset: int = 0) -> mx.array:
    """Causal sliding-window SDPA with a shared latent KV and a per-head sink (denominator-only).

    ``q``: ``[B,T,H,D]``; ``kv``: ``[B,S,D]`` (single head, broadcast to all query heads); ``sink``:
    ``[H]``. Query ``t`` (absolute position ``offset+t``) attends to keys ``s`` with
    ``offset+t-window < s <= offset+t``. Returns ``[B,T,H,D]``."""
    b, t, h, d = q.shape
    s = kv.shape[1]
    scores = mx.einsum("bthd,bsd->bths", q, kv) * scale          # [B,T,H,S]
    qi = (mx.arange(t) + offset)[:, None]                        # [T,1] absolute query pos
    ki = mx.arange(s)[None, :]                                   # [1,S] key pos
    allow = (ki <= qi) & (ki > qi - window)                      # [T,S]
    scores = scores + mx.where(allow, 0.0, -1e9)[None, :, None, :]
    m = mx.max(scores, axis=-1, keepdims=True)                   # [B,T,H,1]
    ex = mx.exp(scores - m)                                      # [B,T,H,S]
    denom = mx.sum(ex, axis=-1) + mx.exp(sink[None, None, :] - m[..., 0])   # [B,T,H]
    num = mx.einsum("bths,bsd->bthd", ex, kv)                    # [B,T,H,D]
    return num / denom[..., None]


def grouped_o(o: mx.array, wo_a: mx.array, wo_b: mx.array, n_groups: int, o_lora_rank: int) -> mx.array:
    """Grouped low-rank output projection. ``o``: ``[B,T,H,D]``; ``wo_a``: ``[g*o_lora_rank, H*D/g]``;
    ``wo_b``: ``[dim, g*o_lora_rank]``. Returns ``[B,T,dim]``."""
    b, t, h, d = o.shape
    og = o.reshape(b, t, n_groups, (h * d) // n_groups)         # [B,T,g, H*D/g]
    wa = wo_a.reshape(n_groups, o_lora_rank, -1)                # [g, o_lora_rank, H*D/g]
    proj = mx.einsum("btgd,grd->btgr", og, wa)                  # [B,T,g,o_lora_rank]
    proj = proj.reshape(b, t, n_groups * o_lora_rank)           # [B,T, g*o_lora_rank]
    return proj @ wo_b.T                                        # [B,T,dim]


def rope_tables(cfg: DeepSeekV4Config, layer_id: int, t: int, offset: int = 0) -> tuple[mx.array, mx.array]:
    """The layer's RoPE ``(cos, sin)`` for absolute positions ``[offset, offset+t)``."""
    rd = cfg.rope_head_dim
    orig, theta = cfg.attn_rope(layer_id)
    cos, sin = rope_cos_sin(rd, offset + t, orig, theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
    return cos[offset:offset + t], sin[offset:offset + t]


def project_qkv(x: mx.array, p: dict, cfg: DeepSeekV4Config, cos: mx.array, sin: mx.array
                ) -> tuple[mx.array, mx.array, mx.array]:
    """Shared q/kv projection: returns ``(qr, q, kv)`` — ``qr`` (post-q_norm low-rank, fed to the
    indexer), ``q`` ``[B,T,H,head_dim]`` and the latent ``kv`` ``[B,T,head_dim]`` (both partial-RoPE'd)."""
    b, t, _ = x.shape
    nh, hd, rd, eps = cfg.num_attention_heads, cfg.head_dim, cfg.rope_head_dim, cfg.norm_eps
    qr = _rms_w(x @ p["wq_a"].T, p["q_norm"], eps)              # [B,T,q_lora_rank]
    q = (qr @ p["wq_b"].T).reshape(b, t, nh, hd)               # [B,T,H,head_dim]
    q = _rms(q.astype(mx.float32), eps).astype(x.dtype)        # unweighted per-head RMS
    q = rope_partial(q, cos, sin, rd)
    kv = _rms_w(x @ p["wkv"].T, p["kv_norm"], eps)             # [B,T,head_dim]
    kv = rope_partial(kv, cos, sin, rd)
    return qr, q, kv


def output_proj(o: mx.array, p: dict, cfg: DeepSeekV4Config, cos: mx.array, sin: mx.array) -> mx.array:
    """Inverse-RoPE the attention output's rotated dims, then grouped low-rank O -> ``[B,T,dim]``."""
    o = rope_partial(o, cos, sin, cfg.rope_head_dim, inverse=True)
    return grouped_o(o, p["wo_a"], p["wo_b"], cfg.o_groups, cfg.o_lora_rank)


def attention_dense(x: mx.array, p: dict, cfg: DeepSeekV4Config, layer_id: int,
                    offset: int = 0) -> mx.array:
    """Pure sliding-window attention (ratio-0 layers). ``x``: ``[B,T,dim]``; ``p``: loader
    ``attention(layer_id)`` dict. Returns ``[B,T,dim]``."""
    cos, sin = rope_tables(cfg, layer_id, x.shape[1], offset)
    _, q, kv = project_qkv(x, p, cfg, cos, sin)
    o = sdpa_window_sink(q.astype(mx.float32), kv.astype(mx.float32),
                         p["attn_sink"].astype(mx.float32), cfg.attn_scale,
                         cfg.sliding_window, offset)
    return output_proj(o, p, cfg, cos, sin)
