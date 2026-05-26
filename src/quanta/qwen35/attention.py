"""Gated GQA full-attention for Qwen3.5 (the 15 ``full_attention`` layers), MLX-native.

Qwen3-Next-style gated grouped-query attention with **partial mRoPE** + per-head QK-norm:

* ``q_proj`` 4096 -> 16384 = 32 heads × 256 × **2** (``attn_output_gate``): reshape to
  ``[B,T,32,2*256]`` and split the per-head last dim into the **query** (first 256) and a fused
  **per-head output gate** (last 256). The gate is ``sigmoid(gate_half)`` and multiplies the
  attention output (per head, per dim) **before** ``o_proj``.
* ``k_proj``/``v_proj`` 4096 -> 512 = 2 KV heads × 256; GQA repeat ``n_rep=16``.
* per-head **RMSNorm** ``q_norm``/``k_norm`` (over the 256 head dim) applied **before** RoPE.
* **partial RoPE**: only the first ``rotary_dim=64`` of each 256-dim head is rotated (the rest pass
  through), ``rope_theta=1e7``, with **dynamic YaRN** frequency interpolation scaled by
  :meth:`Qwen35Config.effective_yarn_factor` for the sequence length. mRoPE sections ``[11,11,10]``
  are the multimodal temporal/height/width split; for **text** (1D positions, the parity/PPL path)
  all three sections share the token position, so it collapses to standard 1D RoPE on the rotated
  dims — the grounded text-decoder form (the true interleaved-2D mRoPE matters only for image/video
  position ids, a deferred vision stage). ``o_proj`` 8192 -> 4096.

Two equivalent paths (gated in ``parity/qwen35_forward_test.py``):

* fast (default): ``mx.fast.rope(freqs=...)`` + ``mx.fast.scaled_dot_product_attention`` (tiled,
  never materializes a T×T score matrix — memory-safe at the 1M context).
* naive: explicit interleaved RoPE + manual softmax — the short-sequence parity reference.

Incremental decode (KV cache) is gated == prefill: the same q/k/v/gate per position regardless of
chunking.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from quanta.cache_quant import dequantize_last_axis, quantize_last_axis
from quanta.qwen35.config import Qwen35Config


class KVCache:
    """Plain GQA KV cache: stores ``[B, n_kv, S, head_dim]`` k/v, grows along the seq axis (axis=2).

    Used by the 15 full-attention layers of the hybrid (linear-attention layers carry an O(1)
    recurrent state instead — see :class:`quanta.qwen35.decode._GDNLayerState`). Two storage modes:

    * ``quantized=False`` (default): k/v held as **bf16** ``[B, n_kv, S, head_dim]`` — the
      historical path every existing prefill / decode parity gate runs against.
    * ``quantized=True``: k/v held as **affine int8** per-token, per-group over ``head_dim`` via
      :mod:`quanta.cache_quant`. Storage is codes ``[B, n_kv, S, head_dim/4]`` + scales/biases
      ``[B, n_kv, S, head_dim/group_size]`` ≈ 8.25 bpp (vs bf16's 16 bpp). ``update`` dequantizes
      the full cache for the SDPA return so the attention path is unchanged. The win is
      steady-state memory at 1M context (#114) — 15 full-attn layers × 2 KV × 256 head_dim = 8 KB/
      token; int8 halves that so 1M-token serving is achievable under the 490 GiB ceiling.
    """

    def __init__(self, *, quantized: bool = False, group_size: int = 64) -> None:
        self.quantized = quantized
        self.group_size = group_size
        # bf16 mode
        self.k: mx.array | None = None
        self.v: mx.array | None = None
        # int8 mode (codes + per-group scales/biases)
        self.k_q: mx.array | None = None
        self.k_s: mx.array | None = None
        self.k_b: mx.array | None = None
        self.v_q: mx.array | None = None
        self.v_s: mx.array | None = None
        self.v_b: mx.array | None = None

    @property
    def offset(self) -> int:
        if self.quantized:
            return 0 if self.k_q is None else self.k_q.shape[2]
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        if not self.quantized:
            if self.k is None:
                self.k, self.v = k, v
            else:
                self.k = mx.concatenate([self.k, k], axis=2)
                self.v = mx.concatenate([self.v, v], axis=2)
            return self.k, self.v
        # int8 path
        k_qn, k_sn, k_bn = quantize_last_axis(k, self.group_size)
        v_qn, v_sn, v_bn = quantize_last_axis(v, self.group_size)
        if self.k_q is None:
            self.k_q, self.k_s, self.k_b = k_qn, k_sn, k_bn
            self.v_q, self.v_s, self.v_b = v_qn, v_sn, v_bn
        else:
            self.k_q = mx.concatenate([self.k_q, k_qn], axis=2)
            self.k_s = mx.concatenate([self.k_s, k_sn], axis=2)
            self.k_b = mx.concatenate([self.k_b, k_bn], axis=2)
            self.v_q = mx.concatenate([self.v_q, v_qn], axis=2)
            self.v_s = mx.concatenate([self.v_s, v_sn], axis=2)
            self.v_b = mx.concatenate([self.v_b, v_bn], axis=2)
        k_full = dequantize_last_axis(self.k_q, self.k_s, self.k_b, self.group_size, dtype=k.dtype)
        v_full = dequantize_last_axis(self.v_q, self.v_s, self.v_b, self.group_size, dtype=v.dtype)
        return k_full, v_full


def _rms(x: mx.array, w: mx.array, eps: float) -> mx.array:
    """Weighted RMSNorm over the last dim, computed in fp32 (per-head q/k norm)."""
    xf = x.astype(mx.float32)
    xf = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (w.astype(mx.float32) * xf).astype(x.dtype)


def yarn_inv_freq(cfg: Qwen35Config, seq_len: int) -> mx.array:
    """YaRN-corrected per-pair inverse frequencies for the rotated dims, shape ``[rotary_dim//2]``.

    Dynamic YaRN: the scaling ``factor`` comes from :meth:`Qwen35Config.effective_yarn_factor`
    (1.0 when the sequence fits the native window ⇒ plain RoPE, no rescale). Construction mirrors
    the HF/DeepSeek YaRN ramp used elsewhere in quanta (``modeling.rope.build_yarn_inv_freq``).
    """
    dim = cfg.rotary_dim
    base = cfg.rope_theta
    factor = cfg.effective_yarn_factor(seq_len)
    idx = mx.arange(0, dim, 2, dtype=mx.float32)
    freq_extra = 1.0 / (base ** (idx / dim))                  # un-scaled (extrapolation)
    if factor <= 1.0:
        return freq_extra
    freq_inter = freq_extra / factor                          # scaled (interpolation)
    orig_max = cfg.yarn_original_max
    beta_fast, beta_slow = 32.0, 1.0                          # YaRN ramp bounds (HF defaults)

    def corr(num_rot: float) -> float:
        return dim * math.log(orig_max / (num_rot * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(corr(beta_fast)), 0)
    high = min(math.ceil(corr(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = mx.clip((mx.arange(dim // 2, dtype=mx.float32) - low) / (high - low), 0.0, 1.0)
    mask = 1.0 - ramp                                         # 1 on low (extrapolated) dims
    return freq_inter * (1.0 - mask) + freq_extra * mask


def _rope_fast(x: mx.array, inv_freq: mx.array, rd: int, offset: int) -> mx.array:
    """``mx.fast.rope`` over the first ``rd`` dims (interleaved), the rest pass through.

    mlx ``freqs`` is the *period* (angle = pos/freqs), so pass ``1/inv_freq``. ``traditional=True``
    rotates consecutive pairs — dot-product-equivalent to the explicit de-interleaved form."""
    return mx.fast.rope(x, dims=rd, traditional=True, base=None, scale=1.0, offset=offset,
                        freqs=1.0 / inv_freq)


def _rope_explicit(x: mx.array, inv_freq: mx.array, rd: int, offset: int) -> mx.array:
    """Explicit interleaved RoPE on the first ``rd`` dims (the rest pass through). ``x`` ``[B,H,T,D]``.

    Reference for :func:`_rope_fast`: rotate consecutive pairs ``(x0,x1)`` by the position angle.
    """
    b, h, t, d = x.shape
    pos = (mx.arange(t, dtype=mx.float32) + offset)[:, None]           # [T,1]
    ang = pos * inv_freq[None, :]                                      # [T, rd/2]
    cos = mx.cos(ang)[None, None]                                      # [1,1,T,rd/2]
    sin = mx.sin(ang)[None, None]
    xr = x[..., :rd].reshape(b, h, t, rd // 2, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    rot = mx.stack([o0, o1], axis=-1).reshape(b, h, t, rd)
    return mx.concatenate([rot, x[..., rd:]], axis=-1)


def _causal_mask(q_len: int, kv_len: int, dtype: mx.Dtype) -> mx.array:
    """Lower-right causal additive mask (query j at abs pos kv_len-q_len+j)."""
    off = kv_len - q_len
    j = mx.arange(q_len)[:, None]
    i = mx.arange(kv_len)[None, :]
    return mx.where(i <= j + off, mx.array(0.0, dtype), mx.array(float("-inf"), dtype))


class Qwen35Attention(nn.Module):
    """Gated GQA + partial-mRoPE + per-head QK-norm full-attention layer."""

    def __init__(self, cfg: Qwen35Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.nh = cfg.num_attention_heads          # 32
        self.nkv = cfg.num_key_value_heads         # 2
        self.hd = cfg.head_dim                     # 256
        self.rep = cfg.n_rep                       # 16
        self.rd = cfg.rotary_dim                   # 64
        self.scale = cfg.attn_scale
        self.eps = cfg.norm_eps
        self.gated = cfg.attn_output_gate
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.q_proj_out, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.q_dim, cfg.hidden_size, bias=False)
        self.q_norm = mx.ones((self.hd,))
        self.k_norm = mx.ones((self.hd,))

    def _project(self, x):
        """q,k,v -> [B,H,T,D]; plus the per-head sigmoid output gate (or None)."""
        b, t, _ = x.shape
        if self.gated:
            qg = self.q_proj(x).reshape(b, t, self.nh, 2 * self.hd)
            q = qg[..., : self.hd]                              # [B,T,H,D]
            gate = mx.sigmoid(qg[..., self.hd:])                # [B,T,H,D]
        else:
            q = self.q_proj(x).reshape(b, t, self.nh, self.hd)
            gate = None
        k = self.k_proj(x).reshape(b, t, self.nkv, self.hd)
        v = self.v_proj(x).reshape(b, t, self.nkv, self.hd)
        q = _rms(q, self.q_norm, self.eps)                     # per-head QK-norm BEFORE RoPE
        k = _rms(k, self.k_norm, self.eps)
        q = mx.transpose(q, (0, 2, 1, 3))                      # [B,H,T,D]
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        return q, k, v, gate

    def __call__(self, x, *, cache=None, use_fast=True, seq_hint=None):
        b, t, _ = x.shape
        offset = cache.offset if cache is not None else 0
        q, k, v, gate = self._project(x)
        # YaRN factor depends on the full sequence length, not the chunk; seq_hint lets the model
        # pass the total so chunked prefill == single-shot (else use the cache-aware length).
        seq_len = seq_hint if seq_hint is not None else (offset + t)
        inv_freq = yarn_inv_freq(self.cfg, seq_len)
        rope = _rope_fast if use_fast else _rope_explicit
        q = rope(q, inv_freq, self.rd, offset)
        k = rope(k, inv_freq, self.rd, offset)
        if cache is not None:
            k, v = cache.update(k, v)
        kv_len = k.shape[2]
        kr = mx.repeat(k, self.rep, axis=1)                    # GQA: kv head -> its query group
        vr = mx.repeat(v, self.rep, axis=1)
        if use_fast:
            mask = "causal" if t > 1 else None
            out = mx.fast.scaled_dot_product_attention(q, kr, vr, scale=self.scale, mask=mask)
        else:
            scores = (q @ mx.swapaxes(kr, -1, -2)) * self.scale + _causal_mask(t, kv_len, q.dtype)
            w = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
            out = w @ vr
        out = mx.transpose(out, (0, 2, 1, 3))                  # [B,T,H,D]
        if gate is not None:
            out = out * gate.astype(out.dtype)                 # fused per-head output gate
        out = out.reshape(b, t, self.nh * self.hd)
        return self.o_proj(out)
