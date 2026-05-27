"""Gated GQA attention for Qwen2.5-14B-Instruct-1M, MLX-native.

Qwen2 (vs Qwen3 / Qwen3-Next) specifics — every divergence from the Qwen3.5 attention path:

* **QKV biases ON.** ``q_proj``/``k_proj``/``v_proj`` are ``nn.Linear(bias=True)``; ``o_proj`` is
  ``bias=False`` (confirmed empirically — only q/k/v keys carry ``.bias`` in the safetensors index).
* **No QK-norm.** Qwen3 added per-head RMSNorm; Qwen2 does not.
* **No output gate.** Qwen3-Next added the fused per-head sigmoid; Qwen2's q_proj is plain ``q_dim``.
* **Full RoPE.** Qwen2.5 rotates the full ``head_dim``; no partial-rotary, no mRoPE sections, no YaRN
  scaling. ``rope_theta=1e7`` is *already* the long-context base — no further rescale.

Two equivalent paths (the standard quanta gate):

* fast (default): ``mx.fast.rope`` + ``mx.fast.scaled_dot_product_attention`` (tiled — never
  materializes a T×T score matrix, memory-safe at the 1M context).
* naive: explicit RoPE + manual softmax — the short-sequence parity reference.

**Dual Chunk Attention (DCA) — Qwen2.5-1M's >256K long-context method:** stubbed here. Below
``cfg.dca_original_max`` (262144) plain RoPE attention is exact, and ``rope_theta=1e7`` gives the
model meaningful headroom past 256K via plain extrapolation (degrades gracefully). True DCA — which
splits the sequence into chunks of ``cfg.dca_chunk_size`` with a ``cfg.dca_local_size`` overlap and
remaps cross-chunk positions to live within the trained window — is a follow-up task; the current
attention forward will warn (and still run plain RoPE) when ``seq_len > cfg.dca_original_max``.
"""

from __future__ import annotations

import warnings

import mlx.core as mx
import mlx.nn as nn

from quanta.cache_quant import dequantize_last_axis, quantize_last_axis
from quanta.qwen25.config import Qwen25Config


class KVCache:
    """Plain GQA KV cache: ``[B, n_kv, S, head_dim]`` k/v growing along the seq axis.

    Two storage modes (mirror :class:`quanta.qwen35.attention.KVCache`):

    * ``quantized=False`` (default): bf16 verbatim — parity reference / short-context decode path.
    * ``quantized=True``: per-token, per-group affine int8 over ``head_dim`` via
      :mod:`quanta.cache_quant`. ``update`` dequantizes the full cache for the SDPA return so the
      attention path is unchanged. Cuts steady-state cache memory ~half — at 1M context this is
      the dominant memory cost (48 layers × 8 KV × 128 head_dim × 2 = 192 KB/token in bf16; ~96 KB
      in int8).
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


def _rope_fast(x: mx.array, base: float, offset: int) -> mx.array:
    """Full RoPE on the last dim via ``mx.fast.rope`` (traditional, interleaved pairs)."""
    return mx.fast.rope(x, dims=x.shape[-1], traditional=True, base=base, scale=1.0, offset=offset)


def _rope_explicit(x: mx.array, base: float, offset: int) -> mx.array:
    """Explicit RoPE on the full last dim (parity reference for :func:`_rope_fast`).

    ``x`` ``[B, H, T, D]`` with even ``D``; rotates consecutive pairs ``(x0, x1)`` by the position
    angle. ``inv_freq = 1 / base ** (idx / D)`` with ``idx`` even-indexed (0, 2, 4, …, D-2)."""
    b, h, t, d = x.shape
    idx = mx.arange(0, d, 2, dtype=mx.float32)
    inv_freq = 1.0 / (base ** (idx / d))                                # [D/2]
    pos = (mx.arange(t, dtype=mx.float32) + offset)[:, None]            # [T, 1]
    ang = pos * inv_freq[None, :]                                       # [T, D/2]
    cos = mx.cos(ang)[None, None]                                       # [1, 1, T, D/2]
    sin = mx.sin(ang)[None, None]
    xr = x.reshape(b, h, t, d // 2, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return mx.stack([o0, o1], axis=-1).reshape(b, h, t, d)


def _causal_mask(q_len: int, kv_len: int, dtype: mx.Dtype) -> mx.array:
    """Lower-right causal additive mask (query j at abs pos kv_len-q_len+j)."""
    off = kv_len - q_len
    j = mx.arange(q_len)[:, None]
    i = mx.arange(kv_len)[None, :]
    return mx.where(i <= j + off, mx.array(0.0, dtype), mx.array(float("-inf"), dtype))


class Qwen25Attention(nn.Module):
    """GQA + QKV biases + full RoPE attention layer for Qwen2.5-14B-Instruct-1M."""

    def __init__(self, cfg: Qwen25Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.nh = cfg.num_attention_heads          # 40
        self.nkv = cfg.num_key_value_heads         # 8
        self.hd = cfg.head_dim                     # 128
        self.rep = cfg.n_rep                       # 5
        self.scale = cfg.attn_scale
        self.theta = cfg.rope_theta                # 1e7
        # Qwen2 q/k/v have biases (o does not). attention_bias defaults True in the config.
        bias = bool(cfg.attention_bias)
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.q_dim, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=bias)
        self.o_proj = nn.Linear(cfg.q_dim, cfg.hidden_size, bias=False)

    def _project(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """q,k,v -> [B, H, T, D] / [B, n_kv, T, D]."""
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.nh, self.hd)
        k = self.k_proj(x).reshape(b, t, self.nkv, self.hd)
        v = self.v_proj(x).reshape(b, t, self.nkv, self.hd)
        q = mx.transpose(q, (0, 2, 1, 3))                       # [B, H, T, D]
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        return q, k, v

    def __call__(self, x: mx.array, *, cache: KVCache | None = None, use_fast: bool = True,
                 seq_hint: int | None = None) -> mx.array:
        b, t, _ = x.shape
        offset = cache.offset if cache is not None else 0
        seq_len = seq_hint if seq_hint is not None else (offset + t)
        if self.cfg.use_dca and seq_len > self.cfg.dca_original_max:
            # Plain extrapolation past the trained DCA window. The model still produces sensible
            # output (rope_theta=1e7 has meaningful headroom), but DCA is the proper long-context
            # method — track follow-up via the task tracker.
            warnings.warn(
                f"seq_len={seq_len} > dca_original_max={self.cfg.dca_original_max}: DCA not yet "
                f"implemented; falling back to plain RoPE extrapolation (degrades past ~512K).",
                stacklevel=2,
            )

        q, k, v = self._project(x)
        q = (_rope_fast if use_fast else _rope_explicit)(q, self.theta, offset)
        k = (_rope_fast if use_fast else _rope_explicit)(k, self.theta, offset)
        if cache is not None:
            k, v = cache.update(k, v)
        kv_len = k.shape[2]
        kr = mx.repeat(k, self.rep, axis=1)                     # GQA: kv head -> its query group
        vr = mx.repeat(v, self.rep, axis=1)
        if use_fast:
            mask = "causal" if t > 1 else None
            out = mx.fast.scaled_dot_product_attention(q, kr, vr, scale=self.scale, mask=mask)
        else:
            scores = (q @ mx.swapaxes(kr, -1, -2)) * self.scale + _causal_mask(t, kv_len, q.dtype)
            w = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
            out = w @ vr
        out = mx.transpose(out, (0, 2, 1, 3))                   # [B, T, H, D]
        out = out.reshape(b, t, self.nh * self.hd)
        return self.o_proj(out)
