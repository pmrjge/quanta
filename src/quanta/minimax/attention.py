"""GQA attention for MiniMax-M2.7 (all 62 layers are full softmax), MLX-native.

Grouped-query attention with **partial RoPE** and **per-layer QK-norm** — the M2 attention block
(``attn_type_list`` is all-1s, so every layer is dense softmax; M2 is *not* the lightning/linear
hybrid M1 was). 48 query heads / 8 KV heads (``n_rep=6``), ``head_dim=128``.

Per-head pipeline (faithful to HF ``MiniMaxM2Attention``):

* project ``q`` ``[B,T,H,head_dim]`` / ``k``,``v`` ``[B,T,H_kv,head_dim]``;
* **QK-norm**: a *weighted* RMSNorm over the full ``head_dim`` (``q_norm``/``k_norm``,
  ``use_qk_norm``/``qk_norm_type="per_layer"``) applied to q and k **before** RoPE;
* **partial RoPE**: only the first ``rotary_dim=64`` of each 128-dim head is rotated
  (``theta=5e6``), the trailing ``head_dim-rotary_dim`` dims pass through unrotated;
* GQA repeat KV heads to query-head count, scaled dot-product attention (causal), output proj.

Two equivalent paths (gated in :mod:`parity.minimax_forward_test`):

* fast (default): ``mx.fast.rope`` on the rotated slice + ``mx.fast.scaled_dot_product_attention``
  (``mask="causal"``); the flash/tiled path that never materializes a TxT score matrix.
* naive: explicit ``rotate_half`` RoPE (HF NeoX / non-interleaved form) + manual softmax — the
  short-sequence parity reference only.

``mx.fast.rope(traditional=False)`` on the leading ``rotary_dim`` dims is bit-equivalent to HF's
``rotate_half`` over that slice (verified), so the two paths agree to fp tolerance. KV heads are
repeated to the query-head count before attention so the result is independent of the SDPA kernel's
GQA support. Only one layer's weights are resident at a time (rule 8); the heavy real-weight forward
is deferred to a GPU session.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.cache_quant import dequantize_last_axis, quantize_last_axis
from quanta.minimax.config import MiniMaxConfig


class KVCache:
    """Plain GQA KV cache: stores ``[B, n_kv, S, head_dim]`` k/v, grows along the seq axis (axis=2).

    Two storage modes:

    * ``quantized=False`` (default, historical): k/v held as **bf16** ``[B, n_kv, S, head_dim]`` —
      exact, the path every existing prefill / decode parity gate runs against.
    * ``quantized=True``: k/v held as **affine int8** per-token, per-group over ``head_dim`` —
      ``mx.quantize``/``mx.dequantize`` via :mod:`quanta.cache_quant`. Storage is codes
      ``[B, n_kv, S, head_dim/4]`` + scales/biases ``[B, n_kv, S, head_dim/group_size]`` ≈ 8.25
      bpp (vs bf16's 16 bpp); ``update`` dequantizes the full cache for the return so SDPA-time
      compute is unchanged. The win is **steady-state memory** at long context — MiniMax's per-token
      KV is 4 KB / token / layer (8 KV × 128 head_dim × 2) × 62 layers ≈ 248 GB at 1M ctx in bf16;
      int8 halves storage so 1M is achievable under the 490 GiB ceiling.

    Rule 6 (no silent wrong-length state): every quantized field is sliced together by ``truncate``
    in :class:`quanta.minimax.decode._LayerKVCache`, so rollback is lossless on either mode.
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
        # int8 path: quantize new tokens (per-token rows along the last axis), append along S (axis=2),
        # then dequantize the full cache for the SDPA return so the caller's path is unchanged.
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


def _causal_mask(q_len: int, kv_len: int, dtype: mx.Dtype) -> mx.array:
    """Lower-right causal additive mask (query j at abs pos kv_len-q_len+j)."""
    off = kv_len - q_len
    j = mx.arange(q_len)[:, None]
    i = mx.arange(kv_len)[None, :]
    return mx.where(i <= j + off, mx.array(0.0, dtype), mx.array(float("-inf"), dtype))


def _rotate_half(x: mx.array) -> mx.array:
    """HF NeoX ``rotate_half``: split the last dim in two halves ``[a, b] -> [-b, a]``."""
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    return mx.concatenate([-b, a], axis=-1)


def _rope_naive(x: mx.array, rotary_dim: int, base: float, offset: int) -> mx.array:
    """Explicit partial RoPE (HF non-interleaved form) on the **first** ``rotary_dim`` dims of ``x``.

    ``x``: ``[B, H, T, head_dim]``. The leading ``rotary_dim`` dims are rotated by NeoX
    ``rotate_half``; the trailing ``head_dim - rotary_dim`` dims pass through unrotated. Computed in
    float32 then cast back (matches ``mx.fast.rope``'s internal precision)."""
    t = x.shape[2]
    xr = x[..., :rotary_dim].astype(mx.float32)
    xp = x[..., rotary_dim:]
    inv = 1.0 / (base ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim))  # [rd/2]
    pos = (mx.arange(t, dtype=mx.float32) + offset)
    ang = pos[:, None] * inv[None, :]                                # [T, rd/2]
    cos = mx.concatenate([mx.cos(ang), mx.cos(ang)], axis=-1)[None, None]  # [1,1,T,rd]
    sin = mx.concatenate([mx.sin(ang), mx.sin(ang)], axis=-1)[None, None]
    out = (xr * cos + _rotate_half(xr) * sin).astype(x.dtype)
    return mx.concatenate([out, xp], axis=-1)


def _rope_fast(x: mx.array, rotary_dim: int, base: float, offset: int) -> mx.array:
    """``mx.fast.rope`` partial RoPE: rope the leading ``rotary_dim`` dims, pass the rest through."""
    xr = mx.fast.rope(x[..., :rotary_dim], dims=rotary_dim, traditional=False,
                      base=base, scale=1.0, offset=offset)
    return mx.concatenate([xr, x[..., rotary_dim:]], axis=-1)


class MiniMaxAttention(nn.Module):
    """GQA + partial RoPE + per-layer weighted QK-norm. One ``layer_id`` may select a non-full
    attention type in a future hybrid variant; here every layer is full softmax, but the guard keeps
    a linear-attention layer from silently routing through softmax (fail loud, rule 6)."""

    def __init__(self, cfg: MiniMaxConfig, layer_id: int = 0) -> None:
        super().__init__()
        if not cfg.is_full_attention(layer_id):
            raise ValueError(f"L{layer_id}: attn_type_list marks this layer non-full-softmax; "
                             f"MiniMaxAttention only implements full GQA softmax")
        self.cfg = cfg
        self.layer_id = layer_id
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.rep = cfg.n_rep
        self.rotary_dim = cfg.rotary_dim
        self.scale = cfg.attn_scale
        self.theta = cfg.rope_theta
        self.eps = cfg.norm_eps
        self.use_qk_norm = cfg.use_qk_norm
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)
        # Per-layer QK RMSNorm over head_dim (weighted). nn.RMSNorm so its weight loads/casts like
        # every other norm; a no-op identity weight if the config disables QK-norm.
        self.q_norm = nn.RMSNorm(self.hd, eps=self.eps)
        self.k_norm = nn.RMSNorm(self.hd, eps=self.eps)

    def __call__(self, x, *, offset=0, cache=None, use_fast=True):
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.nh, self.hd)
        k = self.k_proj(x).reshape(b, t, self.nkv, self.hd)
        v = self.v_proj(x).reshape(b, t, self.nkv, self.hd)
        if self.use_qk_norm:                              # weighted RMSNorm over head_dim, pre-RoPE
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = mx.transpose(q, (0, 2, 1, 3))                 # [B,H,T,hd]
        k = mx.transpose(k, (0, 2, 1, 3))                 # [B,H_kv,T,hd]
        v = mx.transpose(v, (0, 2, 1, 3))
        if cache is not None:
            offset = cache.offset
        rope = _rope_fast if use_fast else _rope_naive
        q = rope(q, self.rotary_dim, self.theta, offset)
        k = rope(k, self.rotary_dim, self.theta, offset)
        if cache is not None:
            k, v = cache.update(k, v)
        kv_len = k.shape[2]
        kr = mx.repeat(k, self.rep, axis=1)               # GQA: kv head -> its query-head group
        vr = mx.repeat(v, self.rep, axis=1)
        if use_fast:
            mask = "causal" if t > 1 else None            # single decode query attends all cached keys
            out = mx.fast.scaled_dot_product_attention(q, kr, vr, scale=self.scale, mask=mask)
        else:
            scores = (q @ mx.swapaxes(kr, -1, -2)) * self.scale + _causal_mask(t, kv_len, q.dtype)
            w = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
            out = w @ vr
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, t, self.nh * self.hd)
        return self.o_proj(out)
