"""GQA attention for Nemotron-H (the 8 ``*`` layers), MLX-native.

Standard grouped-query attention with full RoPE (theta=10000, no YaRN, no MLA): 32 query
heads, 2 KV heads, head_dim 128. Two equivalent paths (gated in the test):

* fast (default): ``mx.fast.rope`` + ``mx.fast.scaled_dot_product_attention`` (mask="causal").
* naive: explicit rotate-half RoPE + manual softmax — the parity reference.

KV heads are repeated to query-head count before attention (so it works regardless of the
SDPA kernel's GQA support). Only the 8 ``*`` attention layers carry a growing KV cache — at
the model's 1M context that's ~8 GB total bf16 / ~4 GB int8 (2 KV heads × 128 head_dim × 8
layers); the 40 Mamba layers keep an O(1) recurrent state regardless of length, which is why
1M context is cheap on this architecture (~8 GB KV on top of the ~68 GB int4 weights). This
runtime operates at the model's full **1M context by default**. ``max_position_embeddings=262144``
(256K) is the *trained* RoPE window, **not** a runtime cap — RoPE here is built from
``rope_theta=10000`` alone (``mx.fast.rope`` never consumes max_position_embeddings), so positions
extend past 262144 by **plain extrapolation** with **no** ``rope_scaling`` (identical numerics to
the long-context guard-bypass upstream exposes: same theta, no rescale). The fast path uses tiled
``mx.fast.scaled_dot_product_attention`` (never materializes a TxT score matrix), so a 1M prefill
is memory-safe; the naive softmax path (TxT) is the short-sequence parity reference only.
Long-context validation confirms the 8 global-attention layers stay coherent under that 4x RoPE
extrapolation (the Mamba pathway is length-agnostic).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.cache_quant import dequantize_last_axis, quantize_last_axis
from quanta.nemotron.config import NemotronHConfig


class KVCache:
    """GQA KV cache for Nemotron: stores ``[B, n_kv, S, head_dim]`` k/v, grows along the seq axis.

    Two storage modes (#133 audit lever — long-context steady-state memory):

    * ``quantized=True`` (default since #133): **affine int8** per-token, per-group over
      ``head_dim`` — codes ``[B, n_kv, S, head_dim/4]`` + scales/biases ``[B, n_kv, S,
      head_dim/group_size]`` via :mod:`quanta.cache_quant`. ``update`` dequantizes the full
      cache for the SDPA return so the attention path is unchanged (same scheme as the Kimi
      MLA cache since #47 and the GLM/MiniMax/Qwen3.5 caches since #122). At long context the
      steady-state KV memory drops from 16 bpp to ~8.25 bpp (8 bits + ``32/group_size`` for
      scale+bias), trading ≈2× per-step dequant cost for halved KV residency — the right call
      at 1M context where memory is the bottleneck, not per-step decode latency.
    * ``quantized=False``: bf16 ``[B, n_kv, S, head_dim]`` (the historical mode; kept for
      parity gates and short-context debugging).

    Speculative-decode rollback (``truncate(length)``) slices the active stream(s) cleanly along
    the seq axis — per-position storage makes the rolled-back state bit-identical to a fresh cache
    fed only those ``length`` positions. ``max_rollback`` declares the deepest rollback the cache
    will accept for multi-step spec-decode (``k`` MTP drafts → rollback up to ``k`` tokens); a
    deeper ``truncate`` fails LOUD (rule 6: never silently keep a diverged state).
    """

    def __init__(self, *, quantized: bool = True, group_size: int = 128,
                 max_rollback: int = 1) -> None:
        if max_rollback < 0:
            raise ValueError(f"max_rollback {max_rollback} < 0")
        self.quantized = quantized
        self.group_size = group_size
        self.max_rollback = int(max_rollback)
        # bf16 mode
        self.k: mx.array | None = None
        self.v: mx.array | None = None
        # int8 mode: codes + per-group scales/biases on the head_dim axis (the last axis); grow
        # along the seq axis (axis=2) so the trio still shares the leading ``[B, n_kv, S]`` prefix.
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
        """Append the new token's k/v; return the full streams (bf16 in both modes)."""
        if not self.quantized:
            if self.k is None:
                self.k, self.v = k, v
            else:
                self.k = mx.concatenate([self.k, k], axis=2)
                self.v = mx.concatenate([self.v, v], axis=2)
            return self.k, self.v
        # int8: quantize the new tokens along head_dim (last axis), then append along the seq axis
        # (axis=2). Dequantize the full cache to bf16 so the SDPA call site is unchanged.
        kq, ks, kb = quantize_last_axis(k, self.group_size)
        vq, vs, vb = quantize_last_axis(v, self.group_size)
        if self.k_q is None:
            self.k_q, self.k_s, self.k_b = kq, ks, kb
            self.v_q, self.v_s, self.v_b = vq, vs, vb
        else:
            self.k_q = mx.concatenate([self.k_q, kq], axis=2)
            self.k_s = mx.concatenate([self.k_s, ks], axis=2)
            self.k_b = mx.concatenate([self.k_b, kb], axis=2)
            self.v_q = mx.concatenate([self.v_q, vq], axis=2)
            self.v_s = mx.concatenate([self.v_s, vs], axis=2)
            self.v_b = mx.concatenate([self.v_b, vb], axis=2)
        k_full = dequantize_last_axis(self.k_q, self.k_s, self.k_b, self.group_size, dtype=k.dtype)
        v_full = dequantize_last_axis(self.v_q, self.v_s, self.v_b, self.group_size, dtype=v.dtype)
        return k_full, v_full

    def truncate(self, length: int) -> None:
        """Roll the cache back to exactly ``length`` cached positions (drop the rolled-back tail).
        Slicing along the seq axis is lossless (per-position storage). Fails LOUD if the rollback
        depth exceeds ``max_rollback`` (rule 6) — the cache must not silently retain a diverged
        state past its declared rollback budget."""
        if length < 0:
            raise ValueError(f"truncate length {length} < 0")
        cur = self.offset
        if length >= cur:
            return
        drop = cur - length
        if drop > self.max_rollback:
            raise ValueError(
                f"truncate({length}) rolls back {drop} tokens from offset {cur}, exceeding "
                f"max_rollback={self.max_rollback}. Multi-step spec-decode needs the cache built "
                f"with max_rollback >= k (the per-round chained-draft depth).")
        if length == 0:
            self.k = self.v = None
            self.k_q = self.k_s = self.k_b = None
            self.v_q = self.v_s = self.v_b = None
            return
        if self.quantized:
            self.k_q = self.k_q[:, :, :length]
            self.k_s = self.k_s[:, :, :length]
            self.k_b = self.k_b[:, :, :length]
            self.v_q = self.v_q[:, :, :length]
            self.v_s = self.v_s[:, :, :length]
            self.v_b = self.v_b[:, :, :length]
        else:
            self.k = self.k[:, :, :length]
            self.v = self.v[:, :, :length]


def _causal_mask(q_len: int, kv_len: int, dtype: mx.Dtype) -> mx.array:
    """Lower-right causal additive mask (query j at abs pos kv_len-q_len+j)."""
    off = kv_len - q_len
    j = mx.arange(q_len)[:, None]
    i = mx.arange(kv_len)[None, :]
    return mx.where(i <= j + off, mx.array(0.0, dtype), mx.array(float("-inf"), dtype))


class NemotronAttention(nn.Module):
    def __init__(self, cfg: NemotronHConfig) -> None:
        super().__init__()
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.rep = self.nh // self.nkv
        self.scale = self.hd ** -0.5
        self.theta = cfg.rope_theta
        bias = cfg.attention_bias
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=bias)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=bias)

    def __call__(self, x, *, offset=0, cache=None, use_fast=True):
        b, t, _ = x.shape
        q = mx.transpose(self.q_proj(x).reshape(b, t, self.nh, self.hd), (0, 2, 1, 3))
        k = mx.transpose(self.k_proj(x).reshape(b, t, self.nkv, self.hd), (0, 2, 1, 3))
        v = mx.transpose(self.v_proj(x).reshape(b, t, self.nkv, self.hd), (0, 2, 1, 3))
        if cache is not None:
            offset = cache.offset
        q = mx.fast.rope(q, dims=self.hd, traditional=False, base=self.theta, scale=1.0, offset=offset)
        k = mx.fast.rope(k, dims=self.hd, traditional=False, base=self.theta, scale=1.0, offset=offset)
        if cache is not None:
            k, v = cache.update(k, v)
        kv_len = k.shape[2]
        kr = mx.repeat(k, self.rep, axis=1)  # GQA: kv head -> its query-head group
        vr = mx.repeat(v, self.rep, axis=1)
        if use_fast:
            mask = "causal" if t > 1 else None  # single decode query attends all cached keys
            out = mx.fast.scaled_dot_product_attention(q, kr, vr, scale=self.scale, mask=mask)
        else:
            scores = (q @ mx.swapaxes(kr, -1, -2)) * self.scale + _causal_mask(t, kv_len, q.dtype)
            w = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
            out = w @ vr
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, t, self.nh * self.hd)
        return self.o_proj(out)
