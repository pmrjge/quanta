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

from quanta.nemotron.config import NemotronHConfig


class KVCache:
    """Plain GQA KV cache: stores ``[B, n_kv, S, head_dim]`` k/v, grows along the seq axis."""

    def __init__(self) -> None:
        self.k: mx.array | None = None
        self.v: mx.array | None = None

    @property
    def offset(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = mx.concatenate([self.k, k], axis=2)
            self.v = mx.concatenate([self.v, v], axis=2)
        return self.k, self.v


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
