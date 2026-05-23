"""GQA attention for Nemotron-H (the 8 ``*`` layers), MLX-native.

Standard grouped-query attention with full RoPE (theta=10000, no YaRN, no MLA): 32 query
heads, 2 KV heads, head_dim 128. Two equivalent paths (gated in the test):

* fast (default): ``mx.fast.rope`` + ``mx.fast.scaled_dot_product_attention`` (mask="causal").
* naive: explicit rotate-half RoPE + manual softmax — the parity reference.

KV heads are repeated to query-head count before attention (so it works regardless of the
SDPA kernel's GQA support). Only the 8 attention layers carry a growing KV cache — at 256K
that's ~2 GB total (2 KV heads), which is why long context is cheap on this architecture.
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
