"""Multi-head Latent Attention (MLA) for Kimi-K2.6, MLX-native.

Mirrors ``DeepseekV3Attention`` (q_lora path). Two compute paths share identical
projections / RoPE frequencies / softmax scale:

* naive (default): explicit ``softmax(QKᵀ·scale + mask)V`` — the parity reference,
  fine only at small T.
* fast (``use_fast=True``): ``mx.fast.rope`` + ``mx.fast.scaled_dot_product_attention``
  with ``mask="causal"`` — the flash/tiled path required for long context. V is
  zero-padded from ``v_head_dim`` to ``q_head_dim`` so SDPA gets a uniform head dim,
  and the output is sliced back. Must be proven equivalent to naive by the harness.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.cache import MLACache
from quanta.config import KimiTextConfig
from quanta.modeling.rope import (
    apply_rope_explicit,
    apply_rope_fast,
    attention_softmax_scale,
    build_yarn_inv_freq,
    yarn_cos_sin,
)
from quanta.modeling.xattention import XAttnConfig, gather_sparse_attention, sparse_prefill_mask

_SUBNORM_EPS = 1e-6  # HF DeepseekV3RMSNorm default for q_a / kv_a layernorms


class MLAAttention(nn.Module):
    def __init__(self, cfg: KimiTextConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_attention_heads
        self.nope = cfg.qk_nope_head_dim
        self.rope = cfg.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.q_head_dim = cfg.q_head_dim
        self.scale = attention_softmax_scale(cfg)

        bias = cfg.attention_bias
        self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=bias)
        self.q_a_layernorm = nn.RMSNorm(cfg.q_lora_rank, eps=_SUBNORM_EPS)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(cfg.hidden_size, cfg.kv_lora_rank + self.rope, bias=bias)
        self.kv_a_layernorm = nn.RMSNorm(cfg.kv_lora_rank, eps=_SUBNORM_EPS)
        self.kv_b_proj = nn.Linear(
            cfg.kv_lora_rank, self.num_heads * (self.nope + self.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, cfg.hidden_size, bias=bias)

        # Decode-optimal MLA: absorb W_UK into the query and attend over the compressed
        # latent c_kv (kv_lora-wide) instead of materialized per-head K/V, then up-project
        # with W_UV. Output-equivalent to the expanded path; cheaper only at decode
        # (Sq=1) — at prefill it is more FLOPs. Off until parity-proven (a CLAUDE.md suspect).
        self.absorbed = False

        # XAttention block-sparse prefill (lossy; ppl-gated, not parity). None = dense.
        # Applies only to from-scratch prefill (t == kv_len) at/above cfg.min_seq.
        self.sparse: XAttnConfig | None = None

    def __call__(
        self,
        x: mx.array,
        positions: mx.array,
        *,
        use_fast: bool = False,
        cache: MLACache | None = None,
    ) -> mx.array:
        b, t, _ = x.shape
        h, nope, rope, vhd = self.num_heads, self.nope, self.rope, self.v_head_dim
        kv_lora = self.cfg.kv_lora_rank

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = mx.transpose(q.reshape(b, t, h, self.q_head_dim), (0, 2, 1, 3))  # [B,H,T,qhd]
        q_nope, q_pe = q[..., :nope], q[..., nope:]

        ckv = self.kv_a_proj_with_mqa(x)
        c_kv, k_pe = ckv[..., :kv_lora], ckv[..., kv_lora:]
        c_kv = self.kv_a_layernorm(c_kv)  # [B,m,kv_lora] latent for the new tokens
        k_pe = mx.transpose(k_pe.reshape(b, t, 1, rope), (0, 2, 1, 3))  # [B,1,m,rope]

        if use_fast:
            inv_freq = build_yarn_inv_freq(self.cfg)
            offset = int(positions[0].item()) if t > 0 else 0
            q_pe = apply_rope_fast(q_pe, inv_freq, offset=offset)
            k_pe = apply_rope_fast(k_pe, inv_freq, offset=offset)
        else:
            cos, sin = yarn_cos_sin(self.cfg, positions)
            cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
            q_pe = apply_rope_explicit(q_pe, cos, sin)
            k_pe = apply_rope_explicit(k_pe, cos, sin)

        if cache is not None:
            c_kv, k_pe = cache.update(c_kv, k_pe)  # full [B,S,kv_lora], [B,1,S,rope]
        kv_len = c_kv.shape[1]
        k_pe = mx.broadcast_to(k_pe, (b, h, kv_len, rope))  # [B,H,S,rope]

        if self.absorbed:
            return self._absorbed(q_nope, q_pe, c_kv, k_pe, b, t, kv_len)

        kv = self.kv_b_proj(c_kv)
        kv = mx.transpose(kv.reshape(b, kv_len, h, nope + vhd), (0, 2, 1, 3))  # [B,H,S,nope+vhd]
        k_nope, value = kv[..., :nope], kv[..., nope:]
        query = mx.concatenate([q_nope, q_pe], axis=-1)  # [B,H,m,qhd]
        key = mx.concatenate([k_nope, k_pe], axis=-1)  # [B,H,S,qhd]

        is_sparse_prefill = self.sparse is not None and t == kv_len and t >= self.sparse.min_seq
        if is_sparse_prefill and self.sparse.gather:  # block-gathered execution (the speed path)
            out = gather_sparse_attention(query, key, value, self.scale, self.sparse)
            out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, t, h * vhd)
            return self.o_proj(out)

        sparse_mask = sparse_prefill_mask(query, key, self.scale, self.sparse) if is_sparse_prefill else None

        if use_fast:
            v_pad = mx.concatenate([value, mx.zeros((b, h, kv_len, self.q_head_dim - vhd), value.dtype)], -1)
            mask = sparse_mask if sparse_mask is not None else "causal"
            out = mx.fast.scaled_dot_product_attention(query, key, v_pad, scale=self.scale, mask=mask)
            out = out[..., :vhd]
        else:
            scores = (query @ mx.transpose(key, (0, 1, 3, 2))) * self.scale  # [B,H,m,S]
            mask = sparse_mask if sparse_mask is not None else _causal_additive_mask(t, kv_len, scores.dtype)
            weights = mx.softmax((scores + mask).astype(mx.float32), axis=-1).astype(query.dtype)
            out = weights @ value  # [B,H,m,vhd]

        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, t, h * vhd)
        return self.o_proj(out)

    def _absorbed(
        self,
        q_nope: mx.array,
        q_pe: mx.array,
        c_kv: mx.array,
        k_pe: mx.array,
        b: int,
        t: int,
        kv_len: int,
    ) -> mx.array:
        """MLA attending over the compressed latent (decode-optimal). Explicit softmax."""
        h, nope, vhd, kv_lora = self.num_heads, self.nope, self.v_head_dim, self.cfg.kv_lora_rank
        kvb = self.kv_b_proj
        wd = (mx.dequantize(kvb.weight, kvb.scales, kvb.biases, group_size=kvb.group_size, bits=kvb.bits)
              if isinstance(kvb, nn.QuantizedLinear) else kvb.weight)  # absorb needs the dense W_UK/W_UV
        w = wd.reshape(h, nope + vhd, kv_lora)
        w_uk, w_uv = w[:, :nope, :], w[:, nope:, :]  # [H,nope,kv_lora], [H,vhd,kv_lora]

        q_absorb = q_nope @ w_uk[None]  # [B,H,m,kv_lora] = q_nope folded through W_UK
        c = c_kv[:, None]  # [B,1,S,kv_lora], shared across heads
        scores = q_absorb @ mx.transpose(c, (0, 1, 3, 2)) + q_pe @ mx.transpose(k_pe, (0, 1, 3, 2))
        scores = scores * self.scale + _causal_additive_mask(t, kv_len, scores.dtype)
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q_nope.dtype)
        out_latent = weights @ c  # [B,H,m,kv_lora]
        out = out_latent @ mx.transpose(w_uv, (0, 2, 1))[None]  # [B,H,m,vhd] up-project via W_UV
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, t, h * vhd)
        return self.o_proj(out)


def _causal_additive_mask(q_len: int, kv_len: int, dtype: mx.Dtype) -> mx.array:
    """Lower-right causal: query j (abs pos kv_len-q_len+j) attends keys i ≤ that pos."""
    off = kv_len - q_len
    j = mx.arange(q_len)[:, None]
    i = mx.arange(kv_len)[None, :]
    return mx.where(i <= j + off, mx.array(0.0, dtype), mx.array(float("-inf"), dtype))
