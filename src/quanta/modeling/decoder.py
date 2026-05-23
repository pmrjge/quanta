"""Dense decoder layer (L0) — pre-norm residual around MLA attention and SwiGLU MLP.

Mirrors ``DeepseekV3DecoderLayer`` for ``layer_idx < first_k_dense_replace``. MoE
layers (L1+) reuse the same attention + norms but swap the MLP for router/experts.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.cache import MLACache
from quanta.config import KimiTextConfig
from quanta.modeling.attention import MLAAttention
from quanta.modeling.mlp import DenseMLP
from quanta.modeling.moe import SparseMoE


class DenseDecoderLayer(nn.Module):
    def __init__(self, cfg: KimiTextConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = MLAAttention(cfg)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = DenseMLP(cfg)

    def __call__(
        self, x: mx.array, positions: mx.array, *, use_fast: bool = False, cache: MLACache | None = None
    ) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), positions, use_fast=use_fast, cache=cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MoEDecoderLayer(nn.Module):
    """L1+ layer: same attention + norms as dense, MLP replaced by the sparse MoE block."""

    def __init__(self, cfg: KimiTextConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = MLAAttention(cfg)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = SparseMoE(cfg)

    def __call__(
        self, x: mx.array, positions: mx.array, *, use_fast: bool = False, cache: MLACache | None = None
    ) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), positions, use_fast=use_fast, cache=cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
