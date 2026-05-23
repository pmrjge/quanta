"""Text-model hyperparameters for Kimi-K2.6 (DeepSeek-V3-style).

Parsed from the source ``config.json`` ``text_config`` block into a frozen
dataclass. Pure Python + ``json`` — no ``torch``/``transformers``/``mlx`` import,
so it is safe to use on the runtime hot path and in the bake pipeline alike.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class YarnRope:
    """YaRN RoPE scaling parameters (DeepSeek-V3 convention)."""

    factor: float
    beta_fast: float
    beta_slow: float
    mscale: float
    mscale_all_dim: float
    original_max_position_embeddings: int
    rope_theta: float


@dataclass(frozen=True)
class KimiTextConfig:
    """Hyperparameters of the Kimi-K2.6 text decoder (DeepSeek-V3 architecture)."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int

    # MLA (multi-head latent attention) dims.
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    # MoE routing.
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    n_group: int
    topk_group: int
    topk_method: str
    scoring_func: str
    norm_topk_prob: bool
    routed_scaling_factor: float
    first_k_dense_replace: int
    moe_layer_freq: int

    rms_norm_eps: float
    hidden_act: str
    attention_bias: bool
    max_position_embeddings: int
    bos_token_id: int
    eos_token_id: int

    rope: YarnRope

    @property
    def q_head_dim(self) -> int:
        """Per-head query/key width: nope + rope = 128 + 64 = 192."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def is_dense_layer(self, layer_idx: int) -> bool:
        """L0 is dense; L1..L60 are MoE (``first_k_dense_replace == 1``)."""
        return layer_idx < self.first_k_dense_replace

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> KimiTextConfig:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        tc = cfg["text_config"]
        rs = tc["rope_scaling"]
        rope = YarnRope(
            factor=float(rs["factor"]),
            beta_fast=float(rs["beta_fast"]),
            beta_slow=float(rs["beta_slow"]),
            mscale=float(rs["mscale"]),
            mscale_all_dim=float(rs["mscale_all_dim"]),
            original_max_position_embeddings=int(rs["original_max_position_embeddings"]),
            rope_theta=float(tc["rope_theta"]),
        )
        return cls(
            vocab_size=int(tc["vocab_size"]),
            hidden_size=int(tc["hidden_size"]),
            intermediate_size=int(tc["intermediate_size"]),
            moe_intermediate_size=int(tc["moe_intermediate_size"]),
            num_hidden_layers=int(tc["num_hidden_layers"]),
            num_attention_heads=int(tc["num_attention_heads"]),
            num_key_value_heads=int(tc["num_key_value_heads"]),
            q_lora_rank=int(tc["q_lora_rank"]),
            kv_lora_rank=int(tc["kv_lora_rank"]),
            qk_nope_head_dim=int(tc["qk_nope_head_dim"]),
            qk_rope_head_dim=int(tc["qk_rope_head_dim"]),
            v_head_dim=int(tc["v_head_dim"]),
            n_routed_experts=int(tc["n_routed_experts"]),
            n_shared_experts=int(tc["n_shared_experts"]),
            num_experts_per_tok=int(tc["num_experts_per_tok"]),
            n_group=int(tc["n_group"]),
            topk_group=int(tc["topk_group"]),
            topk_method=str(tc["topk_method"]),
            scoring_func=str(tc["scoring_func"]),
            norm_topk_prob=bool(tc["norm_topk_prob"]),
            routed_scaling_factor=float(tc["routed_scaling_factor"]),
            first_k_dense_replace=int(tc["first_k_dense_replace"]),
            moe_layer_freq=int(tc["moe_layer_freq"]),
            rms_norm_eps=float(tc["rms_norm_eps"]),
            hidden_act=str(tc["hidden_act"]),
            attention_bias=bool(tc["attention_bias"]),
            max_position_embeddings=int(tc["max_position_embeddings"]),
            bos_token_id=int(tc["bos_token_id"]),
            eos_token_id=int(tc["eos_token_id"]),
            rope=rope,
        )
