"""Nemotron-H text-model hyperparameters (NVIDIA-Nemotron-3-Super-120B-A12B).

Parsed from the source ``config.json`` into a frozen dataclass. Pure Python + ``json`` — no
``torch``/``transformers``/``mlx`` import. Nemotron-H is a hybrid: each of the 88 layers is
one of ``{mamba, moe, attention}`` selected by ``hybrid_override_pattern`` (letters
``M``/``E``/``*``). See :mod:`quanta.nemotron.quant_policy` for the per-tensor quant mix and
:mod:`quanta.nemotron.tokenizer` for encode/chat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# hybrid_override_pattern letters -> layer kind (verified empirically against the weight map)
_LETTER_TO_KIND = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}


@dataclass(frozen=True)
class NemotronHConfig:
    """Hyperparameters of the Nemotron-H hybrid decoder."""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    hybrid_override_pattern: str

    # attention (GQA, standard RoPE — no MLA, no YaRN)
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attention_bias: bool
    rope_theta: float
    partial_rotary_factor: float

    # mamba-2 (SSD)
    mamba_num_heads: int
    mamba_head_dim: int
    mamba_n_groups: int
    ssm_state_size: int
    conv_kernel: int
    expand: int
    mamba_hidden_act: str
    chunk_size: int
    use_conv_bias: bool

    # latent MoE (relu^2 experts: up/down only, on a low-rank latent)
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    moe_intermediate_size: int
    moe_latent_size: int
    moe_shared_expert_intermediate_size: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    n_group: int
    topk_group: int

    norm_eps: float
    max_position_embeddings: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    num_nextn_predict_layers: int
    tie_word_embeddings: bool

    # --- derived dims ---------------------------------------------------------
    @property
    def mamba_d_inner(self) -> int:
        return self.expand * self.hidden_size

    @property
    def mamba_conv_dim(self) -> int:
        """conv1d acts on x+B+C: d_inner + 2*n_groups*ssm_state."""
        return self.mamba_d_inner + 2 * self.mamba_n_groups * self.ssm_state_size

    @property
    def mamba_in_proj_dim(self) -> int:
        """in_proj output = [z, x, B, C, dt] = 2*d_inner + 2*n_groups*ssm_state + num_heads."""
        return (2 * self.mamba_d_inner + 2 * self.mamba_n_groups * self.ssm_state_size
                + self.mamba_num_heads)

    @property
    def attn_q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def attn_kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def layers_block_type(self) -> list[str]:
        kinds = [_LETTER_TO_KIND[c] for c in self.hybrid_override_pattern]
        if len(kinds) != self.num_hidden_layers:
            raise ValueError(f"pattern length {len(kinds)} != num_hidden_layers "
                             f"{self.num_hidden_layers}")
        return kinds

    def layer_kind(self, i: int) -> str:
        return self.layers_block_type[i]

    def count(self, kind: str) -> int:
        return sum(k == kind for k in self.layers_block_type)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> NemotronHConfig:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        eos = cfg["eos_token_id"]
        return cls(
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            hybrid_override_pattern=str(cfg["hybrid_override_pattern"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg["num_key_value_heads"]),
            head_dim=int(cfg["head_dim"]),
            attention_bias=bool(cfg.get("attention_bias", False)),
            rope_theta=float(cfg["rope_theta"]),
            partial_rotary_factor=float(cfg.get("partial_rotary_factor", 1.0)),
            mamba_num_heads=int(cfg["mamba_num_heads"]),
            mamba_head_dim=int(cfg["mamba_head_dim"]),
            mamba_n_groups=int(cfg["n_groups"]),
            ssm_state_size=int(cfg["ssm_state_size"]),
            conv_kernel=int(cfg["conv_kernel"]),
            expand=int(cfg["expand"]),
            mamba_hidden_act=str(cfg.get("mamba_hidden_act", "silu")),
            chunk_size=int(cfg["chunk_size"]),
            use_conv_bias=bool(cfg.get("use_conv_bias", True)),
            n_routed_experts=int(cfg["n_routed_experts"]),
            num_experts_per_tok=int(cfg["num_experts_per_tok"]),
            n_shared_experts=int(cfg["n_shared_experts"]),
            moe_intermediate_size=int(cfg["moe_intermediate_size"]),
            moe_latent_size=int(cfg["moe_latent_size"]),
            moe_shared_expert_intermediate_size=int(cfg["moe_shared_expert_intermediate_size"]),
            routed_scaling_factor=float(cfg["routed_scaling_factor"]),
            norm_topk_prob=bool(cfg["norm_topk_prob"]),
            n_group=int(cfg["n_group"]),
            topk_group=int(cfg["topk_group"]),
            norm_eps=float(cfg.get("norm_eps", cfg.get("layer_norm_epsilon", 1e-5))),
            max_position_embeddings=int(cfg["max_position_embeddings"]),
            bos_token_id=int(cfg["bos_token_id"]),
            eos_token_id=int(eos if isinstance(eos, int) else eos[0]),
            pad_token_id=int(cfg["pad_token_id"]),
            num_nextn_predict_layers=int(cfg.get("num_nextn_predict_layers", 0)),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        )
