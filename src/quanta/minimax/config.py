"""MiniMax-M2.7 hyperparameters (``model_type="minimax_m2"``, ``MiniMaxM2ForCausalLM``).

Parsed from the source ``config.json`` into a frozen dataclass. Pure Python + ``json`` — no
``torch``/``transformers``/``mlx`` import. Grounded in the on-disk checkpoint at
``~/models/MiniMax-M2.7`` (``config.json`` + ``generation_config.json`` + the safetensors
weight-key index), NOT assumed from a sibling model — per the project's "verify each model's
formats empirically" rule (assuming DeepSeek parity once cost us the int4 offset-binary bug).

Architecture (empirically confirmed from ``config.json`` + ``model.safetensors.index.json``):

* **62 decoder layers, ALL MoE** (no dense layer-0 like Kimi), **all FULL softmax attention** —
  ``attn_type_list`` is all-1s, so M2 is *not* the lightning/linear-attention hybrid M1 was.
* **Attention: GQA** — 48 query heads / 8 KV heads (``n_rep=6``), ``head_dim=128``
  (``q_proj`` 3072->6144, ``k_proj``/``v_proj`` 3072->1024, ``o_proj`` 6144->3072). **Partial
  RoPE**: only the first ``rotary_dim=64`` of each 128-dim head is rotated, ``theta=5e6``. **Per-layer
  QK-norm**: RMSNorm ``q_norm``/``k_norm`` (``use_qk_norm``, ``qk_norm_type="per_layer"``).
* **MoE**: 256 routed experts, top-8, **sigmoid** scoring with an ``e_score_correction_bias``
  (noaux_tc-style), per-expert FFN width 1536. Mixtral expert naming: ``w1``=gate, ``w3``=up,
  ``w2``=down. **NO shared expert** (``shared_intermediate_size=0``).
* **Native MTP**: ``use_mtp``, ``num_mtp_modules=3``, ``mtp_transformer_layers=1``. NOTE: the MTP
  module weights are **NOT present** in this checkpoint's index (no ``nextn``/``mtp``/``eh_proj``
  keys) — the MTP stage must source them separately or confirm their key layout empirically before
  loading. Do not assume DeepSeek-style ``eh_proj``/``enorm``/``hnorm`` naming.
* **Source quantization**: block-fp8 (``float8_e4m3fn``, ``weight_block_size=[128,128]``, dynamic
  activation). ``q/k/v/o`` + experts ``w1/w2/w3`` are fp8 (each carries a ``.weight_scale_inv``
  block-scale grid, dequant ``= fp8.float() * scale_inv``); ``gate``, ``e_score_correction_bias``,
  ``lm_head``, ``embed_tokens`` and the norms stay bf16 (``modules_to_not_convert``).
* **Tokens**: bos 200019, eos 200020 (generation_config), vocab 200064, untied ``lm_head``.

Fields not present in ``config.json`` (MoE top-k normalization / scaling) are read with documented
defaults and **must be confirmed against the reference forward in the MoE-parity stage** — they are
not structural invariants, so ``from_pretrained`` does not refuse on them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MiniMaxConfig:
    """Hyperparameters of the MiniMax-M2.7 decoder (+ native MTP modules)."""

    vocab_size: int
    hidden_size: int                 # 3072
    moe_intermediate_size: int       # per-expert FFN width (1536); HF key "intermediate_size"
    num_hidden_layers: int           # 62

    # attention (GQA, full softmax every layer)
    num_attention_heads: int         # 48 query heads
    num_key_value_heads: int         # 8 kv heads
    head_dim: int                    # 128
    rotary_dim: int                  # rotated dims per head (64); partial RoPE
    use_qk_norm: bool                # per-layer RMSNorm on q,k before RoPE
    qk_norm_type: str                # "per_layer"
    attn_type_list: tuple[int, ...]  # per-layer attention type (all 1 = full softmax); len == n_layers

    # MoE (256 routed experts, top-8, sigmoid noaux_tc, no shared expert)
    num_local_experts: int           # 256
    num_experts_per_tok: int         # top-8
    shared_intermediate_size: int    # 0 (no shared expert)
    scoring_func: str                # "sigmoid"
    use_routing_bias: bool           # e_score_correction_bias present
    norm_topk_prob: bool             # default True — confirm in MoE parity
    routed_scaling_factor: float     # default 1.0 — confirm in MoE parity

    # native multi-token prediction (weights absent from this checkpoint; see module docstring)
    use_mtp: bool
    num_mtp_modules: int             # 3
    mtp_transformer_layers: int      # 1

    # norm / RoPE / positions / activation / tokens
    hidden_act: str                  # "silu"
    norm_eps: float                  # rms_norm_eps (1e-6)
    rope_theta: float                # 5e6
    max_position_embeddings: int     # 204800
    bos_token_id: int
    eos_token_id: int
    eos_token_ids: tuple[int, ...]
    tie_word_embeddings: bool

    # source quantization (block-fp8: float8_e4m3fn + [128,128] block scale_inv)
    quantization_config: dict = field(default_factory=dict)

    # --- derived geometry ------------------------------------------------------
    @property
    def q_dim(self) -> int:
        """Projected query dim = ``num_attention_heads * head_dim`` (6144)."""
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        """Projected key/value dim = ``num_key_value_heads * head_dim`` (1024)."""
        return self.num_key_value_heads * self.head_dim

    @property
    def n_rep(self) -> int:
        """GQA repeat factor = ``num_attention_heads // num_key_value_heads`` (6)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def attn_scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def partial_rotary_factor(self) -> float:
        """Fraction of each head rotated by RoPE = ``rotary_dim / head_dim`` (0.5)."""
        return self.rotary_dim / self.head_dim

    @property
    def has_shared_expert(self) -> bool:
        return self.shared_intermediate_size > 0

    @property
    def weight_block_size(self) -> tuple[int, int]:
        """Source fp8 block-scale grid (rows, cols) — ``[128, 128]``."""
        wbs = self.quantization_config.get("weight_block_size", [128, 128])
        return int(wbs[0]), int(wbs[1])

    @property
    def quant_modules_to_skip(self) -> tuple[str, ...]:
        """Module-name substrings kept in bf16 at the source (not fp8)."""
        return tuple(self.quantization_config.get("modules_to_not_convert", ()))

    # --- per-layer typing ------------------------------------------------------
    def is_full_attention(self, layer_id: int) -> bool:
        """``attn_type_list[layer_id] == 1`` (full softmax). All layers here, but kept per-layer so a
        future hybrid variant can't silently route a linear-attention layer through softmax."""
        if not 0 <= layer_id < len(self.attn_type_list):
            raise IndexError(f"layer_id {layer_id} out of range for attn_type_list "
                             f"(len {len(self.attn_type_list)})")
        return int(self.attn_type_list[layer_id]) == 1

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> MiniMaxConfig:
        d = Path(model_dir)
        cfg = json.loads((d / "config.json").read_text())
        gen: dict = {}
        gp = d / "generation_config.json"
        if gp.exists():
            gen = json.loads(gp.read_text())

        n_layers = int(cfg["num_hidden_layers"])
        attn_types = tuple(int(x) for x in cfg.get("attn_type_list", [1] * n_layers))
        if len(attn_types) != n_layers:
            raise ValueError(f"attn_type_list length {len(attn_types)} != num_hidden_layers "
                             f"{n_layers} (cannot type layers; refusing to guess)")

        head_dim = int(cfg["head_dim"])
        rotary_dim = int(cfg.get("rotary_dim", head_dim))

        eos = gen.get("eos_token_id", cfg.get("eos_token_id", 200020))
        eos_ids = tuple(int(x) for x in (eos if isinstance(eos, list) else [eos]))

        return cls(
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            moe_intermediate_size=int(cfg["intermediate_size"]),
            num_hidden_layers=n_layers,
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg["num_key_value_heads"]),
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            use_qk_norm=bool(cfg.get("use_qk_norm", False)),
            qk_norm_type=str(cfg.get("qk_norm_type", "per_layer")),
            attn_type_list=attn_types,
            num_local_experts=int(cfg["num_local_experts"]),
            num_experts_per_tok=int(cfg["num_experts_per_tok"]),
            shared_intermediate_size=int(cfg.get("shared_intermediate_size", 0)),
            scoring_func=str(cfg.get("scoring_func", "sigmoid")),
            use_routing_bias=bool(cfg.get("use_routing_bias", True)),
            norm_topk_prob=bool(cfg.get("norm_topk_prob", True)),
            routed_scaling_factor=float(cfg.get("routed_scaling_factor", 1.0)),
            use_mtp=bool(cfg.get("use_mtp", False)),
            num_mtp_modules=int(cfg.get("num_mtp_modules", 0)),
            mtp_transformer_layers=int(cfg.get("mtp_transformer_layers", 1)),
            hidden_act=str(cfg.get("hidden_act", "silu")),
            norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            rope_theta=float(cfg.get("rope_theta", 1e6)),
            max_position_embeddings=int(cfg.get("max_position_embeddings", 204800)),
            bos_token_id=int(gen.get("bos_token_id", cfg.get("bos_token_id", 200019))),
            eos_token_id=int(eos_ids[0]),
            eos_token_ids=eos_ids,
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            quantization_config=dict(cfg.get("quantization_config") or {}),
        )
