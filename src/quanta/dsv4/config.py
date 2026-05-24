"""DeepSeek-V4-Flash hyperparameters (``model_type="deepseek_v4"``, ``DeepseekV4ForCausalLM``).

Parsed from the source ``config.json`` into a frozen dataclass. Pure Python + ``json`` — no
``torch``/``transformers``/``mlx`` import. DSV4-Flash is a *very* non-standard sparse-MoE backbone;
the per-layer helpers here encode the structural facts so downstream code can't diverge on them:

* **Hyper-Connections (HC).** The residual stream carries ``hc_mult=4`` parallel copies; every block
  mixes them with a Sinkhorn-normalized learned combine (see :mod:`quanta.dsv4.hyper`).
* **Three attention regimes**, selected per layer by ``compress_ratios`` (length ``n_layers+1`` —
  the last entry is the MTP block): ratio ``0`` = pure sliding-window; ratio ``4`` = window +
  **Lightning Indexer** (DeepSeek Sparse Attention, top-``index_topk``) over a 4x **Compressor**;
  ratio ``128`` = window + a 128x **Compressor** (no indexer). :meth:`compress_ratio`,
  :meth:`has_indexer`, :meth:`has_compressor`.
* **RoPE differs by regime** (:meth:`attn_rope`): compressed layers use YaRN (``original_seq_len``,
  ``compress_rope_theta``); pure sliding-window layers use the base ``rope_theta`` with YaRN off.
  Only the last ``rope_head_dim`` of each head's ``head_dim`` is rotated (partial RoPE).
* **Routing differs by depth** (:meth:`is_hash`): the first ``n_hash_layers`` MoE layers route by a
  fixed token->expert hash table (``tid2eid``); the rest use ``sqrtsoftplus`` ``noaux_tc`` scoring.

Source weights are **block-fp8** (non-experts: F8_E4M3 + e8m0 block-128) and **fp4** (experts:
e2m1 packed + e8m0 group-32); both scale grids are OCP-MX **e8m0** (power-of-two). See
:mod:`quanta.dsv4.fp`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DeepSeekV4Config:
    """Hyperparameters of the DeepSeek-V4-Flash decoder (+ native MTP block)."""

    vocab_size: int
    hidden_size: int                 # dim
    num_hidden_layers: int           # n_layers (main decoder blocks)
    moe_intermediate_size: int       # per-expert FFN width

    # attention (MLA-style, single shared latent kv head)
    num_attention_heads: int         # n_heads
    head_dim: int                    # full per-head dim (512)
    rope_head_dim: int               # rotated dims per head (64); nope = head_dim - rope_head_dim
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    sliding_window: int              # window_size

    # Lightning Indexer (DeepSeek Sparse Attention) — present on ratio==4 layers
    index_n_heads: int
    index_head_dim: int
    index_topk: int

    # per-layer KV compression: ratio per block; len == num_hidden_layers + n_mtp_layers
    compress_ratios: tuple[int, ...]
    compress_rope_theta: float

    # MoE (256 experts top-6 + 1 shared; hash routing for the first n_hash_layers)
    n_routed_experts: int
    num_experts_per_tok: int         # n_activated_experts
    n_shared_experts: int
    n_hash_layers: int
    scoring_func: str                # "sqrtsoftplus"
    topk_method: str                 # "noaux_tc"
    norm_topk_prob: bool
    routed_scaling_factor: float     # route_scale (1.5)
    swiglu_limit: float

    # Hyper-Connections
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float

    # native multi-token prediction
    n_mtp_layers: int                # num_nextn_predict_layers

    # norm / RoPE / positions / tokens
    norm_eps: float
    rope_theta: float                # base theta for pure sliding-window layers
    rope_scaling: dict               # {factor, beta_fast, beta_slow, original_max_position_embeddings, type}
    max_position_embeddings: int
    bos_token_id: int
    eos_token_id: int
    eos_token_ids: tuple[int, ...]
    tie_word_embeddings: bool

    # source quantization (fp8 non-experts + fp4 experts; e8m0/MX scales)
    quantization_config: dict = field(default_factory=dict)
    expert_dtype: str = "fp4"

    # --- derived geometry ------------------------------------------------------
    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def attn_scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def original_seq_len(self) -> int:
        return int(self.rope_scaling.get("original_max_position_embeddings", 0) or 0)

    @property
    def rope_factor(self) -> float:
        return float(self.rope_scaling.get("factor", 1.0) or 1.0)

    @property
    def beta_fast(self) -> float:
        return float(self.rope_scaling.get("beta_fast", 32))

    @property
    def beta_slow(self) -> float:
        return float(self.rope_scaling.get("beta_slow", 1))

    # --- per-layer typing ------------------------------------------------------
    def compress_ratio(self, layer_id: int) -> int:
        """KV compression ratio for a block (``layer_id`` may index the MTP block at ``n_layers``)."""
        if not 0 <= layer_id < len(self.compress_ratios):
            raise IndexError(f"layer_id {layer_id} out of range for compress_ratios "
                             f"(len {len(self.compress_ratios)})")
        return int(self.compress_ratios[layer_id])

    def has_compressor(self, layer_id: int) -> bool:
        return self.compress_ratio(layer_id) != 0

    def has_indexer(self, layer_id: int) -> bool:
        """The Lightning Indexer (DSA) is attached only on ratio==4 layers."""
        return self.compress_ratio(layer_id) == 4

    def is_hash(self, layer_id: int) -> bool:
        """First ``n_hash_layers`` blocks route by the fixed ``tid2eid`` table (no learned bias)."""
        return layer_id < self.n_hash_layers

    def overlap(self, layer_id: int) -> bool:
        """Compressor uses overlapping windows (coff=2) iff ratio==4."""
        return self.compress_ratio(layer_id) == 4

    def compressor_coff(self, layer_id: int) -> int:
        return 2 if self.overlap(layer_id) else 1

    def attn_rope(self, layer_id: int) -> tuple[int, float]:
        """``(original_seq_len, theta)`` for this block's RoPE: compressed layers use YaRN with
        ``compress_rope_theta``; pure sliding-window layers use the base ``rope_theta`` (YaRN off,
        signalled by ``original_seq_len == 0``)."""
        if self.has_compressor(layer_id):
            return self.original_seq_len, self.compress_rope_theta
        return 0, self.rope_theta

    @property
    def mix_hc(self) -> int:
        """Width of the HC mix vector: ``(2 + hc_mult) * hc_mult``."""
        return (2 + self.hc_mult) * self.hc_mult

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> DeepSeekV4Config:
        d = Path(model_dir)
        cfg = json.loads((d / "config.json").read_text())
        gen: dict = {}
        gp = d / "generation_config.json"
        if gp.exists():
            gen = json.loads(gp.read_text())

        n_layers = int(cfg["num_hidden_layers"])
        n_mtp = int(cfg.get("num_nextn_predict_layers", 0))
        ratios = tuple(int(x) for x in cfg["compress_ratios"])
        if len(ratios) != n_layers + n_mtp:
            raise ValueError(f"compress_ratios length {len(ratios)} != n_layers+n_mtp "
                             f"{n_layers + n_mtp} (cannot type layers; refusing to guess)")

        eos = gen.get("eos_token_id", cfg.get("eos_token_id", 1))
        eos_ids = tuple(int(x) for x in (eos if isinstance(eos, list) else [eos]))

        return cls(
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            num_hidden_layers=n_layers,
            moe_intermediate_size=int(cfg["moe_intermediate_size"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            head_dim=int(cfg["head_dim"]),
            rope_head_dim=int(cfg["qk_rope_head_dim"]),
            q_lora_rank=int(cfg["q_lora_rank"]),
            o_lora_rank=int(cfg["o_lora_rank"]),
            o_groups=int(cfg["o_groups"]),
            sliding_window=int(cfg["sliding_window"]),
            index_n_heads=int(cfg["index_n_heads"]),
            index_head_dim=int(cfg["index_head_dim"]),
            index_topk=int(cfg["index_topk"]),
            compress_ratios=ratios,
            compress_rope_theta=float(cfg["compress_rope_theta"]),
            n_routed_experts=int(cfg["n_routed_experts"]),
            num_experts_per_tok=int(cfg["num_experts_per_tok"]),
            n_shared_experts=int(cfg.get("n_shared_experts", 1)),
            n_hash_layers=int(cfg.get("num_hash_layers", 0)),
            scoring_func=str(cfg.get("scoring_func", "sqrtsoftplus")),
            topk_method=str(cfg.get("topk_method", "noaux_tc")),
            norm_topk_prob=bool(cfg.get("norm_topk_prob", True)),
            routed_scaling_factor=float(cfg.get("routed_scaling_factor", 1.0)),
            swiglu_limit=float(cfg.get("swiglu_limit", 0.0)),
            hc_mult=int(cfg.get("hc_mult", 1)),
            hc_sinkhorn_iters=int(cfg.get("hc_sinkhorn_iters", 0)),
            hc_eps=float(cfg.get("hc_eps", 1e-6)),
            n_mtp_layers=n_mtp,
            norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            rope_theta=float(cfg.get("rope_theta", 10000.0)),
            rope_scaling=dict(cfg.get("rope_scaling") or {}),
            max_position_embeddings=int(cfg.get("max_position_embeddings", 1048576)),
            bos_token_id=int(gen.get("bos_token_id", cfg.get("bos_token_id", 0))),
            eos_token_id=int(eos_ids[0]),
            eos_token_ids=eos_ids,
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            quantization_config=dict(cfg.get("quantization_config") or {}),
            expert_dtype=str(cfg.get("expert_dtype", "fp4")),
        )
