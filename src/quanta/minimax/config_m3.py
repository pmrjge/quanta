"""MiniMax-M3-VL hyperparameters (``model_type="minimax_m3_vl"``,
``MiniMaxM3SparseForConditionalGeneration``).

Parsed from the source ``config.json`` (the nested ``text_config`` + ``vision_config`` +
multimodal wrapper fields) into frozen dataclasses. Pure Python + ``json`` — no
``torch``/``transformers``/``mlx`` import. Grounded in the on-disk checkpoint at
``~/models/MiniMax-M3`` (``config.json`` + ``generation_config.json`` + the safetensors
weight-key index), NOT assumed from the sibling **M2.7** module (:mod:`quanta.minimax.config`),
which is a DIFFERENT architecture — per the project's "verify each model's formats empirically"
rule.

Architecture (empirically confirmed from ``config.json`` + ``model.safetensors.index.json`` +
the per-shard safetensors headers — see the N0/M0 fit-test):

* **60 decoder layers**: **layers 0–2 dense** (``mlp.{gate,up,down}_proj``, width
  ``dense_intermediate_size=12288``), **layers 3–59 MoE** (``block_sparse_moe.*``). The split
  is the explicit ``moe_layer_freq`` list (1 == MoE), NOT assumed.
* **Attention: GQA** — 64 query heads / 4 KV heads (``n_rep=16``), ``head_dim=128``
  (``q_proj`` 6144->8192, ``k_proj``/``v_proj`` 6144->512, ``o_proj`` 8192->6144). **Partial
  RoPE**: only the first ``rotary_dim=64`` of each 128-dim head is rotated, ``theta=5e6``.
  **Per-head QK-norm** (RMSNorm ``q_norm``/``k_norm`` of width ``head_dim``,
  ``qk_norm_type="per_head"``). **Gemma ``(1+w)`` RMSNorm** everywhere (``use_gemma_norm``).
* **Native TRAINED block-sparse attention** on the MoE layers (3–59): each ships
  ``self_attn.index_{q,k}_proj`` + ``index_{q,k}_norm`` (a lightweight learned indexer:
  ``sparse_num_index_heads=4`` query index heads of ``sparse_index_dim=128`` + a shared index
  key; selects the top ``sparse_topk_blocks=16`` key-blocks of ``sparse_block_size=128`` by
  ``sparse_score_type="max"``, always keeping ``sparse_init_block`` sinks + ``sparse_local_block``
  recent). At ``T <= ~2048`` tokens the top-16 blocks ARE all blocks ⇒ sparse == dense, so the
  parity reference can run dense and the indexer is a long-context efficiency milestone.
* **MoE**: 128 routed experts, top-4, **+1 shared expert** (``n_shared_experts=1``,
  ``shared_intermediate_size=3072``), **sigmoid** scoring with an ``e_score_correction_bias``
  (noaux_tc-style), ``routed_scaling_factor=2.0``, per-expert FFN width ``intermediate_size=3072``
  (``w1``=gate, ``w3``=up, ``w2``=down). Router ``gate`` + bias are **fp32**.
* **Activation: clamped SwiGLU-OpenAI** (``hidden_act="swigluoai"``, ``swiglu_alpha=1.702``,
  ``swiglu_limit=7.0``) for BOTH the dense FFN and the experts — NOT plain silu-SwiGLU.
* **Native MTP**: ``num_mtp_modules=7`` is DECLARED but the checkpoint ships **ZERO**
  ``mtp.*``/``nextn.*`` weights (exactly the Nex case) ⇒ ``from_pretrained`` refines
  ``num_mtp_modules -> 0`` by index presence (rule 6). Native-MTP spec-decode is N/A for M3.
* **1M native context** (``max_position_embeddings=1048576``).
* **Vision tower** (full VL): a CLIP-style ViT (``vision_config``: 32 layers, hidden 1280,
  patch 14, 3D-RoPE) + ``multi_modal_projector`` (-> 6144) + ``patch_merge_mlp``; ``image_token``
  id 200025, ``video_token`` id 200026, dynamic-res tiling (``image_grid_pinpoints``).
* **Tokens**: bos 200019, eos 200020 (``generation_config.json``; the chat template ends a turn
  on the lone eos ``[e~[`` 200020 — unlike Nex no second turn-ender to derive). vocab 200064,
  untied ``lm_head``. Reasoning markers ``<mm:think>``/``</mm:think>``; tool calls are namespaced
  nested-XML (``]<]minimax[>[<tool_call>`` ...) — handled in the serving shim, not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MiniMaxVisionConfig:
    """CLIP-style ViT sub-config of MiniMax-M3-VL (``vision_config``)."""

    hidden_size: int                 # 1280
    num_hidden_layers: int           # 32
    num_attention_heads: int         # 16
    intermediate_size: int           # 5120
    patch_size: int                  # 14
    image_size: int                  # 2016
    projection_dim: int              # 6144 (== text hidden)
    rope_theta: float                # 10000.0
    rope_mode: str                   # "3d"
    layer_norm_eps: float            # 1e-5
    hidden_act: str                  # "gelu"
    num_channels: int                # 3
    spatial_merge_size: int          # 2 (patch_merge compression)
    temporal_patch_size: int         # 2

    @classmethod
    def from_dict(cls, v: dict) -> MiniMaxVisionConfig:
        comp = v.get("img_token_compression_config", {}) or {}
        return cls(
            hidden_size=int(v["hidden_size"]),
            num_hidden_layers=int(v["num_hidden_layers"]),
            num_attention_heads=int(v["num_attention_heads"]),
            intermediate_size=int(v["intermediate_size"]),
            patch_size=int(v["patch_size"]),
            image_size=int(v["image_size"]),
            projection_dim=int(v.get("projection_dim", 6144)),
            rope_theta=float(v.get("rope_theta", 10000.0)),
            rope_mode=str(v.get("rope_mode", "3d")),
            layer_norm_eps=float(v.get("layer_norm_eps", 1e-5)),
            hidden_act=str(v.get("hidden_act", "gelu")),
            num_channels=int(v.get("num_channels", 3)),
            spatial_merge_size=int(comp.get("spatial_merge_size", 2)),
            temporal_patch_size=int(comp.get("temporal_patch_size", 2)),
        )


@dataclass(frozen=True)
class MiniMaxM3Config:
    """Hyperparameters of the MiniMax-M3-VL decoder (text backbone + vision tower)."""

    # --- text backbone ---------------------------------------------------------
    vocab_size: int                  # 200064
    hidden_size: int                 # 6144
    moe_intermediate_size: int       # per-expert FFN width (3072); HF key "intermediate_size"
    dense_intermediate_size: int     # dense-layer FFN width (12288); layers 0-2
    shared_intermediate_size: int    # shared-expert FFN width (3072)
    num_hidden_layers: int           # 60

    # attention (GQA, partial RoPE, per-head QK-norm, Gemma (1+w) norms)
    num_attention_heads: int         # 64 query heads
    num_key_value_heads: int         # 4 kv heads
    head_dim: int                    # 128
    rotary_dim: int                  # rotated dims per head (64); partial RoPE
    use_qk_norm: bool                # per-head RMSNorm on q,k before RoPE
    qk_norm_type: str                # "per_head"
    use_gemma_norm: bool             # (1+w) RMSNorm fold
    attention_output_gate: bool      # False for M3
    rope_theta: float                # 5e6

    # MoE (128 routed experts, top-4, sigmoid noaux_tc + bias, 1 shared expert)
    num_local_experts: int           # 128
    num_experts_per_tok: int         # top-4
    n_shared_experts: int            # 1
    scoring_func: str                # "sigmoid"
    use_routing_bias: bool           # e_score_correction_bias present
    norm_topk_prob: bool             # default True — confirm in MoE parity
    routed_scaling_factor: float     # 2.0
    moe_layer_freq: tuple[int, ...]  # per-layer 1==MoE / 0==dense; len == n_layers

    # activation (clamped SwiGLU-OpenAI)
    hidden_act: str                  # "swigluoai"
    swiglu_alpha: float              # 1.702
    swiglu_limit: float              # 7.0

    # native trained block-sparse attention (the DSA-style indexer)
    use_sparse_attention: bool
    sparse_topk_blocks: int          # 16
    sparse_block_size: int           # 128
    sparse_num_index_heads: int      # 4
    sparse_index_dim: int            # 128
    sparse_init_block: int           # 0 (sink blocks always kept)
    sparse_local_block: int          # 1 (recent blocks always kept)
    sparse_score_type: str           # "max"
    sparse_attention_freq: tuple[int, ...]  # per-layer 1==sparse-eligible; len == n_layers

    # native multi-token prediction (DECLARED but weights absent from this checkpoint; refined to 0)
    num_mtp_modules: int             # effective (0 after presence-refine); declared 7
    num_mtp_modules_declared: int    # 7 (what config.json says, for the gate)
    num_nextn_predict_layers: int    # 1

    # norm / positions / tokens
    norm_eps: float                  # rms_norm_eps (1e-6)
    max_position_embeddings: int     # 1048576 (1M native)
    bos_token_id: int                # 200019
    eos_token_id: int                # 200020 (primary turn-ender)
    eos_token_ids: tuple[int, ...]   # stop set
    tie_word_embeddings: bool        # False

    # --- multimodal wrapper ----------------------------------------------------
    image_token_index: int           # 200025
    video_token_index: int           # 200026
    image_seq_length: int            # 576
    vision_feature_layer: int        # -1
    vision_feature_select_strategy: str  # "full"
    projector_hidden_act: str        # "gelu"
    multimodal_projector_bias: bool  # True
    vision: MiniMaxVisionConfig | None = None

    # raw blocks kept for downstream (sparse cfg dict, rope, etc.)
    sparse_attention_config: dict = field(default_factory=dict)

    # --- derived geometry ------------------------------------------------------
    @property
    def q_dim(self) -> int:
        """Projected query dim = ``num_attention_heads * head_dim`` (8192)."""
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        """Projected key/value dim = ``num_key_value_heads * head_dim`` (512)."""
        return self.num_key_value_heads * self.head_dim

    @property
    def n_rep(self) -> int:
        """GQA repeat factor = ``num_attention_heads // num_key_value_heads`` (16)."""
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
        return self.n_shared_experts > 0 and self.shared_intermediate_size > 0

    # --- per-layer typing (validated; refuse to guess on a malformed schedule) --
    def _check_layer(self, layer_id: int, seq: tuple[int, ...], name: str) -> int:
        if not 0 <= layer_id < self.num_hidden_layers:
            raise IndexError(f"layer_id {layer_id} out of range [0,{self.num_hidden_layers})")
        if len(seq) != self.num_hidden_layers:
            raise ValueError(f"{name} length {len(seq)} != num_hidden_layers "
                             f"{self.num_hidden_layers} (cannot type layers; refusing to guess)")
        return int(seq[layer_id])

    def is_moe_layer(self, layer_id: int) -> bool:
        """``moe_layer_freq[layer_id] == 1`` (block_sparse_moe). Layers 0–2 are dense."""
        return self._check_layer(layer_id, self.moe_layer_freq, "moe_layer_freq") == 1

    def is_dense_layer(self, layer_id: int) -> bool:
        return not self.is_moe_layer(layer_id)

    def is_sparse_attention_layer(self, layer_id: int) -> bool:
        """``sparse_attention_freq[layer_id] == 1`` AND sparse attention enabled — the layers that
        carry the trained ``index_*`` projections (3–59). Dense layers 0–2 are full attention."""
        if not self.use_sparse_attention:
            return False
        return self._check_layer(layer_id, self.sparse_attention_freq, "sparse_attention_freq") == 1

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> MiniMaxM3Config:
        d = Path(model_dir)
        cfg = json.loads((d / "config.json").read_text())
        if "text_config" not in cfg:
            raise ValueError("config.json has no text_config — not a minimax_m3_vl checkpoint "
                             "(use quanta.minimax.config.MiniMaxConfig for the flat M2 schema)")
        tc = cfg["text_config"]
        gen: dict = {}
        gp = d / "generation_config.json"
        if gp.exists():
            gen = json.loads(gp.read_text())

        n_layers = int(tc["num_hidden_layers"])
        moe_freq = tuple(int(x) for x in tc.get("moe_layer_freq", [1] * n_layers))
        if len(moe_freq) != n_layers:
            raise ValueError(f"moe_layer_freq length {len(moe_freq)} != num_hidden_layers {n_layers}")

        sac = dict(tc.get("sparse_attention_config") or {})
        sparse_freq = tuple(int(x) for x in sac.get("sparse_attention_freq", [0] * n_layers))
        if sac and len(sparse_freq) != n_layers:
            raise ValueError(f"sparse_attention_freq length {len(sparse_freq)} != "
                             f"num_hidden_layers {n_layers}")

        head_dim = int(tc["head_dim"])
        rotary_dim = int(tc.get("rotary_dim", round(head_dim * float(tc.get(
            "partial_rotary_factor", 1.0)))))

        # eos: trust generation_config (M3 ships a correct one); the chat template ends a turn on the
        # lone eos 200020. Fall back to config eos, then the canonical 200020, if absent (rule 6).
        eos = gen.get("eos_token_id", tc.get("eos_token_id", cfg.get("eos_token_id", 200020)))
        eos_ids = tuple(int(x) for x in (eos if isinstance(eos, list) else [eos]))
        bos = int(gen.get("bos_token_id", tc.get("bos_token_id", cfg.get("bos_token_id", 200019))))

        # MTP presence-refine (rule 6): config declares num_mtp_modules but M3 ships NO mtp.*/nextn.*
        # weights. Trust the weights — refine the effective count to what the index actually carries.
        declared_mtp = int(tc.get("num_mtp_modules", 0))
        mtp_eff = declared_mtp
        idx_path = d / "model.safetensors.index.json"
        if idx_path.exists():
            wm = json.loads(idx_path.read_text()).get("weight_map", {})
            has_mtp = any((".mtp." in k or k.startswith("mtp.") or "nextn" in k
                           or "mtp_" in k) for k in wm)
            mtp_eff = declared_mtp if has_mtp else 0

        vc = cfg.get("vision_config")
        vision = MiniMaxVisionConfig.from_dict(vc) if vc else None

        return cls(
            vocab_size=int(tc["vocab_size"]),
            hidden_size=int(tc["hidden_size"]),
            moe_intermediate_size=int(tc["intermediate_size"]),
            dense_intermediate_size=int(tc.get("dense_intermediate_size", tc["intermediate_size"])),
            shared_intermediate_size=int(tc.get("shared_intermediate_size", 0)),
            num_hidden_layers=n_layers,
            num_attention_heads=int(tc["num_attention_heads"]),
            num_key_value_heads=int(tc["num_key_value_heads"]),
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            use_qk_norm=bool(tc.get("use_qk_norm", False)),
            qk_norm_type=str(tc.get("qk_norm_type", "per_head")),
            use_gemma_norm=bool(tc.get("use_gemma_norm", False)),
            attention_output_gate=bool(tc.get("attention_output_gate", False)),
            rope_theta=float(tc.get("rope_theta", 5e6)),
            num_local_experts=int(tc["num_local_experts"]),
            num_experts_per_tok=int(tc["num_experts_per_tok"]),
            n_shared_experts=int(tc.get("n_shared_experts", 0)),
            scoring_func=str(tc.get("scoring_func", "sigmoid")),
            use_routing_bias=bool(tc.get("use_routing_bias", True)),
            norm_topk_prob=bool(tc.get("norm_topk_prob", True)),
            routed_scaling_factor=float(tc.get("routed_scaling_factor", 1.0)),
            moe_layer_freq=moe_freq,
            hidden_act=str(tc.get("hidden_act", "swigluoai")),
            swiglu_alpha=float(tc.get("swiglu_alpha", 1.702)),
            swiglu_limit=float(tc.get("swiglu_limit", 7.0)),
            use_sparse_attention=bool(sac.get("use_sparse_attention", False)),
            sparse_topk_blocks=int(sac.get("sparse_topk_blocks", 0)),
            sparse_block_size=int(sac.get("sparse_block_size", 128)),
            sparse_num_index_heads=int(sac.get("sparse_num_index_heads", 0)),
            sparse_index_dim=int(sac.get("sparse_index_dim", 0)),
            sparse_init_block=int(sac.get("sparse_init_block", 0)),
            sparse_local_block=int(sac.get("sparse_local_block", 0)),
            sparse_score_type=str(sac.get("sparse_score_type", "max")),
            sparse_attention_freq=sparse_freq,
            num_mtp_modules=mtp_eff,
            num_mtp_modules_declared=declared_mtp,
            num_nextn_predict_layers=int(tc.get("num_nextn_predict_layers", 1)),
            norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
            max_position_embeddings=int(tc.get("max_position_embeddings", 1048576)),
            bos_token_id=bos,
            eos_token_id=int(eos_ids[0]),
            eos_token_ids=eos_ids,
            tie_word_embeddings=bool(tc.get("tie_word_embeddings", False)),
            image_token_index=int(cfg.get("image_token_index", 200025)),
            video_token_index=int(cfg.get("video_token_index", 200026)),
            image_seq_length=int(cfg.get("image_seq_length", 576)),
            vision_feature_layer=int(cfg.get("vision_feature_layer", -1)),
            vision_feature_select_strategy=str(cfg.get("vision_feature_select_strategy", "full")),
            projector_hidden_act=str(cfg.get("projector_hidden_act", "gelu")),
            multimodal_projector_bias=bool(cfg.get("multimodal_projector_bias", True)),
            vision=vision,
            sparse_attention_config=sac,
        )
