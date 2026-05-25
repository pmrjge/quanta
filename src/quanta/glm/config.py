"""GLM-5.1 (``model_type="glm_moe_dsa"``, ``GlmMoeDsaForCausalLM``) hyperparameters.

GLM-5.1 is a DeepSeek-V3.2-style model: **MLA** attention (low-rank q/kv via ``q_a/q_b`` and
``kv_a_proj_with_mqa``/``kv_b``) with a **DSA Lightning-Indexer** (top-``index_topk`` sparse attention),
**noaux_tc sigmoid MoE** (256 routed experts, top-8, 1 shared, ``e_score_correction_bias``,
``routed_scaling_factor``), the first ``first_k_dense_replace`` layers dense, and **one native MTP
head** (``num_nextn_predict_layers == 1``). RoPE is partial (only ``qk_rope_head_dim`` of the head is
rotated) and **interleaved** (``rope_interleave``) with ``rope_theta`` and no YaRN.

This dataclass is the self-contained config the GLM loader / forward / bake consume — it mirrors
:class:`quanta.dsv4.config.DeepSeekV4Config` and the Kimi config. All fields are read straight from the
checkpoint's ``config.json`` (``from_pretrained`` / ``from_dict``); nothing is hand-transcribed, so a
config change in a re-release is picked up automatically. The component dims are cross-checked against
the empirically-confirmed tensor shapes in ``parity/glm_config_test.py``.

Source is **plain bf16** (no ``quantization_config``); the quanta bake target is int4-AWQ **g64**
routed experts + int8 non-experts (≈403 GiB, under the 490.4 GiB ceiling — see the bake unit).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GLMConfig:
    """DeepSeek-V3.2-style GLM-5.1 config (MLA + DSA indexer + noaux_tc MoE + 1 MTP head)."""

    # --- core ---
    vocab_size: int = 154880
    hidden_size: int = 6144
    intermediate_size: int = 12288            # dense FFN (first_k_dense_replace layers)
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    attention_bias: bool = False

    # --- MLA (multi-head latent attention) ---
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    qk_head_dim: int = 256                     # == qk_nope_head_dim + qk_rope_head_dim
    v_head_dim: int = 256

    # --- DSA Lightning Indexer (sparse attention selector) ---
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_topk: int = 2048
    indexer_rope_interleave: bool = True

    # --- MoE ---
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    moe_intermediate_size: int = 2048
    first_k_dense_replace: int = 3            # layers [0, first_k_dense_replace) are dense FFN
    moe_layer_freq: int = 1
    n_group: int = 1
    topk_group: int = 1
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True

    # --- MTP (native multi-token-prediction head) ---
    num_nextn_predict_layers: int = 1

    # --- RoPE (partial, interleaved, no YaRN) ---
    rope_theta: float = 1000000.0
    rope_interleave: bool = True
    max_position_embeddings: int = 202752

    # --- tokens ---
    bos_token_id: int | None = None
    eos_token_id: tuple[int, ...] = (154820, 154827, 154829)
    pad_token_id: int = 154820

    model_type: str = "glm_moe_dsa"

    # ---- constructors -------------------------------------------------------
    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "GLMConfig":
        """Build from a parsed ``config.json`` dict, tolerating a ``text_config`` wrapper and ignoring
        unknown keys (so a transformers re-release with extra fields still loads)."""
        c = dict(cfg.get("text_config", cfg))
        rope = c.get("rope_parameters") or c.get("rope_scaling") or {}
        eos = c.get("eos_token_id")
        eos_t = tuple(eos) if isinstance(eos, (list, tuple)) else ((eos,) if eos is not None else ())
        known = {f for f in cls.__dataclass_fields__}
        kw: dict[str, Any] = {k: c[k] for k in c if k in known}
        kw["eos_token_id"] = eos_t
        if "rope_theta" not in c and "rope_theta" in rope:
            kw["rope_theta"] = float(rope["rope_theta"])
        return cls(**kw)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "GLMConfig":
        """Load from a checkpoint directory (or a direct path to ``config.json``)."""
        p = Path(path)
        if p.is_dir():
            p = p / "config.json"
        return cls.from_dict(json.loads(p.read_text()))

    # ---- derived helpers ----------------------------------------------------
    def is_dense_layer(self, layer_id: int) -> bool:
        """The first ``first_k_dense_replace`` layers use a dense FFN; the rest use the MoE."""
        return layer_id < self.first_k_dense_replace

    def is_moe_layer(self, layer_id: int) -> bool:
        return not self.is_dense_layer(layer_id)

    @property
    def mtp_layer_id(self) -> int:
        """Index of the native MTP block in the checkpoint (it follows the main stack)."""
        return self.num_hidden_layers  # layer 78 for the 78-layer (0..77) main stack

    @property
    def softmax_scale(self) -> float:
        """MLA attention scale ``1/sqrt(qk_head_dim)`` — no YaRN ``mscale`` (rope_type='default')."""
        return self.qk_head_dim ** -0.5

    @property
    def head_dim(self) -> int:
        return self.qk_head_dim

    def __post_init__(self) -> None:
        # fail loud on an inconsistent config rather than silently mis-shaping the forward (rule 6)
        if self.qk_head_dim != self.qk_nope_head_dim + self.qk_rope_head_dim:
            raise ValueError(
                f"qk_head_dim {self.qk_head_dim} != qk_nope {self.qk_nope_head_dim} + "
                f"qk_rope {self.qk_rope_head_dim}")
        if self.num_nextn_predict_layers != 1:
            raise ValueError(
                f"GLM-5.1 ships exactly 1 MTP head; got {self.num_nextn_predict_layers}")
