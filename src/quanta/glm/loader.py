"""Streamed bf16 source loader for GLM-5.1 (``glm_moe_dsa``).

The checkpoint is **plain bf16** (no ``quantization_config``), ~1.5 TB across 282 shards, so unlike the
Kimi (int4) / DSV4 (fp4/fp8/e8m0) loaders there is **no dequant** — accessors just stream the needed
tensors from the sharded safetensors via ``model.safetensors.index.json``. Tensors are returned lazily
(``mx.load`` memory-maps the shard; only the requested tensor materializes on ``eval``), and the
per-kind accessors hand back **one layer's** params at a time so a consumer (the bf16 reference forward
or the bake) never holds more than a layer resident (rule 8). ``expert_stacks`` stacks the 256 routed
experts of one layer into ``[E, out, in]`` for the gather/quantizer; that single stack is the largest
working set and is built + consumed + dropped per layer.

The accessor surface mirrors :class:`quanta.dsv4.loader.DeepSeekV4SourceCheckpoint` (minus the
Hyper-Connections, which GLM does not have) so the GLM bake's ``load_block_params`` consumes it the
same way: ``block_norms`` / ``attention`` (incl. the DSA indexer) / ``moe_router`` / ``shared_expert``
/ ``expert_stacks`` / ``dense_mlp`` / ``mtp`` plus ``embed`` / ``final_norm`` / ``lm_head``. Tensor key
templates are the empirically-confirmed checkpoint names (``parity/glm_loader_test.py`` round-trips
them on a tiny synthetic checkpoint).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from quanta.glm.config import GLMConfig

EMBED_KEY = "model.embed_tokens.weight"
FINAL_NORM_KEY = "model.norm.weight"
LM_HEAD_KEY = "lm_head.weight"


class GLMSourceCheckpoint:
    """Lazy, sharded reader for a GLM-5.1 bf16 checkpoint directory."""

    def __init__(self, model_dir: str | Path, cfg: GLMConfig | None = None) -> None:
        self.dir = Path(model_dir)
        self.cfg = cfg if cfg is not None else GLMConfig.from_pretrained(self.dir)
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self._wm: dict[str, str] = index["weight_map"]
        self._cache_file: str | None = None      # single-entry shard mmap cache
        self._cache: dict[str, mx.array] = {}

    @property
    def num_layers(self) -> int:
        return self.cfg.num_hidden_layers

    # ---- tensor access ------------------------------------------------------
    def _tensor(self, key: str) -> mx.array:
        """The tensor for ``key``, streamed from its shard (fails loud if the key is absent — rule 6)."""
        shard = self._wm.get(key)
        if shard is None:
            raise KeyError(f"{key!r} not in GLM weight_map ({self.dir})")
        if shard != self._cache_file:
            self._cache = mx.load(str(self.dir / shard))
            self._cache_file = shard
        return self._cache[key]

    def has(self, key: str) -> bool:
        return key in self._wm

    # ---- top-level ----------------------------------------------------------
    def embed(self) -> mx.array:
        return self._tensor(EMBED_KEY)

    def final_norm(self) -> mx.array:
        return self._tensor(FINAL_NORM_KEY)

    def lm_head(self) -> mx.array:
        """Output projection — separate from the embedding (``tie_word_embeddings=False``)."""
        key = EMBED_KEY if self.cfg.tie_word_embeddings else LM_HEAD_KEY
        return self._tensor(key)

    # ---- per-layer kinds ----------------------------------------------------
    def block_norms(self, i: int) -> dict[str, mx.array]:
        p = f"model.layers.{i}."
        return {
            "input_layernorm": self._tensor(p + "input_layernorm.weight"),
            "post_attention_layernorm": self._tensor(p + "post_attention_layernorm.weight"),
        }

    def attention(self, i: int) -> dict[str, mx.array]:
        """MLA projections + norms for layer ``i``, with the DSA indexer as a nested sub-dict."""
        p = f"model.layers.{i}.self_attn."
        return {
            "q_a_proj": self._tensor(p + "q_a_proj.weight"),
            "q_a_layernorm": self._tensor(p + "q_a_layernorm.weight"),
            "q_b_proj": self._tensor(p + "q_b_proj.weight"),
            "kv_a_proj_with_mqa": self._tensor(p + "kv_a_proj_with_mqa.weight"),
            "kv_a_layernorm": self._tensor(p + "kv_a_layernorm.weight"),
            "kv_b_proj": self._tensor(p + "kv_b_proj.weight"),
            "o_proj": self._tensor(p + "o_proj.weight"),
            "indexer": self.indexer(i),
        }

    def indexer(self, i: int) -> dict[str, mx.array]:
        """DSA Lightning-Indexer params for layer ``i`` (query from the q-latent, MQA key from hidden)."""
        p = f"model.layers.{i}.self_attn.indexer."
        return {
            "wq_b": self._tensor(p + "wq_b.weight"),
            "wk": self._tensor(p + "wk.weight"),
            "weights_proj": self._tensor(p + "weights_proj.weight"),
            "k_norm_weight": self._tensor(p + "k_norm.weight"),
            "k_norm_bias": self._tensor(p + "k_norm.bias"),
        }

    def moe_router(self, i: int) -> dict[str, mx.array]:
        p = f"model.layers.{i}.mlp.gate."
        return {
            "weight": self._tensor(p + "weight"),
            "e_score_correction_bias": self._tensor(p + "e_score_correction_bias"),  # f32 control
        }

    def shared_expert(self, i: int) -> dict[str, mx.array]:
        p = f"model.layers.{i}.mlp.shared_experts."
        return {proj: self._tensor(f"{p}{proj}.weight") for proj in ("gate_proj", "up_proj", "down_proj")}

    def expert_stacks(self, i: int) -> dict[str, mx.array]:
        """The 256 routed experts of layer ``i`` stacked to ``[E, out, in]`` per projection.

        This is the bake's per-layer working set (the gather/quantizer input); built one layer at a
        time and dropped by the caller (rule 8)."""
        e = self.cfg.n_routed_experts
        p = f"model.layers.{i}.mlp.experts."
        out: dict[str, mx.array] = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            out[proj] = mx.stack([self._tensor(f"{p}{j}.{proj}.weight") for j in range(e)])
        return out

    def dense_mlp(self, i: int) -> dict[str, mx.array]:
        """Dense FFN for a ``first_k_dense_replace`` layer (``i < first_k_dense_replace``)."""
        p = f"model.layers.{i}.mlp."
        return {proj: self._tensor(f"{p}{proj}.weight") for proj in ("gate_proj", "up_proj", "down_proj")}

    def mtp(self, j: int = 0) -> dict[str, mx.array]:
        """The native MTP block (a full decoder layer + the embed/hidden combine).

        ``j`` selects the MTP head; GLM-5.1 has exactly one, at layer index ``num_hidden_layers``.
        """
        if j != 0:
            raise IndexError(f"GLM-5.1 has 1 MTP head; got j={j}")
        i = self.cfg.mtp_layer_id
        p = f"model.layers.{i}."
        return {
            "enorm": self._tensor(p + "enorm.weight"),
            "hnorm": self._tensor(p + "hnorm.weight"),
            "eh_proj": self._tensor(p + "eh_proj.weight"),
            "shared_head_norm": self._tensor(p + "shared_head.norm.weight"),
            "input_layernorm": self._tensor(p + "input_layernorm.weight"),
            "post_attention_layernorm": self._tensor(p + "post_attention_layernorm.weight"),
            "attention": self.attention(i),
            "router": self.moe_router(i),
            "shared": self.shared_expert(i),
            "experts": self.expert_stacks(i),
        }
