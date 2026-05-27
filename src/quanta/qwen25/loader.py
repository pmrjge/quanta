"""Streamed bf16 source loader for Qwen2.5-14B-Instruct-1M (``qwen2``).

The checkpoint is **plain bf16** (no ``quantization_config``), ~28 GB across 8 shards, so — like the
GLM/Qwen3.5 loaders and unlike the Kimi (int4) / DSV4 (fp4/fp8) loaders — there is **no dequant**.
Accessors just stream the needed tensors from the sharded safetensors via
``model.safetensors.index.json``; ``mx.load`` memory-maps the shard and only the requested tensor
materializes on ``eval``. Per-kind accessors hand back **one layer's** params at a time so a consumer
(the bf16 reference forward or the bake) never holds more than a layer resident (rule 8).

The accessor surface mirrors :class:`quanta.qwen35.loader.Qwen35SourceCheckpoint` minus everything
Qwen2.5 lacks (MoE, MTP, linear-attention, QK-norm): ``embed`` / ``final_norm`` / ``lm_head`` plus
per-layer ``block_norms`` / ``attention`` (q/k/v/o + QKV biases — Qwen2 specific) / ``mlp``
(gate/up/down SwiGLU). No expert stacks, no MTP block — there is no source for them.

Tensors come back in their **native dtype** verbatim (all bf16 in this checkpoint); never silently
downcast (rule 6).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from quanta.qwen25.config import Qwen25Config

# --- top-level text tensors (Qwen2 is flat — no `model.language_model.` namespace) -------------------
MODEL_PREFIX = "model."
EMBED_KEY = MODEL_PREFIX + "embed_tokens.weight"
FINAL_NORM_KEY = MODEL_PREFIX + "norm.weight"
LM_HEAD_KEY = "lm_head.weight"

# --- per-kind suffix sets (declarative — the bake's policy partition mirrors these) ------------------
# Attention weights (matmul) — int8-quantizable, 2-D ``[out, in]``.
ATTN_WEIGHT_SUFFIXES: tuple[str, ...] = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
)
# Attention biases (Qwen2 specific — Qwen3 drops them). Stored bf16 verbatim; never quantized.
ATTN_BIAS_SUFFIXES: tuple[str, ...] = (
    "q_proj.bias",
    "k_proj.bias",
    "v_proj.bias",
)
# SwiGLU FFN — int4-quantizable (dominates byte count; bf16 source tolerates int4 g64 well).
MLP_SUFFIXES: tuple[str, ...] = (
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)


class Qwen25SourceCheckpoint:
    """Lazy, sharded reader for a Qwen2.5-14B-Instruct-1M bf16 checkpoint directory."""

    def __init__(self, model_dir: str | Path, cfg: Qwen25Config | None = None) -> None:
        self.dir = Path(model_dir)
        self.cfg = cfg if cfg is not None else Qwen25Config.from_pretrained(self.dir)
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self._wm: dict[str, str] = index["weight_map"]
        self._cache_file: str | None = None      # single-entry shard mmap cache
        self._cache: dict[str, mx.array] = {}

    @property
    def num_layers(self) -> int:
        return self.cfg.num_hidden_layers

    # ---- tensor access ------------------------------------------------------
    def _tensor(self, key: str) -> mx.array:
        """The tensor for ``key``, streamed from its shard (fails loud if absent — rule 6)."""
        shard = self._wm.get(key)
        if shard is None:
            raise KeyError(f"{key!r} not in Qwen2.5 weight_map ({self.dir})")
        if shard != self._cache_file:
            self._cache = mx.load(str(self.dir / shard))
            self._cache_file = shard
        return self._cache[key]

    def has(self, key: str) -> bool:
        return key in self._wm

    def release(self) -> None:
        """Drop the current shard handle so its mmap can be released."""
        self._cache = {}
        self._cache_file = None

    # ---- top-level ----------------------------------------------------------
    def embed(self) -> mx.array:
        return self._tensor(EMBED_KEY)

    def final_norm(self) -> mx.array:
        return self._tensor(FINAL_NORM_KEY)

    def lm_head(self) -> mx.array:
        """Output projection — separate from the embedding for Qwen2.5 (``tie_word_embeddings=False``)."""
        key = EMBED_KEY if self.cfg.tie_word_embeddings else LM_HEAD_KEY
        return self._tensor(key)

    # ---- per-layer kinds ----------------------------------------------------
    def block_norms(self, i: int) -> dict[str, mx.array]:
        """The two RMSNorms on every layer."""
        p = f"{MODEL_PREFIX}layers.{i}."
        return {
            "input_layernorm": self._tensor(p + "input_layernorm.weight"),
            "post_attention_layernorm": self._tensor(p + "post_attention_layernorm.weight"),
        }

    def attention(self, i: int) -> dict[str, mx.array]:
        """GQA attention tensors for layer ``i``: q/k/v/o weights + q/k/v biases (Qwen2 quirk).

        ``o_proj`` has **no bias** in Qwen2 (verified empirically from the safetensors index);
        only q/k/v carry biases. Returned as a suffix-keyed dict so the bake's policy table can
        partition the weight (int8) vs bias (bf16) entries by suffix.
        """
        p = f"{MODEL_PREFIX}layers.{i}.self_attn."
        out: dict[str, mx.array] = {s: self._tensor(p + s) for s in ATTN_WEIGHT_SUFFIXES}
        if self.cfg.attention_bias:
            for s in ATTN_BIAS_SUFFIXES:
                out[s] = self._tensor(p + s)
        return out

    def mlp(self, i: int) -> dict[str, mx.array]:
        """SwiGLU FFN tensors for layer ``i``: gate_proj / up_proj / down_proj."""
        p = f"{MODEL_PREFIX}layers.{i}.mlp."
        return {s: self._tensor(p + s) for s in MLP_SUFFIXES}
