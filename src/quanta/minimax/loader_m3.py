"""Streamed bf16 source loader for MiniMax-M3-VL — the TEXT decoder, one layer at a time (rule 8).

The checkpoint is plain bf16 (809.5 GiB / 59 shards / 23,416 tensors). Per-kind accessors stream
**one layer's** tensors from the sharded safetensors via ``model.safetensors.index.json``; ``mx.load``
memory-maps the shard and only the requested tensor materializes on ``eval``, so a consumer (the
fp32 reference forward, or the M2 bake) never holds more than a layer's source weights resident.

Two layout facts make this loader different from its Nex/qwen35 sibling
(:class:`quanta.qwen35.loader.Qwen35SourceCheckpoint`):

* **Routed experts ship PER-EXPERT** (``block_sparse_moe.experts.{e}.{w1,w2,w3}.weight``), not
  pre-stacked. :meth:`moe` pre-stacks them into the ``[E, 2*inter, hidden]`` (w1 over w3) and
  ``[E, hidden, inter]`` (w2) ``mx.gather_mm``/``gather_qmm``-ready layout that
  :meth:`quanta.minimax.model_m3.MiniMaxM3MoE.set_experts` expects. The 128-expert stacking is a
  bounded, non-hot **load-time** loop (rule 3 permits coarse load/IO loops) over lazy mmap views —
  no tensor materializes until the caller evals.
* **The MoE layers (3–59) carry the native trained block-sparse indexer**
  (``self_attn.index_{q,k}_proj`` + ``index_{q,k}_norm``). :meth:`sparse_index` streams them; the
  short-context parity forward leaves them inert (top-k blocks == all blocks at
  ``T <= sparse_topk_blocks*sparse_block_size`` ⇒ sparse == dense), but loading them keeps rule-6
  coverage honest (every layer tensor has an accessor — none is silently dropped).

This loader bakes the **text decoder only** (``language_model.model.*``; ``lm_head`` sits at
``language_model.lm_head.weight``, not under ``language_model.model.``). The **vision tower**
(``vision_tower.*`` / ``multi_modal_projector.*`` / ``patch_merge_mlp.*``) is part of the full-VL
build but a SEPARATE track with its own loader; the text accessors here **refuse** a vision key
(rule 6 — reaching for one through a text accessor is a bug, not a fallback), they do not drop it.

Tensors come back in their **native source dtype** verbatim: most are BF16, but the router
``gate.weight`` and ``e_score_correction_bias`` are **F32** in the checkpoint (routing precision) and
pass through unchanged. The caller folds Gemma ``(1+w)`` on the norms and casts to fp32/quantizes as
it sees fit — this loader never alters values.

**Safety:** the accessor *code* mmaps and reads real tensors, so this is a SOLO / real-weight path —
the model-free layer gate (``parity/minimax_m3_layer_test.py``) exercises the math on synthetic
weights and never touches disk; the real-weight at-scale gate is ``parity/minimax_m3_layer_parity.py``
(loads L0 + L3 only, streamed + released).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.quant_policy_m3 import (
    EMBED_KEY,
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    LM_PREFIX,
    VISION_PREFIXES,
)

# --- per-kind suffix sets (declarative so the bake / parity gate can mirror them) -------------------
ATTN_SUFFIXES: tuple[str, ...] = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "q_norm.weight",
    "k_norm.weight",
)
SPARSE_INDEX_SUFFIXES: tuple[str, ...] = (
    "index_q_proj.weight",
    "index_k_proj.weight",
    "index_q_norm.weight",
    "index_k_norm.weight",
)
DENSE_MLP_PROJS: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
SHARED_EXPERT_PROJS: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
EXPERT_WEIGHTS: tuple[str, ...] = ("w1", "w2", "w3")  # w1=gate, w3=up, w2=down


class MiniMaxM3SourceCheckpoint:
    """Lazy, sharded reader for a MiniMax-M3-VL bf16 checkpoint directory (text decoder only)."""

    def __init__(self, model_dir: str | Path, cfg: MiniMaxM3Config | None = None) -> None:
        self.dir = Path(model_dir)
        self.cfg = cfg if cfg is not None else MiniMaxM3Config.from_pretrained(self.dir)
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self._wm: dict[str, str] = index["weight_map"]
        self._cache_file: str | None = None      # single-entry shard mmap cache
        self._cache: dict[str, mx.array] = {}

    @property
    def num_layers(self) -> int:
        return self.cfg.num_hidden_layers

    # ---- tensor access ------------------------------------------------------
    def _tensor(self, key: str) -> mx.array:
        """The tensor for ``key``, streamed from its shard. Fails loud (rule 6) on an absent key or a
        vision-tower key (the vision tower is a separate track — reaching for it through a text
        accessor is a bug, never a silent wrong tensor)."""
        if key.startswith(VISION_PREFIXES):
            raise KeyError(f"{key!r} is a vision-tower tensor; the M3 text loader is "
                           f"language-model-only (the ViT has its own loader in the VL track)")
        shard = self._wm.get(key)
        if shard is None:
            raise KeyError(f"{key!r} not in MiniMax-M3 weight_map ({self.dir})")
        if shard != self._cache_file:
            self._cache = mx.load(str(self.dir / shard))
            self._cache_file = shard
        return self._cache[key]

    def has(self, key: str) -> bool:
        return key in self._wm

    def release(self) -> None:
        """Drop the cached shard mmap so a layer's source tensors can be freed (rule 8: one text
        layer resident at a time). The next ``_tensor`` access re-mmaps its shard lazily."""
        self._cache = {}
        self._cache_file = None

    # ---- top-level ----------------------------------------------------------
    def embed(self) -> mx.array:
        return self._tensor(EMBED_KEY)

    def final_norm(self) -> mx.array:
        return self._tensor(FINAL_NORM_KEY)

    def lm_head(self) -> mx.array:
        """Output projection (``tie_word_embeddings=False`` ⇒ separate from the embedding)."""
        return self._tensor(EMBED_KEY if self.cfg.tie_word_embeddings else LM_HEAD_KEY)

    # ---- per-layer kinds ----------------------------------------------------
    def block_norms(self, i: int) -> dict[str, mx.array]:
        """The two pre-norms present on every decoder layer (raw weights — caller folds ``(1+w)``)."""
        p = f"{LM_PREFIX}layers.{i}."
        return {
            "input_layernorm": self._tensor(p + "input_layernorm.weight"),
            "post_attention_layernorm": self._tensor(p + "post_attention_layernorm.weight"),
        }

    def attention(self, i: int) -> dict[str, mx.array]:
        """GQA q/k/v/o projections + per-head q/k RMSNorm — present on every layer. Raw weights; the
        caller folds Gemma ``(1+w)`` on q_norm/k_norm (rule: ``(1+w)`` applies to all non-gated
        norms)."""
        p = f"{LM_PREFIX}layers.{i}.self_attn."
        return {suffix: self._tensor(p + suffix) for suffix in ATTN_SUFFIXES}

    def sparse_index(self, i: int) -> dict[str, mx.array]:
        """The trained block-sparse indexer (``index_{q,k}_proj`` + ``index_{q,k}_norm``) for a
        sparse-attention layer (3–59). Refuses on a dense layer (rule 6 — those keys do not exist)."""
        if not self.cfg.is_sparse_attention_layer(i):
            raise ValueError(f"layer {i} carries no trained sparse indexer "
                             f"(is_sparse_attention_layer is False)")
        p = f"{LM_PREFIX}layers.{i}.self_attn."
        return {suffix: self._tensor(p + suffix) for suffix in SPARSE_INDEX_SUFFIXES}

    def dense_mlp(self, i: int) -> dict[str, mx.array]:
        """Dense feed-forward (layers 0–2): gate/up/down projections (width
        ``dense_intermediate_size``). Refuses on a MoE layer (rule 6)."""
        if not self.cfg.is_dense_layer(i):
            raise ValueError(f"layer {i} is a MoE layer; use moe()")
        p = f"{LM_PREFIX}layers.{i}.mlp."
        return {proj: self._tensor(p + f"{proj}.weight") for proj in DENSE_MLP_PROJS}

    def moe(self, i: int) -> dict[str, mx.array]:
        """MoE block (layers 3–59): router (``gate`` F32 + ``e_score_correction_bias`` F32), the
        shared expert, and the 128 routed experts **pre-stacked** into the gather-ready layout.

        Routed experts ship per-expert (``experts.{e}.{w1,w2,w3}``), so they are stacked here at load
        time (bounded loop, rule 3): ``experts_gate_up`` ``[E, 2*inter, hidden]`` = per expert
        ``concat([w1, w3], axis=0)`` (gate over up), ``experts_down`` ``[E, hidden, inter]`` = ``w2``.
        The stack references lazy mmap views; nothing materializes until the caller evals. Refuses on
        a dense layer (rule 6)."""
        if not self.cfg.is_moe_layer(i):
            raise ValueError(f"layer {i} is a dense layer; use dense_mlp()")
        mp = f"{LM_PREFIX}layers.{i}.block_sparse_moe."
        out: dict[str, mx.array] = {
            "gate": self._tensor(mp + "gate.weight"),                       # [E, hidden] F32
            "e_score_correction_bias": self._tensor(mp + "e_score_correction_bias"),  # [E] F32
        }
        sp = mp + "shared_experts."
        for proj in SHARED_EXPERT_PROJS:
            out[f"shared_{proj}"] = self._tensor(sp + f"{proj}.weight")
        ep = mp + "experts."
        e = self.cfg.num_local_experts
        gate_up = [mx.concatenate([self._tensor(f"{ep}{j}.w1.weight"),
                                   self._tensor(f"{ep}{j}.w3.weight")], axis=0) for j in range(e)]
        down = [self._tensor(f"{ep}{j}.w2.weight") for j in range(e)]
        out["experts_gate_up"] = mx.stack(gate_up, axis=0)                  # [E, 2*inter, hidden]
        out["experts_down"] = mx.stack(down, axis=0)                        # [E, hidden, inter]
        return out
