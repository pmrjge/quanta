"""Streamed bf16 source loader for Qwen3.5-397B-A17B (``qwen3_5_moe``).

The checkpoint is **plain bf16** (no ``quantization_config``), ~775 GB across 94 shards, so — like the
GLM-5.1 loader and unlike the Kimi (int4) / DSV4 (fp4/fp8/e8m0) loaders — there is **no dequant**.
Accessors just stream the needed tensors from the sharded safetensors via ``model.safetensors.index.json``;
``mx.load`` memory-maps the shard and only the requested tensor materializes on ``eval``. The per-kind
accessors hand back **one layer's** params at a time so a consumer (the bf16 reference forward or the bake)
never holds more than a layer resident (rule 8).

This is a **multimodal** checkpoint: the text decoder lives under ``model.language_model.*`` and a ViT
under ``model.visual.*``. This loader bakes the **language model only** — it **never references a single
``model.visual.*`` key** (the vision tower is a deferred stage). ``lm_head`` is the one text tensor that
sits at the top level (``lm_head.weight``), not under ``model.language_model.``.

The accessor surface mirrors :class:`quanta.glm.loader.GLMSourceCheckpoint` /
:class:`quanta.dsv4.loader.DeepSeekV4SourceCheckpoint` so the Qwen3.5 bake consumes it the same way:
``embed`` / ``final_norm`` / ``lm_head`` plus per-layer ``block_norms`` / ``linear_attn`` (Gated
DeltaNet) **or** ``full_attn`` (gated GQA) chosen by :meth:`Qwen35Config.is_linear_attention` /
:meth:`~Qwen35Config.is_full_attention`, ``moe`` (router + pre-stacked routed experts + shared expert),
and the native ``mtp`` block.

Layout note — routed experts are stored **pre-stacked + gate/up-fused 3-D** on **every** MoE block,
the main decoder and the native MTP block alike (``mlp.experts.gate_up_proj`` ``[E, 2*moe_inter,
hidden]`` and ``mlp.experts.down_proj`` ``[E, hidden, moe_inter]`` — already ``gather_qmm``-ready), so
this loader never stacks experts itself. (Verified empirically on Qwen3.6-35B-A3B: the only
``experts.*`` children across the whole checkpoint are ``gate_up_proj`` / ``down_proj``, on all 40
decoder layers + the 1 MTP layer; no per-expert ``experts.{e}.*`` keys exist.)

Tensors are returned in their **native source dtype** verbatim: most weights are BF16, but the Gated
DeltaNet ``A_log`` and per-head ``norm.weight`` are F32 in the checkpoint (SSM state precision) — they
pass through unchanged (no cast), so the bf16 reference / bake see the source precision exactly.

**Safety:** although the loader *code* memory-maps and reads tensors, the heavy real-checkpoint load is
**deferred to a future GPU session** — the model-free key/shape gate
(``parity/qwen35_loader_keys_test.py``) exercises only the index ``weight_map`` and safetensors headers
and **never materializes a tensor**.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from quanta.qwen35.config import Qwen35Config

# --- top-level text tensors (the only text tensor outside model.language_model.* is lm_head) ---------
LM_PREFIX = "model.language_model."
EMBED_KEY = LM_PREFIX + "embed_tokens.weight"
FINAL_NORM_KEY = LM_PREFIX + "norm.weight"
LM_HEAD_KEY = "lm_head.weight"

# --- per-kind suffix sets (keep enumeration declarative so the gate can mirror it) ------------------
LINEAR_ATTN_SUFFIXES: tuple[str, ...] = (
    "in_proj_qkv.weight",
    "in_proj_a.weight",
    "in_proj_b.weight",
    "in_proj_z.weight",
    "conv1d.weight",
    "A_log",
    "dt_bias",
    "norm.weight",
    "out_proj.weight",
)
FULL_ATTN_SUFFIXES: tuple[str, ...] = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "q_norm.weight",
    "k_norm.weight",
)
SHARED_EXPERT_PROJS: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")


class Qwen35SourceCheckpoint:
    """Lazy, sharded reader for a Qwen3.5-397B-A17B bf16 checkpoint directory (text decoder only)."""

    def __init__(self, model_dir: str | Path, cfg: Qwen35Config | None = None) -> None:
        self.dir = Path(model_dir)
        self.cfg = cfg if cfg is not None else Qwen35Config.from_pretrained(self.dir)
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self._wm: dict[str, str] = index["weight_map"]
        self._cache_file: str | None = None      # single-entry shard mmap cache
        self._cache: dict[str, mx.array] = {}

    @property
    def num_layers(self) -> int:
        return self.cfg.num_hidden_layers

    # ---- tensor access ------------------------------------------------------
    def _tensor(self, key: str) -> mx.array:
        """The tensor for ``key``, streamed from its shard (fails loud if the key is absent — rule 6).

        Refuses any ``model.visual.*`` key: the vision tower is out of scope for this language-model
        loader, so reaching for one is a bug, not a fallback (rule 6 — no silent wrong tensor).
        """
        if key.startswith("model.visual."):
            raise KeyError(f"{key!r} is a vision-tower tensor; Qwen35 loader is language-model-only")
        shard = self._wm.get(key)
        if shard is None:
            raise KeyError(f"{key!r} not in Qwen3.5 weight_map ({self.dir})")
        if shard != self._cache_file:
            self._cache = mx.load(str(self.dir / shard))
            self._cache_file = shard
        return self._cache[key]

    def has(self, key: str) -> bool:
        return key in self._wm

    def release(self) -> None:
        """Drop the cached shard mmap so a layer's source tensors can be freed (rule-8: one text layer
        resident at a time). The next ``_tensor`` access re-mmaps its shard lazily. Mirrors
        :meth:`quanta.qwen35.artifact.Qwen35Artifact.release` so the bake/reference treat the source
        checkpoint and the baked artifact identically."""
        self._cache = {}
        self._cache_file = None

    # ---- top-level ----------------------------------------------------------
    def embed(self) -> mx.array:
        return self._tensor(EMBED_KEY)

    def final_norm(self) -> mx.array:
        return self._tensor(FINAL_NORM_KEY)

    def lm_head(self) -> mx.array:
        """Output projection — separate from the embedding (``tie_word_embeddings=False``).

        ``lm_head`` sits at the top level, NOT under ``model.language_model.``.
        """
        key = EMBED_KEY if self.cfg.tie_word_embeddings else LM_HEAD_KEY
        return self._tensor(key)

    # ---- per-layer kinds ----------------------------------------------------
    def block_norms(self, i: int) -> dict[str, mx.array]:
        """The two RMSNorms present on every layer (linear and full alike)."""
        p = f"{LM_PREFIX}layers.{i}."
        return {
            "input_layernorm": self._tensor(p + "input_layernorm.weight"),
            "post_attention_layernorm": self._tensor(p + "post_attention_layernorm.weight"),
        }

    def linear_attn(self, i: int) -> dict[str, mx.array]:
        """Gated-DeltaNet (linear-attention) tensors for layer ``i``.

        Refuses if layer ``i`` is a full-attention layer (per the config schedule) — fail loud rather
        than read keys that do not exist (rule 6). ``A_log`` and ``norm.weight`` come back F32 (their
        source dtype); everything else is bf16.
        """
        if not self.cfg.is_linear_attention(i):
            raise ValueError(f"layer {i} is not a linear-attention layer "
                             f"({self.cfg.layer_types[i]!r}); use full_attn()")
        p = f"{LM_PREFIX}layers.{i}.linear_attn."
        return {suffix: self._tensor(p + suffix) for suffix in LINEAR_ATTN_SUFFIXES}

    def full_attn(self, i: int) -> dict[str, mx.array]:
        """Gated-GQA (full-attention) tensors for layer ``i`` (q/k/v/o + per-head q/k RMSNorm).

        Refuses if layer ``i`` is a linear-attention layer (per the config schedule) — fail loud
        (rule 6).
        """
        if not self.cfg.is_full_attention(i):
            raise ValueError(f"layer {i} is not a full-attention layer "
                             f"({self.cfg.layer_types[i]!r}); use linear_attn()")
        p = f"{LM_PREFIX}layers.{i}.self_attn."
        return {suffix: self._tensor(p + suffix) for suffix in FULL_ATTN_SUFFIXES}

    def moe(self, i: int) -> dict[str, mx.array]:
        """MoE block for layer ``i`` — present on **every** layer (router + routed + shared).

        Routed experts are loaded **pre-stacked 3D** straight from the checkpoint (no Python stacking):
        ``experts_gate_up`` ``[E, 2*moe_inter, hidden]`` and ``experts_down`` ``[E, hidden, moe_inter]``
        — already in the ``[E, out, in]`` layout ``mx.gather_qmm`` / the quantizer want. That stacked
        pair is the bake's per-layer working set; built one layer at a time and dropped by the caller
        (rule 8).
        """
        p = f"{LM_PREFIX}layers.{i}.mlp."
        out: dict[str, mx.array] = {
            "gate": self._tensor(p + "gate.weight"),
            "experts_gate_up": self._tensor(p + "experts.gate_up_proj"),  # [E, 2*moe_inter, hidden]
            "experts_down": self._tensor(p + "experts.down_proj"),        # [E, hidden, moe_inter]
            "shared_expert_gate": self._tensor(p + "shared_expert_gate.weight"),  # [1, hidden] sigmoid
        }
        sp = p + "shared_expert."
        for proj in SHARED_EXPERT_PROJS:
            out[f"shared_{proj}"] = self._tensor(sp + f"{proj}.weight")
        return out

    # ---- native MTP block ---------------------------------------------------
    def mtp(self, j: int = 0) -> dict[str, mx.array]:
        """The native MTP module: the ``fc`` embed/hidden combine + one full-attn + MoE decoder block.

        ``j`` selects the MTP head; Qwen3.5 has exactly one (``mtp_num_hidden_layers=1``). The MTP
        decoder block is **always full-attention**, and its MoE — exactly like the main decoder — ships
        routed experts **pre-stacked + gate/up-fused** (``mlp.experts.gate_up_proj`` / ``down_proj``),
        so the ``moe`` sub-dict matches :meth:`moe` verbatim (``experts_gate_up`` / ``experts_down``;
        no per-expert python stacking).
        """
        if j != 0:
            raise IndexError(f"Qwen3.5 has {self.cfg.num_mtp_modules} MTP head(s); got j={j}")
        # --- fc combine + its norms (under the top-level mtp.* prefix) ---
        out: dict[str, mx.array] = {
            "fc": self._tensor("mtp.fc.weight"),                              # [hidden, 2*hidden]
            "pre_fc_norm_embedding": self._tensor("mtp.pre_fc_norm_embedding.weight"),
            "pre_fc_norm_hidden": self._tensor("mtp.pre_fc_norm_hidden.weight"),
            "norm": self._tensor("mtp.norm.weight"),
        }
        # --- the inherited full-attn + MoE decoder block (mtp.layers.0.*) ---
        lp = "mtp.layers.0."
        out["input_layernorm"] = self._tensor(lp + "input_layernorm.weight")
        out["post_attention_layernorm"] = self._tensor(lp + "post_attention_layernorm.weight")
        ap = lp + "self_attn."
        out["attention"] = {suffix: self._tensor(ap + suffix) for suffix in FULL_ATTN_SUFFIXES}
        mp = lp + "mlp."
        moe: dict[str, mx.array] = {
            "gate": self._tensor(mp + "gate.weight"),
            "shared_expert_gate": self._tensor(mp + "shared_expert_gate.weight"),
            "experts_gate_up": self._tensor(mp + "experts.gate_up_proj"),  # [E, 2*moe_inter, hidden]
            "experts_down": self._tensor(mp + "experts.down_proj"),        # [E, hidden, moe_inter]
        }
        sp = mp + "shared_expert."
        for proj in SHARED_EXPERT_PROJS:
            moe[f"shared_{proj}"] = self._tensor(sp + f"{proj}.weight")
        out["moe"] = moe
        return out
