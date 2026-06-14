"""Streamed reader over the baked MiniMax-M3-VL quanta artifact — dequantizes to bf16.

Duck-types :class:`quanta.minimax.loader_m3.MiniMaxM3SourceCheckpoint`: it exposes the same
``embed`` / ``final_norm`` / ``lm_head`` / ``block_norms`` / ``attention`` / ``sparse_index`` /
``dense_mlp`` / ``moe`` / ``has`` / ``release`` surface and returns the **same suffix-keyed dicts**,
so the M3 reference forward / ppl harness run against either unchanged. This is the **parity
reference for the M3 quant gate**: the artifact's packed weights dequantized back to bf16 and run
through the proven :mod:`quanta.minimax.model_m3` forward — the float baseline the resident
``gather_qmm`` runtime must then match.

Storage contract (mirrors :class:`quanta.bake.artifact.ArtifactWriter` + :mod:`quanta.minimax.bake_m3`):

* ``dense``         → ``{key}`` verbatim (all RMSNorms; per-head q/k norm; the trained sparse indexer
  ``index_{q,k}_proj/norm``; the router ``gate`` + ``e_score_correction_bias`` — kept f32; embeddings /
  lm_head / final norm; the whole vision tower), cast to bf16 on read (``read``); raw on ``get``.
* ``affine_packed`` → an affine int weight at ``{base}`` (the key **without** ``.weight``), stored as
  ``{base}.weight_packed`` + ``.weight_scale`` + ``.weight_bias``; ``mx.dequantize`` inverts it.
  Covers the int8 non-experts (2-D: GQA q/k/v/o, dense-FFN gate/up/down, shared expert) **and** the
  int6 routed-expert stacks (**pre-stacked 3-D** ``[E, out, in]`` — dequantized over the trailing
  ``in`` dim in one shot at the manifest-recorded width).

One text layer resident at a time (rule 8): per-layer accessors return only that layer's tensors and
shard handles are dropped on :meth:`release`. Unknown / unmapped formats fail loud (rule 6) — never a
silent default, never a wrong-bits dequant. **Text decoder only** (matches loader_m3): a vision read
path is added with the vision track; vision weights are present in the bundle (baked dense) but a
text accessor refuses a vision key.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.loader_m3 import (
    ATTN_SUFFIXES,
    DENSE_MLP_PROJS,
    SHARED_EXPERT_PROJS,
    SPARSE_INDEX_SUFFIXES,
)
from quanta.minimax.quant_policy_m3 import (
    EMBED_KEY,
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    LM_PREFIX,
    VISION_PREFIXES,
)

_WEIGHT_SUFFIX = ".weight"


class MiniMaxM3Artifact:
    """Lazy, streamed, dequantizing reader over a baked MiniMax-M3-VL quanta artifact (text decoder)."""

    def __init__(self, art_dir: str | Path) -> None:
        self.dir = Path(art_dir)
        self.cfg = MiniMaxM3Config.from_pretrained(art_dir)  # self-contained config (native 1M)
        self.weight_map: dict[str, str] = json.loads(
            (self.dir / "model.safetensors.index.json").read_text()
        )["weight_map"]
        man = json.loads((self.dir / "manifest.json").read_text())
        if man.get("format") != "quanta":
            raise ValueError(f"{self.dir}: not a quanta artifact (format={man.get('format')!r})")
        self.manifest: dict[str, dict] = man["tensors"]
        self._shards: dict[str, dict[str, mx.array]] = {}

    @property
    def num_layers(self) -> int:
        return self.cfg.num_hidden_layers

    # --- low-level shard access ------------------------------------------------
    def get(self, name: str) -> mx.array:
        """Load one tensor verbatim by its exact safetensors key (lazy / mmap; shard handles cached)."""
        if name not in self.weight_map:
            raise KeyError(f"tensor not in artifact index: {name}")
        fn = self.weight_map[name]
        shard = self._shards.get(fn)
        if shard is None:
            shard = mx.load(str(self.dir / fn))  # lazy / mmap
            self._shards[fn] = shard
        return shard[name]

    def has(self, key: str) -> bool:
        """True iff a logical key resolves (a dense entry, or a quantized base)."""
        if key in self.weight_map:
            return True
        base = key[: -len(_WEIGHT_SUFFIX)] if key.endswith(_WEIGHT_SUFFIX) else key
        return self.manifest.get(base, {}).get("format") == "affine_packed"

    def release(self) -> None:
        """Drop cached shard handles so materialized tensors can be freed (rule 8)."""
        self._shards.clear()

    # --- manifest resolution ---------------------------------------------------
    def _meta(self, key: str) -> tuple[str, dict]:
        """Resolve a logical key to ``(base, meta)``: a dense entry at the full key, else the
        ``affine_packed`` base (the key minus a trailing ``.weight``). Fail loud if neither (rule 6) —
        ``KeyError`` when no entry exists, ``ValueError`` on a recognized entry with an unknown format
        (never silently guess or dequant at the wrong bits)."""
        meta = self.manifest.get(key)
        if meta is not None and meta.get("format") == "dense":
            return key, meta
        base = key[: -len(_WEIGHT_SUFFIX)] if key.endswith(_WEIGHT_SUFFIX) else key
        bmeta = self.manifest.get(base)
        if bmeta is not None and bmeta.get("format") == "affine_packed":
            return base, bmeta
        for cand, cmeta in ((key, meta), (base, bmeta)):  # entry exists but format unrecognized
            if cmeta is not None:
                raise ValueError(f"{cand}: unknown manifest format {cmeta.get('format')!r} "
                                 f"(refusing to guess / dequant at wrong bits)")
        raise KeyError(f"{key}: not in artifact manifest (no dense entry, no quantized base {base!r})")

    # --- dequant ---------------------------------------------------------------
    def _dequant(self, base: str, meta: dict) -> mx.array:
        """Dequantize a packed weight at ``base`` → bf16 (2-D non-expert or 3-D expert stack)."""
        gs, bits = int(meta["group_size"]), int(meta["bits"])
        w = mx.dequantize(
            self.get(base + ".weight_packed"),
            self.get(base + ".weight_scale"),
            self.get(base + ".weight_bias"),
            group_size=gs,
            bits=bits,
        )
        return w.astype(mx.bfloat16)

    def read(self, key: str) -> mx.array:
        """Return a DEQUANTIZED bf16 weight for a logical ``key`` (dense verbatim→bf16, else dequant).

        Refuses a vision key (rule 6) — text accessor only; the ViT has its own read path (the weights
        ARE in the bundle, baked dense, but reached through the vision track, not here)."""
        if key.startswith(VISION_PREFIXES):
            raise KeyError(f"{key!r} is a vision-tower tensor; the M3 artifact text reader is "
                           f"language-model-only (the ViT has its own read path in the VL track)")
        base, meta = self._meta(key)
        if meta["format"] == "dense":
            return self.get(base).astype(mx.bfloat16)
        return self._dequant(base, meta)

    def raw(self, key: str) -> mx.array:
        """The packed codes verbatim (``{base}.weight_packed``) for the ``gather_qmm`` decode path —
        siblings fetch ``.weight_scale`` / ``.weight_bias`` via :meth:`get`. Fail loud on a dense key
        (it has no packed codes)."""
        base, meta = self._meta(key)
        if meta["format"] != "affine_packed":
            raise ValueError(f"{key}: format {meta['format']!r} has no packed codes (not quantized)")
        return self.get(base + ".weight_packed")

    # --- top-level tensors (same surface as MiniMaxM3SourceCheckpoint) ---------
    def embed(self) -> mx.array:
        return self.read(EMBED_KEY)

    def final_norm(self) -> mx.array:
        return self.read(FINAL_NORM_KEY)

    def lm_head(self) -> mx.array:
        return self.read(EMBED_KEY if self.cfg.tie_word_embeddings else LM_HEAD_KEY)

    # --- per-layer kinds (same suffix-keyed dicts loader_m3 returns) -----------
    def block_norms(self, i: int) -> dict[str, mx.array]:
        p = f"{LM_PREFIX}layers.{i}."
        out = {"input_layernorm": self.read(p + "input_layernorm.weight"),
               "post_attention_layernorm": self.read(p + "post_attention_layernorm.weight")}
        mx.eval(list(out.values()))
        return out

    def attention(self, i: int) -> dict[str, mx.array]:
        """GQA q/k/v/o (int8 dequantized) + per-head q/k norm (bf16). Same keys as loader_m3."""
        p = f"{LM_PREFIX}layers.{i}.self_attn."
        out = {suffix: self.read(p + suffix) for suffix in ATTN_SUFFIXES}
        mx.eval(list(out.values()))
        return out

    def sparse_index(self, i: int) -> dict[str, mx.array]:
        """The trained block-sparse indexer (bf16) for a sparse-attention layer (3–59)."""
        if not self.cfg.is_sparse_attention_layer(i):
            raise ValueError(f"layer {i} carries no trained sparse indexer "
                             f"(is_sparse_attention_layer is False)")
        p = f"{LM_PREFIX}layers.{i}.self_attn."
        out = {suffix: self.read(p + suffix) for suffix in SPARSE_INDEX_SUFFIXES}
        mx.eval(list(out.values()))
        return out

    def dense_mlp(self, i: int) -> dict[str, mx.array]:
        """Dense feed-forward (layers 0–2): gate/up/down (int8 dequantized). Refuses on a MoE layer."""
        if not self.cfg.is_dense_layer(i):
            raise ValueError(f"layer {i} is a MoE layer; use moe()")
        p = f"{LM_PREFIX}layers.{i}.mlp."
        out = {proj: self.read(p + f"{proj}.weight") for proj in DENSE_MLP_PROJS}
        mx.eval(list(out.values()))
        return out

    def moe(self, i: int) -> dict[str, mx.array]:
        """MoE block (layers 3–59). Routed experts come back **pre-stacked 3-D** bf16 (the int6 stacks
        dequantized in one shot); shared expert dequantized (int8); router ``gate`` +
        ``e_score_correction_bias`` bf16 (read-cast from the stored f32) — the exact dict
        :meth:`quanta.minimax.model_m3.MiniMaxM3MoE.set_experts` + the block consume."""
        if not self.cfg.is_moe_layer(i):
            raise ValueError(f"layer {i} is a dense layer; use dense_mlp()")
        mp = f"{LM_PREFIX}layers.{i}.block_sparse_moe."
        # router gate + bias are F32 in the checkpoint (routing precision) — return them at native F32
        # via get() (NOT read(), which would bf16-downcast and could flip a top-k tie ⇒ a different
        # expert), matching loader_m3 verbatim. Everything else here is bf16 in source ⇒ read() is faithful.
        out: dict[str, mx.array] = {
            "gate": self.get(mp + "gate.weight"),                       # F32 verbatim
            "e_score_correction_bias": self.get(mp + "e_score_correction_bias"),  # F32 verbatim
            "experts_gate_up": self.read(mp + "experts.gate_up_proj"),  # [E, 2*moe_inter, hidden]
            "experts_down": self.read(mp + "experts.down_proj"),        # [E, hidden, moe_inter]
        }
        for proj in SHARED_EXPERT_PROJS:
            out[f"shared_{proj}"] = self.read(mp + f"shared_experts.{proj}.weight")
        mx.eval(list(out.values()))
        return out

    def _packed_triplet(self, base: str) -> dict:
        """A routed-expert stack's packed affine triplet for the resident ``gather_qmm`` path —
        ``{packed, scale, bias, group_size, bits}`` held **verbatim** (NO dequant), the decode width
        read from the manifest (rule 6: never a hardcoded width). Fail loud if ``base`` is not an
        ``affine_packed`` weight."""
        meta = self.manifest.get(base)
        if meta is None or meta.get("format") != "affine_packed":
            raise ValueError(f"{base}: not an affine_packed weight (format="
                             f"{None if meta is None else meta.get('format')!r}); cannot pack "
                             f"experts (rule 6)")
        return {"packed": self.get(base + ".weight_packed"),
                "scale": self.get(base + ".weight_scale"),
                "bias": self.get(base + ".weight_bias"),
                "group_size": int(meta["group_size"]),
                "bits": int(meta["bits"])}

    def moe_packed(self, i: int) -> dict:
        """MoE block for layer ``i`` with the routed experts kept **packed int6** (NOT dequantized) —
        the memory-lean sibling of :meth:`moe` for the resident ``mx.gather_qmm`` serving path (M3).
        ``experts_gate_up`` / ``experts_down`` come back as affine-triplet dicts; router + shared
        expert + bias stay bf16/f32 dequantized."""
        if not self.cfg.is_moe_layer(i):
            raise ValueError(f"layer {i} is a dense layer; use dense_mlp()")
        mp = f"{LM_PREFIX}layers.{i}.block_sparse_moe."
        gu = self._packed_triplet(mp + "experts.gate_up_proj")  # [E, 2*moe_inter, hidden] int6
        dn = self._packed_triplet(mp + "experts.down_proj")     # [E, hidden, moe_inter]   int6
        out: dict[str, object] = {
            "gate": self.get(mp + "gate.weight"),                       # F32 verbatim (routing precision)
            "e_score_correction_bias": self.get(mp + "e_score_correction_bias"),  # F32 verbatim
            "experts_gate_up": gu,
            "experts_down": dn,
        }
        for proj in SHARED_EXPERT_PROJS:
            out[f"shared_{proj}"] = self.read(mp + f"shared_experts.{proj}.weight")
        mx.eval([out["gate"], out["e_score_correction_bias"],
                 *(out[f"shared_{proj}"] for proj in SHARED_EXPERT_PROJS),
                 gu["packed"], gu["scale"], gu["bias"], dn["packed"], dn["scale"], dn["bias"]])
        return out
