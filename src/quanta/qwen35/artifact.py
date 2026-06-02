"""Streamed reader over the baked Qwen3.5-397B-A17B quanta artifact — dequantizes to bf16.

Duck-types :class:`quanta.qwen35.loader.Qwen35SourceCheckpoint`: it exposes the same
``embed`` / ``final_norm`` / ``lm_head`` / ``block_norms`` / ``linear_attn`` / ``full_attn`` /
``moe`` / ``mtp`` / ``has`` / ``release`` surface and returns the **same suffix-keyed dicts**, so
the bf16 reference forward / ppl harness run against it unchanged. This is the **parity reference for
the Qwen3.5 quant gate**: the artifact's packed weights dequantized back to bf16 and run through the
proven naive forward — the float baseline the resident ``gather_qmm`` / ``quantized_matmul`` runtime
must then match.

Storage contract (mirrors :class:`quanta.bake.artifact.ArtifactWriter` + :mod:`quanta.qwen35.bake`):

* ``dense``         → ``{key}`` verbatim (SSM control ``A_log`` / ``dt_bias`` / ``conv1d`` / DeltaNet
  ``norm``; all RMSNorms; router ``gate``; ``shared_expert_gate``; embeddings / lm_head / final norm;
  MTP ``fc`` + pre-norms), cast to bf16 on read.
* ``affine_packed`` → an affine int weight at ``{base}`` (the key **without** ``.weight``), stored as
  ``{base}.weight_packed`` + ``.weight_scale`` + ``.weight_bias``; ``mx.dequantize`` inverts it.
  Covers both the int8 non-experts (2-D) **and** the int4 routed-expert stacks (**pre-stacked 3-D**
  ``[E, out, in]`` — quantized/dequantized over the trailing ``in`` dim in one shot).

One layer resident at a time (rule-8): per-layer accessors return only that layer's tensors, and the
shard handles are dropped on :meth:`release`. Unknown / unmapped formats fail loud (rule-6) — never a
silent default, never a wrong-bits dequant.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.loader import (
    EMBED_KEY,
    FINAL_NORM_KEY,
    FULL_ATTN_SUFFIXES,
    LINEAR_ATTN_SUFFIXES,
    LM_HEAD_KEY,
    LM_PREFIX,
    SHARED_EXPERT_PROJS,
)

_WEIGHT_SUFFIX = ".weight"


class Qwen35Artifact:
    """Lazy, streamed, dequantizing reader over a baked Qwen3.5-397B-A17B quanta artifact."""

    def __init__(self, art_dir: str | Path) -> None:
        self.dir = Path(art_dir)
        self.cfg = Qwen35Config.from_pretrained(art_dir)  # self-contained config (incl. baked YaRN)
        self.weight_map: dict[str, str] = json.loads(
            (self.dir / "model.safetensors.index.json").read_text()
        )["weight_map"]
        man = json.loads((self.dir / "manifest.json").read_text())
        if man.get("format") != "quanta":
            raise ValueError(f"{self.dir}: not a quanta artifact (format={man.get('format')!r})")
        self.manifest: dict[str, dict] = man["tensors"]
        self._shards: dict[str, dict[str, mx.array]] = {}

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
        """Drop cached shard handles so materialized tensors can be freed."""
        self._shards.clear()

    # --- manifest resolution ---------------------------------------------------
    def _meta(self, key: str) -> tuple[str, dict]:
        """Resolve a logical key to ``(base, meta)``: a dense entry at the full key, else the
        ``affine_packed`` base (the key minus a trailing ``.weight``). Fail loud if neither
        (rule-6) — ``KeyError`` when no manifest entry exists, ``ValueError`` on a recognized entry
        with an unknown format (never silently guess or dequant at the wrong bits)."""
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
        """Return a DEQUANTIZED bf16 weight for a logical ``key`` (dense verbatim, else dequant)."""
        base, meta = self._meta(key)
        if meta["format"] == "dense":
            return self.get(base).astype(mx.bfloat16)
        return self._dequant(base, meta)

    def raw(self, key: str) -> mx.array:
        """Return the packed codes verbatim (``{base}.weight_packed``) for the ``gather_qmm`` /
        ``quantized_matmul`` decode path — siblings fetch ``.weight_scale`` / ``.weight_bias`` via
        :meth:`get`. Fail loud on a dense key (it has no packed codes)."""
        base, meta = self._meta(key)
        if meta["format"] != "affine_packed":
            raise ValueError(f"{key}: format {meta['format']!r} has no packed codes (not quantized)")
        return self.get(base + ".weight_packed")

    # --- top-level tensors (same surface as Qwen35SourceCheckpoint) ------------
    def embed(self) -> mx.array:
        return self.read(EMBED_KEY)

    def final_norm(self) -> mx.array:
        return self.read(FINAL_NORM_KEY)

    def lm_head(self) -> mx.array:
        return self.read(EMBED_KEY if self.cfg.tie_word_embeddings else LM_HEAD_KEY)

    # --- per-layer kinds -------------------------------------------------------
    def block_norms(self, i: int) -> dict[str, mx.array]:
        p = f"{LM_PREFIX}layers.{i}."
        out = {"input_layernorm": self.read(p + "input_layernorm.weight"),
               "post_attention_layernorm": self.read(p + "post_attention_layernorm.weight")}
        mx.eval(list(out.values()))
        return out

    def linear_attn(self, i: int) -> dict[str, mx.array]:
        """Gated-DeltaNet tensors for layer ``i`` (int8 projections dequantized; SSM control bf16)."""
        if not self.cfg.is_linear_attention(i):
            raise ValueError(f"layer {i} is not a linear-attention layer "
                             f"({self.cfg.layer_types[i]!r}); use full_attn()")
        p = f"{LM_PREFIX}layers.{i}.linear_attn."
        out = {suffix: self.read(p + suffix) for suffix in LINEAR_ATTN_SUFFIXES}
        mx.eval(list(out.values()))
        return out

    def full_attn(self, i: int) -> dict[str, mx.array]:
        """Gated-GQA tensors for layer ``i`` (int8 q/k/v/o dequantized; per-head q/k norm bf16)."""
        if not self.cfg.is_full_attention(i):
            raise ValueError(f"layer {i} is not a full-attention layer "
                             f"({self.cfg.layer_types[i]!r}); use linear_attn()")
        p = f"{LM_PREFIX}layers.{i}.self_attn."
        out = {suffix: self.read(p + suffix) for suffix in FULL_ATTN_SUFFIXES}
        mx.eval(list(out.values()))
        return out

    def moe(self, i: int) -> dict[str, mx.array]:
        """MoE block for layer ``i`` (every layer). Routed experts come back **pre-stacked 3-D** bf16
        (the int4 stacks dequantized in one shot); shared expert dequantized; router gate +
        shared-gate bf16 — the exact dict :func:`quanta.qwen35.moe.qwen35_moe` consumes."""
        p = f"{LM_PREFIX}layers.{i}.mlp."
        out: dict[str, mx.array] = {
            "gate": self.read(p + "gate.weight"),
            "experts_gate_up": self.read(p + "experts.gate_up_proj"),  # [E, 2*moe_inter, hidden]
            "experts_down": self.read(p + "experts.down_proj"),        # [E, hidden, moe_inter]
            "shared_expert_gate": self.read(p + "shared_expert_gate.weight"),
        }
        for proj in SHARED_EXPERT_PROJS:
            out[f"shared_{proj}"] = self.read(p + f"shared_expert.{proj}.weight")
        mx.eval(list(out.values()))
        return out

    def _packed_triplet(self, base: str) -> dict:
        """A routed-expert stack's packed affine triplet for the resident ``gather_qmm`` path —
        ``{packed, scale, bias, group_size, bits}`` held **verbatim** (NO dequant), the decode width
        read from the manifest (rule-6: the baked manifest is the single source of truth, never a
        hardcoded width). Fail loud if ``base`` is not an ``affine_packed`` weight (it has no packed
        codes). The dict shape :meth:`quanta.qwen35.model.Qwen35MoEModule.set_experts_packed` consumes;
        the analogue of :func:`quanta.qwen35.runtime._load_quant_triplet` for the 3-D expert stacks."""
        meta = self.manifest.get(base)
        if meta is None or meta.get("format") != "affine_packed":
            raise ValueError(f"{base}: not an affine_packed weight (format="
                             f"{None if meta is None else meta.get('format')!r}); cannot pack "
                             f"experts (rule-6)")
        return {"packed": self.get(base + ".weight_packed"),
                "scale": self.get(base + ".weight_scale"),
                "bias": self.get(base + ".weight_bias"),
                "group_size": int(meta["group_size"]),
                "bits": int(meta["bits"])}

    def moe_packed(self, i: int) -> dict:
        """MoE block for layer ``i`` with the routed experts kept **packed int4** (NOT dequantized).

        The memory-lean sibling of :meth:`moe`: ``experts_gate_up`` / ``experts_down`` come back as
        affine triplet dicts (``{packed, scale, bias, group_size, bits}``) for the resident
        ``mx.gather_qmm`` path (:func:`quanta.qwen35.moe._routed_sparse_packed`) — the routed experts
        stay int4-resident (~4× lighter; the ~79→~30 GiB lever) instead of the dequantized-bf16
        ``[E,*,*]`` stacks :meth:`moe` returns. The router ``gate``, the shared expert, and the
        shared-gate stay **bf16** (CLAUDE.md: the always-on shared expert is never quantized). The
        exact dict :func:`quanta.qwen35.moe.qwen35_moe` consumes when it auto-detects packed experts."""
        p = f"{LM_PREFIX}layers.{i}.mlp."
        gu = self._packed_triplet(p + "experts.gate_up_proj")  # [E, 2*moe_inter, hidden] int4
        dn = self._packed_triplet(p + "experts.down_proj")     # [E, hidden, moe_inter]   int4
        out: dict[str, object] = {
            "gate": self.read(p + "gate.weight"),
            "experts_gate_up": gu,
            "experts_down": dn,
            "shared_expert_gate": self.read(p + "shared_expert_gate.weight"),
        }
        for proj in SHARED_EXPERT_PROJS:
            out[f"shared_{proj}"] = self.read(p + f"shared_expert.{proj}.weight")
        mx.eval([out["gate"], out["shared_expert_gate"],
                 *(out[f"shared_{proj}"] for proj in SHARED_EXPERT_PROJS),
                 gu["packed"], gu["scale"], gu["bias"], dn["packed"], dn["scale"], dn["bias"]])
        return out

    # --- native MTP block ------------------------------------------------------
    def mtp(self, j: int = 0) -> dict[str, mx.array]:
        """The native MTP block: ``fc`` fusion + pre-norms (bf16), one full-attn (int8) + MoE block.

        Mirrors :meth:`quanta.qwen35.loader.Qwen35SourceCheckpoint.mtp`: the MoE has the SAME fused
        pre-stacked layout as a main-decoder block, so the routed experts come back as ``experts_gate_up``
        ``[E, 2*moe_inter, hidden]`` / ``experts_down`` ``[E, hidden, moe_inter]`` (the int4 g64 stacks
        dequantized in one shot) — identical to :meth:`moe`.
        """
        if j != 0:
            raise IndexError(f"Qwen3.5 has {self.cfg.num_mtp_modules} MTP head(s); got j={j}")
        p = f"mtp.{j}."
        out: dict[str, mx.array] = {
            "fc": self.read(p + "fc.weight"),
            "pre_fc_norm_embedding": self.read(p + "pre_fc_norm_embedding.weight"),
            "pre_fc_norm_hidden": self.read(p + "pre_fc_norm_hidden.weight"),
            "norm": self.read(p + "norm.weight"),
            "input_layernorm": self.read(p + "input_layernorm.weight"),
            "post_attention_layernorm": self.read(p + "post_attention_layernorm.weight"),
        }
        ap = p + "self_attn."
        out["attention"] = {suffix: self.read(ap + suffix) for suffix in FULL_ATTN_SUFFIXES}
        mp = p + "mlp."
        moe: dict[str, mx.array] = {
            "gate": self.read(mp + "gate.weight"),
            "shared_expert_gate": self.read(mp + "shared_expert_gate.weight"),
            "experts_gate_up": self.read(mp + "experts.gate_up_proj"),  # [E, 2*moe_inter, hidden]
            "experts_down": self.read(mp + "experts.down_proj"),        # [E, hidden, moe_inter]
        }
        for proj in SHARED_EXPERT_PROJS:
            moe[f"shared_{proj}"] = self.read(mp + f"shared_expert.{proj}.weight")
        out["moe"] = moe
        mx.eval([out["fc"], out["attention"]["q_proj.weight"], moe["experts_gate_up"]])
        return out
