"""Streamed reader over the baked Qwen2.5-14B-Instruct-1M quanta artifact — dequantizes to bf16.

Duck-types :class:`quanta.qwen25.loader.Qwen25SourceCheckpoint`: exposes the same
``embed`` / ``final_norm`` / ``lm_head`` / ``block_norms`` / ``attention`` / ``mlp`` / ``has`` /
``release`` surface and returns the **same suffix-keyed dicts**, so the bf16 reference forward runs
against it unchanged. This is the **parity reference for the Qwen2.5 quant gate**: the artifact's
packed weights dequantized back to bf16 and run through the proven naive forward — the float
baseline the resident ``quantized_matmul`` runtime must then match.

Storage contract (mirrors :class:`quanta.bake.artifact.ArtifactWriter` + :mod:`quanta.qwen25.bake`):

* ``dense``         → ``{key}`` verbatim (RMSNorms, q/k/v biases, embed_tokens, lm_head,
  final norm), cast to bf16 on read.
* ``affine_packed`` → an affine int weight at ``{base}`` (the key **without** ``.weight``), stored as
  ``{base}.weight_packed`` + ``.weight_scale`` + ``.weight_bias``; ``mx.dequantize`` inverts it.
  Covers both the int8 attention weights and the int4 SwiGLU FFN weights (2-D ``[out, in]``).

One layer resident at a time (rule-8): per-layer accessors return only that layer's tensors, and
shard handles are dropped on :meth:`release`. Unknown formats fail loud (rule-6).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from quanta.qwen25.config import Qwen25Config
from quanta.qwen25.loader import (
    ATTN_BIAS_SUFFIXES,
    ATTN_WEIGHT_SUFFIXES,
    EMBED_KEY,
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    MLP_SUFFIXES,
    MODEL_PREFIX,
)

_WEIGHT_SUFFIX = ".weight"


class Qwen25Artifact:
    """Lazy, streamed, dequantizing reader over a baked Qwen2.5-14B-Instruct-1M quanta artifact."""

    def __init__(self, art_dir: str | Path) -> None:
        self.dir = Path(art_dir)
        self.cfg = Qwen25Config.from_pretrained(art_dir)  # self-contained (incl. DCA policy)
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
        """Load one tensor verbatim by exact safetensors key (lazy / mmap; shard handles cached)."""
        if name not in self.weight_map:
            raise KeyError(f"tensor not in artifact index: {name}")
        fn = self.weight_map[name]
        shard = self._shards.get(fn)
        if shard is None:
            shard = mx.load(str(self.dir / fn))                # lazy / mmap
            self._shards[fn] = shard
        return shard[name]

    def has(self, key: str) -> bool:
        """True iff a logical key resolves (a dense entry or a quantized base)."""
        if key in self.weight_map:
            return True
        base = key[: -len(_WEIGHT_SUFFIX)] if key.endswith(_WEIGHT_SUFFIX) else key
        return self.manifest.get(base, {}).get("format") == "affine_packed"

    def release(self) -> None:
        """Drop cached shard handles so materialized tensors can be freed."""
        self._shards.clear()

    # --- manifest resolution ---------------------------------------------------
    def _meta(self, key: str) -> tuple[str, dict]:
        """Resolve a logical key to ``(base, meta)``: dense at the full key, else ``affine_packed``
        at the base (key minus ``.weight``). Fail loud if neither (rule-6)."""
        meta = self.manifest.get(key)
        if meta is not None and meta.get("format") == "dense":
            return key, meta
        base = key[: -len(_WEIGHT_SUFFIX)] if key.endswith(_WEIGHT_SUFFIX) else key
        bmeta = self.manifest.get(base)
        if bmeta is not None and bmeta.get("format") == "affine_packed":
            return base, bmeta
        for cand, cmeta in ((key, meta), (base, bmeta)):
            if cmeta is not None:
                raise ValueError(f"{cand}: unknown manifest format {cmeta.get('format')!r}")
        raise KeyError(f"{key}: not in artifact manifest")

    # --- dequant ---------------------------------------------------------------
    def _dequant(self, base: str, meta: dict) -> mx.array:
        """Dequantize a packed weight at ``base`` → bf16."""
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
        """Packed codes verbatim (``{base}.weight_packed``) for the ``quantized_matmul`` decode path.

        Siblings fetch ``.weight_scale`` / ``.weight_bias`` via :meth:`get`. Fail loud on a dense key
        (it has no packed codes).
        """
        base, meta = self._meta(key)
        if meta["format"] != "affine_packed":
            raise ValueError(f"{key}: format {meta['format']!r} has no packed codes (not quantized)")
        return self.get(base + ".weight_packed")

    # --- top-level tensors (same surface as Qwen25SourceCheckpoint) ------------
    def embed(self) -> mx.array:
        return self.read(EMBED_KEY)

    def final_norm(self) -> mx.array:
        return self.read(FINAL_NORM_KEY)

    def lm_head(self) -> mx.array:
        return self.read(EMBED_KEY if self.cfg.tie_word_embeddings else LM_HEAD_KEY)

    # --- per-layer kinds -------------------------------------------------------
    def block_norms(self, i: int) -> dict[str, mx.array]:
        p = f"{MODEL_PREFIX}layers.{i}."
        out = {"input_layernorm": self.read(p + "input_layernorm.weight"),
               "post_attention_layernorm": self.read(p + "post_attention_layernorm.weight")}
        mx.eval(list(out.values()))
        return out

    def attention(self, i: int) -> dict[str, mx.array]:
        """Qwen2 self-attention tensors for layer ``i``: int8 q/k/v/o weights + bf16 q/k/v biases."""
        p = f"{MODEL_PREFIX}layers.{i}.self_attn."
        out: dict[str, mx.array] = {s: self.read(p + s) for s in ATTN_WEIGHT_SUFFIXES}
        if self.cfg.attention_bias:
            for s in ATTN_BIAS_SUFFIXES:
                if self.has(p + s):                            # absent if bake config disabled biases
                    out[s] = self.read(p + s)
        mx.eval(list(out.values()))
        return out

    def mlp(self, i: int) -> dict[str, mx.array]:
        """SwiGLU FFN tensors for layer ``i``: int4 gate_proj / up_proj / down_proj (dequantized)."""
        p = f"{MODEL_PREFIX}layers.{i}.mlp."
        out = {s: self.read(p + s) for s in MLP_SUFFIXES}
        mx.eval(list(out.values()))
        return out
