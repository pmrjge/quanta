"""Streamed reader over the baked InternLM2.5-7B-Chat-1M quanta artifact — dequantizes to bf16.

Duck-types :class:`quanta.internlm2.loader.InternLM2SourceCheckpoint`: exposes the same
``embed`` / ``final_norm`` / ``lm_head`` / ``block_norms`` / ``attention`` / ``mlp`` / ``has`` /
``release`` surface and returns the **same suffix-keyed dicts**, so the bf16 reference forward runs
against it unchanged. This is the **parity reference for the InternLM2.5 quant gate**: the
artifact's packed weights dequantized back to bf16 and run through the proven naive forward —
the float baseline the resident ``quantized_matmul`` runtime must then match.

Storage contract (mirrors :class:`quanta.bake.artifact.ArtifactWriter` + :mod:`quanta.internlm2.bake`):

* ``dense``         → ``{key}`` verbatim (RMSNorms, embed_tokens, output, final norm), cast to
  bf16 on read.
* ``affine_packed`` → an affine int weight at ``{base}`` (the key **without** ``.weight``), stored as
  ``{base}.weight_packed`` + ``.weight_scale`` + ``.weight_bias``; ``mx.dequantize`` inverts it.
  Covers both the int8 attention weights (wq/wk/wv/wo) and the int4 SwiGLU FFN weights
  (w1/w3/w2), all 2-D ``[out, in]``.

One layer resident at a time (rule-8): per-layer accessors return only that layer's tensors, and
shard handles are dropped on :meth:`release`. Unknown formats fail loud (rule-6). The fused
``wqkv`` no longer exists in the artifact — the bake stored three separate (already-split)
projections under the standard ``attention.wq/wk/wv`` keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from quanta.internlm2.config import InternLM2Config
from quanta.internlm2.loader import (
    ATTN_WEIGHT_SUFFIXES,
    EMBED_KEY,
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    MLP_SUFFIXES,
    MODEL_PREFIX,
)

_WEIGHT_SUFFIX = ".weight"


class InternLM2Artifact:
    """Lazy, streamed, dequantizing reader over a baked InternLM2.5-7B-Chat-1M quanta artifact."""

    def __init__(self, art_dir: str | Path) -> None:
        self.dir = Path(art_dir)
        self.cfg = InternLM2Config.from_pretrained(art_dir)  # self-contained (incl. NTK policy)
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

    # --- top-level tensors (same surface as InternLM2SourceCheckpoint) ---------
    def embed(self) -> mx.array:
        return self.read(EMBED_KEY)

    def final_norm(self) -> mx.array:
        return self.read(FINAL_NORM_KEY)

    def lm_head(self) -> mx.array:
        return self.read(EMBED_KEY if self.cfg.tie_word_embeddings else LM_HEAD_KEY)

    # --- per-layer kinds -------------------------------------------------------
    def block_norms(self, i: int) -> dict[str, mx.array]:
        p = f"{MODEL_PREFIX}layers.{i}."
        out = {"attention_norm": self.read(p + "attention_norm.weight"),
               "ffn_norm":       self.read(p + "ffn_norm.weight")}
        mx.eval(list(out.values()))
        return out

    def attention(self, i: int) -> dict[str, mx.array]:
        """InternLM2 self-attention tensors for layer ``i``: int8 wq/wk/wv/wo (already split)."""
        p = f"{MODEL_PREFIX}layers.{i}.attention."
        out: dict[str, mx.array] = {s: self.read(p + s) for s in ATTN_WEIGHT_SUFFIXES}
        mx.eval(list(out.values()))
        return out

    def mlp(self, i: int) -> dict[str, mx.array]:
        """SwiGLU FFN tensors for layer ``i``: int4 w1 / w3 / w2 (dequantized)."""
        p = f"{MODEL_PREFIX}layers.{i}.feed_forward."
        out = {s: self.read(p + s) for s in MLP_SUFFIXES}
        mx.eval(list(out.values()))
        return out
