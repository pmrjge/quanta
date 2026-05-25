"""Streamed reader over the baked GLM-5.1 (``glm_moe_dsa``) quanta artifact — dequantizes to bf16.

Duck-types :class:`quanta.glm.loader.GLMSourceCheckpoint`: it exposes the same
``embed`` / ``final_norm`` / ``lm_head`` / ``block_norms`` / ``attention`` (incl. the nested DSA
``indexer``) / ``moe_router`` / ``shared_expert`` / ``expert_stacks`` / ``dense_mlp`` / ``mtp`` / ``has``
surface and returns the **same logical sub-keyed dicts**, so :func:`quanta.glm.model.load_block` /
:func:`quanta.glm.model.glm_logits` and the bf16 ppl harness run against it unchanged. This is the
**parity reference for the GLM quant gate**: the artifact's packed weights dequantized back to bf16 and
run through the proven naive forward — the float baseline the resident ``gather_qmm`` runtime must match.

Storage contract (the consume side of :class:`quanta.bake.artifact.ArtifactWriter`, written by
:mod:`quanta.glm.bake`):

* ``dense``         → ``{key}`` verbatim (norms, MLA q/kv sub-norms, indexer ``k_norm.{weight,bias}``,
  router ``gate.weight`` + ``e_score_correction_bias``, embeddings / final norm), returned in its
  stored native dtype (the consumer re-casts — bf16 weights, f32 the correction bias).
* ``affine_packed`` → int8 affine weight at ``{key}`` (the base, **without** ``.weight``), stored as
  ``{key}.weight_packed`` + ``.weight_scale`` + ``.weight_bias``; ``mx.dequantize`` inverts it. Covers
  the MLA q/kv/o + indexer matmuls, the dense-FFN layers, the shared expert, ``eh_proj`` and ``lm_head``.
* ``awq_packed``    → int4 AWQ routed-expert weight at ``{key}``: the affine codes are of
  ``W·diag(s)``; recover ``W = dequantize(...) / s`` with the stored per-input-channel ``{key}.awq_scale``
  (``s=1`` for cold / RTN experts → identity), matching :class:`quanta.nemotron.artifact.NemotronArtifact`
  / :class:`quanta.dsv4.artifact.DSV4Artifact`.

One layer resident at a time (rule 8): expert stacks are dequant-streamed, evaluated, and the shard
handles released every 16 experts. Unknown / unmapped formats fail loud (rule 6) — never a silent
default, never a wrong-bits dequant.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from quanta.glm.config import GLMConfig

# Exact GLM source key strings for the top-level tensors (see quanta.glm.loader).
EMBED_KEY = "model.embed_tokens.weight"
FINAL_NORM_KEY = "model.norm.weight"
LM_HEAD_KEY = "lm_head.weight"

_WEIGHT_SUFFIX = ".weight"
_DEQUANT_FORMATS = ("affine_packed", "awq_packed")
_EXPERT_PROJ = ("gate_proj", "up_proj", "down_proj")


class GLMArtifact:
    """Lazy, streamed, dequantizing reader over a baked GLM-5.1 quanta artifact."""

    def __init__(self, art_dir: str | Path) -> None:
        self.dir = Path(art_dir)
        self.cfg = GLMConfig.from_pretrained(art_dir)  # self-contained config
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
        """True iff ``key`` resolves to a manifest entry (dense at the key, or a quantized base)."""
        if key in self.manifest:
            return True
        base = key[: -len(_WEIGHT_SUFFIX)] if key.endswith(_WEIGHT_SUFFIX) else key
        return base in self.manifest

    def release(self) -> None:
        """Drop cached shard handles so materialized tensors can be freed."""
        self._shards.clear()

    # --- manifest resolution ---------------------------------------------------
    def _meta(self, key: str) -> tuple[str, dict]:
        """Resolve a logical key to ``(base, meta)``: a dense entry at the full key, else the quantized
        base (the key minus a trailing ``.weight``). Fail loud if neither (rule 6) — ``KeyError`` when
        no manifest entry exists, ``ValueError`` on a recognized entry with an unknown format (never
        silently guess or dequant at the wrong bits)."""
        meta = self.manifest.get(key)
        if meta is not None and meta.get("format") == "dense":
            return key, meta
        base = key[: -len(_WEIGHT_SUFFIX)] if key.endswith(_WEIGHT_SUFFIX) else key
        bmeta = self.manifest.get(base)
        if bmeta is not None and bmeta.get("format") in _DEQUANT_FORMATS:
            return base, bmeta
        for cand, cmeta in ((key, meta), (base, bmeta)):  # entry exists but format unrecognized
            if cmeta is not None:
                raise ValueError(f"{cand}: unknown manifest format {cmeta.get('format')!r} "
                                 f"(refusing to guess / dequant at wrong bits)")
        raise KeyError(f"{key}: not in artifact manifest (no dense entry, no quantized base {base!r})")

    # --- dequant ---------------------------------------------------------------
    def _dequant(self, base: str, meta: dict) -> mx.array:
        """Dequantize a packed weight at ``base`` → bf16 ``[out, in]`` (AWQ unscaled by ``1/s``).

        AWQ codes are of ``W·diag(s)``, so the original weight is recovered as ``dequantize(...) / s``
        per input channel (``s=1`` for cold / RTN experts → identity), matching the Nemotron / DSV4
        readers."""
        gs, bits = int(meta["group_size"]), int(meta["bits"])
        w = mx.dequantize(
            self.get(base + ".weight_packed"),
            self.get(base + ".weight_scale"),
            self.get(base + ".weight_bias"),
            group_size=gs,
            bits=bits,
        )
        if meta["format"] == "awq_packed":  # codes are of W·diag(s); recover W per input channel
            s = self.get(base + ".awq_scale")
            w = w / s[None, :]
        return w.astype(mx.bfloat16)

    def read(self, key: str) -> mx.array:
        """Return a weight for a logical ``key``: dense verbatim (its **stored native dtype** — the
        consumer re-casts), else the dequantized bf16 weight."""
        base, meta = self._meta(key)
        if meta["format"] == "dense":
            return self.get(base)
        return self._dequant(base, meta)

    def raw(self, key: str) -> mx.array:
        """Return the packed codes verbatim (``{base}.weight_packed``) for the gather_qmm decode path —
        siblings fetch the companion ``.weight_scale`` / ``.weight_bias`` / ``.awq_scale`` via
        :meth:`get`. Fail loud on a dense key (it has no packed codes)."""
        base, meta = self._meta(key)
        if meta["format"] not in _DEQUANT_FORMATS:
            raise ValueError(f"{key}: format {meta['format']!r} has no packed codes (not quantized)")
        return self.get(base + ".weight_packed")

    # --- key helpers (match the source checkpoint layout) ----------------------
    def _bp(self, i: int) -> str:
        return f"layers.{i}."

    # --- top-level tensors (same dicts as GLMSourceCheckpoint) -----------------
    def embed(self) -> mx.array:
        return self.read(EMBED_KEY)

    def final_norm(self) -> mx.array:
        return self.read(FINAL_NORM_KEY)

    def lm_head(self) -> mx.array:
        """Output projection (untied) — dequantized int8; tied falls back to the embedding."""
        return self.read(EMBED_KEY if self.cfg.tie_word_embeddings else LM_HEAD_KEY)

    # --- per-block helpers -----------------------------------------------------
    def block_norms(self, i: int) -> dict[str, mx.array]:
        p = self._bp(i)
        return {"input_layernorm": self.read(p + "input_layernorm.weight"),
                "post_attention_layernorm": self.read(p + "post_attention_layernorm.weight")}

    def indexer(self, i: int) -> dict[str, mx.array]:
        """DSA Lightning-Indexer params for layer ``i`` (matmuls dequantized, k_norm weight+bias dense)."""
        p = self._bp(i) + "self_attn.indexer."
        out = {"wq_b": self.read(p + "wq_b.weight"),
               "wk": self.read(p + "wk.weight"),
               "weights_proj": self.read(p + "weights_proj.weight"),
               "k_norm_weight": self.read(p + "k_norm.weight"),
               "k_norm_bias": self.read(p + "k_norm.bias")}
        mx.eval(list(out.values()))
        return out

    def attention(self, i: int) -> dict[str, mx.array]:
        """MLA low-rank q/kv + o-proj + q/kv sub-norms, with the DSA indexer as a nested sub-dict."""
        p = self._bp(i) + "self_attn."
        out = {
            "q_a_proj": self.read(p + "q_a_proj.weight"),
            "q_a_layernorm": self.read(p + "q_a_layernorm.weight"),
            "q_b_proj": self.read(p + "q_b_proj.weight"),
            "kv_a_proj_with_mqa": self.read(p + "kv_a_proj_with_mqa.weight"),
            "kv_a_layernorm": self.read(p + "kv_a_layernorm.weight"),
            "kv_b_proj": self.read(p + "kv_b_proj.weight"),
            "o_proj": self.read(p + "o_proj.weight"),
            "indexer": self.indexer(i),
        }
        mx.eval([v for v in out.values() if isinstance(v, mx.array)])
        return out

    # --- MoE -------------------------------------------------------------------
    def moe_router(self, i: int) -> dict[str, mx.array]:
        """Router: gate.weight (bf16) + e_score_correction_bias (f32 control) — both dense verbatim."""
        p = self._bp(i) + "mlp.gate."
        out = {"weight": self.read(p + "weight"),
               "e_score_correction_bias": self.read(p + "e_score_correction_bias")}
        mx.eval(list(out.values()))
        return out

    def shared_expert(self, i: int) -> dict[str, mx.array]:
        """Always-on shared expert (int8 in the artifact) → dequantized ``gate/up/down_proj`` bf16."""
        p = self._bp(i) + "mlp.shared_experts."
        out = {proj: self.read(f"{p}{proj}.weight") for proj in _EXPERT_PROJ}
        mx.eval(list(out.values()))
        return out

    def dense_mlp(self, i: int) -> dict[str, mx.array]:
        """Dense FFN for a ``first_k_dense_replace`` layer (int8 in the artifact) → dequantized bf16."""
        p = self._bp(i) + "mlp."
        out = {proj: self.read(f"{p}{proj}.weight") for proj in _EXPERT_PROJ}
        mx.eval(list(out.values()))
        return out

    def expert_stacks(self, i: int, n_experts: int | None = None) -> dict[str, mx.array]:
        """Dequant routed experts into ``[E, out, in]`` bf16 stacks (gate/up/down_proj).

        Streamed per expert (dequant → place → drop shard handles every 16) so only the bf16 stacks
        stay resident. ``gate/up_proj``: ``[E, moe_inter, hidden]``; ``down_proj``: ``[E, hidden, moe_inter]``.
        """
        ne = n_experts if n_experts is not None else self.cfg.n_routed_experts

        def ek(e: int, proj: str) -> str:
            return f"{self._bp(i)}mlp.experts.{e}.{proj}.weight"

        first = {proj: self.read(ek(0, proj)) for proj in _EXPERT_PROJ}
        stacks = {proj: mx.zeros((ne, *first[proj].shape), first[proj].dtype) for proj in first}
        for proj in first:
            stacks[proj][0] = first[proj]
        for e in range(1, ne):
            for proj in _EXPERT_PROJ:
                stacks[proj][e] = self.read(ek(e, proj))
            if e % 16 == 15:
                mx.eval(list(stacks.values()))
                self.release()
        mx.eval(list(stacks.values()))
        self.release()
        return stacks

    # --- native MTP block ------------------------------------------------------
    def mtp(self, j: int = 0) -> dict[str, mx.array]:
        """The native MTP block (one full MoE decoder layer at ``num_hidden_layers``): the embed/hidden
        combine (``eh_proj`` + ``enorm``/``hnorm``/``shared_head.norm``) plus the inherited
        attention / router / shared / routed experts — the same dict shape as the source loader."""
        if j != 0:
            raise IndexError(f"GLM-5.1 has 1 MTP head; got j={j}")
        i = self.cfg.mtp_layer_id
        p = f"layers.{i}."
        out: dict = {
            "enorm": self.read(p + "enorm.weight"),
            "hnorm": self.read(p + "hnorm.weight"),
            "eh_proj": self.read(p + "eh_proj.weight"),
            "shared_head_norm": self.read(p + "shared_head.norm.weight"),
            "input_layernorm": self.read(p + "input_layernorm.weight"),
            "post_attention_layernorm": self.read(p + "post_attention_layernorm.weight"),
            "attention": self.attention(i),
            "router": self.moe_router(i),
            "shared": self.shared_expert(i),
            "experts": self.expert_stacks(i),
        }
        return out
