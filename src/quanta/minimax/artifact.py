"""Streamed reader over the baked MiniMax-M2.7 quanta artifact — dequantizes to bf16.

Duck-types :class:`quanta.minimax.loader.MiniMaxSourceCheckpoint`: it exposes the same
``embed`` / ``lm_head`` / ``final_norm`` / ``block_norms`` / ``attention`` / ``moe_router`` /
``expert_stacks`` / ``moe`` / ``read`` / ``release`` surface and returns the **same logical
sub-keyed dicts**, so :func:`quanta.minimax.model.load_block` / :func:`quanta.minimax.model.minimax_logits`
and the bf16 ppl harness run against it unchanged. This is the **parity reference for the int6 quant
gate**: the artifact's packed weights dequantized back to bf16 and run through the proven naive forward
— the float baseline the resident ``gather_qmm`` runtime must then match.

Storage contract (mirrors :class:`quanta.bake.artifact.ArtifactWriter` + :mod:`quanta.minimax.bake`):

* ``dense``         -> ``{key}`` verbatim (router ``gate.weight`` + ``e_score_correction_bias``, all
  norms incl. per-layer ``q_norm``/``k_norm``, embeddings / ``lm_head`` / final norm), cast to bf16.
* ``affine_packed`` -> affine int weight at ``{key}`` (the base, **without** ``.weight``), stored as
  ``{key}.weight_packed`` + ``.weight_scale`` + ``.weight_bias``; ``mx.dequantize`` inverts it. Used
  for **both** the int8 GQA projections and the int6-GPTQ experts (uniform runtime path — the bits
  live in the manifest). **No AWQ path** (GPTQ stores plain affine codes) and **no shared expert**.

One layer resident at a time (rule 8): expert stacks dequant streamed, evaluated and the shard handles
released every 16 experts. Unknown / unmapped formats fail loud (rule 6) — never a silent default,
never a wrong-bits dequant.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from quanta.minimax.config import MiniMaxConfig
from quanta.minimax.loader import EMBED_KEY, FINAL_NORM_KEY, LM_HEAD_KEY

_WEIGHT_SUFFIX = ".weight"


class MiniMaxArtifact:
    """Lazy, streamed, dequantizing reader over a baked MiniMax-M2.7 quanta artifact."""

    def __init__(self, art_dir: str | Path, cfg: MiniMaxConfig | None = None) -> None:
        self.dir = Path(art_dir)
        self.cfg = cfg if cfg is not None else MiniMaxConfig.from_pretrained(art_dir)
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

    def release(self) -> None:
        """Drop cached shard handles so materialized tensors can be freed."""
        self._shards.clear()

    def has(self, key: str) -> bool:
        return key in self.weight_map

    # --- manifest resolution ---------------------------------------------------
    def _meta(self, key: str) -> tuple[str, dict]:
        """Resolve a logical key to ``(base, meta)``: a dense entry at the full key, else the
        affine-packed base (the key minus a trailing ``.weight``). Fail loud if neither (rule 6) —
        ``KeyError`` when no manifest entry exists, ``ValueError`` on a recognized entry with an
        unknown format (never silently guess or dequant at the wrong bits)."""
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
        """Dequantize a packed affine weight at ``base`` -> bf16 ``[out, in]`` via ``mx.dequantize``."""
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

    def read_dequant(self, key: str, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
        """Loader-shaped alias for :meth:`read` (the source reader's fp8 dequant accessor)."""
        return self.read(key).astype(dtype)

    def raw(self, key: str) -> mx.array:
        """Return the packed codes verbatim (``{base}.weight_packed``) for the gather_qmm decode path —
        siblings fetch the companion ``.weight_scale`` / ``.weight_bias`` via :meth:`get`. Fail loud on
        a dense key (it has no packed codes)."""
        base, meta = self._meta(key)
        if meta["format"] != "affine_packed":
            raise ValueError(f"{key}: format {meta['format']!r} has no packed codes (not quantized)")
        return self.get(base + ".weight_packed")

    # --- key helpers (match the source checkpoint layout) ----------------------
    def _bp(self, i: int) -> str:
        return f"model.layers.{i}."

    # --- top-level tensors (same as MiniMaxSourceCheckpoint) -------------------
    def embed(self) -> mx.array:
        return self.read(EMBED_KEY)

    def lm_head(self) -> mx.array:
        key = EMBED_KEY if self.cfg.tie_word_embeddings else LM_HEAD_KEY
        return self.read(key)

    def final_norm(self) -> mx.array:
        return self.read(FINAL_NORM_KEY)

    # --- per-layer helpers -----------------------------------------------------
    def block_norms(self, i: int) -> dict[str, mx.array]:
        p = self._bp(i)
        return {
            "input_layernorm": self.read(p + "input_layernorm.weight"),
            "post_attention_layernorm": self.read(p + "post_attention_layernorm.weight"),
        }

    def attention(self, i: int) -> dict[str, mx.array]:
        """GQA q/k/v/o (int8 -> bf16) + per-layer QK RMSNorm weights (bf16) for layer ``i``."""
        p = self._bp(i) + "self_attn."
        out = {
            "q_proj": self.read(p + "q_proj.weight"),
            "k_proj": self.read(p + "k_proj.weight"),
            "v_proj": self.read(p + "v_proj.weight"),
            "o_proj": self.read(p + "o_proj.weight"),
            "q_norm": self.read(p + "q_norm.weight"),
            "k_norm": self.read(p + "k_norm.weight"),
        }
        mx.eval(list(out.values()))
        return out

    # --- MoE -------------------------------------------------------------------
    def moe_router(self, i: int) -> dict[str, mx.array]:
        """Router: ``gate.weight`` (bf16) + ``e_score_correction_bias`` (bf16) — both dense."""
        p = self._bp(i) + "block_sparse_moe."
        out = {
            "weight": self.read(p + "gate.weight"),
            "e_score_correction_bias": self.read(p + "e_score_correction_bias"),
        }
        mx.eval(list(out.values()))
        return out

    def expert_stacks(self, i: int, n_experts: int | None = None) -> dict[str, mx.array]:
        """Dequant routed experts into ``[E, out, in]`` bf16 stacks (w1 gate, w3 up, w2 down).

        Streamed per expert (dequant -> place -> drop shard handles every 16) so only the bf16 stacks
        stay resident. ``w1``/``w3``: ``[E, moe_inter, hidden]``; ``w2``: ``[E, hidden, moe_inter]``.
        """
        ne = n_experts if n_experts is not None else self.cfg.num_local_experts

        def ek(e: int, proj: str) -> str:
            return f"{self._bp(i)}block_sparse_moe.experts.{e}.{proj}.weight"

        first = {proj: self.read(ek(0, proj)) for proj in ("w1", "w2", "w3")}
        stacks = {proj: mx.zeros((ne, *first[proj].shape), first[proj].dtype) for proj in first}
        for proj in first:
            stacks[proj][0] = first[proj]
        for e in range(1, ne):
            for proj in ("w1", "w2", "w3"):
                stacks[proj][e] = self.read(ek(e, proj))
            if e % 16 == 15:
                mx.eval(list(stacks.values()))
                self.release()
        mx.eval(list(stacks.values()))
        self.release()
        return stacks

    def moe(self, i: int) -> dict[str, mx.array]:
        """Full MoE block of layer ``i``: router (gate + bias) + routed expert stacks.

        No shared expert in this checkpoint (``shared_intermediate_size == 0``); refuse to invent one."""
        if self.cfg.has_shared_expert:
            raise ValueError(f"L{i}: config reports a shared expert (shared_intermediate_size="
                             f"{self.cfg.shared_intermediate_size}) but this artifact has none")
        return {"router": self.moe_router(i), "experts": self.expert_stacks(i)}
