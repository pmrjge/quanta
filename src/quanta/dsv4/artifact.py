"""Streamed reader over the baked DeepSeek-V4-Flash quanta artifact — dequantizes to bf16.

Duck-types :class:`quanta.dsv4.loader.DeepSeekV4SourceCheckpoint`: it exposes the same
``embed`` / ``head`` / ``final_norm`` / ``final_hc`` / ``block_norms`` / ``block_hc`` /
``attention`` / ``moe_router`` / ``shared_expert`` / ``expert_stacks`` / ``mtp`` / ``read`` /
``release`` surface and returns the **same logical sub-keyed dicts**, so
:func:`quanta.dsv4.model.load_block_params` / :func:`quanta.dsv4.model.dsv4_logits` and the
bf16 ppl harness run against it unchanged. This is the **parity reference for the DSV4 quant
gate**: the artifact's packed weights dequantized back to bf16 and run through the proven naive
forward — the float baseline the resident ``gather_qmm`` runtime must then match.

Storage contract (mirrors :class:`quanta.bake.artifact.ArtifactWriter`):

* ``dense``         → ``{key}`` verbatim (norms, router gate / bias / tid2eid, attn sink, HC
  params, embeddings / head / final norm), cast to bf16.
* ``affine_packed`` → affine int weight at ``{key}`` (the base, **without** ``.weight``), stored
  as ``{key}.weight_packed`` + ``.weight_scale`` + ``.weight_bias``; ``mx.dequantize`` inverts it.
* ``awq_packed``    → AWQ int weight at ``{key}``: the affine codes are of ``W·diag(s)``; recover
  ``W = dequantize(...) · s`` with the stored per-input-channel ``{key}.awq_scale`` (``s=1`` for
  cold / RTN experts → identity), matching :class:`quanta.nemotron.artifact.NemotronArtifact`.

One layer resident at a time (rule-8): expert stacks dequant streamed, evaluated and the shard
handles released every 16 experts. Unknown / unmapped formats fail loud (rule-6) — never a silent
default, never a wrong-bits dequant.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from quanta.dsv4.config import DeepSeekV4Config

# Exact DSV4 source key strings for the top-level tensors (see quanta.dsv4.loader).
EMBED_KEY = "embed.weight"
LM_HEAD_KEY = "head.weight"
FINAL_NORM_KEY = "norm.weight"

_WEIGHT_SUFFIX = ".weight"
_DEQUANT_FORMATS = ("affine_packed", "awq_packed")


class DSV4Artifact:
    """Lazy, streamed, dequantizing reader over a baked DeepSeek-V4-Flash quanta artifact."""

    def __init__(self, art_dir: str | Path) -> None:
        self.dir = Path(art_dir)
        self.cfg = DeepSeekV4Config.from_pretrained(art_dir)  # self-contained config
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

    def release(self) -> None:
        """Drop cached shard handles so materialized tensors can be freed."""
        self._shards.clear()

    # --- manifest resolution ---------------------------------------------------
    def _meta(self, key: str) -> tuple[str, dict]:
        """Resolve a logical key to ``(base, meta)``: a dense entry at the full key, else the
        quantized base (the key minus a trailing ``.weight``). Fail loud if neither (rule-6) —
        ``KeyError`` when no manifest entry exists, ``ValueError`` on a recognized entry with an
        unknown format (never silently guess or dequant at the wrong bits)."""
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

        Matches :meth:`quanta.nemotron.artifact.NemotronArtifact._dequant`: AWQ codes are of
        ``W·diag(s)``, so the original weight is recovered as ``dequantize(...) / s`` per input
        channel (``s=1`` for cold / RTN experts → identity)."""
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
        """Return a DEQUANTIZED bf16 weight for a logical ``key`` (dense verbatim, else dequant)."""
        base, meta = self._meta(key)
        if meta["format"] == "dense":
            return self.get(base).astype(mx.bfloat16)
        return self._dequant(base, meta)

    def raw(self, key: str) -> mx.array:
        """Return the packed codes verbatim (``{base}.weight_packed``) for the gather_qmm decode
        path — siblings fetch the companion ``.weight_scale`` / ``.weight_bias`` / ``.awq_scale``
        via :meth:`get`. Fail loud on a dense key (it has no packed codes)."""
        base, meta = self._meta(key)
        if meta["format"] not in _DEQUANT_FORMATS:
            raise ValueError(f"{key}: format {meta['format']!r} has no packed codes (not quantized)")
        return self.get(base + ".weight_packed")

    # --- key helpers (match the source checkpoint layout) ----------------------
    def _bp(self, i: int) -> str:
        return f"layers.{i}."

    # --- top-level tensors (same dicts as DeepSeekV4SourceCheckpoint) ----------
    def embed(self) -> mx.array:
        return self.read(EMBED_KEY)

    def head(self) -> mx.array:
        return self.read(LM_HEAD_KEY)

    def final_norm(self) -> mx.array:
        return self.read(FINAL_NORM_KEY)

    def final_hc(self) -> dict[str, mx.array]:
        return {k: self.read(f"hc_head_{k}") for k in ("fn", "base", "scale")}

    # --- per-block helpers -----------------------------------------------------
    def block_norms(self, i: int) -> dict[str, mx.array]:
        p = self._bp(i)
        return {"attn_norm": self.read(p + "attn_norm.weight"),
                "ffn_norm": self.read(p + "ffn_norm.weight")}

    def block_hc(self, i: int) -> dict[str, mx.array]:
        """HC mixing params for the attn + ffn sub-blocks (fn / base / scale per sub-block)."""
        p = self._bp(i)
        out: dict[str, mx.array] = {}
        for which in ("attn", "ffn"):
            for k in ("fn", "base", "scale"):
                out[f"hc_{which}_{k}"] = self.read(p + f"hc_{which}_{k}")
        mx.eval(list(out.values()))
        return out

    def _compressor(self, prefix: str) -> dict[str, mx.array]:
        """ape, norm, wkv/wgate for a Compressor at ``prefix`` (matches the source loader)."""
        out = {"ape": self.read(prefix + "ape"),
               "norm": self.read(prefix + "norm.weight"),
               "wkv": self.read(prefix + "wkv.weight"),
               "wgate": self.read(prefix + "wgate.weight")}
        mx.eval(list(out.values()))
        return out

    def indexer(self, i: int) -> dict[str, mx.array]:
        p = self._bp(i) + "attn.indexer."
        out = {"wq_b": self.read(p + "wq_b.weight"),
               "weights_proj": self.read(p + "weights_proj.weight"),
               "compressor": self._compressor(p + "compressor.")}
        mx.eval([out["wq_b"], out["weights_proj"]])
        return out

    def attention(self, i: int) -> dict[str, mx.array]:
        """Low-rank q/kv + grouped-O attention tensors (+ compressor / indexer when present)."""
        p = self._bp(i) + "attn."
        out: dict = {
            "wq_a": self.read(p + "wq_a.weight"),
            "q_norm": self.read(p + "q_norm.weight"),
            "wq_b": self.read(p + "wq_b.weight"),
            "wkv": self.read(p + "wkv.weight"),
            "kv_norm": self.read(p + "kv_norm.weight"),
            "wo_a": self.read(p + "wo_a.weight"),
            "wo_b": self.read(p + "wo_b.weight"),
            "attn_sink": self.read(p + "attn_sink"),
        }
        if self.cfg.has_compressor(i):
            out["compressor"] = self._compressor(p + "compressor.")
        if self.cfg.has_indexer(i):
            out["indexer"] = self.indexer(i)
        return out

    # --- MoE -------------------------------------------------------------------
    def moe_router(self, i: int) -> dict[str, mx.array]:
        """Router: gate.weight (bf16) + hash table (tid2eid) for hash layers, else score bias."""
        p = self._bp(i) + "ffn.gate."
        out = {"weight": self.read(p + "weight")}
        if self.cfg.is_hash(i):
            tkey = p + "tid2eid"
            if not self._has_dense(tkey):
                raise ValueError(f"L{i} is a hash layer but tid2eid is missing")
            out["tid2eid"] = self.get(tkey)                     # [vocab, topk] int, no dequant/cast
        else:
            bkey = p + "bias"
            if self._has_dense(bkey):
                out["bias"] = self.read(bkey)                   # [n_experts]
        return out

    def shared_expert(self, i: int) -> dict[str, mx.array]:
        p = self._bp(i) + "ffn.shared_experts."
        out = {proj: self.read(p + f"{proj}.weight") for proj in ("w1", "w2", "w3")}
        mx.eval(list(out.values()))
        return out

    def expert_stacks(self, i: int, n_experts: int | None = None) -> dict[str, mx.array]:
        """Dequant routed experts into ``[E, out, in]`` bf16 stacks (w1 gate, w3 up, w2 down).

        Streamed per expert (dequant → place → drop shard handles every 16) so only the bf16
        stacks stay resident. ``w1/w3``: ``[E, moe_inter, hidden]``; ``w2``: ``[E, hidden, moe_inter]``.
        """
        ne = n_experts if n_experts is not None else self.cfg.n_routed_experts

        def ek(e: int, proj: str) -> str:
            return f"{self._bp(i)}ffn.experts.{e}.{proj}.weight"

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

    # --- native MTP block ------------------------------------------------------
    def mtp(self, j: int = 0) -> dict[str, mx.array]:
        """MTP block tensors: projections / norms + the inherited HC head params."""
        p = f"mtp.{j}."
        out = {
            "e_proj": self.read(p + "e_proj.weight"),
            "h_proj": self.read(p + "h_proj.weight"),
            "enorm": self.read(p + "enorm.weight"),
            "hnorm": self.read(p + "hnorm.weight"),
            "norm": self.read(p + "norm.weight"),
        }
        for k in ("fn", "base", "scale"):
            out[f"hc_head_{k}"] = self.read(p + f"hc_head_{k}")
        mx.eval(list(out.values()))
        return out

    # --- internal --------------------------------------------------------------
    def _has_dense(self, key: str) -> bool:
        """True iff ``key`` is present as a dense tensor in the index (no dequant resolution)."""
        return key in self.weight_map
