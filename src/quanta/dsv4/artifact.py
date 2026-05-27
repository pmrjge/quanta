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

    def _has_dense(self, key: str) -> bool:
        """True iff ``key`` is present in the weight index (works for any format, but used by the
        loader helpers to gate optional dense control tensors like router ``bias`` / ``tid2eid``)."""
        return key in self.weight_map

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

    def expert_stacks_packed(self, i: int, n_experts: int | None = None) -> dict[str, dict[str, mx.array]]:
        """Packed routed experts → ``{w1/w3/w2: {packed, scale, bias, awq_scale, group_size, bits}}``.

        Unlike :meth:`expert_stacks` this does **not** dequantize: it stacks the per-expert int4
        AWQ codes / scales / biases / per-input-channel ``awq_scale`` into ``[E, ...]`` arrays
        consumed by :func:`quanta.dsv4.moe._swiglu_stack_packed` via ``mx.gather_qmm``. Resident
        cost per layer is ~3.0 GB (vs ~12 GB bf16), the win that makes the full 43-layer DSV4
        fit the 490 GB working-set ceiling (#141).

        Streamed per expert in groups of 16; shard handles released between groups so peak load
        residency stays bounded. ``s=1`` cold/RTN experts have an ``awq_scale`` of all ones (the
        bake writes it uniformly so the runtime path is uniform: same div-by-scale for warm/cold).
        """
        ne = n_experts if n_experts is not None else self.cfg.n_routed_experts
        projs = ("w1", "w2", "w3")

        # Resolve int4/group_size from the first expert (uniform across all experts of a layer).
        first_meta = self.manifest[f"{self._bp(i)}ffn.experts.0.w1"]
        bits, group_size = int(first_meta["bits"]), int(first_meta["group_size"])

        # Per-projection accumulators of [E] lists, then stacked at end of layer.
        per_proj: dict[str, dict[str, list[mx.array]]] = {
            proj: {"packed": [], "scale": [], "bias": [], "awq_scale": []} for proj in projs
        }
        for e in range(ne):
            for proj in projs:
                base = f"{self._bp(i)}ffn.experts.{e}.{proj}"
                per_proj[proj]["packed"].append(self.get(base + ".weight_packed"))
                per_proj[proj]["scale"].append(self.get(base + ".weight_scale"))
                per_proj[proj]["bias"].append(self.get(base + ".weight_bias"))
                per_proj[proj]["awq_scale"].append(self.get(base + ".awq_scale"))
            if e % 16 == 15:
                for proj in projs:
                    mx.eval([per_proj[proj][k][-16:] for k in ("packed", "scale", "bias", "awq_scale")])
                self.release()

        stacks: dict[str, dict[str, mx.array]] = {}
        for proj in projs:
            stacks[proj] = {
                "packed": mx.stack(per_proj[proj]["packed"], axis=0),
                "scale": mx.stack(per_proj[proj]["scale"], axis=0),
                "bias": mx.stack(per_proj[proj]["bias"], axis=0),
                "awq_scale": mx.stack(per_proj[proj]["awq_scale"], axis=0),
                "group_size": group_size,
                "bits": bits,
            }
            mx.eval([stacks[proj][k] for k in ("packed", "scale", "bias", "awq_scale")])
            self.release()
        return stacks

    # --- native MTP block ------------------------------------------------------
    def mtp(self, j: int = 0) -> dict[str, mx.array]:
        """MTP-specific tensors (combine + head). For the **full** MTP decoder-block params (attn,
        router, experts, shared, norms, HC) under the same ``mtp.{j}.`` prefix, use
        :class:`MTPArtifactView` — :func:`quanta.dsv4.model.load_block_params` reads it unchanged."""
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


class MTPArtifactView:
    """Proxies a :class:`DSV4Artifact` but reroutes ``layers.{i}.*`` reads to ``mtp.{j}.*``.

    Mirror of :class:`quanta.dsv4.loader.MTPCheckpointView` on the artifact side: by overriding
    ``_bp`` it lets :func:`quanta.dsv4.model.load_block_params` (and the packed-expert path)
    assemble the MTP block's inherited decoder params (attn, router, shared, experts, norms, HC)
    from the artifact unchanged. The view exposes the same surface
    :class:`DSV4Artifact` does, plus the wrapped instance's ``mtp(j)`` for the combine/head.

    Implementation note: ``__getattr__`` rebinds class-level methods to ``self`` (the view) so the
    method's internal ``self._bp(i)`` resolves to the view's override (returning ``mtp.{j}.``)
    rather than the wrapped artifact's ``layers.{i}.``. Non-method attributes (``weight_map``,
    ``manifest``, ``_shards`` …) still forward to ``_art`` so the shard cache stays single-instance.
    """

    def __init__(self, art: DSV4Artifact, j: int = 0) -> None:
        self._art = art
        self._j = j
        self.cfg = art.cfg

    def _bp(self, _i: int) -> str:
        return f"mtp.{self._j}."

    def __getattr__(self, name: str):
        # Look up on the wrapped artifact's class — if it's a method, rebind to `self` so internal
        # `self._bp(i)` calls hit our override. Otherwise forward the instance attribute.
        cls_attr = getattr(type(self._art), name, None)
        if callable(cls_attr) and not isinstance(cls_attr, type):
            return cls_attr.__get__(self, type(self._art))
        return getattr(self._art, name)
