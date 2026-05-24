"""Streamed reader over the baked Nemotron-H quanta artifact — dequantizes to bf16.

Duck-types :class:`quanta.nemotron.loader.NemotronSourceCheckpoint`: it exposes the same
``mamba_tensors`` / ``attention_tensors`` / ``moe_nonexpert_tensors`` / ``expert_stacks`` /
``read`` / ``release`` surface and returns the **same suffix-keyed dicts**, so
:func:`quanta.nemotron.model.load_block` and the bf16 ppl harness run against it unchanged.
This is the **parity reference for the int4 quant gate (#38)**: the artifact's packed weights
dequantized back to bf16 and run through the proven naive forward — the float baseline the
resident ``quantized_matmul`` runtime (#39) must then match.

Storage contract (mirrors :class:`quanta.bake.artifact.ArtifactWriter`):

* ``dense``         → ``{key}`` verbatim (SSM core, conv1d [already ``[dim,k]``], every norm,
  router gate, embeddings/head/norm_f).
* ``affine_packed`` → int8 g128 at ``{base}`` (the suffix **without** ``.weight``), as
  ``{base}.weight_packed`` + ``.weight_scale`` + ``.weight_bias``; ``mx.dequantize`` inverts it.
* ``awq_packed``    → int4 g128 experts at ``{base}`` (per-expert ``up_proj``/``down_proj``):
  the affine codes are of ``W·diag(s)``; recover ``W = dequantize(...) / s`` with the stored
  per-input-channel ``{base}.awq_scale`` (``s=1`` for cold/RTN experts → identity).

One layer resident at a time (rule-8): expert stacks dequant streamed, evaluated and the shard
handles released every 32 experts. Unmapped keys fail loud (rule-6) — never a silent default.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from quanta.nemotron.loader import (
    ATTENTION_SUFFIXES,
    MAMBA_SUFFIXES,
    MOE_NONEXPERT_SUFFIXES,
    PREFIX,
)


class NemotronArtifact:
    """Lazy, streamed, dequantizing reader over a baked Nemotron-H quanta artifact."""

    def __init__(self, art_dir: str | Path) -> None:
        self.dir = Path(art_dir)
        self.weight_map: dict[str, str] = json.loads(
            (self.dir / "model.safetensors.index.json").read_text()
        )["weight_map"]
        man = json.loads((self.dir / "manifest.json").read_text())
        if man.get("format") != "quanta":
            raise ValueError(f"{self.dir}: not a quanta artifact (format={man.get('format')!r})")
        self.manifest: dict[str, dict] = man["tensors"]
        self._shards: dict[str, dict[str, mx.array]] = {}

    # --- low-level shard access ------------------------------------------------
    def _lazy(self, key: str) -> mx.array:
        if key not in self.weight_map:
            raise KeyError(f"tensor not in artifact index: {key}")
        fn = self.weight_map[key]
        shard = self._shards.get(fn)
        if shard is None:
            shard = mx.load(str(self.dir / fn))  # lazy / mmap
            self._shards[fn] = shard
        return shard[key]

    def _read_raw(self, key: str) -> mx.array:
        arr = self._lazy(key)
        mx.eval(arr)
        return arr

    def release(self) -> None:
        """Drop cached shard handles so materialized tensors can be freed."""
        self._shards.clear()

    # --- dequant ---------------------------------------------------------------
    def _dequant(self, base: str, meta: dict) -> mx.array:
        """Dequantize a packed weight at ``base`` → bf16 ``[out, in]`` (AWQ unscaled by ``1/s``)."""
        gs, bits = int(meta["group_size"]), int(meta["bits"])
        w = mx.dequantize(
            self._read_raw(base + ".weight_packed"),
            self._read_raw(base + ".weight_scale"),
            self._read_raw(base + ".weight_bias"),
            group_size=gs,
            bits=bits,
        )
        if meta["format"] == "awq_packed":  # codes are of W·diag(s); recover W per input channel
            s = self._read_raw(base + ".awq_scale")
            w = w / s[None, :]
        return w.astype(mx.bfloat16)

    def _materialize(self, key: str) -> mx.array:
        """Resolve a source-style key to a bf16 tensor: dense verbatim, else dequant the packed
        weight stored at the key minus its trailing ``.weight``. Fail loud if neither (rule-6)."""
        meta = self.manifest.get(key)
        if meta is not None and meta["format"] == "dense":
            return self._read_raw(key)
        base = key[: -len(".weight")] if key.endswith(".weight") else key
        bmeta = self.manifest.get(base)
        if bmeta is not None and bmeta["format"] in ("affine_packed", "awq_packed"):
            return self._dequant(base, bmeta)
        raise KeyError(f"{key}: not in artifact manifest (no dense entry, no quantized base {base!r})")

    def read(self, key: str) -> mx.array:
        """Materialize one tensor by full key (dense or, if packed, dequantized)."""
        return self._materialize(key)

    def raw(self, key: str) -> mx.array:
        """Materialize a stored tensor verbatim (no dequant) — e.g. ``{base}.weight_packed`` for
        the resident ``gather_qmm``/``quantized_matmul`` path that consumes packed codes directly."""
        return self._read_raw(key)

    # --- key helpers (match the source checkpoint) -----------------------------
    def mixer_key(self, layer_idx: int, suffix: str) -> str:
        return f"{PREFIX}layers.{layer_idx}.mixer.{suffix}"

    def norm_key(self, layer_idx: int) -> str:
        return f"{PREFIX}layers.{layer_idx}.norm.weight"

    # --- per-kind loaders (same dicts as NemotronSourceCheckpoint) -------------
    def _layer_dict(self, layer_idx: int, suffixes: tuple[str, ...]) -> dict[str, mx.array]:
        out = {suf: self._materialize(self.mixer_key(layer_idx, suf)) for suf in suffixes}
        out["layer_norm"] = self._materialize(self.norm_key(layer_idx))
        mx.eval(list(out.values()))
        return out

    def mamba_tensors(self, layer_idx: int) -> dict[str, mx.array]:
        return self._layer_dict(layer_idx, MAMBA_SUFFIXES)

    def attention_tensors(self, layer_idx: int) -> dict[str, mx.array]:
        return self._layer_dict(layer_idx, ATTENTION_SUFFIXES)

    def moe_nonexpert_tensors(self, layer_idx: int) -> dict[str, mx.array]:
        return self._layer_dict(layer_idx, MOE_NONEXPERT_SUFFIXES)

    def expert_stacks(self, layer_idx: int, n_experts: int) -> dict[str, mx.array]:
        """Dequantize all routed experts into stacked bf16 ``[E, out, in]`` (AWQ int4 → W).

        Streamed per expert (dequant → place → drop shard handles every 32) so only the growing
        bf16 stacks stay resident, not the whole shard set."""
        def ekey(e: int, proj: str) -> str:
            return f"{PREFIX}layers.{layer_idx}.mixer.experts.{e}.{proj}"

        up0 = self._materialize(ekey(0, "up_proj"))
        down0 = self._materialize(ekey(0, "down_proj"))
        up = mx.zeros((n_experts, *up0.shape), mx.bfloat16)
        down = mx.zeros((n_experts, *down0.shape), mx.bfloat16)
        up[0], down[0] = up0, down0
        for e in range(1, n_experts):
            up[e] = self._materialize(ekey(e, "up_proj"))
            down[e] = self._materialize(ekey(e, "down_proj"))
            if e % 32 == 31:
                mx.eval(up, down)
                self.release()
        mx.eval(up, down)
        self.release()
        return {"up": up, "down": down}
