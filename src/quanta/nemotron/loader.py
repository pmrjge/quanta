"""Streamed, layer-by-layer loader for the Nemotron-H bf16 source checkpoint (pure MLX).

Reads tensors from the sharded ``safetensors`` via ``mx.load`` (lazy / mmap), one layer
resident at a time (memory discipline). Unlike Kimi's int4 source, Nemotron is bf16, so
experts are read directly (no dequant). Tensors live under ``backbone.layers.{i}.mixer.*``
with per-layer input norm ``backbone.layers.{i}.norm.weight``; embeddings/head are
``backbone.embeddings.weight`` / ``lm_head.weight`` / ``backbone.norm_f.weight``.

``shape(key)`` reads a tensor's shape from the lazy handle without materializing data
(cheap wiring checks); ``read(key)`` materializes one tensor; ``release()`` drops shard
handles so materialized layers can be freed.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

PREFIX = "backbone."

MAMBA_SUFFIXES: tuple[str, ...] = (
    "in_proj.weight", "out_proj.weight", "conv1d.weight", "conv1d.bias",
    "A_log", "D", "dt_bias", "norm.weight",
)
ATTENTION_SUFFIXES: tuple[str, ...] = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
)
MOE_NONEXPERT_SUFFIXES: tuple[str, ...] = (
    "gate.weight", "gate.e_score_correction_bias",
    "fc1_latent_proj.weight", "fc2_latent_proj.weight",
    "shared_experts.up_proj.weight", "shared_experts.down_proj.weight",
)


class NemotronSourceCheckpoint:
    """Lazy, streamed reader over the sharded Nemotron-H source checkpoint."""

    def __init__(self, model_dir: str | Path) -> None:
        self.dir = Path(model_dir)
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._shards: dict[str, dict[str, mx.array]] = {}

    def _lazy(self, key: str) -> mx.array:
        if key not in self.weight_map:
            raise KeyError(f"tensor not in source index: {key}")
        fn = self.weight_map[key]
        shard = self._shards.get(fn)
        if shard is None:
            shard = mx.load(str(self.dir / fn))  # lazy / mmap
            self._shards[fn] = shard
        return shard[key]

    def shape(self, key: str) -> tuple[int, ...]:
        """Tensor shape from the lazy handle — no data materialized."""
        return tuple(self._lazy(key).shape)

    def read(self, key: str) -> mx.array:
        arr = self._lazy(key)
        mx.eval(arr)
        return arr

    def release(self) -> None:
        self._shards.clear()

    # --- key helpers -----------------------------------------------------------
    def mixer_key(self, layer_idx: int, suffix: str) -> str:
        return f"{PREFIX}layers.{layer_idx}.mixer.{suffix}"

    def norm_key(self, layer_idx: int) -> str:
        return f"{PREFIX}layers.{layer_idx}.norm.weight"

    def expert_key(self, layer_idx: int, expert: int, proj: str) -> str:
        return f"{PREFIX}layers.{layer_idx}.mixer.experts.{expert}.{proj}.weight"

    # --- per-kind loaders (materialized; one layer resident) -------------------
    def mamba_tensors(self, layer_idx: int) -> dict[str, mx.array]:
        out = {suf: self.read(self.mixer_key(layer_idx, suf)) for suf in MAMBA_SUFFIXES}
        out["conv1d.weight"] = out["conv1d.weight"].reshape(self.shape(self.mixer_key(layer_idx, "conv1d.weight"))[0], -1)
        out["layer_norm"] = self.read(self.norm_key(layer_idx))
        mx.eval(list(out.values()))
        return out

    def attention_tensors(self, layer_idx: int) -> dict[str, mx.array]:
        out = {suf: self.read(self.mixer_key(layer_idx, suf)) for suf in ATTENTION_SUFFIXES}
        out["layer_norm"] = self.read(self.norm_key(layer_idx))
        mx.eval(list(out.values()))
        return out

    def moe_nonexpert_tensors(self, layer_idx: int) -> dict[str, mx.array]:
        out = {suf: self.read(self.mixer_key(layer_idx, suf)) for suf in MOE_NONEXPERT_SUFFIXES}
        out["layer_norm"] = self.read(self.norm_key(layer_idx))
        mx.eval(list(out.values()))
        return out

    def expert_stacks(self, layer_idx: int, n_experts: int) -> dict[str, mx.array]:
        """Stack routed experts into ``[E, out, in]`` (bf16, no dequant — source is bf16).

        Streamed per expert (read -> place -> drop shard handles) so only the bf16 stacks
        stay resident, never the whole shard set.
        """
        up0 = self.read(self.expert_key(layer_idx, 0, "up_proj"))
        down0 = self.read(self.expert_key(layer_idx, 0, "down_proj"))
        up = mx.zeros((n_experts, *up0.shape), up0.dtype)
        down = mx.zeros((n_experts, *down0.shape), down0.dtype)
        up[0], down[0] = up0, down0
        for e in range(1, n_experts):
            up[e] = self.read(self.expert_key(layer_idx, e, "up_proj"))
            down[e] = self.read(self.expert_key(layer_idx, e, "down_proj"))
            if e % 32 == 31:
                mx.eval(up, down)
                self.release()
        mx.eval(up, down)
        self.release()
        return {"up": up, "down": down}
