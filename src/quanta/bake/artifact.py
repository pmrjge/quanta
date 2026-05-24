"""Self-contained baked-artifact writer.

Emits an immutable, relocatable bundle the resident runtime can load with no reference to
the source checkpoint: sharded ``*.safetensors`` + ``model.safetensors.index.json``
(weight_map) + ``config.json`` (the source text_config, copied in) + ``manifest.json``
(per-weight quant metadata: bits / group_size / format). **All references are relative
filenames** — no absolute, source, symlink, or cache paths — so the directory can be moved
or copied wholesale. ``manifest.json`` is written once at finalize and never mutated at
runtime; runtime offload state belongs in a sibling ``<artifact>_offload`` dir, never here.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

SHARD_MAX_BYTES = 8 * 1024**3  # ~8 GiB per shard


class ArtifactWriter:
    def __init__(self, out_dir: str | Path, source_config_path: str | Path,
                 shard_max_bytes: int = SHARD_MAX_BYTES) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_max = shard_max_bytes
        self._source_config = json.loads(Path(source_config_path).read_text())
        self._cur: dict[str, mx.array] = {}
        self._cur_bytes = 0
        self._shard = 0
        self.weight_map: dict[str, str] = {}
        self.manifest: dict[str, dict] = {}

    def _flush(self) -> None:
        if not self._cur:
            return
        fn = f"model-{self._shard:05d}.safetensors"
        mx.save_safetensors(str(self.dir / fn), self._cur)
        for k in self._cur:
            self.weight_map[k] = fn  # relative filename only
        self._cur, self._cur_bytes, self._shard = {}, 0, self._shard + 1

    def _put(self, name: str, arr: mx.array) -> None:
        mx.eval(arr)
        self._cur[name] = arr
        self._cur_bytes += arr.nbytes
        if self._cur_bytes >= self.shard_max:
            self._flush()

    def add_quantized(self, key: str, packed: mx.array, scales: mx.array, biases: mx.array,
                      bits: int, group_size: int) -> None:
        """Add an affine-quantized weight (packed codes + scales + biases) under ``key``."""
        self._put(f"{key}.weight_packed", packed)
        self._put(f"{key}.weight_scale", scales)
        self._put(f"{key}.weight_bias", biases)
        self.manifest[key] = {"format": "affine_packed", "bits": bits, "group_size": group_size}

    def add_awq_quantized(self, key: str, packed: mx.array, scales: mx.array, biases: mx.array,
                          awq_scale: mx.array, bits: int, group_size: int) -> None:
        """Add an AWQ int weight: affine codes of ``W·diag(s)`` + the per-input-channel scale ``s``.
        The runtime applies ``x·diag(1/s)`` before the matmul (folded per expert in the gather)."""
        self._put(f"{key}.weight_packed", packed)
        self._put(f"{key}.weight_scale", scales)
        self._put(f"{key}.weight_bias", biases)
        self._put(f"{key}.awq_scale", awq_scale)
        self.manifest[key] = {"format": "awq_packed", "bits": bits, "group_size": group_size}

    def add_dense(self, name: str, arr: mx.array) -> None:
        """Add an unquantized tensor verbatim (norms, router, shared expert, biases)."""
        self._put(name, arr)
        self.manifest[name] = {"format": "dense", "dtype": str(arr.dtype).split(".")[-1]}

    def finalize(self, quant_policy: dict) -> None:
        """Flush the last shard and write index.json, config.json, manifest.json."""
        self._flush()
        (self.dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": self.weight_map}, indent=0)
        )
        config = dict(self._source_config)
        config["quantization_config"] = quant_policy  # self-contained: policy travels with it
        (self.dir / "config.json").write_text(json.dumps(config, indent=2))
        (self.dir / "manifest.json").write_text(
            json.dumps({"format": "quanta", "tensors": self.manifest}, indent=0)
        )
