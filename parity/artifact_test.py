"""Validate the artifact writer: structure, self-containment, relative refs, round-trip.

Writes a tiny artifact (one quantized weight + one dense tensor), then re-reads it: the
index/manifest must use relative filenames only, config.json must carry text_config +
quantization_config (loadable with no source), and the quantized weight must dequantize
exactly via mx.dequantize from the on-disk shard.

    uv run python -m parity.artifact_test
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlx.core as mx

from quanta.bake.artifact import ArtifactWriter
from quanta.bake.quant import quantize_affine

MODEL = "/Users/pmrj/models/Kimi-K2.6"


def run() -> None:
    out = Path(tempfile.mkdtemp()) / "kimi-quanta_int3"
    w = ArtifactWriter(out, Path(MODEL) / "config.json")

    weight = mx.random.normal((64, 256))
    bits, gs = 3, 128
    packed, scales, biases = quantize_affine(weight, bits, gs)
    qkey = "language_model.model.layers.1.mlp.experts.0.gate_proj"
    w.add_quantized(qkey, packed, scales, biases, bits, gs)
    w.add_dense("language_model.model.layers.1.input_layernorm.weight", mx.random.normal((256,)))
    w.finalize({"experts": "int3/int4 gptq g128", "non_experts": "int8 g128", "shared": "bf16"})

    idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = json.loads((out / "config.json").read_text())
    man = json.loads((out / "manifest.json").read_text())

    rel_only = all(("/" not in v and not v.startswith(".") and ":" not in v) for v in idx.values())
    self_contained = "text_config" in cfg and "quantization_config" in cfg
    has_manifest = man["tensors"][qkey]["bits"] == 3 and man["tensors"][qkey]["format"] == "affine_packed"

    shard = mx.load(str(out / idx[qkey + ".weight_packed"]))
    recon = mx.dequantize(shard[qkey + ".weight_packed"], shard[qkey + ".weight_scale"],
                          shard[qkey + ".weight_bias"], group_size=gs, bits=bits)
    ref = mx.dequantize(packed, scales, biases, group_size=gs, bits=bits)
    drift = mx.max(mx.abs(recon - ref)).item()

    print("\n=== artifact writer ===")
    print(f"weight_map entries   : {len(idx)}  relative-only: {rel_only}")
    print(f"config self-contained: {self_contained}  (text_config + quantization_config)")
    print(f"manifest quant meta  : {has_manifest}")
    print(f"dequant round-trip   : max_abs {drift:.3e}")
    assert rel_only and self_contained and has_manifest and drift < 1e-6
    print(f"artifact OK at {out}")


if __name__ == "__main__":
    run()
