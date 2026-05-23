"""Resident quantized runtime — load a baked artifact and build runnable decoder layers.

The artifact stores attention / dense-MLP / lm_head weights as affine int8 (mx.quantize
layout), routed experts as packed int3/int4, and shared expert + norms + router as bf16.
Because ``MLAAttention``/``DenseMLP`` invoke their projections as ``proj(x)``, we just swap
``nn.Linear`` → ``nn.QuantizedLinear`` (same call) and the existing forward runs unchanged;
routed experts use :class:`QuantizedSparseMoE` (gather_qmm). Norms/shared/router load bf16.

``ResidentModel`` mirrors :class:`KimiModel.__call__`, so :func:`quanta.generate.generate`
and the ppl harness run on it directly. Built for the full ~427 GB artifact held RAM-resident
(``mx.set_wired_limit``); ``build_resident_layer`` is the per-layer unit, validatable on a
single baked layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from quanta.config import KimiTextConfig
from quanta.loader import TEXT_PREFIX
from quanta.modeling.decoder import DenseDecoderLayer, MoEDecoderLayer
from quanta.modeling.quantized import QuantizedSparseMoE


class ResidentArtifact:
    """Reader over a baked artifact dir (config.json + manifest.json + index + safetensors)."""

    def __init__(self, art_dir: str | Path) -> None:
        self.dir = Path(art_dir)
        self.cfg = KimiTextConfig.from_pretrained(art_dir)  # self-contained config
        self.weight_map: dict[str, str] = json.loads(
            (self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.manifest: dict[str, dict] = json.loads((self.dir / "manifest.json").read_text())["tensors"]
        self._shards: dict[str, dict[str, mx.array]] = {}

    def get(self, name: str) -> mx.array:
        fn = self.weight_map[name]
        if fn not in self._shards:
            self._shards[fn] = mx.load(str(self.dir / fn))
        return self._shards[fn][name]

    def release(self) -> None:
        self._shards.clear()


def _quant_linear(art: ResidentArtifact, key: str) -> nn.QuantizedLinear:
    """Build an nn.QuantizedLinear from an int8/affine ``key`` (weight_packed/scale/bias)."""
    m = art.manifest[key]
    packed = art.get(f"{key}.weight_packed")
    out, in_packed = packed.shape
    in_ = in_packed * 32 // m["bits"]
    ql = nn.QuantizedLinear(in_, out, bias=False, group_size=m["group_size"], bits=m["bits"])
    ql.weight = packed
    ql.scales = art.get(f"{key}.weight_scale")
    ql.biases = art.get(f"{key}.weight_bias")
    return ql


def _load_quant_attention(layer, art: ResidentArtifact, pre: str) -> None:
    for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
        setattr(layer.self_attn, proj, _quant_linear(art, f"{pre}self_attn.{proj}"))
    layer.self_attn.q_a_layernorm.weight = art.get(f"{pre}self_attn.q_a_layernorm.weight")
    layer.self_attn.kv_a_layernorm.weight = art.get(f"{pre}self_attn.kv_a_layernorm.weight")
    layer.input_layernorm.weight = art.get(f"{pre}input_layernorm.weight")
    layer.post_attention_layernorm.weight = art.get(f"{pre}post_attention_layernorm.weight")


def build_resident_layer(art: ResidentArtifact, layer_idx: int):
    """Build a runnable quantized decoder layer (dense L0 or MoE) from the artifact."""
    cfg = art.cfg
    pre = f"{TEXT_PREFIX}layers.{layer_idx}."
    if cfg.is_dense_layer(layer_idx):
        layer = DenseDecoderLayer(cfg)
        _load_quant_attention(layer, art, pre)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            setattr(layer.mlp, proj, _quant_linear(art, f"{pre}mlp.{proj}"))
        return layer

    layer = MoEDecoderLayer(cfg)
    _load_quant_attention(layer, art, pre)
    qmoe = QuantizedSparseMoE(cfg)
    qmoe.gate.weight = art.get(f"{pre}mlp.gate.weight")
    qmoe.gate.e_score_correction_bias = art.get(f"{pre}mlp.gate.e_score_correction_bias")
    for proj in ("gate_proj", "up_proj", "down_proj"):
        setattr(qmoe.shared_experts, proj, _quant_linear_or_dense(art, qmoe.shared_experts, proj, pre))
    qmoe.set_experts(*_load_expert_stacks(art, cfg, pre))
    layer.mlp = qmoe
    return layer


def _quant_linear_or_dense(art, module, proj, pre):
    """Shared expert is bf16-dense in the artifact → keep the nn.Linear, just load its weight."""
    lin = getattr(module, proj)
    lin.weight = art.get(f"{pre}mlp.shared_experts.{proj}.weight")
    return lin


def _load_expert_stacks(art: ResidentArtifact, cfg: KimiTextConfig, pre: str):
    """Group routed experts by quant width into per-width packed stacks for gather_qmm."""
    per_width: dict[int, dict[str, list]] = {}
    expert_bits = [0] * cfg.n_routed_experts
    slots = [0] * cfg.n_routed_experts
    for e in range(cfg.n_routed_experts):
        bits = art.manifest[f"{pre}mlp.experts.{e}.gate_proj"]["bits"]
        expert_bits[e] = bits
        d = per_width.setdefault(bits, {f"{p}_{c}": [] for p in ("gate", "up", "down")
                                        for c in ("packed", "scale", "bias")})
        slots[e] = len(d["gate_packed"])
        for p in ("gate", "up", "down"):
            k = f"{pre}mlp.experts.{e}.{p}_proj"
            d[f"{p}_packed"].append(art.get(f"{k}.weight_packed"))
            d[f"{p}_scale"].append(art.get(f"{k}.weight_scale"))
            d[f"{p}_bias"].append(art.get(f"{k}.weight_bias"))
    stacks = {bits: {k: mx.stack(v) for k, v in d.items()} for bits, d in per_width.items()}
    return stacks, mx.array(expert_bits), mx.array(slots)
