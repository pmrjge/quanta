"""Parity: NemotronQuantizedMoE (gather_qmm, packed) == NemotronLatentMoE (gather_mm, dequant).

The resident decode path runs the routed experts through ``mx.gather_qmm`` over the artifact's
**packed** int4 stacks (the 4-bit bandwidth win); this gates it output-equivalent to the bf16
module fed the **dequantized** weights — both dequantize the same codes, so they must match to
gather_qmm-vs-gather_mm tolerance. Real layer-1 experts + fc1/fc2/shared/gate from the int4-g64
artifact; random hidden input. This is the #39 correctness gate for the MoE (94% of active params).

    uv run python -m parity.nemotron_qmoe_test
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.nemotron.artifact import NemotronArtifact
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.moe import NemotronLatentMoE, NemotronQuantizedMoE

ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"
LAYER = 1  # first MoE (E) layer


def _qlin(art: NemotronArtifact, key: str) -> nn.QuantizedLinear:
    m = art.manifest[key]
    packed = art.raw(key + ".weight_packed")
    out, in_packed = packed.shape
    in_ = in_packed * 32 // m["bits"]
    ql = nn.QuantizedLinear(in_, out, bias=False, group_size=m["group_size"], bits=m["bits"])
    ql.weight, ql.scales, ql.biases = packed, art.raw(key + ".weight_scale"), art.raw(key + ".weight_bias")
    return ql


def _packed_stack(art: NemotronArtifact, pre: str, proj: str, n: int) -> dict[str, mx.array]:
    p = [art.raw(f"{pre}experts.{e}.{proj}.weight_packed") for e in range(n)]
    s = [art.raw(f"{pre}experts.{e}.{proj}.weight_scale") for e in range(n)]
    b = [art.raw(f"{pre}experts.{e}.{proj}.weight_bias") for e in range(n)]
    return {"packed": mx.stack(p), "scale": mx.stack(s), "bias": mx.stack(b)}


def run() -> None:
    cfg = NemotronHConfig.from_pretrained(ART)
    art = NemotronArtifact(ART)
    pre = f"backbone.layers.{LAYER}.mixer."
    egs = art.manifest[f"{pre}experts.0.up_proj"]["group_size"]
    ebits = art.manifest[f"{pre}experts.0.up_proj"]["bits"]
    n = cfg.n_routed_experts

    # bf16 reference: dequantized weights into the plain module
    ref = NemotronLatentMoE(cfg)
    t = art.moe_nonexpert_tensors(LAYER)
    ref.gate_weight, ref.e_score_correction_bias = t["gate.weight"], t["gate.e_score_correction_bias"]
    ref.fc1_latent_proj.weight, ref.fc2_latent_proj.weight = t["fc1_latent_proj.weight"], t["fc2_latent_proj.weight"]
    ref.shared_up.weight, ref.shared_down.weight = t["shared_experts.up_proj.weight"], t["shared_experts.down_proj.weight"]
    es = art.expert_stacks(LAYER, n)
    ref.set_experts(es["up"], es["down"])

    # quantized: packed weights + gather_qmm
    q = NemotronQuantizedMoE(cfg, group_size=egs, bits=ebits)
    q.gate_weight, q.e_score_correction_bias = t["gate.weight"], t["gate.e_score_correction_bias"]
    q.fc1_latent_proj, q.fc2_latent_proj = _qlin(art, pre + "fc1_latent_proj"), _qlin(art, pre + "fc2_latent_proj")
    q.shared_up, q.shared_down = _qlin(art, pre + "shared_experts.up_proj"), _qlin(art, pre + "shared_experts.down_proj")
    q.set_experts(_packed_stack(art, pre, "up_proj", n), _packed_stack(art, pre, "down_proj", n))

    mx.random.seed(0)
    x = mx.random.normal([1, 64, cfg.hidden_size]).astype(mx.bfloat16)
    yr, yq = ref(x), q(x)
    rel = (mx.linalg.norm((yq - yr).astype(mx.float32)) / mx.linalg.norm(yr.astype(mx.float32))).item()
    print(f"=== NemotronQuantizedMoE parity (layer {LAYER}, {n} experts, int{ebits} g{egs}) ===")
    print(f"output rel err (gather_qmm vs gather_mm) : {rel:.4%}  | {'PASS' if rel < 0.02 else 'FAIL (>2%)'}")


if __name__ == "__main__":
    run()
