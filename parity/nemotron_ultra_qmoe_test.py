"""Parity (Ultra U4 / M1): NemotronQuantizedMoE (gather_qmm, packed int4-g64) ==
NemotronLatentMoE (gather_mm, dequant) at **Ultra-550B scale** on the shipped int4-RTN artifact.

U3 shipped int4-RTN routed experts; the resident decode path runs those 512 experts per MoE layer
through ``mx.gather_qmm`` over the artifact's **packed** int4 stacks (the 4-bit bandwidth win). This
gates that path output-equivalent to the bf16 ``NemotronLatentMoE`` fed the **dequantized** same
codes — both decode the identical int4 grid (RTN => s=1, so there is no AWQ activation rescale),
hence must match to gather_qmm-vs-gather_mm tolerance.

Unlike the Super-120B sibling gate (``nemotron_qmoe_test``), the quantized side here is built by the
**real runtime constructor** ``build_resident_block(art, cfg, 1).mixer`` — so the gate exercises the
exact resident wiring (per-projection expert stacking, group_size/bits read from the manifest), not
a hand re-wire. Layer 1 = the first MoE (E) layer; rule-8: one MoE layer resident (~5.4 GiB packed +
~21.5 GiB bf16 reference stacks). The #38/U4 correctness gate for the routed experts (94% of the
active params on a MoE layer).

    uv run python -m parity.nemotron_ultra_qmoe_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.nemotron.artifact import NemotronArtifact
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.moe import NemotronLatentMoE, NemotronQuantizedMoE
from quanta.nemotron.runtime import _block_arrays, build_resident_block

ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-quanta_int4rtn_g64"
LAYER = 1  # first MoE (E) layer


def run() -> None:
    cfg = NemotronHConfig.from_pretrained(ART)
    art = NemotronArtifact(ART)
    n = cfg.n_routed_experts
    assert cfg.layer_kind(LAYER) == "moe", f"layer {LAYER} kind={cfg.layer_kind(LAYER)}, expected moe"
    # Fail loud if the wrong artifact slipped in — this gate is Ultra-scale (rule 6).
    assert (n, cfg.moe_latent_size, cfg.moe_intermediate_size) == (512, 2048, 5120), (
        f"unexpected dims: experts={n} latent={cfg.moe_latent_size} inter={cfg.moe_intermediate_size}")

    # Quantized side FIRST = the real resident MoE the decode path builds (gather_qmm over packed
    # int4 g64). Eval its arrays now so the packed stacks are materialized before expert_stacks()
    # below releases the shard mmaps.
    blk = build_resident_block(art, cfg, LAYER)
    mx.eval(_block_arrays(blk))
    q = blk.mixer
    assert isinstance(q, NemotronQuantizedMoE), f"resident mixer is {type(q).__name__}"

    # bf16 reference: the artifact's dequantized weights into the plain gather_mm module.
    ref = NemotronLatentMoE(cfg)
    t = art.moe_nonexpert_tensors(LAYER)
    ref.gate_weight, ref.e_score_correction_bias = t["gate.weight"], t["gate.e_score_correction_bias"]
    ref.fc1_latent_proj.weight = t["fc1_latent_proj.weight"]
    ref.fc2_latent_proj.weight = t["fc2_latent_proj.weight"]
    ref.shared_up.weight = t["shared_experts.up_proj.weight"]
    ref.shared_down.weight = t["shared_experts.down_proj.weight"]
    es = art.expert_stacks(LAYER, n)
    ref.set_experts(es["up"], es["down"])

    mx.random.seed(0)
    x = mx.random.normal([1, 64, cfg.hidden_size]).astype(mx.bfloat16)
    yr, yq = ref(x), q(x)
    rel = (mx.linalg.norm((yq - yr).astype(mx.float32))
           / mx.linalg.norm(yr.astype(mx.float32))).item()
    print(f"=== NemotronQuantizedMoE parity @ ULTRA (layer {LAYER}, {n} experts, "
          f"int{q.bits} g{q.group_size} RTN) ===")
    print(f"  latent={cfg.moe_latent_size}  inter={cfg.moe_intermediate_size}  hidden={cfg.hidden_size}")
    print(f"  output rel err (gather_qmm vs gather_mm dequant) : {rel:.4%}  | "
          f"{'PASS' if rel < 0.02 else 'FAIL (>2%)'}")


if __name__ == "__main__":
    run()
