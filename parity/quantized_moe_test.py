"""Validate dynamic mixed int3/int4 QuantizedSparseMoE == bf16 SparseMoE on dequant weights.

Widths are assigned **per (expert, projection)** — experts are deliberately mixed *within*
themselves (e.g. expert 0 is gate=int3, up=int3, down=int4), the real-artifact case the old
per-expert assumption missed. The gather_qmm path (per-(proj,width) stacks, remapped indices,
select by the row-expert's width for that projection) must equal running the same dequantized
experts through the bf16 SparseMoE. Router + shared shared between the two.

    uv run python -m parity.quantized_moe_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.config import KimiTextConfig, YarnRope
from quanta.modeling.moe import SparseMoE
from quanta.modeling.quantized import QuantizedSparseMoE

HID, INTER, E, TOPK, N, GS = 256, 128, 6, 2, 8, 128
# per-(expert, projection) widths — mixed within experts so no expert is uniform across projs
PBITS = {
    "gate": [3, 3, 4, 4, 3, 4],
    "up":   [3, 4, 3, 4, 4, 3],
    "down": [4, 3, 3, 4, 3, 4],
}


def _cfg() -> KimiTextConfig:
    return KimiTextConfig(
        vocab_size=100, hidden_size=HID, intermediate_size=INTER * 2, moe_intermediate_size=INTER,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, q_lora_rank=8,
        kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=2, v_head_dim=4, n_routed_experts=E,
        n_shared_experts=1, num_experts_per_tok=TOPK, n_group=1, topk_group=1, topk_method="noaux_tc",
        scoring_func="sigmoid", norm_topk_prob=True, routed_scaling_factor=2.0, first_k_dense_replace=1,
        moe_layer_freq=1, rms_norm_eps=1e-6, hidden_act="silu", attention_bias=False,
        max_position_embeddings=1024, bos_token_id=1, eos_token_id=2,
        rope=YarnRope(8.0, 32.0, 1.0, 1.0, 1.0, 512, 10000.0),
    )


def run() -> None:
    mx.random.seed(0)
    cfg = _cfg()
    shapes = {"gate": (INTER, HID), "up": (INTER, HID), "down": (HID, INTER)}
    W = {p: mx.random.normal((E,) + s) for p, s in shapes.items()}

    # per-(proj, width) packed stacks + per-(expert,proj) dequant for the bf16 reference
    stacks: dict[str, dict[int, dict[str, mx.array]]] = {p: {} for p in W}
    deq: dict[str, list] = {p: [None] * E for p in W}
    pslot = {p: [0] * E for p in W}
    for p in W:
        by = {b: {"packed": [], "scale": [], "bias": []} for b in (3, 4)}
        for e in range(E):
            bits = PBITS[p][e]
            qd, s, b = mx.quantize(W[p][e], group_size=GS, bits=bits)
            pslot[p][e] = len(by[bits]["packed"])
            by[bits]["packed"].append(qd)
            by[bits]["scale"].append(s)
            by[bits]["bias"].append(b)
            deq[p][e] = mx.dequantize(qd, s, b, group_size=GS, bits=bits)
        stacks[p] = {bits: {k: mx.stack(v) for k, v in d.items()} for bits, d in by.items() if d["packed"]}
    pbits = {p: mx.array(PBITS[p], mx.int32) for p in W}
    pslot = {p: mx.array(pslot[p], mx.int32) for p in W}

    ref = SparseMoE(cfg)
    ref.gate.weight = mx.random.normal((E, HID))
    ref.set_experts(*(mx.stack([deq[p][e] for e in range(E)]) for p in ("gate", "up", "down")))

    q = QuantizedSparseMoE(cfg, group_size=GS)
    q.gate.weight = ref.gate.weight
    q.gate.e_score_correction_bias = ref.gate.e_score_correction_bias
    q.shared_experts = ref.shared_experts  # identical router + shared isolate the expert path
    q.set_experts(stacks, pbits, pslot)

    x = mx.random.normal((1, N, HID)).astype(mx.bfloat16)
    err = mx.max(mx.abs(ref(x) - q(x))).item()
    print("\n=== dynamic mixed int3/int4 QuantizedSparseMoE (per-projection widths) ===")
    print(f"widths mixed within experts;  gather_qmm vs bf16(dequant): max_abs {err:.3e}")
    # bf16-level (gather_qmm's fused dequant+matmul vs explicit dequant->gather_mm); a logic
    # bug (wrong expert/width/projection select) would be O(1), not sub-percent.
    assert err < 2e-2, "quantized MoE must match the dequant path (bf16 tol)"
    print("dynamic mixed quantized MoE matches dequant path (bf16 precision)")


if __name__ == "__main__":
    run()
