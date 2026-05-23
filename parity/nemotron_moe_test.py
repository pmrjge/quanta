"""Parity for the Nemotron-H latent relu^2 MoE: sparse gather == dense, chunk-invariant.

Small config (via dataclasses.replace) so it's tiny and safe alongside the bake. Validates
the gather dispatch + relu^2 experts + fc1/fc2 latent projections + shared expert against a
dead-simple dense reference, and that token-chunking is output-equivalent.

    uv run python -m parity.nemotron_moe_test
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx

from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.moe import NemotronLatentMoE, relu2

MODEL = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"


def _rel(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)) / (mx.max(mx.abs(b)) + 1e-6))


def run() -> None:
    mx.random.seed(0)
    cfg = replace(
        NemotronHConfig.from_pretrained(MODEL),
        hidden_size=32, moe_latent_size=16, moe_intermediate_size=24,
        moe_shared_expert_intermediate_size=20, n_routed_experts=8, num_experts_per_tok=3,
    )
    e, lat, inter, topk = cfg.n_routed_experts, cfg.moe_latent_size, cfg.moe_intermediate_size, cfg.num_experts_per_tok

    moe = NemotronLatentMoE(cfg)
    moe.gate_weight = mx.random.normal((e, cfg.hidden_size))
    moe.e_score_correction_bias = mx.random.normal((e,))
    moe.up_stack = mx.random.normal((e, inter, lat)) * 0.1
    moe.down_stack = mx.random.normal((e, lat, inter)) * 0.1

    x = mx.random.normal((1, 6, cfg.hidden_size)) * 0.5
    out = moe(x)

    # dense reference: compute each token's top-k experts explicitly in latent, then fc2 + shared
    xf = x.reshape(6, cfg.hidden_size)
    idx, w = moe._route(xf)
    latent = moe.fc1_latent_proj(xf)
    rows = []
    for tk in range(6):
        acc = mx.zeros((lat,))
        for s in range(topk):
            ex = int(idx[tk, s].item())
            acc = acc + w[tk, s] * (moe.down_stack[ex] @ relu2(moe.up_stack[ex] @ latent[tk]))
        rows.append(acc)
    ref = moe.fc2_latent_proj(mx.stack(rows, 0)) + moe.shared_down(relu2(moe.shared_up(xf)))
    dense_ok = _rel(out.reshape(6, cfg.hidden_size), ref) < 1e-4

    # token-chunking is output-equivalent
    moe.token_chunk = 2
    out_chunked = moe(x)
    chunk_ok = _rel(out_chunked, out) < 1e-5

    print("\n=== Nemotron-H latent MoE parity ===")
    print(f"sparse gather == dense reference     : {dense_ok}  rel={_rel(out.reshape(6, cfg.hidden_size), ref):.2e}")
    print(f"token-chunking invariant             : {chunk_ok}")
    assert all([dense_ok, chunk_ok])
    print("Nemotron-H latent MoE OK (gather == dense; chunk-invariant)")


if __name__ == "__main__":
    run()
