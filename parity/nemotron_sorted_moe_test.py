"""Nemotron sorted-MoE dispatch parity (model-free, tiny synthetic tensors).

Proves that toggling :attr:`quanta.nemotron.moe.NemotronLatentMoE.sort_dispatch` (the new #133
audit lever — mirrors Kimi's #11 sorted dispatch) is **output-equivalent** to the default
unsorted path. The win is on the post-bake ``gather_qmm`` quantized path where pre-sorting by
expert id gives the gather kernel contiguous expert groups (a known Apple Silicon win for
indexed dequant); parity must hold bit-for-bit modulo the reduction reorder that comes from
sorted vs unsorted GEMM-tile order.

Model-free: a few KB of random tensors, fixed seed, runs in ms on CPU. SAFE alongside a live
GPU job (the running DSV4 bake).

    uv run python -m parity.nemotron_sorted_moe_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.moe import NemotronLatentMoE


def _tiny_cfg() -> NemotronHConfig:
    """Tiny Nemotron-H config with enough experts/topk to exercise sorted dispatch."""
    return NemotronHConfig(
        vocab_size=64, hidden_size=128, num_hidden_layers=1, hybrid_override_pattern="M",
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        attention_bias=False, rope_theta=10000.0, partial_rotary_factor=1.0,
        mamba_num_heads=1, mamba_head_dim=64, mamba_n_groups=1, ssm_state_size=64,
        conv_kernel=4, expand=1, mamba_hidden_act="silu", chunk_size=64, use_conv_bias=True,
        n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
        moe_intermediate_size=32, moe_latent_size=32, moe_shared_expert_intermediate_size=32,
        routed_scaling_factor=1.0, norm_topk_prob=True, n_group=1, topk_group=1,
        norm_eps=1e-6, max_position_embeddings=128,
        bos_token_id=0, eos_token_id=1, pad_token_id=0,
        num_nextn_predict_layers=0, tie_word_embeddings=False,
    )


def run() -> None:
    cfg = _tiny_cfg()
    mx.random.seed(0)
    moe = NemotronLatentMoE(cfg)
    # randomize expert stacks + projections so the unsorted/sorted dispatch produces non-trivial
    # output — same weights drive both paths so any drift is purely from the dispatch reorder.
    e, inter, lat = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.moe_latent_size
    mx.random.seed(1)
    up = mx.random.normal((e, inter, lat)) * 0.1
    down = mx.random.normal((e, lat, inter)) * 0.1
    moe.set_experts(up, down)
    moe.gate_weight = mx.random.normal((e, cfg.hidden_size)) * 0.1
    mx.eval(moe.parameters())

    n = 20  # >> topk*experts so most experts are hit; sort makes a non-trivial permutation
    x = mx.random.normal((1, n, cfg.hidden_size)).astype(mx.bfloat16)

    moe.sort_dispatch = False
    y_unsorted = moe(x)
    moe.sort_dispatch = True
    y_sorted = moe(x)
    mx.eval(y_unsorted, y_sorted)

    max_diff = float(mx.max(mx.abs(y_unsorted - y_sorted)).item())
    ref_scale = float(mx.max(mx.abs(y_unsorted)).item()) + 1e-6
    rel = max_diff / ref_scale

    # The two paths use the exact same per-row math; sorted only changes the GEMM-tile order so
    # any drift is the reorder of an associative reduction in bf16. A few ULPs is expected; argmax
    # over the hidden axis (the row-shape we care about for routing) must be identical.
    arg_u = mx.argmax(y_unsorted, axis=-1)
    arg_s = mx.argmax(y_sorted, axis=-1)
    agree = float(mx.mean((arg_u == arg_s).astype(mx.float32)).item())
    REL_TOL = 1.0e-2
    ok = rel < REL_TOL and agree >= 0.99
    print(f"n={n}  max_abs_diff={max_diff:.4e}  rel={rel:.3e}  argmax_agree={agree:.3f}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
