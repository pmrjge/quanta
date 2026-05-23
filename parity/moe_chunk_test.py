"""MoE token-chunking equivalence (instant, synthetic, CPU — no model load).

MoE is per-token independent, so running tokens through the experts in chunks must be
output-equivalent to one shot. Tiny synthetic SparseMoE; compares ``token_chunk`` small
(forced multi-chunk, incl. a partial last chunk) vs huge (single shot), for both
``sort_dispatch`` modes and ``return_parts``.

    uv run python -m parity.moe_chunk_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.config import KimiTextConfig, YarnRope
from quanta.modeling.moe import SparseMoE

HIDDEN, INTER, E, TOPK, T = 16, 8, 6, 2, 7


def _cfg() -> KimiTextConfig:
    return KimiTextConfig(
        vocab_size=100, hidden_size=HIDDEN, intermediate_size=INTER * 2, moe_intermediate_size=INTER,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=2, v_head_dim=4,
        n_routed_experts=E, n_shared_experts=1, num_experts_per_tok=TOPK, n_group=1, topk_group=1,
        topk_method="noaux_tc", scoring_func="sigmoid", norm_topk_prob=True, routed_scaling_factor=2.0,
        first_k_dense_replace=1, moe_layer_freq=1, rms_norm_eps=1e-6, hidden_act="silu",
        attention_bias=False, max_position_embeddings=1024, bos_token_id=1, eos_token_id=2,
        rope=YarnRope(8.0, 32.0, 1.0, 1.0, 1.0, 512, 10000.0),
    )


def run() -> None:
    mx.random.seed(0)
    moe = SparseMoE(_cfg())
    moe.gate.weight = mx.random.normal((E, HIDDEN))  # varied per-token routing
    moe.set_experts(
        mx.random.normal((E, INTER, HIDDEN)),
        mx.random.normal((E, INTER, HIDDEN)),
        mx.random.normal((E, HIDDEN, INTER)),
    )
    x = mx.random.normal((1, T, HIDDEN))

    print("\n=== MoE token-chunking equivalence (T=7, chunk=3) ===")
    worst = 0.0
    for srt in (False, True):
        moe.sort_dispatch = srt
        moe.token_chunk = 10**9  # single shot
        full = moe(x)
        _, rf, sf = moe(x, return_parts=True)
        moe.token_chunk = 3  # forced multi-chunk [0:3,3:6,6:7]
        chunked = moe(x)
        _, rc, sc = moe(x, return_parts=True)
        d = max(
            mx.max(mx.abs(full - chunked)).item(),
            mx.max(mx.abs(rf - rc)).item(),
            mx.max(mx.abs(sf - sc)).item(),
        )
        worst = max(worst, d)
        print(f"sort_dispatch={str(srt):5s}: chunked vs single max_abs {d:.3e}")
    assert worst < 1e-5, "token-chunking changed the MoE output"
    print("token-chunking is output-equivalent")


if __name__ == "__main__":
    run()
