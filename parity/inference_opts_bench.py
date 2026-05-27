"""Model-free perf bench for the #133 inference optimizations (tiny synthetic, MLX-only).

Times three levers head-to-head on architecture-faithful but small shapes so the bench runs in
seconds without competing with the live DSV4 bake on the GPU:

1. **Kimi MLA absorbed decode** — explicit-softmax (parity baseline) vs ``mx.fast.scaled_dot_product_attention``
   over MQA-shaped ``(q_absorb || q_pe, c || k_pe, c)``. Single-token decode over a growing
   KV (the long-decode hot path).
2. **Nemotron GQA decode cache** — bf16 ``KVCache`` vs the new ``quantized=True`` int8 g128
   (#133). Append + read at decode time; the cache hands SDPA back a bf16 stream in both modes
   so the measured delta is purely the int8 round-trip on the stored k/v.
3. **Nemotron MoE dispatch** — unsorted ``gather_mm`` vs the new ``sort_dispatch=True`` path
   (mirror of Kimi #11). Same expert stack + routing, different gather order; output is
   bit-identical (verified in ``nemotron_sorted_moe_test``) so this measures the dispatch.

The numbers below are NOT the production decode throughput on the real models (those depend on
quantized weights, fused decode-step compile, and the long-context KV residency that's the
real win for int8 KV). They are *which path the kernels prefer at the same shape*, so we can
rank optimizations before paying for an end-to-end ppl gate.

    uv run python -m parity.inference_opts_bench
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.config import KimiTextConfig, YarnRope
from quanta.modeling.attention import MLAAttention
from quanta.nemotron.attention import KVCache, NemotronAttention
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.moe import NemotronLatentMoE


# ---- common timing helper ---------------------------------------------------
def _time(fn, warmup: int = 5, iters: int = 20) -> float:
    """Run ``fn`` ``iters`` times after ``warmup`` warmups; return per-call seconds."""
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        mx.eval(out)
    return (time.perf_counter() - t0) / iters


# ---- 1) Kimi MLA absorbed decode --------------------------------------------
def _kimi_attn() -> MLAAttention:
    """Architecture-faithful absorbed-MLA module: H=4 (tiny), real kv_lora=512/rope=64/nope=128."""
    cfg = KimiTextConfig(
        vocab_size=128, hidden_size=256, intermediate_size=512, moe_intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        q_lora_rank=64, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        n_routed_experts=1, n_shared_experts=1, num_experts_per_tok=1,
        n_group=1, topk_group=1, topk_method="noaux_tc", scoring_func="sigmoid",
        norm_topk_prob=True, routed_scaling_factor=1.0,
        first_k_dense_replace=1, moe_layer_freq=1,
        rms_norm_eps=1e-6, hidden_act="silu", attention_bias=False,
        max_position_embeddings=16384, bos_token_id=0, eos_token_id=1,
        rope=YarnRope(factor=64.0, beta_fast=32.0, beta_slow=1.0, mscale=1.0, mscale_all_dim=1.0,
                      original_max_position_embeddings=4096, rope_theta=50000.0),
    )
    mx.random.seed(0)
    attn = MLAAttention(cfg)
    attn.absorbed = True
    mx.eval(attn.parameters())
    return attn


def bench_kimi_absorbed_decode() -> None:
    print("\n=== Kimi MLA absorbed decode: explicit-softmax vs fast SDPA ===")
    attn = _kimi_attn()
    b, h, t = 1, attn.num_heads, 1
    rope, kv_lora = attn.rope, attn.cfg.kv_lora_rank

    # Force the W_UK/W_UV cache to land so we don't time first-call dequantize.
    attn._absorbed(
        mx.zeros((b, h, t, attn.nope)).astype(mx.bfloat16),
        mx.zeros((b, h, t, rope)).astype(mx.bfloat16),
        mx.zeros((b, 32, kv_lora)).astype(mx.bfloat16),
        mx.broadcast_to(mx.zeros((b, 1, 32, rope)).astype(mx.bfloat16), (b, h, 32, rope)),
        b, t, 32, use_fast=True,
    )
    mx.eval(attn._w_uk_cache, attn._w_uv_cache)

    print(f"{'kv_len':>8} {'ref_ms':>10} {'fast_ms':>10} {'speedup':>10}")
    for kv_len in (1024, 4096, 16384):
        mx.random.seed(42)
        q_nope = mx.random.normal((b, h, t, attn.nope)).astype(mx.bfloat16)
        q_pe = mx.random.normal((b, h, t, rope)).astype(mx.bfloat16)
        c_kv = mx.random.normal((b, kv_len, kv_lora)).astype(mx.bfloat16)
        k_pe_1 = mx.random.normal((b, 1, kv_len, rope)).astype(mx.bfloat16)
        k_pe = mx.broadcast_to(k_pe_1, (b, h, kv_len, rope))
        mx.eval(q_nope, q_pe, c_kv, k_pe)

        def ref():
            return attn._absorbed(q_nope, q_pe, c_kv, k_pe, b, t, kv_len, use_fast=False)

        def fast():
            return attn._absorbed(q_nope, q_pe, c_kv, k_pe, b, t, kv_len, use_fast=True)

        t_ref = _time(ref, warmup=3, iters=15)
        t_fast = _time(fast, warmup=3, iters=15)
        print(f"{kv_len:>8} {t_ref * 1e3:>10.3f} {t_fast * 1e3:>10.3f} "
              f"{t_ref / t_fast:>9.2f}x")


# ---- 2) Nemotron int8 KV cache ----------------------------------------------
def _nemo_cfg() -> NemotronHConfig:
    return NemotronHConfig(
        vocab_size=256, hidden_size=128, num_hidden_layers=1, hybrid_override_pattern="*",
        num_attention_heads=4, num_key_value_heads=2, head_dim=128,
        attention_bias=False, rope_theta=10000.0, partial_rotary_factor=1.0,
        mamba_num_heads=1, mamba_head_dim=64, mamba_n_groups=1, ssm_state_size=64,
        conv_kernel=4, expand=1, mamba_hidden_act="silu", chunk_size=64, use_conv_bias=True,
        n_routed_experts=1, num_experts_per_tok=1, n_shared_experts=0,
        moe_intermediate_size=64, moe_latent_size=64, moe_shared_expert_intermediate_size=64,
        routed_scaling_factor=1.0, norm_topk_prob=True, n_group=1, topk_group=1,
        norm_eps=1e-6, max_position_embeddings=4096,
        bos_token_id=0, eos_token_id=1, pad_token_id=0,
        num_nextn_predict_layers=0, tie_word_embeddings=False,
    )


def bench_nemotron_int8_kv() -> None:
    print("\n=== Nemotron KV cache decode: bf16 vs int8 g128 (#133) ===")
    cfg = _nemo_cfg()
    mx.random.seed(0)
    attn = NemotronAttention(cfg)
    mx.eval(attn.parameters())

    def drive(quantized: bool, T: int) -> float:
        cache = KVCache(quantized=quantized, group_size=128)
        # warmup not strictly needed (each step touches a fresh KV chunk); time the full decode.
        x_stream = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.bfloat16)
        mx.eval(x_stream)

        def step():
            out = []
            for t in range(T):
                out.append(attn(x_stream[:, t:t + 1], cache=cache))
            return out[-1]

        # Reset between runs (cache is stateful).
        def step_with_reset():
            for k in ("k", "v", "k_q", "k_s", "k_b", "v_q", "v_s", "v_b"):
                setattr(cache, k, None)
            return step()

        return _time(step_with_reset, warmup=2, iters=5)

    print(f"{'T':>6} {'bf16_ms':>10} {'int8_ms':>10} {'overhead':>10}")
    for T in (64, 256, 1024):
        t_bf16 = drive(False, T) * 1e3
        t_int8 = drive(True, T) * 1e3
        # int8 cache trades **memory** for a quantize+dequantize per step. At decode the per-step
        # cost is dominated by the dequantize over the growing cache; this prints the ratio so we
        # know what the SDPA-against-bf16-view path costs vs the bf16-stored path.
        print(f"{T:>6} {t_bf16:>10.3f} {t_int8:>10.3f} "
              f"{t_int8 / t_bf16:>9.2f}x")


# ---- 3) Nemotron sorted MoE dispatch ----------------------------------------
def _nemo_moe_cfg() -> NemotronHConfig:
    cfg = _nemo_cfg()
    return NemotronHConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size, num_hidden_layers=1,
        hybrid_override_pattern="M",
        num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        attention_bias=False, rope_theta=10000.0, partial_rotary_factor=1.0,
        mamba_num_heads=1, mamba_head_dim=64, mamba_n_groups=1, ssm_state_size=64,
        conv_kernel=4, expand=1, mamba_hidden_act="silu", chunk_size=64, use_conv_bias=True,
        n_routed_experts=64, num_experts_per_tok=8, n_shared_experts=1,
        moe_intermediate_size=128, moe_latent_size=64, moe_shared_expert_intermediate_size=64,
        routed_scaling_factor=1.0, norm_topk_prob=True, n_group=1, topk_group=1,
        norm_eps=1e-6, max_position_embeddings=4096,
        bos_token_id=0, eos_token_id=1, pad_token_id=0,
        num_nextn_predict_layers=0, tie_word_embeddings=False,
    )


def bench_nemotron_sorted_moe() -> None:
    print("\n=== Nemotron MoE dispatch (bf16 gather_mm): unsorted vs sorted ===")
    cfg = _nemo_moe_cfg()
    mx.random.seed(0)
    moe = NemotronLatentMoE(cfg)
    e, inter, lat = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.moe_latent_size
    mx.random.seed(1)
    up = mx.random.normal((e, inter, lat)) * 0.1
    down = mx.random.normal((e, lat, inter)) * 0.1
    moe.set_experts(up, down)
    moe.gate_weight = mx.random.normal((e, cfg.hidden_size)) * 0.1
    mx.eval(moe.parameters())

    print(f"{'n_tok':>8} {'unsorted_ms':>14} {'sorted_ms':>12} {'speedup':>10}")
    for n in (64, 256, 1024):
        x = mx.random.normal((1, n, cfg.hidden_size)).astype(mx.bfloat16)
        mx.eval(x)
        moe.sort_dispatch = False
        t_u = _time(lambda: moe(x), warmup=3, iters=10)
        moe.sort_dispatch = True
        t_s = _time(lambda: moe(x), warmup=3, iters=10)
        print(f"{n:>8} {t_u * 1e3:>14.3f} {t_s * 1e3:>12.3f} "
              f"{t_u / t_s:>9.2f}x")


def run() -> None:
    print("Inference-optimization bench (model-free, tiny synthetic shapes — runs in seconds).")
    bench_kimi_absorbed_decode()
    bench_nemotron_int8_kv()
    bench_nemotron_sorted_moe()


if __name__ == "__main__":
    run()
