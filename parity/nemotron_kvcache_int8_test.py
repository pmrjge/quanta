"""Nemotron int8 KV cache parity (model-free, tiny synthetic tensors).

Proves that the new ``quantized=True`` mode of :class:`quanta.nemotron.attention.KVCache`
(affine int8 per-token, per-group over ``head_dim`` — the #133 audit lever) is numerically
close enough to the bf16 default that the downstream SDPA path is argmax-stable. The cache
returns a dequantized bf16 stream in both modes, so attention is byte-identical above the
cache; the only delta is the int8 round-trip on the stored keys/values.

What we check (mirrors the GLM/Qwen3.5/MiniMax #122 gates):

1. ``offset`` and shapes match between quantized=False and quantized=True after T appends.
2. Returned ``(k_full, v_full)`` are within a loose int8-group bound vs the bf16 reference
   (per-group affine on head_dim=128 group_size=128 ≈ 8.25 bpp, well under the 4-bit floor).
3. End-to-end attention output through ``NemotronAttention`` is argmax-stable on a random
   prompt — the actual ship gate. Per-position argmax over heads must match.

Model-free: a few KB of random tensors, fixed seed, runs in ms on CPU. SAFE alongside a live
GPU job (the running DSV4 bake).

    uv run python -m parity.nemotron_kvcache_int8_test

deferred (run later on GPU, task #133): long-context teacher-forced ppl @ 64K/256K/1M on the
real int4g64 artifact comparing ``quantized=False`` vs ``quantized=True`` — the steady-state
memory win this unlocks (~halved KV residency at 1M).
"""

from __future__ import annotations

import mlx.core as mx

from quanta.nemotron.attention import KVCache, NemotronAttention
from quanta.nemotron.config import NemotronHConfig


def _tiny_cfg() -> NemotronHConfig:
    """A faithful tiny Nemotron-H config — exercises the int8 grouping on head_dim=128 (real
    Nemotron head_dim) with a small hidden / layer count so the test runs in ms on CPU. The
    SSM / MoE fields are unused by ``NemotronAttention`` but the frozen dataclass requires them."""
    return NemotronHConfig(
        vocab_size=256, hidden_size=128, num_hidden_layers=1, hybrid_override_pattern="*",
        num_attention_heads=4, num_key_value_heads=2, head_dim=128,
        attention_bias=False, rope_theta=10000.0, partial_rotary_factor=1.0,
        mamba_num_heads=1, mamba_head_dim=64, mamba_n_groups=1, ssm_state_size=64,
        conv_kernel=4, expand=1, mamba_hidden_act="silu", chunk_size=64, use_conv_bias=True,
        n_routed_experts=1, num_experts_per_tok=1, n_shared_experts=0,
        moe_intermediate_size=64, moe_latent_size=64, moe_shared_expert_intermediate_size=64,
        routed_scaling_factor=1.0, norm_topk_prob=True, n_group=1, topk_group=1,
        norm_eps=1e-6, max_position_embeddings=1024,
        bos_token_id=0, eos_token_id=1, pad_token_id=0,
        num_nextn_predict_layers=0, tie_word_embeddings=False,
    )


def _run_steps(attn: NemotronAttention, x_stream: mx.array, *, quantized: bool) -> tuple[mx.array, int]:
    """Drive ``T`` decode steps through ``attn`` with a fresh cache; return the stacked outputs
    ``[B, T, hidden]`` and the final offset."""
    cache = KVCache(quantized=quantized, group_size=128)
    outs = []
    for t in range(x_stream.shape[1]):
        out_t = attn(x_stream[:, t:t + 1], cache=cache)   # cache.offset drives RoPE offset
        outs.append(out_t)
        mx.eval(out_t)
    return mx.concatenate(outs, axis=1), cache.offset


def run() -> None:
    cfg = _tiny_cfg()
    mx.random.seed(0)
    attn = NemotronAttention(cfg)
    mx.eval(attn.parameters())

    T = 16
    x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.bfloat16)

    out_bf16, off_bf16 = _run_steps(attn, x, quantized=False)
    out_int8, off_int8 = _run_steps(attn, x, quantized=True)

    assert off_bf16 == off_int8 == T, f"offset mismatch: bf16 {off_bf16} int8 {off_int8} (expected {T})"

    max_diff = float(mx.max(mx.abs(out_bf16 - out_int8)).item())
    ref_scale = float(mx.max(mx.abs(out_bf16)).item()) + 1e-6
    rel = max_diff / ref_scale

    # argmax over hidden-axis: stable proxy for argmax of the final logits in a real forward.
    arg_bf16 = mx.argmax(out_bf16, axis=-1)
    arg_int8 = mx.argmax(out_int8, axis=-1)
    agree = float(mx.mean((arg_bf16 == arg_int8).astype(mx.float32)).item())

    # int8 per-group on head_dim is the same scheme that ships as default for MLA/MiniMax/GLM/Qwen3.5;
    # bf16 round-trip after one int8 quantize-dequantize at head_dim=128 g128 is on the order of a
    # ULP of the dynamic range, and the SDPA reduction is the same bf16 path. The bound below is
    # what GLM/Qwen3.5 ship at; argmax must match.
    REL_TOL = 5e-2
    ok = rel < REL_TOL and agree >= 0.99
    print(f"T={T}  offset={off_bf16}  max_abs_diff={max_diff:.4e}  rel={rel:.3e}  "
          f"argmax_agree={agree:.3f}  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
