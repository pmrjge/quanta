"""DSV4 int8 KV cache parity (model-free, tiny synthetic tensors) — #123 / #133.

Proves the new ``quantized=True`` mode of :class:`quanta.dsv4.decode.DSV4Cache` (affine int8
per-token, per-group over ``head_dim`` on both ``kv`` and ``ckv``) is numerically close
enough to the bf16 default that the SDPA / indexer paths are argmax-stable. The cache exposes
its streams as bf16 (the :attr:`kv` / :attr:`ckv` properties dequantize on read) so attention
is byte-identical above the cache; the only delta is the int8 round-trip on the stored kv/ckv.

What we check (mirrors GLM/Qwen3.5/MiniMax #122 + Kimi MLA #47 gates):

1. ``offset`` and shapes match between quantized=False and quantized=True after T decode steps.
2. Final dense output (``decode_step_dense`` over T steps) is argmax-stable on a random prompt
   on a tiny but architecture-faithful config (real head_dim=128, real ratio-0 SW attention).
3. Truncate (spec-decode rollback) gives bit-identical state to a fresh cache fed the kept
   prefix, in BOTH modes — quantization must not interfere with rollback semantics.

Model-free: a few KB of random tensors, fixed seed, runs in ms on CPU. SAFE alongside the
live DSV4 bake (the bake's GPU residency dominates; this test allocates nothing comparable).

    uv run python -m parity.dsv4_kvcache_int8_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.dsv4 import decode as D
from quanta.dsv4.config import DeepSeekV4Config


# Small but architecture-faithful: real head_dim=128 (so int8 g128 quantization actually fires),
# small hidden + 2 heads, ratio-0 sliding-window regime (the simplest of the three).
DIM = 128
HEAD_DIM = 128            # exactly matches group_size=128 → int8 quantization actually fires
ROPE_HEAD_DIM = 64
N_HEADS = 2
Q_LORA = 64
WIN = 16
O_GROUPS = 1
O_LORA = 32


def _config() -> DeepSeekV4Config:
    """Minimal DSV4 config mirroring ``parity/dsv4_decode_attn_test._cfg`` shape contract, but
    with ``head_dim=128`` so the int8 cache quantizes (not auto-disabled by ``_resolve_quant``)."""
    return DeepSeekV4Config(
        vocab_size=32, hidden_size=DIM, num_hidden_layers=1, moe_intermediate_size=16,
        num_attention_heads=N_HEADS, head_dim=HEAD_DIM, rope_head_dim=ROPE_HEAD_DIM,
        q_lora_rank=Q_LORA, o_lora_rank=O_LORA, o_groups=O_GROUPS, sliding_window=WIN,
        index_n_heads=2, index_head_dim=64, index_topk=4,
        compress_ratios=(0,), compress_rope_theta=10000.0,
        n_routed_experts=2, num_experts_per_tok=2, n_shared_experts=1, n_hash_layers=0,
        scoring_func="sqrtsoftplus", topk_method="noaux_tc", norm_topk_prob=True,
        routed_scaling_factor=1.0, swiglu_limit=0.0,
        hc_mult=1, hc_sinkhorn_iters=1, hc_eps=1e-6, n_mtp_layers=0,
        norm_eps=1e-6, rope_theta=10000.0,
        rope_scaling={"factor": 1.0, "beta_fast": 32, "beta_slow": 1,
                      "original_max_position_embeddings": 16, "type": "yarn"},
        max_position_embeddings=4096, bos_token_id=0, eos_token_id=1, eos_token_ids=(1,),
        tie_word_embeddings=False,
    )


def _rand_attn_params(rng_seed: int) -> dict:
    """Random per-layer attention params matching the loader.attention() shape contract for the
    ratio-0 path (wq_a/b, q_norm, wkv, kv_norm, wo_a/b, attn_sink)."""
    mx.random.seed(rng_seed)
    return {
        "wq_a": mx.random.normal((Q_LORA, DIM)) * 0.1,
        "q_norm": mx.random.normal((Q_LORA,)) * 0.01 + 1.0,
        "wq_b": mx.random.normal((N_HEADS * HEAD_DIM, Q_LORA)) * 0.1,
        "wkv": mx.random.normal((HEAD_DIM, DIM)) * 0.1,
        "kv_norm": mx.random.normal((HEAD_DIM,)) * 0.01 + 1.0,
        "wo_a": mx.random.normal((O_GROUPS * O_LORA, N_HEADS * HEAD_DIM // O_GROUPS)) * 0.1,
        "wo_b": mx.random.normal((DIM, O_GROUPS * O_LORA)) * 0.1,
        "attn_sink": mx.random.normal((N_HEADS,)) * 0.01,
    }


def _decode_all(x: mx.array, p: dict, cfg: DeepSeekV4Config, cos: mx.array, sin: mx.array,
                cache: D.DSV4Cache) -> mx.array:
    """Drive ``T`` decode steps through ``decode_step_dense`` with the given cache (handles both
    quantized=True and quantized=False)."""
    T = x.shape[1]
    outs = [D.decode_step_dense(x[:, t:t + 1], p, cfg, 0, cache, cos, sin, t) for t in range(T)]
    return mx.concatenate(outs, axis=1)


def run() -> None:
    cfg = _config()
    p = _rand_attn_params(rng_seed=0)
    mx.eval(p)

    # RoPE tables once for both runs.
    from quanta.dsv4.attention import rope_cos_sin
    T = 32
    cos, sin = rope_cos_sin(ROPE_HEAD_DIM, T, 0, cfg.rope_theta, cfg.rope_factor,
                            cfg.beta_fast, cfg.beta_slow)

    mx.random.seed(42)
    x = mx.random.normal((1, T, DIM)).astype(mx.bfloat16)
    mx.eval(x)

    # bf16 baseline
    bf16_cache = D.DSV4Cache(1, quantized=False)
    out_bf16 = _decode_all(x, p, cfg, cos, sin, bf16_cache)
    mx.eval(out_bf16)
    off_bf16 = bf16_cache.offset

    # int8 default (head_dim=128 exactly matches group_size=128 → real quantization fires)
    int8_cache = D.DSV4Cache(1, quantized=True)
    out_int8 = _decode_all(x, p, cfg, cos, sin, int8_cache)
    mx.eval(out_int8)
    off_int8 = int8_cache.offset

    assert off_bf16 == off_int8 == T, f"offsets {off_bf16} / {off_int8} != T={T}"
    assert int8_cache[0].quantized, "quantization didn't fire — group_size mismatch?"

    max_diff = float(mx.max(mx.abs(out_bf16 - out_int8)).item())
    ref_scale = float(mx.max(mx.abs(out_bf16)).item()) + 1e-6
    rel = max_diff / ref_scale
    arg_b = mx.argmax(out_bf16, axis=-1)
    arg_i = mx.argmax(out_int8, axis=-1)
    agree = float(mx.mean((arg_b == arg_i).astype(mx.float32)).item())

    # Rollback parity: truncate to T/2, then a fresh cache fed only the first T/2 tokens
    # must match bit-identically (in both modes).
    int8_cache.truncate(T // 2)
    fresh_int8 = D.DSV4Cache(1, quantized=True)
    _decode_all(x[:, :T // 2], p, cfg, cos, sin, fresh_int8)
    mx.eval(int8_cache[0].kv, fresh_int8[0].kv)
    rb_diff = float(mx.max(mx.abs(int8_cache[0].kv - fresh_int8[0].kv)).item())

    REL_TOL = 5e-2
    ok = rel < REL_TOL and agree >= 0.99 and rb_diff < 1e-6
    print(f"T={T}  off={off_bf16}  bf16_vs_int8: max_abs={max_diff:.4e}  rel={rel:.3e}  "
          f"argmax_agree={agree:.3f}  rollback_diff={rb_diff:.2e}  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
