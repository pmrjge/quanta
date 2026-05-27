"""Absorbed-MLA fast-SDPA parity (model-free, tiny tensors).

Proves the new ``mx.fast.scaled_dot_product_attention``-based ``_absorbed`` path
(MQA-shaped Q[H]/K[1]/V[1], K=concat(c,k_pe), V=zero-pad(c)→slice) is numerically
equivalent to the explicit-softmax reference:

    scores = q_absorb·c.T + q_pe·k_pe.T
    weights = softmax(scores·scale + causal_mask)   # fp32
    out_latent = weights·c

across both shape regimes that matter at decode:

* ``t=1`` (single new token over growing KV) — the decode hot path
* ``t=k+1`` over a long KV (spec-decode verify batch) — the spec hot path

Same random weights, two paths, compare max-abs of the o_proj input (``out``)
in bf16. SDPA tiles + reorders the reduction so we expect a few ULPs of drift,
but it must be argmax-stable end-to-end; the tolerance below is what the
existing dense-vs-explicit SDPA path passes at in the Kimi attention.

Model-free: no checkpoint load, no real weights, random ``nn.Linear`` / synthetic
config. Runs in milliseconds on CPU. SAFE to run alongside a live GPU job.

    uv run python -m parity.mla_absorbed_sdpa_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.config import KimiTextConfig, YarnRope
from quanta.modeling.attention import MLAAttention

# Small but architecture-faithful: same kv_lora=512, rope=64, nope=128, vhd=128
# as Kimi (so the W_UK/W_UV split path exercises the real reshape), but only
# H=4 heads and a tiny hidden so the test runs in milliseconds with no quant.
CFG = KimiTextConfig(
    vocab_size=128,
    hidden_size=256,
    intermediate_size=512,
    moe_intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    q_lora_rank=64,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    n_routed_experts=1, n_shared_experts=1, num_experts_per_tok=1,
    n_group=1, topk_group=1, topk_method="noaux_tc", scoring_func="sigmoid",
    norm_topk_prob=True, routed_scaling_factor=1.0,
    first_k_dense_replace=1, moe_layer_freq=1,
    rms_norm_eps=1e-6, hidden_act="silu", attention_bias=False,
    max_position_embeddings=4096,
    bos_token_id=0, eos_token_id=1,
    rope=YarnRope(factor=64.0, beta_fast=32.0, beta_slow=1.0,
                  mscale=1.0, mscale_all_dim=1.0,
                  original_max_position_embeddings=4096, rope_theta=50000.0),
)


def _make_attn(seed: int = 0) -> MLAAttention:
    mx.random.seed(seed)
    attn = MLAAttention(CFG)
    attn.absorbed = True
    mx.eval(attn.parameters())
    return attn


def _step(attn: MLAAttention, t: int, kv_len: int, *, use_fast: bool) -> mx.array:
    """One absorbed-MLA forward at the given (t, kv_len) shape with both paths sharing
    the same random q_nope/q_pe/c_kv/k_pe — passing through ``_absorbed`` directly so
    the test isolates the SDPA vs explicit-softmax math (no RoPE / cache stateful diff)."""
    b, h, nope, rope_d = 1, attn.num_heads, attn.nope, attn.rope
    kv_lora = attn.cfg.kv_lora_rank

    mx.random.seed(123)
    q_nope = mx.random.normal((b, h, t, nope)).astype(mx.bfloat16)
    q_pe = mx.random.normal((b, h, t, rope_d)).astype(mx.bfloat16)
    c_kv = mx.random.normal((b, kv_len, kv_lora)).astype(mx.bfloat16)
    # Caller broadcasts k_pe to [B,H,S,rope] before _absorbed — replicate that.
    k_pe_unbroadcast = mx.random.normal((b, 1, kv_len, rope_d)).astype(mx.bfloat16)
    k_pe = mx.broadcast_to(k_pe_unbroadcast, (b, h, kv_len, rope_d))

    out = attn._absorbed(q_nope, q_pe, c_kv, k_pe, b, t, kv_len, use_fast=use_fast)
    mx.eval(out)
    return out


def run() -> None:
    attn = _make_attn()
    # Decode (t=1) at growing KV — the hot path.
    # Spec-verify (t=5 = k=4+1) at long KV — the spec hot path.
    shapes = [(1, 64), (1, 1024), (1, 8192), (5, 1024), (5, 8192)]
    # bf16 tolerance: ~1.5e-2 relative is what flash-style SDPA reorders cost
    # over a length-S softmax reduction in bf16 (~7-bit mantissa); the dense MLA
    # path already runs through SDPA in production and is argmax-stable at this
    # scale across the full 60-layer Kimi forward. End-to-end argmax-stability
    # of the spec-decode loop is the actual ship gate (eagle_spec_longprompt_sweep).
    REL_TOL = 1.5e-2
    print(f"{'t':>3} {'kv_len':>7} {'ref_norm':>10} {'fast_norm':>10} "
          f"{'max_abs_diff':>13} {'rel_diff':>10}  argmax_match")
    ok = True
    for t, kv_len in shapes:
        ref = _step(attn, t, kv_len, use_fast=False)
        fast = _step(attn, t, kv_len, use_fast=True)
        max_diff = float(mx.max(mx.abs(ref - fast)).item())
        ref_scale = float(mx.max(mx.abs(ref)).item()) + 1e-6
        rel = max_diff / ref_scale
        ref_norm = float(mx.linalg.norm(ref).item())
        fast_norm = float(mx.linalg.norm(fast).item())
        # argmax match over the last axis (vhd channel) — proxy for logit-argmax
        # stability without a full model forward; should be perfect at this scale.
        argmax_ref = mx.argmax(ref, axis=-1)
        argmax_fast = mx.argmax(fast, axis=-1)
        argmax_agreement = float(mx.mean((argmax_ref == argmax_fast).astype(mx.float32)).item())
        passed = rel < REL_TOL and argmax_agreement >= 0.99
        ok = ok and passed
        marker = "ok" if passed else "FAIL"
        print(f"{t:>3} {kv_len:>7} {ref_norm:>10.4f} {fast_norm:>10.4f} "
              f"{max_diff:>13.4e} {rel:>9.2e}  {argmax_agreement:>6.3f}  {marker}")
    print("\nPASS" if ok else "\nFAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    run()
