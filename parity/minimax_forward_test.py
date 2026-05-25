"""Model-free parity gate for the MiniMax-M2.7 forward (GQA + sigmoid noaux_tc MoE + assembly).

Everything here runs on a **tiny random-weight config** with **tiny tensors** (a few KB) — it never
loads a real checkpoint tensor, never runs a forward on real weights (per the host-OOM safety rule).
The real bf16 teacher-forced PPL gate is **deferred** (needs the weights + GPU memory); its heavy
invocation is documented at the bottom of this file and is **not** run here.

Three parts (the parity-first discipline: optimized == naive on tiny inputs):

(1) **GQA attention** — (a) the fast path (``mx.fast.rope`` partial RoPE + ``mx.fast.sdpa``) equals
    the naive path (explicit ``rotate_half`` RoPE + manual softmax), and (b) stepwise incremental
    decode (growing KV cache) equals a single full-sequence prefill. A forward-math or cache bug is
    O(1), not sub-1%. Exercises partial RoPE (first ``rotary_dim`` dims) + per-layer weighted QK-norm.

(2) **sigmoid noaux_tc MoE** — (a) the sparse ``gather_mm`` dispatch equals a dead-simple dense
    per-token top-k reference, and (b) the top-8 **selection** uses ``scores + e_score_correction_bias``
    while the **weights** are the bias-free sigmoid scores (normalized over the chosen 8) — checked
    against an explicit argsort. NO shared expert.

(3) **assembled forward** — the 62→tiny-layer ``MiniMaxModel`` is finite, and each block's fast path
    equals its naive path (per-layer naive==optimized), so the whole stack is parity-clean.

    uv run --with numpy python -m parity.minimax_forward_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.minimax.attention import KVCache, MiniMaxAttention, _rope_fast, _rope_naive
from quanta.minimax.config import MiniMaxConfig
from quanta.minimax.model import MiniMaxModel
from quanta.minimax.moe import minimax_moe, minimax_route, silu


def _rel(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)) / (mx.max(mx.abs(b)) + 1e-6))


def _absd(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)))


def _tiny_cfg(n_layers: int = 3) -> MiniMaxConfig:
    """A few-KB MiniMax-M2.7 config: real flags (sigmoid routing, bias, partial RoPE, QK-norm, no
    shared expert), tiny dims. Built directly (no disk read) so the gate is fully model-free."""
    return MiniMaxConfig(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=24,
        num_hidden_layers=n_layers,
        num_attention_heads=6,           # 6 q heads
        num_key_value_heads=2,           # 2 kv heads -> n_rep 3 (GQA)
        head_dim=8,
        rotary_dim=4,                    # partial RoPE: rotate first 4 of 8 dims
        use_qk_norm=True,
        qk_norm_type="per_layer",
        attn_type_list=tuple([1] * n_layers),   # all full softmax
        num_local_experts=8,
        num_experts_per_tok=3,           # tiny top-k (real is top-8 of 256)
        shared_intermediate_size=0,      # NO shared expert
        scoring_func="sigmoid",
        use_routing_bias=True,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        use_mtp=True,
        num_mtp_modules=3,
        mtp_transformer_layers=1,
        hidden_act="silu",
        norm_eps=1e-6,
        rope_theta=5e6,
        max_position_embeddings=204800,
        bos_token_id=200019,
        eos_token_id=200020,
        eos_token_ids=(200020,),
        tie_word_embeddings=False,
        quantization_config={},
    )


def _rand_attention(cfg: MiniMaxConfig) -> MiniMaxAttention:
    m = MiniMaxAttention(cfg, 0)
    sc = cfg.head_dim ** -0.5
    m.q_proj.weight = mx.random.normal(m.q_proj.weight.shape) * sc
    m.k_proj.weight = mx.random.normal(m.k_proj.weight.shape) * sc
    m.v_proj.weight = mx.random.normal(m.v_proj.weight.shape) * sc
    m.o_proj.weight = mx.random.normal(m.o_proj.weight.shape) * sc
    # non-trivial QK-norm weights (not the identity-1 default) so the norm is actually exercised
    m.q_norm.weight = 1.0 + mx.random.normal(m.q_norm.weight.shape) * 0.1
    m.k_norm.weight = 1.0 + mx.random.normal(m.k_norm.weight.shape) * 0.1
    return m


def test_attention(cfg: MiniMaxConfig) -> bool:
    m = _rand_attention(cfg)
    x = mx.random.normal((1, 7, cfg.hidden_size)) * 0.5

    # (a) fast (rope+sdpa) == naive (rotate_half + manual softmax), full prefill
    o_fast = m(x, use_fast=True)
    o_naive = m(x, use_fast=False)
    fast_ok = _rel(o_fast, o_naive) < 2e-3

    # sanity: the partial-RoPE primitives themselves agree (fast == naive) and only rotate rd dims
    q = mx.random.normal((1, cfg.num_attention_heads, 7, cfg.head_dim))
    rf = _rope_fast(q, cfg.rotary_dim, cfg.rope_theta, 0)
    rn = _rope_naive(q, cfg.rotary_dim, cfg.rope_theta, 0)
    rope_ok = _absd(rf, rn) < 1e-3 and _absd(rf[..., cfg.rotary_dim:], q[..., cfg.rotary_dim:]) == 0.0

    # (b) stepwise incremental decode (growing KV cache) == single full-sequence prefill
    cache = KVCache()
    steps = [m(x[:, t : t + 1], cache=cache, use_fast=True) for t in range(x.shape[1])]
    o_dec = mx.concatenate(steps, axis=1)
    decode_ok = _rel(o_dec, o_fast) < 2e-3

    print("=== (1) GQA attention (partial RoPE + per-layer QK-norm) ===")
    print(f"  [{'OK' if fast_ok else 'FAIL'}] fast(rope+sdpa) == naive          rel={_rel(o_fast, o_naive):.2e}")
    print(f"  [{'OK' if rope_ok else 'FAIL'}] partial-RoPE fast==naive, pass-through exact")
    print(f"  [{'OK' if decode_ok else 'FAIL'}] incremental decode == prefill      rel={_rel(o_dec, o_fast):.2e}")
    return fast_ok and rope_ok and decode_ok


def _dense_moe_ref(xf: mx.array, router: dict, experts: dict, cfg: MiniMaxConfig) -> mx.array:
    """Dead-simple dense reference: per token, explicit top-k SwiGLU sum (no shared expert)."""
    n, dim = xf.shape
    topk = cfg.num_experts_per_tok
    idx, w = minimax_route(xf, router, cfg)
    rows = []
    for t in range(n):
        acc = mx.zeros((dim,))
        for s in range(topk):
            e = int(idx[t, s].item())
            g = experts["w1"][e] @ xf[t]
            u = experts["w3"][e] @ xf[t]
            d = experts["w2"][e] @ (silu(g) * u)
            acc = acc + w[t, s] * d
        rows.append(acc)
    return mx.stack(rows, 0)


def test_moe(cfg: MiniMaxConfig) -> bool:
    e, inter, h, topk = (cfg.num_local_experts, cfg.moe_intermediate_size,
                         cfg.hidden_size, cfg.num_experts_per_tok)
    router = {
        "weight": mx.random.normal((e, h)),
        "e_score_correction_bias": mx.random.normal((e,)) * 0.5,
    }
    experts = {
        "w1": mx.random.normal((e, inter, h)) * 0.1,
        "w3": mx.random.normal((e, inter, h)) * 0.1,
        "w2": mx.random.normal((e, h, inter)) * 0.1,
    }
    x = mx.random.normal((1, 6, h)) * 0.5
    n = x.shape[1]
    xf = x.reshape(n, h).astype(mx.float32)

    # (a) sparse gather dispatch == dense per-token top-k reference
    out = minimax_moe(x, router, experts, cfg)
    ref = _dense_moe_ref(xf, router, experts, cfg)
    dense_ok = _rel(out.reshape(n, h), ref) < 1e-4

    # (b) selection uses scores+bias; weights are bias-free sigmoid (normalized over chosen topk)
    logits = xf @ router["weight"].T
    scores = mx.sigmoid(logits)
    choice = scores + router["e_score_correction_bias"][None]
    exp_idx = mx.argsort(-choice, axis=-1)[:, :topk]                 # full-sort top-k by score+bias
    idx, w = minimax_route(xf, router, cfg)
    sel_ok = all(set(int(i) for i in idx[t].tolist()) == set(int(i) for i in exp_idx[t].tolist())
                 for t in range(n))
    w_exp = mx.take_along_axis(scores, idx, axis=-1)
    w_exp = w_exp / (mx.sum(w_exp, axis=-1, keepdims=True) + 1e-20)  # norm_topk_prob over chosen 8
    weight_ok = _rel(w, w_exp) < 1e-5

    print("=== (2) sigmoid noaux_tc MoE (256 experts top-8, NO shared expert) ===")
    print(f"  [{'OK' if dense_ok else 'FAIL'}] sparse gather == dense ref         rel={_rel(out.reshape(n, h), ref):.2e}")
    print(f"  [{'OK' if sel_ok else 'FAIL'}] top-k selection uses score+bias")
    print(f"  [{'OK' if weight_ok else 'FAIL'}] weights = bias-free sigmoid (normed) rel={_rel(w, w_exp):.2e}")
    return dense_ok and sel_ok and weight_ok


def _randomize_model(model: MiniMaxModel) -> None:
    """Expert stacks / gate default to zeros — give them real dynamics so routing/SwiGLU are active."""
    cfg = model.cfg
    e, inter, h = cfg.num_local_experts, cfg.moe_intermediate_size, cfg.hidden_size
    for blk in model.layers:
        blk.gate_weight = mx.random.normal((e, h))
        blk.e_score_correction_bias = mx.random.normal((e,)) * 0.5
        blk.w1 = mx.random.normal((e, inter, h)) * 0.1
        blk.w3 = mx.random.normal((e, inter, h)) * 0.1
        blk.w2 = mx.random.normal((e, h, inter)) * 0.1
        a = blk.self_attn
        a.q_norm.weight = 1.0 + mx.random.normal(a.q_norm.weight.shape) * 0.1
        a.k_norm.weight = 1.0 + mx.random.normal(a.k_norm.weight.shape) * 0.1


def test_forward(cfg: MiniMaxConfig) -> bool:
    model = MiniMaxModel(cfg)
    _randomize_model(model)
    ids = mx.random.randint(0, cfg.vocab_size, (9,))

    logits_fast = model(ids, use_fast=True)
    finite_ok = (logits_fast.shape == (1, 9, cfg.vocab_size)
                 and bool(mx.all(mx.isfinite(logits_fast)).item()))

    # per-layer naive == optimized: the whole assembled stack on the naive path matches the fast path
    logits_naive = model(ids, use_fast=False)
    layer_ok = _rel(logits_fast, logits_naive) < 5e-3

    # incremental decode == prefill through the full assembly (KV caches thread across all layers)
    caches = [KVCache() for _ in model.layers]
    steps = [model(ids[t : t + 1], caches=caches, use_fast=True) for t in range(ids.shape[0])]
    logits_dec = mx.concatenate(steps, axis=1)
    decode_ok = _rel(logits_dec, logits_fast) < 5e-3

    print("=== (3) assembled 62-layer forward (one layer resident) ===")
    print(f"  [{'OK' if finite_ok else 'FAIL'}] forward finite  logits{logits_fast.shape}")
    print(f"  [{'OK' if layer_ok else 'FAIL'}] per-layer naive == optimized       rel={_rel(logits_fast, logits_naive):.2e}")
    print(f"  [{'OK' if decode_ok else 'FAIL'}] incremental decode == prefill      rel={_rel(logits_dec, logits_fast):.2e}")
    return finite_ok and layer_ok and decode_ok


def run() -> None:
    mx.random.seed(0)
    cfg = _tiny_cfg()
    a = test_attention(cfg)
    m = test_moe(cfg)
    f = test_forward(cfg)
    print("\nPASS" if all([a, m, f]) else "\nFAIL")
    assert all([a, m, f])


# --- DEFERRED real bf16 teacher-forced PPL gate (needs weights + GPU memory; DO NOT run here) ----
# Heavy: streams the real ~230B-param block-fp8 checkpoint one layer at a time and scores prose. Run
# only in a GPU session with the checkpoint present and the memory headroom (host OOM rebooted the
# box once). Expected: teacher-forced ppl on clean prose with the correct BOS in the low single
# digits; bisect per-layer residuals vs a torch oracle if it is not.
#
#   from quanta.minimax.config import MiniMaxConfig
#   from quanta.minimax.loader import MiniMaxSourceCheckpoint
#   from quanta.minimax.model import teacher_forced_ppl
#   import mlx.core as mx
#   ART = "/Users/pmrj/models/MiniMax-M2.7"
#   cfg = MiniMaxConfig.from_pretrained(ART)
#   ck = MiniMaxSourceCheckpoint(ART, cfg)
#   ids = mx.array([[cfg.bos_token_id, *tokenizer.encode("<clean prose here>")]])
#   print("teacher-forced ppl:", teacher_forced_ppl(ck, ids, cfg))

if __name__ == "__main__":
    run()
