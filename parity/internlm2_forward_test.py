"""Model-free parity gate for the InternLM2.5-7B-Chat-1M bf16 reference forward.

Runs on a **tiny random-weight config** (a few KB) — never loads a real checkpoint tensor (the real
bf16 teacher-forced ppl gate is `parity/internlm2_bf16_ppl.py`, run separately with the weights +
GPU memory). The parity-first discipline: the fast (`mx.fast.rope` + `mx.fast.sdpa`) path must equal
the naive (explicit `rotate_half` RoPE + manual softmax) path, and stepwise incremental decode must
equal a single full-sequence prefill — a forward-math or KV-cache bug is O(1), not sub-1%.

Three parts:

(1) **GQA attention** — (a) fast == naive over a full prefill, (b) the `rotate_half` RoPE primitive
    fast == explicit, (c) growing-cache incremental decode == one prefill. Exercises GQA repeat
    (4 q-heads per kv-head) + Llama-style full-dim RoPE.
(2) **assembled forward** (`InternLM2Model`, tiny dense stack) — fast == naive, and incremental
    decode == prefill, threading one `KVCache` per layer (norm placement, residual structure,
    cache offset all covered).
(3) **dynamic-NTK regime** — fast == naive **still holds** when the base is NTK-rescaled (seq_len >
    max_position_embeddings): both paths read `cfg.ntk_base(seq_len)`, so the rescale is shared.
    NOTE: incremental-decode == prefill is asserted ONLY in the NToff regime (seq_len ≤ max_pos).
    Under dynamic-NTK they legitimately diverge — HF recomputes inv_freq from `max(position_ids)+1`,
    so decode uses a per-step-growing base while a single prefill uses one base(T) for all positions.
    That divergence is correct behavior, not a bug, so we don't assert equality across it.

    uv run --with numpy python -m parity.internlm2_forward_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.internlm2.attention import InternLM2Attention, KVCache, _rope_explicit, _rope_fast
from quanta.internlm2.config import InternLM2Config
from quanta.internlm2.model import InternLM2Model


def _rel(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)) / (mx.max(mx.abs(b)) + 1e-6))


def _absd(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)))


def _tiny_cfg(n_layers: int = 3, *, max_pos: int = 4096) -> InternLM2Config:
    """A few-KB InternLM2.5 config: real flags (dynamic-NTK, GQA 4:1, no biases, untied head), tiny
    dims. Built directly (no disk read) so the gate is fully model-free. ``max_pos`` is tunable so a
    short sequence can be forced above/below the NTK ceiling."""
    return InternLM2Config(
        vocab_size=64, hidden_size=32, num_hidden_layers=n_layers, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, attention_bias=False,
        rope_theta=5e7, rope_scaling_type="dynamic", rope_scaling_factor=2.5,
        max_position_embeddings=max_pos, hidden_act="silu", norm_eps=1e-5,
        tie_word_embeddings=False, eos_token_id=2, eos_token_ids=(2, 92542),
        pad_token_id=2, bos_token_id=1, add_bos_token=True,
    )


def _rand_attention(cfg: InternLM2Config) -> InternLM2Attention:
    m = InternLM2Attention(cfg)
    sc = cfg.head_dim ** -0.5
    m.wq.weight = mx.random.normal(m.wq.weight.shape) * sc
    m.wk.weight = mx.random.normal(m.wk.weight.shape) * sc
    m.wv.weight = mx.random.normal(m.wv.weight.shape) * sc
    m.wo.weight = mx.random.normal(m.wo.weight.shape) * sc
    return m


def test_attention(cfg: InternLM2Config) -> bool:
    m = _rand_attention(cfg)
    x = mx.random.normal((1, 7, cfg.hidden_size)) * 0.5

    # (a) fast (rope+sdpa) == naive (rotate_half + manual softmax), full prefill
    o_fast = m(x, use_fast=True)
    o_naive = m(x, use_fast=False)
    fast_ok = _rel(o_fast, o_naive) < 3e-3

    # (b) rotate_half RoPE primitive: fast == explicit (full head_dim rotation)
    q = mx.random.normal((1, cfg.num_attention_heads, 7, cfg.head_dim))
    rf = _rope_fast(q, cfg.rope_theta, 0)
    rn = _rope_explicit(q, cfg.rope_theta, 0)
    rope_ok = _absd(rf, rn) < 1e-3

    # (c) stepwise incremental decode (growing KV cache) == single full-sequence prefill (NTK off)
    cache = KVCache()
    steps = [m(x[:, t : t + 1], cache=cache, use_fast=True) for t in range(x.shape[1])]
    o_dec = mx.concatenate(steps, axis=1)
    decode_ok = _rel(o_dec, o_fast) < 3e-3

    print("=== (1) GQA attention (Llama rotate_half RoPE + dynamic-NTK) ===")
    print(f"  [{'OK' if fast_ok else 'FAIL'}] fast(rope+sdpa) == naive          rel={_rel(o_fast, o_naive):.2e}")
    print(f"  [{'OK' if rope_ok else 'FAIL'}] rotate_half RoPE fast == explicit  abs={_absd(rf, rn):.2e}")
    print(f"  [{'OK' if decode_ok else 'FAIL'}] incremental decode == prefill      rel={_rel(o_dec, o_fast):.2e}")
    return fast_ok and rope_ok and decode_ok


def _randomize_model(model: InternLM2Model) -> None:
    """Give the dense stack real dynamics (embeddings/projections/norms) so parity is non-trivial."""
    cfg = model.cfg
    model.tok_embeddings.weight = mx.random.normal(model.tok_embeddings.weight.shape) * 0.1
    model.norm.weight = 1.0 + mx.random.normal(model.norm.weight.shape) * 0.1
    model.output.weight = mx.random.normal(model.output.weight.shape) * (cfg.hidden_size ** -0.5)
    sc = cfg.head_dim ** -0.5
    for layer in model.layers:
        layer.attention_norm.weight = 1.0 + mx.random.normal(layer.attention_norm.weight.shape) * 0.1
        layer.ffn_norm.weight = 1.0 + mx.random.normal(layer.ffn_norm.weight.shape) * 0.1
        a = layer.attention
        a.wq.weight = mx.random.normal(a.wq.weight.shape) * sc
        a.wk.weight = mx.random.normal(a.wk.weight.shape) * sc
        a.wv.weight = mx.random.normal(a.wv.weight.shape) * sc
        a.wo.weight = mx.random.normal(a.wo.weight.shape) * sc
        ff = layer.feed_forward
        ff.w1.weight = mx.random.normal(ff.w1.weight.shape) * 0.1
        ff.w3.weight = mx.random.normal(ff.w3.weight.shape) * 0.1
        ff.w2.weight = mx.random.normal(ff.w2.weight.shape) * 0.1


def test_forward(cfg: InternLM2Config) -> bool:
    model = InternLM2Model(cfg)
    _randomize_model(model)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 9))

    logits_fast = model(ids, use_fast=True)
    finite_ok = (logits_fast.shape == (1, 9, cfg.vocab_size)
                 and bool(mx.all(mx.isfinite(logits_fast)).item()))

    # per-layer naive == optimized: the whole assembled stack on the naive path matches the fast path
    logits_naive = model(ids, use_fast=False)
    layer_ok = _rel(logits_fast, logits_naive) < 5e-3

    # incremental decode == prefill through the full assembly (one KVCache per layer; NTK off)
    caches = [KVCache() for _ in model.layers]
    steps = [model(ids[:, t : t + 1], caches=caches, use_fast=True) for t in range(ids.shape[1])]
    logits_dec = mx.concatenate(steps, axis=1)
    decode_ok = _rel(logits_dec, logits_fast) < 5e-3

    print("=== (2) assembled dense forward (InternLM2Model) ===")
    print(f"  [{'OK' if finite_ok else 'FAIL'}] forward finite  logits{logits_fast.shape}")
    print(f"  [{'OK' if layer_ok else 'FAIL'}] per-layer naive == optimized       rel={_rel(logits_fast, logits_naive):.2e}")
    print(f"  [{'OK' if decode_ok else 'FAIL'}] incremental decode == prefill      rel={_rel(logits_dec, logits_fast):.2e}")
    return finite_ok and layer_ok and decode_ok


def test_ntk_regime() -> bool:
    """With seq_len > max_position_embeddings the RoPE base is NTK-rescaled; fast == naive must still
    hold (both read ``cfg.ntk_base(seq_len)``). Forces NTK on by shrinking the ceiling under T."""
    cfg = _tiny_cfg(n_layers=2, max_pos=4)          # T=9 > 4 → dynamic-NTK engaged
    model = InternLM2Model(cfg)
    _randomize_model(model)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 9))
    # sanity: the rescaled base is actually different from the source theta at this seq_len
    engaged = cfg.ntk_base(9) != cfg.rope_theta
    o_fast = model(ids, use_fast=True)
    o_naive = model(ids, use_fast=False)
    fast_ok = _rel(o_fast, o_naive) < 5e-3
    print("=== (3) dynamic-NTK engaged (seq_len > max_pos) ===")
    print(f"  [{'OK' if engaged else 'FAIL'}] NTK base rescaled (base != rope_theta)")
    print(f"  [{'OK' if fast_ok else 'FAIL'}] fast == naive with rescaled base    rel={_rel(o_fast, o_naive):.2e}")
    return engaged and fast_ok


def run() -> None:
    mx.random.seed(0)
    cfg = _tiny_cfg()
    a = test_attention(cfg)
    f = test_forward(cfg)
    n = test_ntk_regime()
    print("\nPASS" if all([a, f, n]) else "\nFAIL")
    assert all([a, f, n])


# --- DEFERRED no longer: the real bf16 teacher-forced PPL gate lives in parity/internlm2_bf16_ppl.py.
# It streams the real ~7B bf16 source one layer at a time and scores clean prose with BOS=1; expect
# low single-digit ppl (a forward bug yields catastrophic ppl — the Kimi lesson). Run it with the
# checkpoint present:  uv run --extra reference --with numpy python -m parity.internlm2_bf16_ppl

if __name__ == "__main__":
    run()
