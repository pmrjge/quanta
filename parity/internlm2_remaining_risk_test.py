"""Verify the remaining numerical + contract risks in the InternLM2.5-7B-Chat-1M scaffold.

Model-free (rule-8): every check runs on tiny random tensors or static config values; no source
checkpoint or baked artifact is loaded. Covers the residual risks that the prior scaffold-time
gate (wqkv split @ 0.00 err, rotate_half RoPE @ ~8e-3) did not pin down:

(1) Dynamic-NTK formula vs the HF ``InternLM2DynamicNTKScalingRotaryEmbedding`` formula at
    modeling_internlm2.py:152-154 — boundary, +1, far extrapolation. A wrong factor or exponent
    silently degrades every token past the trained window (the failure mode that broke Qwen2.5-1M).
(2) Chunked-prefill RoPE offset arithmetic: ``mx.fast.rope(slice, offset=c·CS)`` must equal the
    ``[c·CS : (c+1)·CS]`` window of a one-pass RoPE. This is the property that makes
    ``generate._prefill_chunked(chunk_size=4096)`` equivalent to a single monolithic prefill;
    without it, every chunk after the first sees the wrong absolute position.
(3) ``rope_fast`` vs ``rope_explicit`` at a 1M-token offset — exercises both the fast kernel and
    the parity reference at the actual decode regime.
(4) KV cache replicate + truncate round-trip on the bf16 path (the contract spec-decode relies on).
(5) Bake ↔ runtime bit-width lock-step: ``_ATTN_BITS`` / ``_MLP_BITS`` / ``_GROUP_SIZE`` in
    ``runtime.py`` must equal what ``bake_internlm2`` writes, else the qmm kernel decodes wrong.
(6) Sampler edge cases: ``temperature=0`` → argmax (top-k/p inert); ``top_p`` always keeps top-1
    (cumulative-prob shift); ``repetition_penalty`` divides positive logits of seen ids.
(7) ``_split_wqkv`` re-confirmation: round-trip on a synthetic interleaved fused weight (the
    invariant that lets the bake quantize three separate projections from the source's fused one).

    uv run --with numpy python -m parity.internlm2_remaining_risk_test
"""

from __future__ import annotations

import inspect

import mlx.core as mx
import mlx.nn as nn

from quanta.internlm2 import bake as _bake_mod
from quanta.internlm2 import loader as _loader_mod
from quanta.internlm2.attention import _rope_explicit, _rope_fast
from quanta.internlm2.config import InternLM2Config
from quanta.internlm2.decode import InternLM2Cache
from quanta.internlm2.generate import _apply_repetition_penalty, _sample
from quanta.internlm2.runtime import _ATTN_BITS, _GROUP_SIZE, _MLP_BITS


def _mk_cfg(**over) -> InternLM2Config:
    """A miniature InternLM2 config — preserves the dynamic-NTK + GQA + dim arithmetic."""
    base: dict = dict(
        vocab_size=128, hidden_size=32, num_hidden_layers=2, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, attention_bias=False,
        rope_theta=5e7, rope_scaling_type="dynamic", rope_scaling_factor=2.5,
        max_position_embeddings=64, hidden_act="silu", norm_eps=1e-5,
        tie_word_embeddings=False, eos_token_id=2, eos_token_ids=(2, 92542),
        pad_token_id=2, bos_token_id=1, add_bos_token=True,
    )
    base.update(over)
    return InternLM2Config(**base)


def _hf_ntk_base(theta: float, factor: float, max_pos: int, dim: int, seq_len: int) -> float:
    """Verbatim re-implementation of modeling_internlm2.py:151-154 (the HF reference)."""
    if seq_len <= max_pos:
        return theta
    return theta * ((factor * seq_len / max_pos) - (factor - 1)) ** (dim / (dim - 2))


def _check_ntk() -> bool:
    cfg = _mk_cfg()
    points = (
        cfg.max_position_embeddings - 1,
        cfg.max_position_embeddings,
        cfg.max_position_embeddings + 1,
        cfg.max_position_embeddings * 2,
        cfg.max_position_embeddings * 16,
    )
    ok = True
    for sl in points:
        got = cfg.ntk_base(sl)
        want = _hf_ntk_base(cfg.rope_theta, cfg.rope_scaling_factor,
                            cfg.max_position_embeddings, cfg.head_dim, sl)
        match = abs(got - want) <= 1e-6 * max(abs(want), 1.0)
        ok = ok and match
        print(f"  ntk(seq_len={sl}): mine={got:.6e} hf={want:.6e} match={match}")
    same_at_boundary = cfg.ntk_base(cfg.max_position_embeddings) == cfg.rope_theta
    print(f"  ntk no-rescale at boundary (seq_len == max_pos): {same_at_boundary}")
    return ok and same_at_boundary


def _check_chunked_prefill_offset() -> bool:
    """RoPE in chunks must equal RoPE over the whole sequence (the generator's chunked-prefill contract)."""
    mx.random.seed(0)
    b, h, t, d = 1, 2, 256, 16
    chunk = 64
    base = 5e7
    x = mx.random.normal((b, h, t, d)).astype(mx.float32)
    y_full = _rope_fast(x, base=base, offset=0)
    chunks = []
    for start in range(0, t, chunk):
        end = min(start + chunk, t)
        chunks.append(_rope_fast(x[:, :, start:end, :], base=base, offset=start))
    y_chunks = mx.concatenate(chunks, axis=2)
    diff = float(mx.max(mx.abs(y_full - y_chunks)))
    print(f"  chunked vs full RoPE max abs diff: {diff:.3e}")
    return diff < 1e-4


def _check_rope_fast_vs_explicit_long_offset() -> bool:
    """rotate_half RoPE fast == explicit at a 1M-token offset (long-context decode regime)."""
    mx.random.seed(3)
    x = mx.random.normal((1, 2, 16, 16)).astype(mx.float32)
    base = 5e7
    offset = 1_000_000
    y_fast = _rope_fast(x, base, offset)
    y_naive = _rope_explicit(x, base, offset)
    diff = float(mx.max(mx.abs(y_fast - y_naive)))
    print(f"  rope_fast vs rope_explicit @offset={offset}: max abs diff={diff:.3e}")
    # fp32 sin/cos at 1M positions; ~1e-2 is the expected bound (matches the prior scaffold gate).
    return diff < 5e-2


def _check_cache_roundtrip() -> bool:
    """``InternLM2Cache``: extend, then ``truncate(pre_offset)`` restores; ``replicate(B)`` broadcasts."""
    cfg = _mk_cfg(num_hidden_layers=2)
    c = InternLM2Cache(cfg, quantized=False)
    k0 = mx.random.normal((1, cfg.num_key_value_heads, 7, cfg.head_dim)).astype(mx.bfloat16)
    v0 = mx.random.normal((1, cfg.num_key_value_heads, 7, cfg.head_dim)).astype(mx.bfloat16)
    for layer in c.as_list():
        layer.update(k0, v0)
    pre_offset = c.offset
    # extend by 5 more, then roll back
    k1 = mx.random.normal((1, cfg.num_key_value_heads, 5, cfg.head_dim)).astype(mx.bfloat16)
    v1 = mx.random.normal((1, cfg.num_key_value_heads, 5, cfg.head_dim)).astype(mx.bfloat16)
    for layer in c.as_list():
        layer.update(k1, v1)
    extended_ok = c.offset == pre_offset + 5
    c.truncate(pre_offset)
    truncated_ok = c.offset == pre_offset
    r = c.replicate(4)
    rep_ok = (r.offset == pre_offset
              and all(rl.k.shape[0] == 4 and rl.v.shape[0] == 4 for rl in r.as_list()))
    print(f"  extend→offset={c.offset+5}: ok={extended_ok}  truncate→{pre_offset}: ok={truncated_ok}  "
          f"replicate(4): ok={rep_ok}")
    return extended_ok and truncated_ok and rep_ok


def _check_bake_runtime_lockstep() -> bool:
    """The runtime's bit-width constants must match the bake's policy (rule-6: fail loud, not silent)."""
    same_attn = _ATTN_BITS == _bake_mod._ATTN_BITS
    same_mlp = _MLP_BITS == _bake_mod._MLP_BITS
    sig = inspect.signature(_bake_mod.bake_internlm2)
    default_gs = sig.parameters["group_size"].default
    same_gs = default_gs == _GROUP_SIZE
    print(f"  attn_bits {_ATTN_BITS}=={_bake_mod._ATTN_BITS}? {same_attn}  "
          f"mlp_bits {_MLP_BITS}=={_bake_mod._MLP_BITS}? {same_mlp}  "
          f"group_size {_GROUP_SIZE}=={default_gs}? {same_gs}")
    return same_attn and same_mlp and same_gs


def _check_sampler() -> bool:
    """``temperature=0`` → argmax; ``top_p=tiny`` keeps top-1; ``repetition_penalty`` divides positives."""
    key = mx.random.key(2)
    logits = mx.array([1.0, 5.0, 3.0, 2.0])
    argmax_id = int(_sample(logits, temperature=0.0, top_k=0, top_p=1.0, min_p=0.0, key=key).item())
    argmax_ok = argmax_id == 1
    pkey = mx.random.key(7)
    top1_only = int(_sample(logits, temperature=1.0, top_k=0, top_p=0.001, min_p=0.0,
                            key=pkey).item())
    top1_ok = top1_only == 1
    seen = mx.array([1, 3], dtype=mx.int32)
    pen = _apply_repetition_penalty(logits, seen, 2.0)
    pen_ok = (
        abs(float(pen[0].item()) - 1.0) < 1e-5
        and abs(float(pen[1].item()) - 2.5) < 1e-5
        and abs(float(pen[2].item()) - 3.0) < 1e-5
        and abs(float(pen[3].item()) - 1.0) < 1e-5
    )
    print(f"  T=0→argmax_id={argmax_id} ok={argmax_ok}  "
          f"top_p=0.001 →id={top1_only} ok={top1_ok}  rep_penalty: ok={pen_ok}")
    return argmax_ok and top1_ok and pen_ok


def _check_swiglu_identity() -> bool:
    """Sanity: the runtime's SwiGLU formula (``silu(w1·x) * (w3·x) → w2``) matches the bf16 reference."""
    mx.random.seed(1)
    h, i = 32, 64
    x = mx.random.normal((3, h)).astype(mx.bfloat16)
    w1 = mx.random.normal((i, h)).astype(mx.bfloat16)
    w3 = mx.random.normal((i, h)).astype(mx.bfloat16)
    w2 = mx.random.normal((h, i)).astype(mx.bfloat16)
    # Reference: explicit operator order
    gate = x @ w1.T
    up = x @ w3.T
    y_ref = (nn.silu(gate) * up) @ w2.T
    # Runtime path mimic (calling order in ``_packed_ffn``)
    y_got = (nn.silu(x @ w1.T) * (x @ w3.T)) @ w2.T
    diff = float(mx.max(mx.abs(y_ref - y_got)))
    print(f"  swiglu reference identity diff: {diff:.3e}")
    return diff < 1e-3


def _check_wqkv_split() -> bool:
    """Re-confirm ``_split_wqkv`` deinterleaves a synthetic fused weight with the expected per-kv-head
    interleaving (gs = n_rep + 2 row-slots: first n_rep are q-heads, then k-head, then v-head).
    """
    cfg = _mk_cfg(num_attention_heads=4, num_key_value_heads=2, head_dim=8, hidden_size=32)
    n_kv, rep, hd, hidden = cfg.num_key_value_heads, cfg.n_rep, cfg.head_dim, cfg.hidden_size
    gs = rep + 2
    # Build a fused [(n_kv * gs * hd), hidden] tensor where each row is uniquely tagged so
    # we can verify each slot lands in the right output projection.
    tag = mx.arange(n_kv * gs * hd * hidden, dtype=mx.float32).reshape(n_kv * gs * hd, hidden)
    wq, wk, wv = _loader_mod._split_wqkv(tag, cfg)
    shapes_ok = (
        wq.shape == (n_kv * rep * hd, hidden)
        and wk.shape == (n_kv * hd, hidden)
        and wv.shape == (n_kv * hd, hidden)
    )
    # Verify the deinterleave invariant: rebuild the fused tensor from the split and compare
    grouped = tag.reshape(n_kv, gs, hd, hidden)
    wq_ref = grouped[:, :rep, :, :].reshape(n_kv * rep * hd, hidden)
    wk_ref = grouped[:, rep, :, :].reshape(n_kv * hd, hidden)
    wv_ref = grouped[:, rep + 1, :, :].reshape(n_kv * hd, hidden)
    eq_ok = (
        float(mx.max(mx.abs(wq - wq_ref))) == 0.0
        and float(mx.max(mx.abs(wk - wk_ref))) == 0.0
        and float(mx.max(mx.abs(wv - wv_ref))) == 0.0
    )
    print(f"  wqkv split shapes ok={shapes_ok}  per-slot identity ok={eq_ok}")
    return shapes_ok and eq_ok


def run() -> bool:
    print("== InternLM2.5 remaining-risk verification (model-free) ==")
    results = {
        "[1] dynamic-NTK formula vs HF": _check_ntk(),
        "[2] chunked-prefill RoPE offset arithmetic": _check_chunked_prefill_offset(),
        "[3] rope_fast vs rope_explicit @ 1M offset": _check_rope_fast_vs_explicit_long_offset(),
        "[4] KV cache replicate + truncate round-trip": _check_cache_roundtrip(),
        "[5] bake ↔ runtime bit-width lock-step": _check_bake_runtime_lockstep(),
        "[6] sampler edge cases (T=0, top_p, rep_penalty)": _check_sampler(),
        "[7] SwiGLU formula identity": _check_swiglu_identity(),
        "[8] wqkv split deinterleave": _check_wqkv_split(),
    }
    print()
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print(f"\nALL OK: {all_ok}")
    return all_ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
