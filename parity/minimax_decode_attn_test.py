"""Parity: MiniMax-M2.7 incremental (decode) GQA attention == the prefill path.

THE core decode gate for :mod:`quanta.minimax.decode` (model-free). MiniMax is the simplest regime in
the repo — every layer is plain full-softmax GQA (partial RoPE + per-layer QK-norm, no MLA / no
compressor / no sliding window), so the decode state is just a growing K/V cache and the step is the
existing prefill attention restricted to a single query. With **tiny random params** (small
dim/heads/head_dim, short sequence):

  1. run the existing PREFILL path over the full ``T`` tokens at once (the parity truth, the fast +
     naive prefill agree, so we check both);
  2. run the incremental decode: feed token 0, then step one token at a time, threading a
     :class:`quanta.minimax.decode.MiniMaxCache` (via :func:`quanta.minimax.decode.decode_step`);
  3. assert the per-position outputs of (2) match (1) to a tight fp tolerance.

It also exercises :meth:`quanta.minimax.decode.MiniMaxCache.truncate` (speculative-decode rollback):
decode ``T`` tokens, roll the cache back to ``rb``, re-decode ``[rb, T)``, and assert the result
matches a fresh decode that only ever fed ``[0, T)`` then sliced — i.e. rollback restores exact
state losslessly. And the fail-loud guards: ``truncate`` past the consumed length raises (rule 6).

**Tiny tensors only — never load the real model** (a 398 GiB job may be GPU-resident).

    uv run --with numpy python -m parity.minimax_decode_attn_test
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from quanta.minimax.attention import MiniMaxAttention
from quanta.minimax.config import MiniMaxConfig
from quanta.minimax.decode import MiniMaxCache, _LayerKVCache, decode_step

# tiny geometry (a few KB of params — model-free)
DIM = 12
N_HEADS = 4
N_KV = 2          # n_rep = 2
HEAD_DIM = 8
ROTARY_DIM = 4    # partial RoPE
N_LAYERS = 2


def _cfg() -> MiniMaxConfig:
    """A minimal valid MiniMax config with the tiny geometry above (no checkpoint, no I/O)."""
    return MiniMaxConfig(
        vocab_size=32,
        hidden_size=DIM,
        moe_intermediate_size=16,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        use_qk_norm=True,
        qk_norm_type="per_layer",
        attn_type_list=tuple(1 for _ in range(N_LAYERS)),
        num_local_experts=4,
        num_experts_per_tok=2,
        shared_intermediate_size=0,
        scoring_func="sigmoid",
        use_routing_bias=True,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        use_mtp=False,
        num_mtp_modules=0,
        mtp_transformer_layers=1,
        hidden_act="silu",
        norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        bos_token_id=0,
        eos_token_id=1,
        eos_token_ids=(1,),
        tie_word_embeddings=False,
    )


def _r(rng, *shape):
    return mx.array((rng.standard_normal(shape) * 0.5).astype(np.float32))


def _attn(cfg: MiniMaxConfig, layer_id: int, rng) -> MiniMaxAttention:
    """A random-init GQA attention module (tiny weights; QK-norm weights ~1.0 like a real norm)."""
    a = MiniMaxAttention(cfg, layer_id)
    a.q_proj.weight = _r(rng, cfg.q_dim, DIM)
    a.k_proj.weight = _r(rng, cfg.kv_dim, DIM)
    a.v_proj.weight = _r(rng, cfg.kv_dim, DIM)
    a.o_proj.weight = _r(rng, DIM, cfg.q_dim)
    a.q_norm.weight = _r(rng, HEAD_DIM) * 0.1 + 1.0
    a.k_norm.weight = _r(rng, HEAD_DIM) * 0.1 + 1.0
    mx.eval([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight, a.o_proj.weight,
             a.q_norm.weight, a.k_norm.weight])
    return a


def _maxdiff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _decode_all(attn, x, cache, lo: int, hi: int, *, use_fast: bool) -> mx.array:
    """Decode tokens ``[lo, hi)`` into ``cache`` (one layer's K/V cache); return ``[1, hi-lo, dim]``."""
    outs = [decode_step(x[:, t:t + 1], attn, cache, use_fast=use_fast) for t in range(lo, hi)]
    return mx.concatenate(outs, axis=1)


def _run_layer(cfg: MiniMaxConfig, layer_id: int, T: int, rb: int, rng) -> bool:
    attn = _attn(cfg, layer_id, rng)
    x = _r(rng, 1, T, DIM)

    # (1) prefill over the whole sequence — fast path is the runtime path; naive is the reference.
    ref_fast = attn(x, use_fast=True)                          # [1,T,dim]
    ref_naive = attn(x, use_fast=False)
    d_fastnaive = _maxdiff(ref_fast, ref_naive)               # fast prefill == naive prefill

    # (2) incremental decode, one token at a time, threading a per-layer cache.
    cache = MiniMaxCache(cfg.num_hidden_layers)
    inc = _decode_all(attn, x, cache[layer_id], 0, T, use_fast=True)
    d_inc = _maxdiff(ref_fast, inc)                           # decode == fast prefill
    d_inc_naive = _maxdiff(ref_naive, inc)                    # decode == naive prefill
    absmax = float(mx.max(mx.abs(ref_naive)).item())
    off_ok = cache.offset == T and cache[layer_id].offset == T

    # (3) truncate (rollback) to ``rb``, re-decode [rb, T) and compare against a fresh run that only
    #     ever fed [0, T) (sliced). Rollback must restore exact, lossless state.
    cache.truncate(rb)
    trunc_off_ok = cache.offset == rb
    a = _decode_all(attn, x, cache[layer_id], rb, T, use_fast=True)
    fresh = MiniMaxCache(cfg.num_hidden_layers)
    b = _decode_all(attn, x, fresh[layer_id], 0, T, use_fast=True)[:, rb:]
    d_roll = _maxdiff(a, b)

    good = (inc.shape == ref_fast.shape and d_inc < 1e-4 and d_inc_naive < 2e-3 and d_roll < 1e-4
            and d_fastnaive < 2e-3 and off_ok and trunc_off_ok)
    flags = "" if (off_ok and trunc_off_ok) else " STATE-BAD"
    print(f"  [{'OK' if good else 'FAIL'}] GQA L{layer_id} T={T:2d} rb={rb:2d}  "
          f"|Δdecode−prefill|={d_inc:.2e} |Δdecode−naive|={d_inc_naive:.2e} "
          f"|Δrollback|={d_roll:.2e} |Δfast−naive|={d_fastnaive:.2e} absmax={absmax:.3f}{flags}")
    return good


def _truncate_guards(cfg: MiniMaxConfig, rng) -> bool:
    """``truncate`` is lossless or it raises (rule 6): rolling forward past the consumed length, and a
    negative length, both fail loud; a no-op truncate to the current length leaves state intact."""
    ok = True
    attn = _attn(cfg, 0, rng)
    x = _r(rng, 1, 4, DIM)
    cache = MiniMaxCache(cfg.num_hidden_layers)
    _decode_all(attn, x, cache[0], 0, 4, use_fast=True)

    # roll FORWARD past consumed length -> ValueError (cannot fabricate future K/V)
    try:
        cache.truncate(5)
        good = False
    except ValueError:
        good = True
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] truncate(5) past consumed length -> ValueError")

    # negative length -> ValueError
    try:
        cache.truncate(-1)
        good = False
    except ValueError:
        good = True
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] truncate(-1) -> ValueError")

    # truncate to the current length is a no-op (state unchanged, still decodable)
    before = cache.offset
    cache.truncate(before)
    good = cache.offset == before
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] truncate(current) no-op: offset={cache.offset} (expect {before})")

    # empty per-layer cache: truncate(0) ok, truncate(>0) raises (direct fail-loud guard)
    empty = _LayerKVCache()
    try:
        empty.truncate(0)
        good = empty.offset == 0
    except ValueError:
        good = False
    try:
        empty.truncate(1)
        good = good and False
    except ValueError:
        good = good and True
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] empty per-layer cache truncate: 0 ok, >0 raises")
    return ok


def run() -> None:
    cfg = _cfg()
    rng = np.random.default_rng(0)
    ok = True
    # T crosses several positions; rb rolls back a few tokens (within the lossless K/V slice).
    ok &= _run_layer(cfg, 0, 12, 7, rng)
    ok &= _run_layer(cfg, 1, 12, 11, rng)
    ok &= _run_layer(cfg, 0, 6, 1, rng)
    ok &= _truncate_guards(cfg, rng)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
