"""Parity: DSV4 incremental (decode) attention == the prefill path, for all three regimes.

Model-free gate for :mod:`quanta.dsv4.decode` (task #77, decode half). With **tiny random params**
(small dim/head_dim/window, short sequence), for each of the three per-layer attention regimes:

  1. run the existing PREFILL path over the full ``T`` tokens at once;
  2. run the incremental decode: feed token 0, then ``decode_step_*`` one token at a time, threading
     a :class:`quanta.dsv4.decode.DSV4Cache`;
  3. assert the per-position outputs of (2) match (1) to a tight fp tolerance.

It also exercises ``DSV4Cache.truncate`` (speculative-decode rollback): decode ``T`` tokens, roll the
cache back to ``T-1``, re-decode the last token, and assert the result matches a fresh decode that
only ever fed ``T-1`` then that token — i.e. rollback restores exact state.

The three regimes are ``cfg.compress_ratio(layer_id)``: ratio 0 (dense sliding-window), ratio 4
(compressed + Lightning-Indexer / DSA), and the "ratio-128" regime (compressed, no indexer). The
ratio-128 code path is keyed purely on ``has_indexer``/``overlap`` (both ``== 4``), so any non-{0,4}
ratio exercises it identically; we use a small ratio (3) so a short sequence still produces several
compressed tokens. **Tiny tensors only — never load the real model** (a ~180 GiB job may be resident).

    uv run --with numpy python -m parity.dsv4_decode_attn_test
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from quanta.dsv4 import decode as D
from quanta.dsv4.attention import attention_dense, rope_cos_sin
from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.indexer import attention_compressed

# tiny geometry (a few KB of params — model-free)
DIM = 12
N_HEADS = 2
HEAD_DIM = 8
ROPE_HEAD_DIM = 4
Q_LORA = 6
O_GROUPS = 2
O_LORA = 3
WINDOW = 4
IDX_HEADS = 2
IDX_HEAD_DIM = 4
IDX_TOPK = 2
RATIOS = (0, 4, 3)            # L0 dense ; L1 ratio-4 (+indexer) ; L2 ratio-3 (== ratio-128 regime)


def _cfg() -> DeepSeekV4Config:
    """A minimal valid DSV4 config with the tiny geometry above (no checkpoint, no I/O)."""
    return DeepSeekV4Config(
        vocab_size=32, hidden_size=DIM, num_hidden_layers=len(RATIOS), moe_intermediate_size=16,
        num_attention_heads=N_HEADS, head_dim=HEAD_DIM, rope_head_dim=ROPE_HEAD_DIM,
        q_lora_rank=Q_LORA, o_lora_rank=O_LORA, o_groups=O_GROUPS, sliding_window=WINDOW,
        index_n_heads=IDX_HEADS, index_head_dim=IDX_HEAD_DIM, index_topk=IDX_TOPK,
        compress_ratios=RATIOS, compress_rope_theta=10000.0,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1, n_hash_layers=0,
        scoring_func="sqrtsoftplus", topk_method="noaux_tc", norm_topk_prob=True,
        routed_scaling_factor=1.0, swiglu_limit=0.0,
        hc_mult=1, hc_sinkhorn_iters=1, hc_eps=1e-6, n_mtp_layers=0,
        norm_eps=1e-6, rope_theta=10000.0,
        rope_scaling={"factor": 4.0, "beta_fast": 32, "beta_slow": 1,
                      "original_max_position_embeddings": 16, "type": "yarn"},
        max_position_embeddings=4096, bos_token_id=0, eos_token_id=1, eos_token_ids=(1,),
        tie_word_embeddings=False,
    )


def _r(rng, *shape):
    return mx.array((rng.standard_normal(shape) * 0.5).astype(np.float32))


def _compressor_params(rng, ratio: int, head_dim: int):
    coff = 2 if ratio == 4 else 1
    return {"ape": _r(rng, ratio, coff * head_dim),
            "norm": _r(rng, head_dim) * 0.1 + 1.0,
            "wkv": _r(rng, coff * head_dim, DIM),
            "wgate": _r(rng, coff * head_dim, DIM)}


def _attn_params(rng, cfg: DeepSeekV4Config, layer_id: int) -> dict:
    """Random attention param dict matching the loader's ``attention(layer_id)`` keys/shapes."""
    p = {"wq_a": _r(rng, Q_LORA, DIM),
         "q_norm": _r(rng, Q_LORA) * 0.1 + 1.0,
         "wq_b": _r(rng, N_HEADS * HEAD_DIM, Q_LORA),
         "wkv": _r(rng, HEAD_DIM, DIM),
         "kv_norm": _r(rng, HEAD_DIM) * 0.1 + 1.0,
         "wo_a": _r(rng, O_GROUPS * O_LORA, (N_HEADS * HEAD_DIM) // O_GROUPS),
         "wo_b": _r(rng, DIM, O_GROUPS * O_LORA),
         "attn_sink": _r(rng, N_HEADS)}
    ratio = cfg.compress_ratio(layer_id)
    if cfg.has_compressor(layer_id):
        p["compressor"] = _compressor_params(rng, ratio, HEAD_DIM)
    if cfg.has_indexer(layer_id):
        p["indexer"] = {"wq_b": _r(rng, IDX_HEADS * IDX_HEAD_DIM, Q_LORA),
                        "weights_proj": _r(rng, IDX_HEADS, DIM),
                        "compressor": _compressor_params(rng, 4, IDX_HEAD_DIM)}
    return p


def _rope_full(cfg: DeepSeekV4Config, layer_id: int, seqlen: int):
    """Full ``(cos, sin)`` RoPE tables for absolute positions ``[0, seqlen)`` — identical to what the
    prefill path builds internally (so decode slices are bit-for-bit the same)."""
    orig, theta = cfg.attn_rope(layer_id)
    return rope_cos_sin(cfg.rope_head_dim, seqlen, orig, theta, cfg.rope_factor,
                        cfg.beta_fast, cfg.beta_slow)


def _maxdiff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _decode_all(step, x, p, cfg, layer_id, cos, sin, lo: int, hi: int, cache):
    """Decode tokens ``[lo, hi)`` into ``cache``; return the stacked outputs ``[1, hi-lo, dim]``."""
    outs = [step(x[:, t:t + 1], p, cfg, layer_id, cache, cos, sin, t) for t in range(lo, hi)]
    return mx.concatenate(outs, axis=1)


def _run_regime(cfg: DeepSeekV4Config, layer_id: int, name: str, T: int, rb: int, rng) -> bool:
    p = _attn_params(rng, cfg, layer_id)
    step = D.decode_step_dense if not cfg.has_compressor(layer_id) else D.decode_step_compressed
    prefill = attention_dense if not cfg.has_compressor(layer_id) else attention_compressed
    x = _r(rng, 1, T, DIM)

    # (1) prefill over the whole sequence
    ref = prefill(x, p, cfg, layer_id)                          # [1,T,dim]

    # (2) incremental decode, one token at a time, threading the cache
    cos, sin = _rope_full(cfg, layer_id, T)
    cache = D.DSV4Cache(cfg.num_hidden_layers)
    inc = _decode_all(step, x, p, cfg, layer_id, cos, sin, 0, T, cache)
    d_inc = _maxdiff(ref, inc)
    absmax = float(mx.max(mx.abs(ref)).item())
    off_ok = cache.offset == T

    # (3) truncate (rollback) to ``rb`` (crosses a compressor-window boundary on compressed layers,
    #     so it must drop compressed tokens too), then re-decode [rb, T) and compare against a fresh
    #     run that only ever fed [0, T). Rollback must restore exact state.
    cache.truncate(rb)
    trunc_off_ok = cache.offset == rb
    a = _decode_all(step, x, p, cfg, layer_id, cos, sin, rb, T, cache)
    fresh = D.DSV4Cache(cfg.num_hidden_layers)
    b = _decode_all(step, x, p, cfg, layer_id, cos, sin, 0, T, fresh)[:, rb:]
    d_roll = _maxdiff(a, b)
    # compressed-stream length must match the fresh run after rollback (exercises ckv/ikv truncation)
    ncomp_ok = cache[layer_id].n_comp() == fresh[layer_id].n_comp()

    good = (inc.shape == ref.shape and d_inc < 1e-4 and d_roll < 1e-4 and off_ok
            and trunc_off_ok and ncomp_ok)
    flags = "" if (off_ok and trunc_off_ok and ncomp_ok) else " STATE-BAD"
    print(f"  [{'OK' if good else 'FAIL'}] {name:22s} L{layer_id} ratio={cfg.compress_ratio(layer_id):3d}"
          f" T={T:2d} rb={rb:2d}  |Δprefill|={d_inc:.2e} |Δrollback|={d_roll:.2e}"
          f" ncomp={cache[layer_id].n_comp()} absmax={absmax:.3f}{flags}")
    return good


def run() -> None:
    cfg = _cfg()
    rng = np.random.default_rng(0)
    ok = True
    # T crosses the sliding window (WINDOW=4) and fills several compressor windows; rb crosses a
    # compressor-window boundary (so rollback must drop compressed tokens) within the bounded ring.
    ok &= _run_regime(cfg, 0, "dense (sliding-window)", 12, 7, rng)
    ok &= _run_regime(cfg, 1, "compressed + indexer", 12, 11, rng)
    ok &= _run_regime(cfg, 2, "compressed (no indexer)", 12, 11, rng)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
