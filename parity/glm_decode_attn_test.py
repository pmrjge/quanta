"""Parity: GLM-5.1 incremental (decode) attention == the prefill path, for the MLA + DSA indexer.

THE core decode gate for the GLM serving stack (mirrors ``parity/dsv4_decode_attn_test.py``). Model-free
— a tiny random :class:`quanta.glm.model.GLMDecoderLayer` (small hidden / few heads / few experts /
short sequence), no checkpoint, no artifact, a few KB of tensors — safe while a large GPU job is
resident. For both indexer regimes (keep-all ``index_topk >= T`` AND a biting ``index_topk < T``):

  1. run the PREFILL block over the full ``T`` tokens at once (:meth:`GLMDecoderLayer.__call__`);
  2. run the incremental decode: feed one token at a time through :meth:`GLMDecoderLayer.step`, threading
     a :class:`quanta.glm.decode.GLMCache` (the MLA latent KV + DSA indexer key streams);
  3. assert the per-position outputs of (2) match (1) to a tight fp tolerance — including the case where
     the indexer selection actually *bites* (so the DSA top-k must select the SAME keys per position in
     decode as in prefill, not just when it keeps everything).

It also exercises ``GLMCache.truncate`` (speculative-decode rollback): decode ``T`` tokens, roll the
cache back to ``rb``, re-decode ``[rb, T)``, and assert the result matches a fresh decode that only ever
fed ``[0, T)`` — i.e. rollback restores exact state (and ``offset`` tracks it). Finally a small
structural check that the real :func:`quanta.glm.mtp.mtp_forward` is callable model-free (shape-correct
logits) so the spec-decode head contract is exercised against the real forward, not only a stub.

    uv run --with numpy python -m parity.glm_decode_attn_test
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from quanta.glm.config import GLMConfig
from quanta.glm.decode import GLMCache, decode_step
from quanta.glm.model import GLMDecoderLayer
from quanta.glm.mtp import build_mtp_layer, mtp_forward

# tiny geometry (a few KB of params — model-free)
CFG = {
    "model_type": "glm_moe_dsa", "vocab_size": 40, "hidden_size": 16, "intermediate_size": 24,
    "num_hidden_layers": 3, "num_attention_heads": 2, "num_key_value_heads": 2,
    "q_lora_rank": 8, "kv_lora_rank": 6, "qk_nope_head_dim": 4, "qk_rope_head_dim": 4,
    "qk_head_dim": 8, "v_head_dim": 8,
    "index_head_dim": 6, "index_n_heads": 2, "index_topk": 3,
    "n_routed_experts": 5, "num_experts_per_tok": 2, "n_shared_experts": 1, "moe_intermediate_size": 6,
    "first_k_dense_replace": 1, "num_nextn_predict_layers": 1,
    "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
    "eos_token_id": [39], "pad_token_id": 39, "tie_word_embeddings": False,
}


def _randomize_module(module, rng: np.random.Generator, scale: float = 0.3) -> None:
    """Fill every leaf param of an ``nn.Module`` with small reproducible random values."""
    flat: dict[str, mx.array] = {}

    def walk(prefix, node):
        if isinstance(node, dict):
            items = node.items()
        elif isinstance(node, (list, tuple)):
            items = enumerate(node)
        else:
            flat[prefix.rstrip(".")] = node
            return
        for k, v in items:
            walk(f"{prefix}{k}.", v)

    walk("", module.parameters())
    upd = [(k, mx.array((rng.standard_normal(v.shape) * scale).astype(np.float32))) for k, v in flat.items()]
    module.load_weights(upd, strict=False)


def _build_layer(cfg: GLMConfig, layer_id: int, rng: np.random.Generator) -> GLMDecoderLayer:
    layer = GLMDecoderLayer(cfg, layer_id)
    _randomize_module(layer, rng, scale=0.3)
    if hasattr(layer.mlp, "set_experts"):  # randomize the routed expert stacks (zeros buffers otherwise)
        def r(*s):
            return mx.array((rng.standard_normal(s) * 0.2).astype(np.float32))
        e, inter, h = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size
        layer.mlp.set_experts(r(e, inter, h), r(e, inter, h), r(e, h, inter))
    return layer


def _maxabs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _decode_range(layer: GLMDecoderLayer, h: mx.array, cache: GLMCache, layer_id: int,
                  lo: int, hi: int) -> mx.array:
    """Decode tokens ``[lo, hi)`` into ``cache`` via :func:`decode_step`; return ``[1, hi-lo, dim]``."""
    cols = [decode_step(layer, h[:, k:k + 1], cache[layer_id], k, use_fast=False, use_indexer=True)
            for k in range(lo, hi)]
    return mx.concatenate(cols, axis=1)


def _run_regime(cfg: GLMConfig, name: str, T: int, rb: int, rng: np.random.Generator) -> bool:
    layer_id = 1  # a MoE layer (first_k_dense_replace=1) so the full block (MLA+DSA+MoE) is exercised
    layer = _build_layer(cfg, layer_id, rng)
    h = mx.array((rng.standard_normal((1, T, cfg.hidden_size)) * 0.5).astype(np.float32))
    positions = mx.arange(T)

    # (1) prefill over the whole sequence
    ref = layer(h, positions, use_fast=False, use_indexer=True)          # [1,T,dim]

    # (2) incremental decode, one token at a time, threading the cache
    cache = GLMCache(cfg.num_hidden_layers)
    inc = _decode_range(layer, h, cache, layer_id, 0, T)
    d_inc = _maxabs(ref, inc)
    off_ok = cache.offset == T

    # (3) truncate (rollback) to ``rb``, re-decode [rb, T), and compare against a fresh run that only
    #     ever fed [0, T). Rollback must restore exact state (incl. the indexer key stream).
    cache.truncate(rb)
    trunc_off_ok = cache.offset == rb
    a = _decode_range(layer, h, cache, layer_id, rb, T)
    fresh = GLMCache(cfg.num_hidden_layers)
    b = _decode_range(layer, h, fresh, layer_id, 0, T)[:, rb:]
    d_roll = _maxabs(a, b)

    good = (inc.shape == ref.shape and d_inc < 1e-4 and d_roll < 1e-4 and off_ok and trunc_off_ok)
    flags = "" if (off_ok and trunc_off_ok) else " STATE-BAD"
    print(f"  [{'OK' if good else 'FAIL'}] {name:26s} topk={cfg.index_topk} T={T:2d} rb={rb:2d}  "
          f"|Δprefill|={d_inc:.2e} |Δrollback|={d_roll:.2e} offset={cache.offset}{flags}")
    return good


def _structural_mtp(cfg: GLMConfig, rng: np.random.Generator) -> bool:
    """The real :func:`quanta.glm.mtp.mtp_forward` is callable model-free and returns shape-correct
    logits — so the spec-decode head contract is validated against the real forward (not only a stub)."""
    h, e, mi, v = cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.vocab_size

    def r(*s):
        return mx.array((rng.standard_normal(s) * 0.2).astype(np.float32))

    nh, qk, vhd = cfg.num_attention_heads, cfg.qk_head_dim, cfg.v_head_dim
    p = {
        "enorm": r(h), "hnorm": r(h), "eh_proj": r(h, 2 * h), "shared_head_norm": r(h),
        "input_layernorm": r(h), "post_attention_layernorm": r(h),
        "attention": {
            "q_a_proj": r(cfg.q_lora_rank, h), "q_a_layernorm": r(cfg.q_lora_rank),
            "q_b_proj": r(nh * qk, cfg.q_lora_rank),
            "kv_a_proj_with_mqa": r(cfg.kv_lora_rank + cfg.qk_rope_head_dim, h),
            "kv_a_layernorm": r(cfg.kv_lora_rank),
            "kv_b_proj": r(nh * (cfg.qk_nope_head_dim + vhd), cfg.kv_lora_rank),
            "o_proj": r(h, nh * vhd),
            "indexer": {"wq_b": r(cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank),
                        "wk": r(cfg.index_head_dim, h), "weights_proj": r(cfg.index_n_heads, h),
                        "k_norm_weight": r(cfg.index_head_dim), "k_norm_bias": r(cfg.index_head_dim)},
        },
        "router": {"weight": r(e, h), "e_score_correction_bias": r(e)},
        "shared": {"gate_proj": r(mi, h), "up_proj": r(mi, h), "down_proj": r(h, mi)},
        "experts": {"gate_proj": r(e, mi, h), "up_proj": r(e, mi, h), "down_proj": r(e, h, mi)},
    }
    layer = build_mtp_layer(p, cfg, dtype=mx.float32)
    T = 3
    prev_hidden = r(1, T, h)
    next_ids = mx.array([[1, 2, 3]])
    embed = r(v, h)
    head = r(v, h)
    logits = mtp_forward(prev_hidden, next_ids, embed, head, p, layer, cfg,
                         use_fast=False, use_indexer=True)
    good = tuple(logits.shape) == (1, T, v) and bool(mx.all(mx.isfinite(logits)).item())
    print(f"  [{'OK' if good else 'FAIL'}] mtp_forward callable & shaped {tuple(logits.shape)} finite")
    return good


def run() -> None:
    ok = True
    rng = np.random.default_rng(0)

    # keep-all indexer (topk >= T): decode == prefill is a pure MLA causal check
    cfg_keep = GLMConfig.from_dict({**CFG, "index_topk": 999})
    ok &= _run_regime(cfg_keep, "MLA + DSA (keep-all)", 10, 6, rng)

    # biting indexer (topk < T): the DSA top-k must select the SAME keys per position in decode as
    # in prefill — the real value of the gate (rb crosses several positions within the cache).
    cfg_bite = GLMConfig.from_dict({**CFG, "index_topk": 3})
    ok &= _run_regime(cfg_bite, "MLA + DSA (top-k bites)", 10, 7, rng)

    ok &= _structural_mtp(cfg_keep, rng)

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
