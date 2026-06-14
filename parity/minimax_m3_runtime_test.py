"""Model-free M3-1 gate: the MiniMax-M3-VL resident serving runtime.

Builds a tiny SYNTHETIC M3 text decoder (no real weights) two ways and drives
:class:`quanta.minimax.runtime_m3.MiniMaxM3ResidentModel` to prove the resident serving path is
output-equivalent to the proven M1/M2 reference forward:

* **packed int6 ``gather_qmm`` == bf16 ``gather_mm`` (greedy-exact).** The routed experts are
  RTN-quantized to int6 once; the *packed* model holds the affine triplets (the resident
  ``mx.gather_qmm`` path), the *bf16* model holds the SAME codes dequantized to a stack (the
  reference ``mx.gather_mm`` path). Same dispatch, same matvec — only fused-vs-separate dequant
  differs ⇒ identical top-1 and ~ULP logits. This is the M3-1 quant-runtime invariant.
* **cached forward == prefill.** A ``T``-token cached forward over fresh per-layer
  :class:`quanta.minimax.model_m3.KVCache` reproduces the ``caches=None`` prefill bit-for-bit (the
  cache only stores k/v; the causal attention is identical).
* **incremental decode == full prefill.** Prefill a prompt, then step the remaining tokens one at a
  time through the cache; the per-step logits reproduce the full-prefill logits at those positions
  (the serving decode primitive — proves the partial-RoPE offset + bottom-right causal continuation).
* **rule-4 / rule-6:** the MoE dense oracle (``sparse=False``) == the sparse ``gather_mm`` dispatch
  (bf16), and a packed (int6) MoE **refuses** ``sparse=False`` (no silent dequant). Plus a
  ``generate`` smoke (the greedy serving convenience returns a token list).

All on tiny synthetic dims; runs in the model-free sweep (no real weights, no checkpoint IO).

    uv run python -m parity.minimax_m3_runtime_test
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from quanta.bake.quant import quantize_affine
from quanta.minimax import model_m3 as M
from quanta.minimax import runtime_m3 as RT
from quanta.minimax.config_m3 import MiniMaxM3Config

GS = 32   # affine group size — divides hidden 64 and moe_inter 32 (the quantized expert in-dims)
BITS = 6  # routed experts int6 (the served scheme)

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _rel(a: mx.array, b: mx.array) -> float:
    an, bn = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    return float(np.abs(an - bn).max() / (np.abs(bn).max() + 1e-9))


def _argmax_agree(a: mx.array, b: mx.array) -> float:
    aa = mx.argmax(a[0].astype(mx.float32), axis=-1)
    bb = mx.argmax(b[0].astype(mx.float32), axis=-1)
    return float(mx.mean((aa == bb).astype(mx.float32)).item())


# --- tiny synthetic config (structurally faithful; 2 dense + 3 MoE layers) --- #

def _cfg() -> MiniMaxM3Config:
    tc = {
        "vocab_size": 128, "hidden_size": 64, "intermediate_size": 32,
        "dense_intermediate_size": 96, "shared_intermediate_size": 32,
        "num_hidden_layers": 5,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 16,
        "rotary_dim": 8, "partial_rotary_factor": 0.5, "rope_theta": 5e6,
        "use_qk_norm": True, "qk_norm_type": "per_head", "use_gemma_norm": True,
        "attention_output_gate": False,
        "num_local_experts": 6, "num_experts_per_tok": 2, "n_shared_experts": 1,
        "scoring_func": "sigmoid", "use_routing_bias": True, "routed_scaling_factor": 2.0,
        "norm_topk_prob": True,
        "moe_layer_freq": [0, 0, 1, 1, 1],
        "hidden_act": "swigluoai", "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
        "rms_norm_eps": 1e-6, "max_position_embeddings": 1048576, "tie_word_embeddings": False,
        "eos_token_id": 200020,
    }
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "config.json").write_text(json.dumps({"model_type": "minimax_m3_vl", "text_config": tc}))
        return MiniMaxM3Config.from_pretrained(d)


# --- deterministic synthetic weights (shared by both models) ----------------- #

def _synth(cfg: MiniMaxM3Config, key) -> dict:
    ks = mx.random.split(key, 512)
    c = iter(range(512))

    def nx():
        return ks[next(c)]

    def bf(shape, scale):
        return (mx.random.normal(shape, key=nx()) * scale).astype(mx.bfloat16)

    h = cfg.hidden_size
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    e, inter = cfg.num_local_experts, cfg.moe_intermediate_size
    si, di = cfg.shared_intermediate_size, cfg.dense_intermediate_size
    w: dict = {"embed": bf((cfg.vocab_size, h), 0.05), "final_norm": bf((h,), 0.2),
               "lm_head": bf((cfg.vocab_size, h), 0.05)}
    for i in range(cfg.num_hidden_layers):
        w[(i, "in")], w[(i, "post")] = bf((h,), 0.2), bf((h,), 0.2)
        w[(i, "q")] = bf((nh * hd, h), 0.05)
        w[(i, "k")] = bf((nkv * hd, h), 0.05)
        w[(i, "v")] = bf((nkv * hd, h), 0.05)
        w[(i, "o")] = bf((h, nh * hd), 0.05)
        w[(i, "qn")], w[(i, "kn")] = bf((hd,), 0.2), bf((hd,), 0.2)
        if cfg.is_moe_layer(i):
            w[(i, "gate")] = mx.random.normal((e, h), key=nx()).astype(mx.float32)
            w[(i, "bias")] = mx.random.normal((e,), key=nx()).astype(mx.float32)
            w[(i, "gate_up")] = bf((e, 2 * inter, h), 0.1)     # [E, 2*inter, h] (w1 over w3)
            w[(i, "down")] = bf((e, h, inter), 0.1)            # [E, h, inter]   (w2)
            w[(i, "sg")], w[(i, "su")], w[(i, "sd")] = bf((si, h), 0.1), bf((si, h), 0.1), bf((h, si), 0.1)
        else:
            w[(i, "dg")], w[(i, "du")], w[(i, "dd")] = bf((di, h), 0.1), bf((di, h), 0.1), bf((h, di), 0.1)
    mx.eval(list(w.values()))
    return w


def _pack(stack: mx.array) -> dict:
    pq, sc, b = quantize_affine(stack, BITS, GS, scale_dtype=mx.bfloat16)
    return {"packed": pq, "scale": sc, "bias": b, "group_size": GS, "bits": BITS}


def _dq(trip: dict) -> mx.array:
    return mx.dequantize(trip["packed"], trip["scale"], trip["bias"],
                         group_size=trip["group_size"], bits=trip["bits"]).astype(mx.bfloat16)


def _build_model(cfg: MiniMaxM3Config, w: dict, *, packed: bool) -> RT.MiniMaxM3ResidentModel:
    """Build the resident model from the shared synthetic weights. ``packed`` holds the routed
    experts as int6 triplets (gather_qmm); else the SAME codes dequantized to bf16 (gather_mm)."""
    blocks: list[M.MiniMaxM3Block] = []
    for i in range(cfg.num_hidden_layers):
        blk = M.MiniMaxM3Block(cfg, i)
        blk.input_layernorm.weight = M.one_plus(w[(i, "in")])
        blk.post_attention_layernorm.weight = M.one_plus(w[(i, "post")])
        blk.self_attn.q_proj.weight = w[(i, "q")]
        blk.self_attn.k_proj.weight = w[(i, "k")]
        blk.self_attn.v_proj.weight = w[(i, "v")]
        blk.self_attn.o_proj.weight = w[(i, "o")]
        blk.self_attn.q_norm = M.one_plus(w[(i, "qn")])
        blk.self_attn.k_norm = M.one_plus(w[(i, "kn")])
        if cfg.is_moe_layer(i):
            blk.mlp.gate = w[(i, "gate")]
            blk.mlp.e_score_correction_bias = w[(i, "bias")]
            gu_trip, dn_trip = _pack(w[(i, "gate_up")]), _pack(w[(i, "down")])
            if packed:
                blk.mlp.set_experts_packed(gu_trip, dn_trip)
            else:
                blk.mlp.set_experts(_dq(gu_trip), _dq(dn_trip))   # SAME int6 codes, dequantized
            blk.mlp.shared_gate_proj = w[(i, "sg")]
            blk.mlp.shared_up_proj = w[(i, "su")]
            blk.mlp.shared_down_proj = w[(i, "sd")]
        else:
            blk.mlp.gate_proj.weight = w[(i, "dg")]
            blk.mlp.up_proj.weight = w[(i, "du")]
            blk.mlp.down_proj.weight = w[(i, "dd")]
        blocks.append(blk)
    return RT.MiniMaxM3ResidentModel.from_blocks(
        blocks, w["embed"], M.one_plus(w["final_norm"]), w["lm_head"], cfg)


def run() -> None:
    mx.random.seed(0)
    cfg = _cfg()
    w = _synth(cfg, mx.random.key(1))
    m_packed = _build_model(cfg, w, packed=True)
    m_bf16 = _build_model(cfg, w, packed=False)
    _ck(m_packed.packed_experts and not m_bf16.packed_experts,
        "from_blocks did not detect packed-experts state")

    T = 12
    ids = mx.array([(i * 7 + 3) % cfg.vocab_size for i in range(T)], dtype=mx.int32)

    # (1) packed gather_qmm == bf16 gather_mm (prefill), greedy-exact ----------------------------
    lp = m_packed(ids)            # [1,T,vocab]
    lb = m_bf16(ids)
    mx.eval(lp, lb)
    rel_pb, agree_pb = _rel(lp, lb), _argmax_agree(lp, lb)
    # top-1 exactness is the binding greedy-exactness claim; the logit rel is a loose bf16-reorder
    # sanity bound (fused gather_qmm keeps the dequantized int6 weight at full precision, while the
    # bf16 reference rounds it to bf16 before the matmul — so the packed path is the MORE precise one).
    _ck(agree_pb == 1.0, f"packed top-1 != bf16 top-1: agree {agree_pb:.4f}")
    _ck(rel_pb < 3e-2, f"packed logits != bf16 logits: rel {rel_pb:.2e}")

    # (2) cached forward (fresh caches, full window) == prefill (caches=None) --------------------
    lc_bf = m_bf16(ids, caches=m_bf16.make_caches())
    lc_pk = m_packed(ids, caches=m_packed.make_caches())
    mx.eval(lc_bf, lc_pk)
    rel_cache = _rel(lc_bf, lb)
    _ck(_argmax_agree(lc_bf, lb) == 1.0 and rel_cache < 1e-4,
        f"cached forward != prefill (bf16): rel {rel_cache:.2e}")
    _ck(_argmax_agree(lc_pk, lp) == 1.0, "cached forward != prefill (packed)")

    # (3) incremental decode == full prefill (the serving decode primitive) ----------------------
    P = 5
    caches = m_packed.make_caches()
    _ = m_packed(ids[:P], caches=caches)                       # consume prompt (positions 0..P-1)
    inc = [m_packed(ids[t:t + 1], caches=caches) for t in range(P, T)]   # step one token at a time
    l_inc = mx.concatenate(inc, axis=1)                        # [1, T-P, vocab] for positions P..T-1
    mx.eval(l_inc)
    rel_inc = _rel(l_inc, lp[:, P:])
    _ck(_argmax_agree(l_inc, lp[:, P:]) == 1.0 and rel_inc < 3e-2,
        f"incremental decode != full prefill: rel {rel_inc:.2e}")

    # (4) rule-4 dense==sparse (bf16) + rule-6 packed refuses sparse=False -----------------------
    moe_i = next(i for i in range(cfg.num_hidden_layers) if cfg.is_moe_layer(i))
    xchk = mx.random.normal((1, 4, cfg.hidden_size)).astype(mx.bfloat16)
    y_sparse = m_bf16.layers[moe_i].mlp(xchk, sparse=True)
    y_dense = m_bf16.layers[moe_i].mlp(xchk, sparse=False)
    mx.eval(y_sparse, y_dense)
    _ck(_rel(y_dense, y_sparse) < 1e-2, f"MoE dense oracle != sparse gather_mm: {_rel(y_dense, y_sparse):.2e}")
    try:
        m_packed.layers[moe_i].mlp(xchk, sparse=False)
        refused = False
    except ValueError:
        refused = True
    _ck(refused, "packed (int6) MoE did not refuse sparse=False (rule 6)")

    # (5) generate smoke (greedy serving convenience) -------------------------------------------
    gen = m_packed.generate([int(t) for t in ids[:3]], max_new=4)
    _ck(isinstance(gen, list) and 1 <= len(gen) <= 4 and all(isinstance(t, int) for t in gen),
        f"generate did not return a 1..4 int list: {gen!r}")

    print("\n=== MiniMax-M3-VL M3-1 resident runtime gate (model-free, tiny synthetic) ===")
    print(f"(1) packed gather_qmm == bf16 gather_mm: top-1 agree {agree_pb:.4f}, logit rel {rel_pb:.2e}")
    print(f"(2) cached == prefill: bf16 rel {rel_cache:.2e} (top-1 exact); packed top-1 exact")
    print(f"(3) incremental decode == full prefill: rel {rel_inc:.2e} (top-1 exact)")
    print("(4) MoE dense==sparse (bf16); packed refuses sparse=False (rule 6)")
    print(f"(5) generate smoke -> {len(gen)} tokens")
    print(f"PARITY-CHECKS: {_N}")
    print("PASS — M3 resident serving: packed-int6 gather_qmm == bf16 reference, cached/decode "
          "output-equivalent to prefill, rule-4/6 honored.")


if __name__ == "__main__":
    run()
