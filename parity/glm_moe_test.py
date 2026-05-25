"""Gate: GLM-5.1 noaux_tc sigmoid MoE — model-free, tiny random weights, ~0 GB.

Parity-first (#85): for a tiny random :class:`quanta.glm.moe.SparseMoE` (few experts / small dims),
diff the ``mlx.nn`` module against an **independent plain-numpy transcription** of the DeepSeek-V3
``noaux_tc`` MoE: ``scores = sigmoid(x @ gate.T)``; select top-k by ``scores + e_score_correction_bias``;
weight by the **bias-free** scores, normalized to sum 1, then ``* routed_scaling_factor``; each expert a
SwiGLU ``down(silu(gate(x))·up(x))``; plus one always-on **shared** expert. Then prove the module's two
paths agree (rule 4):

* the routing (selected expert *set* per token) **==** the reference's selection,
* the sparse ``gather_mm`` dispatch (:meth:`SparseMoE.__call__`) **==** the reference output,
* the sparse path **==** :meth:`SparseMoE.dense_reference` (run-every-expert oracle) — the
  ``sparse == dense`` gate that proves the gather dispatch never drops/duplicates a routed expert.

Formula-vs-real-checkpoint correctness is the deferred torch/ppl oracle; here every tensor is a few KB.

    uv run --with numpy python -m parity.glm_moe_test
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from quanta.glm.config import GLMConfig
from quanta.glm.moe import SparseMoE

CFG = {
    "model_type": "glm_moe_dsa", "vocab_size": 40, "hidden_size": 16, "intermediate_size": 24,
    "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 2,
    "q_lora_rank": 8, "kv_lora_rank": 6, "qk_nope_head_dim": 4, "qk_rope_head_dim": 4,
    "qk_head_dim": 8, "v_head_dim": 8,
    "index_head_dim": 6, "index_n_heads": 2, "index_topk": 3,
    "n_routed_experts": 6, "num_experts_per_tok": 2, "n_shared_experts": 1, "moe_intermediate_size": 5,
    "first_k_dense_replace": 1, "num_nextn_predict_layers": 1,
    "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
    "eos_token_id": [39], "pad_token_id": 39, "tie_word_embeddings": False,
}


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _np_moe(x, gate_w, bias, gs, us, ds, sg, su, sd, cfg: GLMConfig):
    """Independent numpy noaux_tc MoE reference (float64). Returns (y[N,dim], idx[N,topk])."""
    n, _ = x.shape
    topk = cfg.num_experts_per_tok
    logits = x @ gate_w.T
    scores = 1.0 / (1.0 + np.exp(-logits))                       # sigmoid
    choice = scores + bias[None]
    idx = np.argsort(-choice, axis=-1)[:, :topk]                 # top-k by scores+bias
    w = np.take_along_axis(scores, idx, axis=-1)                 # bias-free scores
    w = w / (w.sum(-1, keepdims=True) + 1e-20)                   # norm_topk_prob
    w = w * cfg.routed_scaling_factor
    y = np.zeros_like(x)
    for t in range(n):                                           # reference: explicit per-token loop
        for s in range(topk):
            e = idx[t, s]
            g = _silu(x[t] @ gs[e].T) * (x[t] @ us[e].T)
            y[t] += w[t, s] * (g @ ds[e].T)
    sh = _silu(x @ sg.T) * (x @ su.T) @ sd.T                     # shared expert (always on)
    return y + sh, idx


def run() -> None:
    cfg = GLMConfig.from_dict(CFG)
    rng = np.random.default_rng(0)
    e, inter, h = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size

    gs = (rng.standard_normal((e, inter, h)) * 0.3)
    us = (rng.standard_normal((e, inter, h)) * 0.3)
    ds = (rng.standard_normal((e, h, inter)) * 0.3)
    gate_w = (rng.standard_normal((e, h)) * 0.5)
    bias = (rng.standard_normal((e,)) * 0.2)
    sg = (rng.standard_normal((inter, h)) * 0.3)
    su = (rng.standard_normal((inter, h)) * 0.3)
    sd = (rng.standard_normal((h, inter)) * 0.3)

    moe = SparseMoE(cfg)
    moe.set_experts(mx.array(gs.astype(np.float32)), mx.array(us.astype(np.float32)),
                    mx.array(ds.astype(np.float32)))
    moe.gate.weight = mx.array(gate_w.astype(np.float32))
    moe.gate.e_score_correction_bias = mx.array(bias.astype(np.float32))
    moe.load_weights([
        ("shared_gate.weight", mx.array(sg.astype(np.float32))),
        ("shared_up.weight", mx.array(su.astype(np.float32))),
        ("shared_down.weight", mx.array(sd.astype(np.float32))),
    ], strict=False)

    N = 7
    x = (rng.standard_normal((1, N, h)) * 0.5).astype(np.float32)
    ok = True

    y_ref, idx_ref = _np_moe(x.reshape(N, h).astype(np.float64), gate_w, bias, gs, us, ds, sg, su, sd, cfg)

    idx_mx, _ = moe.gate(mx.array(x).reshape(N, h))
    idx_mx = np.array(idx_mx)
    set_ok = all(set(idx_mx[t]) == set(idx_ref[t]) for t in range(N))
    ok = ok and set_ok
    print(f"  [{'OK' if set_ok else 'FAIL'}] routing selected-expert set matches reference")

    y_sparse = np.array(moe(mx.array(x)).reshape(N, h).astype(mx.float32)).astype(np.float64)
    rel = float(np.max(np.abs(y_ref - y_sparse))) / float(np.max(np.abs(y_ref)))
    good = y_sparse.shape == (N, h) and rel < 1e-3
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] sparse gather_mm == numpy reference  rel={rel:.2e}")

    y_dense = np.array(moe.dense_reference(mx.array(x)).reshape(N, h).astype(mx.float32)).astype(np.float64)
    d = float(np.max(np.abs(y_sparse - y_dense)))
    good = d < 1e-4
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] sparse == dense_reference (run-every-expert)  |Δ|={d:.2e}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
