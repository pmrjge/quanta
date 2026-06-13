"""Model-free M1 layer-parity gate for the MiniMax-M3-VL text decoder block.

Cross-framework parity of :mod:`quanta.minimax.model_m3` (MLX) against an INDEPENDENT ``torch``
reference, with the risky M3 formulas pinned to AUTHORITATIVE ``transformers`` code in **isolated**
single-call sub-checks (the transformers ``minimax_m2`` / ``gpt_oss`` modules carry kernelized global
state that makes them non-reproducible when COMPOSED in sequence — so they are touched once each, in
isolation, never inside the full reference forward):

* **clamped SwiGLU-OAI** (:func:`model_m3.swigluoai`) vs the REAL
  ``transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts._apply_gate`` (unbound, fed the
  interleaved gate_up it expects). M3's ``swiglu_alpha=1.702``/``swiglu_limit=7.0`` ARE gpt-oss's.
* **sigmoid-noaux routing** (:func:`model_m3.route_noaux`) vs the REAL
  ``transformers.models.minimax_m2.MiniMaxM2TopKRouter`` (sigmoid; bias for SELECTION only; weights
  gathered from the pure sigmoid; renorm) — then the M3 ``* routed_scaling_factor`` (M2 has none).
* **partial rotate-half RoPE** (:func:`model_m3.rope_explicit`) vs the REAL
  ``minimax_m2.apply_rotary_pos_emb``.

The full decoder block (attention + dense FFN + MoE + shared expert, two-residual pre-norm,
Gemma ``(1+w)`` per-head QK-norm) is then checked MLX-vs-a-PURE-NUMPY-fp64 reference (no torch /
transformers in the block path ⇒ a clean framework-independent oracle; a torch reference proved
fragile — the transformers minimax_m2/gpt_oss kernels carry global state and the heavy mixed mx+torch
CPU workload perturbed plain-torch reductions structurally) on IDENTICAL synthetic weights + input,
for both a dense layer and a MoE layer. Plus the rule-4 internal MLX equivalences: MoE dense oracle
== sparse ``gather_mm``, and ``use_fast`` (``mx.fast`` rope+SDPA) == naive (explicit rope + manual
softmax) attention. All on tiny SYNTHETIC dims; runs in the model-free sweep (needs the ``reference``
extra for the torch/transformers formula pins).

    uv run --extra reference python -m parity.minimax_m3_layer_test
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from quanta.minimax import model_m3 as M
from quanta.minimax.config_m3 import MiniMaxM3Config

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


# --- synthetic config (tiny but structurally faithful) ----------------------- #

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


# --- helpers ----------------------------------------------------------------- #

def _rng(seed: int):
    return np.random.default_rng(seed)


def _w(rng, *shape, scale=0.05):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.tensor(a)   # COPY — never from_numpy (shared buffers get corrupted in-place)


def _mx(a: np.ndarray) -> mx.array:
    return mx.array(a)


def _rel(a: mx.array, b: torch.Tensor) -> float:
    an = np.array(a.astype(mx.float32))
    bn = b.detach().float().numpy()
    return float(np.abs(an - bn).max() / (np.abs(bn).max() + 1e-9))


def _rel_mx(a: mx.array, b: mx.array) -> float:
    an, bn = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    return float(np.abs(an - bn).max() / (np.abs(bn).max() + 1e-9))


def _rel_np(a: mx.array, b: np.ndarray) -> float:
    an = np.array(a.astype(mx.float32)).astype(np.float64)
    return float(np.abs(an - b).max() / (np.abs(b).max() + 1e-9))


# --- isolated transformers pins (touched once each, never composed) ---------- #

def _gptoss_apply_gate(gate: np.ndarray, up: np.ndarray, alpha: float, limit: float) -> torch.Tensor:
    """Call the REAL ``GptOssExperts._apply_gate`` on a shim, feeding the interleaved gate_up it
    expects (gate=[::2], up=[1::2])."""
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

    class _Shim:
        pass
    shim = _Shim()
    shim.alpha, shim.limit = alpha, limit
    g, u = _t(gate), _t(up)
    gate_up = torch.empty(*g.shape[:-1], 2 * g.shape[-1], dtype=g.dtype)
    gate_up[..., ::2] = g
    gate_up[..., 1::2] = u
    return GptOssExperts._apply_gate(shim, gate_up)


def _mm2_router(x: np.ndarray, gate: np.ndarray, bias: np.ndarray, cfg):
    """REAL ``MiniMaxM2TopKRouter`` (sigmoid noaux, renorm); returns (idx, renormed_weights)."""
    from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2TopKRouter

    class _C:
        num_experts_per_tok = cfg.num_experts_per_tok
        num_local_experts = cfg.num_local_experts
        hidden_size = cfg.hidden_size
    router = MiniMaxM2TopKRouter(_C())
    with torch.no_grad():
        router.weight.copy_(_t(gate))
    _, scores, idx = router(_t(x), _t(bias))
    return idx, scores


def _mm2_rope(q: np.ndarray, k: np.ndarray, cfg):
    """REAL ``minimax_m2.apply_rotary_pos_emb`` partial rotate-half RoPE. q ``[B,H,T,D]``."""
    from transformers.models.minimax_m2.modeling_minimax_m2 import apply_rotary_pos_emb

    rd, t = cfg.rotary_dim, q.shape[2]
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
    emb = torch.cat([torch.arange(t).float()[:, None] * inv[None, :]] * 2, dim=-1)
    cos, sin = emb.cos()[None], emb.sin()[None]
    return apply_rotary_pos_emb(_t(q), _t(k), cos, sin)


# --- pure-numpy float64 reference block (deterministic; no torch/transformers) #
# (a torch reference proved fragile here: the transformers minimax_m2/gpt_oss kernels carry global
#  state and the heavy mixed mx+torch CPU workload perturbed plain-torch reductions structurally —
#  numpy fp64 is a clean, framework-independent oracle. The risky FORMULAS are still pinned to the
#  authoritative transformers code in the isolated single-call checks above.)

def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _np_swigluoai(gate, up, alpha, limit):
    g = np.minimum(gate.astype(np.float64), limit)
    u = np.clip(up.astype(np.float64), -limit, limit)
    return (u + 1.0) * (g * _np_sigmoid(alpha * g))


def _np_gemma_rms(x, w_raw, eps):
    xf = x.astype(np.float64)
    xf = xf * (1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps))
    return (1.0 + w_raw.astype(np.float64)) * xf


def _np_rope(x, rd, theta):
    """rotate-half RoPE on the first ``rd`` dims. x ``[B,H,T,D]`` fp64."""
    t = x.shape[2]
    inv = 1.0 / (theta ** (np.arange(0, rd, 2, dtype=np.float64) / rd))
    ang = np.arange(t, dtype=np.float64)[:, None] * inv[None, :]   # [t, rd/2]
    cos, sin = np.cos(ang)[None, None], np.sin(ang)[None, None]
    xr = x[..., :rd]
    x1, x2 = xr[..., : rd // 2], xr[..., rd // 2:]
    rot = np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return np.concatenate([rot, x[..., rd:]], axis=-1)


def _np_attn(x, Wq, Wk, Wv, Wo, qn, kn, cfg):
    b, t, _ = x.shape
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    xd = x.astype(np.float64)
    q = (xd @ Wq.T).reshape(b, t, nh, hd)
    k = (xd @ Wk.T).reshape(b, t, nkv, hd)
    v = (xd @ Wv.T).reshape(b, t, nkv, hd)
    q = np.transpose(_np_gemma_rms(q, qn, cfg.norm_eps), (0, 2, 1, 3))
    k = np.transpose(_np_gemma_rms(k, kn, cfg.norm_eps), (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))
    q = _np_rope(q, cfg.rotary_dim, cfg.rope_theta)
    k = _np_rope(k, cfg.rotary_dim, cfg.rope_theta)
    rep = nh // nkv
    kr = np.repeat(k, rep, axis=1)
    vr = np.repeat(v, rep, axis=1)
    scores = (q @ np.swapaxes(kr, -1, -2)) * cfg.attn_scale
    mask = np.triu(np.full((t, t), -np.inf), 1)
    scores = scores + mask
    scores = scores - scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p = p / p.sum(axis=-1, keepdims=True)
    out = np.transpose(p @ vr, (0, 2, 1, 3)).reshape(b, t, nh * hd)
    return out @ Wo.T


def _np_dense_mlp(x, Wg, Wu, Wd, cfg):
    xd = x.astype(np.float64)
    return _np_swigluoai(xd @ Wg.T, xd @ Wu.T, cfg.swiglu_alpha, cfg.swiglu_limit) @ Wd.T


def _np_route(x, gate, bias, cfg):
    """sigmoid-noaux router (== minimax_m2, pinned in the isolated check) + M3 scaling. fp64."""
    scores = _np_sigmoid(x.astype(np.float64) @ gate.astype(np.float64).T)
    choice = scores + bias.astype(np.float64)[None]
    k = cfg.num_experts_per_tok
    idx = np.argsort(-choice, axis=-1)[:, :k]
    w = np.take_along_axis(scores, idx, axis=-1)
    if cfg.norm_topk_prob:
        w = w / (w.sum(-1, keepdims=True) + 1e-20)
    return idx, w * cfg.routed_scaling_factor


def _np_moe(x, gate, bias, w1, w2, w3, sg, su, sd, cfg):
    b, t, h = x.shape
    xf = x.reshape(-1, h).astype(np.float64)
    idx, w = _np_route(xf, gate, bias, cfg)
    n = xf.shape[0]
    out = np.zeros((n, h), np.float64)
    for tok in range(n):
        for s in range(idx.shape[1]):
            ex = int(idx[tok, s])
            hgu = _np_swigluoai(xf[tok] @ w1[ex].T, xf[tok] @ w3[ex].T,
                                cfg.swiglu_alpha, cfg.swiglu_limit)
            out[tok] += float(w[tok, s]) * (hgu @ w2[ex].T)
    shared = _np_swigluoai(xf @ sg.T, xf @ su.T, cfg.swiglu_alpha, cfg.swiglu_limit) @ sd.T
    return (out + shared).reshape(b, t, h)


def run() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    rng = _rng(0)
    B, T = 1, 6
    h = cfg.hidden_size
    x_np = _w(rng, B, T, h, scale=1.0)
    x_mx = _mx(x_np)

    # ===== isolated transformers pins (each touched once, never composed) ===== #

    # 1. swigluoai (mx) vs the real gpt_oss _apply_gate
    g_np, u_np = _w(rng, 4, 32, scale=3.0), _w(rng, 4, 32, scale=3.0)   # scale-3 ⇒ exercises clamp
    mine = M.swigluoai(_mx(g_np), _mx(u_np), cfg.swiglu_alpha, cfg.swiglu_limit)
    _ck(_rel(mine, _gptoss_apply_gate(g_np, u_np, cfg.swiglu_alpha, cfg.swiglu_limit)) < 1e-5,
        "swigluoai != gpt_oss _apply_gate")
    _ck(bool((np.abs(g_np) > cfg.swiglu_limit).any()), "test inputs must exercise the clamp")

    # 2. route_noaux (mx) vs the real MiniMaxM2TopKRouter (+ M3 routed_scaling_factor)
    e = cfg.num_local_experts
    gate_np, bias_np = _w(rng, e, h, scale=0.3), _w(rng, e, scale=0.5)
    xf_np = x_np.reshape(-1, h)
    idx_mx, w_mx = M.route_noaux(_mx(xf_np), _mx(gate_np), _mx(bias_np), cfg)
    idx_t, w_t = _mm2_router(xf_np, gate_np, bias_np, cfg)
    im, wm = np.array(idx_mx), np.array(w_mx.astype(mx.float32))
    it, wt = idx_t.numpy(), (w_t.numpy() * cfg.routed_scaling_factor)

    def _by_expert(idx, wts):
        m = np.zeros((idx.shape[0], e), np.float32)
        for r in range(idx.shape[0]):
            for s in range(idx.shape[1]):
                m[r, int(idx[r, s])] = wts[r, s]
        return m
    _ck(all(set(im[r]) == set(it[r]) for r in range(im.shape[0])),
        f"route_noaux selection != minimax_m2 router\n mine={im}\n ref={it}")
    _ck(np.abs(_by_expert(im, wm) - _by_expert(it, wt)).max() < 1e-5,
        "route_noaux weights != minimax_m2 router (+ scaling)")

    # 3. partial rotate-half RoPE (mx) vs the real minimax_m2 apply_rotary_pos_emb
    nh, nkv, hd, rd = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rotary_dim)
    q_np, k_np = _w(rng, 1, nh, T, hd, scale=1.0), _w(rng, 1, nkv, T, hd, scale=1.0)
    inv = M.inv_freq(rd, cfg.rope_theta)
    q_rope_mx = M.rope_explicit(_mx(q_np), inv, rd, 0)
    qe_t, ke_t = _mm2_rope(q_np, k_np, cfg)
    _ck(_rel(q_rope_mx, qe_t) < 1e-5, "partial RoPE (mx) != minimax_m2 apply_rotary_pos_emb")

    # ===== full block vs a PURE-torch reference (deterministic) ===== #

    Wq, Wk = _w(rng, nh * hd, h), _w(rng, nkv * hd, h)
    Wv, Wo = _w(rng, nkv * hd, h), _w(rng, h, nh * hd)
    qn, kn = _w(rng, hd, scale=0.2), _w(rng, hd, scale=0.2)

    # 4. attention: mx (naive + fast) vs pure-torch ref
    attn = M.MiniMaxM3Attention(cfg)
    attn.q_proj.weight, attn.k_proj.weight = _mx(Wq), _mx(Wk)
    attn.v_proj.weight, attn.o_proj.weight = _mx(Wv), _mx(Wo)
    attn.q_norm, attn.k_norm = M.one_plus(_mx(qn)), M.one_plus(_mx(kn))
    a_ref = _np_attn(x_np, Wq, Wk, Wv, Wo, qn, kn, cfg)
    _ck(_rel_np(attn(x_mx, use_fast=False), a_ref) < 2e-4, "attention(naive) != numpy ref")
    _ck(_rel_np(attn(x_mx, use_fast=True), a_ref) < 3e-3, "attention(fast) != numpy ref")

    # 5. dense FFN vs numpy ref
    di = cfg.dense_intermediate_size
    Wg, Wu, Wd = _w(rng, di, h, scale=0.1), _w(rng, di, h, scale=0.1), _w(rng, h, di, scale=0.1)
    mlp = M.MiniMaxM3DenseMLP(cfg)
    mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight = _mx(Wg), _mx(Wu), _mx(Wd)
    _ck(_rel_np(mlp(x_mx), _np_dense_mlp(x_np, Wg, Wu, Wd, cfg)) < 2e-4, "dense FFN != numpy ref")

    # 6. MoE: numpy ref, AND the MLX dense oracle == sparse gather_mm
    inter, si = cfg.moe_intermediate_size, cfg.shared_intermediate_size
    w1, w3 = _w(rng, e, inter, h, scale=0.1), _w(rng, e, inter, h, scale=0.1)
    w2 = _w(rng, e, h, inter, scale=0.1)
    sg, su, sd = _w(rng, si, h, scale=0.1), _w(rng, si, h, scale=0.1), _w(rng, h, si, scale=0.1)
    gate_up = np.concatenate([w1, w3], axis=1)                          # [E, 2*inter, h]
    moe = M.MiniMaxM3MoE(cfg)
    moe.gate, moe.e_score_correction_bias = _mx(gate_np), _mx(bias_np)
    moe.set_experts(_mx(gate_up), _mx(w2))
    moe.shared_gate_proj, moe.shared_up_proj, moe.shared_down_proj = _mx(sg), _mx(su), _mx(sd)
    m_sparse, m_dense = moe(x_mx, sparse=True), moe(x_mx, sparse=False)
    m_ref = _np_moe(x_np, gate_np, bias_np, w1, w2, w3, sg, su, sd, cfg)
    _ck(_rel_mx(m_dense, m_sparse) < 1e-5, "MoE dense oracle != sparse gather_mm")
    _ck(_rel_np(m_sparse, m_ref) < 3e-4, "MoE != numpy ref")

    # 7. full block (dense layer 0 + MoE layer 2) vs numpy ref
    def _block(layer_id, fill):
        in_raw, post_raw = _w(rng, h, scale=0.2), _w(rng, h, scale=0.2)
        blk = M.MiniMaxM3Block(cfg, layer_id)
        blk.input_layernorm.weight = M.one_plus(_mx(in_raw))
        blk.post_attention_layernorm.weight = M.one_plus(_mx(post_raw))
        blk.self_attn.q_proj.weight, blk.self_attn.k_proj.weight = _mx(Wq), _mx(Wk)
        blk.self_attn.v_proj.weight, blk.self_attn.o_proj.weight = _mx(Wv), _mx(Wo)
        blk.self_attn.q_norm, blk.self_attn.k_norm = M.one_plus(_mx(qn)), M.one_plus(_mx(kn))
        fill(blk)
        out_mx = blk(x_mx, use_fast=False)
        xr = x_np.astype(np.float64) + _np_attn(_np_gemma_rms(x_np, in_raw, cfg.norm_eps),
                                                Wq, Wk, Wv, Wo, qn, kn, cfg)
        post = _np_gemma_rms(xr, post_raw, cfg.norm_eps)
        if cfg.is_dense_layer(layer_id):   # config schedule, NOT a hardcoded 0-2 (synthetic cfg differs)
            y = _np_dense_mlp(post, Wg, Wu, Wd, cfg)
        else:
            y = _np_moe(post, gate_np, bias_np, w1, w2, w3, sg, su, sd, cfg)
        return out_mx, xr + y

    def _fill_dense(blk):
        blk.mlp.gate_proj.weight, blk.mlp.up_proj.weight, blk.mlp.down_proj.weight = \
            _mx(Wg), _mx(Wu), _mx(Wd)

    def _fill_moe(blk):
        blk.mlp.gate, blk.mlp.e_score_correction_bias = _mx(gate_np), _mx(bias_np)
        blk.mlp.set_experts(_mx(gate_up), _mx(w2))
        blk.mlp.shared_gate_proj, blk.mlp.shared_up_proj, blk.mlp.shared_down_proj = \
            _mx(sg), _mx(su), _mx(sd)

    o0_mx, o0_ref = _block(0, _fill_dense)
    _ck(cfg.is_dense_layer(0) and _rel_np(o0_mx, o0_ref) < 3e-4, "dense block (layer 0) != numpy ref")
    o2_mx, o2_ref = _block(2, _fill_moe)
    _ck(cfg.is_moe_layer(2) and _rel_np(o2_mx, o2_ref) < 3e-4, "MoE block (layer 2) != numpy ref")

    print(f"PARITY-CHECKS: {_N}")
    print(f"PASS — MiniMax-M3-VL M1 layer parity: swigluoai=gpt_oss, router=minimax_m2(+scale), "
          f"rope=minimax_m2; attn/FFN/MoE/block match the numpy fp64 reference ({_N} checks).")


if __name__ == "__main__":
    run()
