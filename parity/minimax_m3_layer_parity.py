"""MiniMax-M3-VL M1b: real-weight at-scale layer parity for the text decoder block.

Loads the **real** layer-0 (dense) and layer-3 (first MoE, which also carries the trained
block-sparse indexer) from the 809.5 GiB checkpoint, runs :mod:`quanta.minimax.model_m3`'s
:class:`~quanta.minimax.model_m3.MiniMaxM3Block` in fp32, and diffs it against a self-contained
**numpy float64** reference computed on the SAME dequantized weights + input. This validates the
loader (:class:`quanta.minimax.loader_m3.MiniMaxM3SourceCheckpoint`) and the real-shape/dtype wiring
at 397B-class scale (hidden 6144, GQA 64q/4kv head_dim 128, dense_inter 12288, 128 experts top-4 + 1
shared) — the formulas themselves were already pinned to the authoritative ``transformers`` siblings
in the model-free M1a gate (``parity/minimax_m3_layer_test.py``), so the oracle here is plain numpy
fp64 (deterministic, framework-independent — no torch dependency).

Checks per layer (rel = max|Δ| / (max|ref| + eps)):

* **dense block L0** — full block (``x + attn(in_norm(x))`` then ``x + dense_mlp(post_norm(x))``,
  Gemma ``(1+w)`` norms, per-head q/k norm, partial RoPE, clamped SwiGLU) vs numpy fp64; plus
  ``use_fast`` (mx.fast rope+SDPA) == naive at the block level (rule 4).
* **MoE block L3** — full block (attention + noaux router + 128 routed clamped-SwiGLU experts + the
  un-gated shared expert) vs numpy fp64; plus the rule-4 internal equivalences: sparse
  ``mx.gather_mm`` dispatch == the dense oracle, ``use_fast`` == naive, and the MLX router selection
  + weights == the numpy router — all on the REAL gate/bias/experts. The trained indexer tensors are
  streamed and shape-checked (rule-6 coverage) but inert at ``T=6`` (top-16 blocks == all blocks ⇒
  sparse == dense at short context; the indexer is the M3 long-context serving lever).

Real-weight / SOLO path (reads ``/Users/pmrj/models/MiniMax-M3``) ⇒ EXCLUDED from the model-free
sweep, run by hand. Layer-streamed (rule 8): L0 lives wholly in shard 1, L3's text backbone (incl.
all 384 expert tensors, no expert split across shards) wholly in shard 3, so one shard is mmapped at
a time and released between layers. Peak residency ≈ one MoE block's fp32 expert stacks (~29 GiB)
<< 490 GiB.

    uv run python -m parity.minimax_m3_layer_parity
"""

from __future__ import annotations

import warnings

import mlx.core as mx
import numpy as np

from quanta.minimax import model_m3 as M
from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.loader_m3 import MiniMaxM3SourceCheckpoint

MINIMAX = "/Users/pmrj/models/MiniMax-M3"
T = 6   # short ctx: sparse == dense (indexer inert); small enough for the bounded numpy fp64 oracle

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


# --- conversions / metrics --------------------------------------------------- #

def _f32(a: mx.array) -> mx.array:
    return a.astype(mx.float32)


def _npf(a: mx.array) -> np.ndarray:
    """mx (any dtype) → numpy float64 (the oracle precision)."""
    return np.asarray(a.astype(mx.float32)).astype(np.float64)


def _rel_np(a: mx.array, b: np.ndarray) -> float:
    an = np.asarray(a.astype(mx.float32)).astype(np.float64)
    return float(np.abs(an - b).max() / (np.abs(b).max() + 1e-9))


def _rel_mx(a: mx.array, b: mx.array) -> float:
    an, bn = np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32))
    return float(np.abs(an - bn).max() / (np.abs(bn).max() + 1e-9))


def _clear() -> None:
    try:
        mx.clear_cache()
    except AttributeError:
        pass


# --- numpy float64 oracle (formulas pinned to transformers in the M1a model-free gate) -------- #

def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _np_swigluoai(gate, up, alpha, limit):
    g = np.minimum(gate.astype(np.float64), limit)          # clamp(max=limit), no lower bound
    u = np.clip(up.astype(np.float64), -limit, limit)
    return (u + 1.0) * (g * _np_sigmoid(alpha * g))


def _np_gemma_rms(x, w_raw, eps):
    xf = x.astype(np.float64)
    xf = xf * (1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps))
    return (1.0 + w_raw.astype(np.float64)) * xf           # Gemma (1+w)


def _np_rope(x, rd, theta):
    """rotate-half RoPE on the first ``rd`` dims. x ``[B,H,T,D]`` fp64."""
    t = x.shape[2]
    inv = 1.0 / (theta ** (np.arange(0, rd, 2, dtype=np.float64) / rd))
    ang = np.arange(t, dtype=np.float64)[:, None] * inv[None, :]
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
    q = np.transpose(_np_gemma_rms(q, qn, cfg.norm_eps), (0, 2, 1, 3))   # per-head q/k norm BEFORE RoPE
    k = np.transpose(_np_gemma_rms(k, kn, cfg.norm_eps), (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))
    q = _np_rope(q, cfg.rotary_dim, cfg.rope_theta)
    k = _np_rope(k, cfg.rotary_dim, cfg.rope_theta)
    rep = nh // nkv
    kr, vr = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
    scores = (q @ np.swapaxes(kr, -1, -2)) * cfg.attn_scale
    scores = scores + np.triu(np.full((t, t), -np.inf), 1)
    scores = scores - scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p = p / p.sum(axis=-1, keepdims=True)
    out = np.transpose(p @ vr, (0, 2, 1, 3)).reshape(b, t, nh * hd)
    return out @ Wo.T


def _np_dense_mlp(x, Wg, Wu, Wd, cfg):
    xd = x.astype(np.float64)
    return _np_swigluoai(xd @ Wg.T, xd @ Wu.T, cfg.swiglu_alpha, cfg.swiglu_limit) @ Wd.T


def _np_route(x, gate, bias, cfg):
    """sigmoid-noaux router (== minimax_m2, pinned in M1a) + M3 routed_scaling_factor. fp64."""
    scores = _np_sigmoid(x.astype(np.float64) @ gate.astype(np.float64).T)
    choice = scores + bias.astype(np.float64)[None]
    k = cfg.num_experts_per_tok
    idx = np.argsort(-choice, axis=-1)[:, :k]
    w = np.take_along_axis(scores, idx, axis=-1)
    if cfg.norm_topk_prob:
        w = w / (w.sum(-1, keepdims=True) + 1e-20)
    return idx, w * cfg.routed_scaling_factor


def _np_moe_real(xf, gate, bias, eg_up, e_down, sg, su, sd, cfg):
    """MoE forward on REAL pre-stacked experts, memory-bounded: only the selected experts are
    upcast to fp64 (transiently, one at a time) — never all 128. ``xf`` ``[N,h]`` fp64; ``eg_up``
    ``[E,2*inter,h]`` / ``e_down`` ``[E,h,inter]`` are the fp32 mx stacks the block uses."""
    n, h = xf.shape
    inter = cfg.moe_intermediate_size
    idx, w = _np_route(xf, gate, bias, cfg)
    out = np.zeros((n, h), np.float64)
    for tok in range(n):
        for s in range(idx.shape[1]):
            ex = int(idx[tok, s])
            gu = _npf(eg_up[ex])                              # [2*inter, h] fp64 (selected only)
            w2 = _npf(e_down[ex])                             # [h, inter] fp64
            g = xf[tok] @ gu[:inter].T
            u = xf[tok] @ gu[inter:].T
            hgu = _np_swigluoai(g, u, cfg.swiglu_alpha, cfg.swiglu_limit)
            out[tok] += float(w[tok, s]) * (hgu @ w2.T)
    shared = _np_swigluoai(xf @ sg.T, xf @ su.T, cfg.swiglu_alpha, cfg.swiglu_limit) @ sd.T
    return out + shared


# --- per-layer parity -------------------------------------------------------- #

def _fill_attn(blk, at):
    """Load the GQA projections + folded per-head q/k norm into the block's attention (fp32)."""
    blk.self_attn.q_proj.weight = _f32(at["q_proj.weight"])
    blk.self_attn.k_proj.weight = _f32(at["k_proj.weight"])
    blk.self_attn.v_proj.weight = _f32(at["v_proj.weight"])
    blk.self_attn.o_proj.weight = _f32(at["o_proj.weight"])
    blk.self_attn.q_norm = M.one_plus(_f32(at["q_norm.weight"]))            # Gemma (1+w)
    blk.self_attn.k_norm = M.one_plus(_f32(at["k_norm.weight"]))


def _np_pre_attn(x_np, in_raw, at, cfg):
    """numpy fp64: residual after the attention sub-block (x + attn(in_norm(x)))."""
    return x_np + _np_attn(_np_gemma_rms(x_np, in_raw, cfg.norm_eps),
                           _npf(at["q_proj.weight"]), _npf(at["k_proj.weight"]),
                           _npf(at["v_proj.weight"]), _npf(at["o_proj.weight"]),
                           _npf(at["q_norm.weight"]), _npf(at["k_norm.weight"]), cfg)


def _dense_parity(cfg: MiniMaxM3Config, ck: MiniMaxM3SourceCheckpoint, L: int) -> dict:
    at, dm, nm = ck.attention(L), ck.dense_mlp(L), ck.block_norms(L)
    in_raw, post_raw = nm["input_layernorm"], nm["post_attention_layernorm"]

    blk = M.MiniMaxM3Block(cfg, L)
    blk.input_layernorm.weight = M.one_plus(_f32(in_raw))
    blk.post_attention_layernorm.weight = M.one_plus(_f32(post_raw))
    _fill_attn(blk, at)
    blk.mlp.gate_proj.weight = _f32(dm["gate_proj"])
    blk.mlp.up_proj.weight = _f32(dm["up_proj"])
    blk.mlp.down_proj.weight = _f32(dm["down_proj"])

    x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.float32)
    y_naive, y_fast = blk(x, use_fast=False), blk(x, use_fast=True)
    mx.eval(y_naive, y_fast)

    x_np = _npf(x)
    xr = _np_pre_attn(x_np, _npf(in_raw), at, cfg)
    post = _np_gemma_rms(xr, _npf(post_raw), cfg.norm_eps)
    ref = xr + _np_dense_mlp(post, _npf(dm["gate_proj"]), _npf(dm["up_proj"]), _npf(dm["down_proj"]), cfg)

    res = {"par": _rel_np(y_naive, ref), "fast_naive": _rel_mx(y_fast, y_naive)}
    del blk, x, y_naive, y_fast
    ck.release()
    _clear()
    return res


def _moe_parity(cfg: MiniMaxM3Config, ck: MiniMaxM3SourceCheckpoint, L: int) -> dict:
    at, mo, nm = ck.attention(L), ck.moe(L), ck.block_norms(L)
    ix = ck.sparse_index(L)
    in_raw, post_raw = nm["input_layernorm"], nm["post_attention_layernorm"]

    # rule-6 coverage: the trained indexer streams with the expected shapes (inert at short ctx).
    iq, ik = ix["index_q_proj.weight"], ix["index_k_proj.weight"]
    iqn, ikn = ix["index_q_norm.weight"], ix["index_k_norm.weight"]
    index_ok = (tuple(iq.shape) == (cfg.sparse_num_index_heads * cfg.sparse_index_dim, cfg.hidden_size)
                and tuple(ik.shape) == (cfg.sparse_index_dim, cfg.hidden_size)
                and tuple(iqn.shape) == (cfg.sparse_index_dim,)
                and tuple(ikn.shape) == (cfg.sparse_index_dim,))

    eg_up, e_down = _f32(mo["experts_gate_up"]), _f32(mo["experts_down"])
    gate, bias = _f32(mo["gate"]), _f32(mo["e_score_correction_bias"])
    sg, su, sd = _f32(mo["shared_gate_proj"]), _f32(mo["shared_up_proj"]), _f32(mo["shared_down_proj"])

    blk = M.MiniMaxM3Block(cfg, L)
    blk.input_layernorm.weight = M.one_plus(_f32(in_raw))
    blk.post_attention_layernorm.weight = M.one_plus(_f32(post_raw))
    _fill_attn(blk, at)
    blk.mlp.gate, blk.mlp.e_score_correction_bias = gate, bias
    blk.mlp.set_experts(eg_up, e_down)
    blk.mlp.shared_gate_proj, blk.mlp.shared_up_proj, blk.mlp.shared_down_proj = sg, su, sd

    x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.float32)
    y_naive = blk(x, use_fast=False, sparse=True)
    y_fast = blk(x, use_fast=True, sparse=True)
    y_dense = blk(x, use_fast=False, sparse=False)     # dense MoE oracle == sparse gather_mm
    mx.eval(y_naive, y_fast, y_dense)

    # router parity (real gate/bias): MLX route_noaux selection + weights == numpy oracle
    x_np = _npf(x)
    xf = x_np.reshape(-1, cfg.hidden_size)
    idx_mx, w_mx = M.route_noaux(mx.array(xf.astype(np.float32)), gate, bias, cfg)
    im, wm = np.asarray(idx_mx), np.asarray(w_mx.astype(mx.float32))
    it, wt = _np_route(xf, _npf(gate), _npf(bias), cfg)
    n = xf.shape[0]
    set_ok = all(set(im[r].tolist()) == set(it[r].tolist()) for r in range(n))

    def _by_expert(idx, wts):
        m = np.zeros((n, cfg.num_local_experts), np.float64)
        for r in range(n):
            for s in range(idx.shape[1]):
                m[r, int(idx[r, s])] = wts[r, s]
        return m
    router_wdiff = float(np.abs(_by_expert(im, wm) - _by_expert(it, wt)).max())

    # full-block numpy fp64 reference
    xr = _np_pre_attn(x_np, _npf(in_raw), at, cfg)
    post = _np_gemma_rms(xr, _npf(post_raw), cfg.norm_eps)
    ref = xr + _np_moe_real(post.reshape(-1, cfg.hidden_size), _npf(gate), _npf(bias),
                            eg_up, e_down, _npf(sg), _npf(su), _npf(sd), cfg).reshape(post.shape)

    res = {"par": _rel_np(y_naive, ref), "fast_naive": _rel_mx(y_fast, y_naive),
           "sparse_dense": _rel_mx(y_dense, y_naive), "router_set": set_ok,
           "router_wdiff": router_wdiff, "index_ok": index_ok}
    del blk, eg_up, e_down, x, y_naive, y_fast, y_dense
    ck.release()
    _clear()
    return res


def run() -> None:
    warnings.filterwarnings("ignore")
    mx.random.seed(0)
    cfg = MiniMaxM3Config.from_pretrained(MINIMAX)
    ck = MiniMaxM3SourceCheckpoint(MINIMAX, cfg)

    i_dense = next(i for i in range(cfg.num_hidden_layers) if cfg.is_dense_layer(i))
    i_moe = next(i for i in range(cfg.num_hidden_layers) if cfg.is_moe_layer(i))

    d = _dense_parity(cfg, ck, i_dense)
    e = _moe_parity(cfg, ck, i_moe)

    # tolerances: fp32-mx-vs-fp64-numpy over hidden 6144 is machine-precision here — measured ~8e-7
    # block, ~1e-7 fast-vs-naive, ~4e-7 sparse-vs-dense, ~6e-8 router weights. 1e-4 keeps a ~100x
    # margin while a real O(1) wiring bug (wrong norm fold, swapped gate/up, mis-stacked experts,
    # F32 router read as bf16) lands at 1e-2+; fast==naive (mx.fast SDPA/rope vs manual fp32) and
    # sparse==dense (gather_mm vs einsum) are same-precision reorderings; router selection is exact.
    print("\n=== MiniMax-M3-VL M1b (real-weight at-scale layer parity) ===")
    print(f"checkpoint  : {MINIMAX}  (hidden {cfg.hidden_size}, GQA {cfg.num_attention_heads}q/"
          f"{cfg.num_key_value_heads}kv hd{cfg.head_dim}, {cfg.num_local_experts}e top-"
          f"{cfg.num_experts_per_tok} + {cfg.n_shared_experts} shared, dense_inter "
          f"{cfg.dense_intermediate_size})  T={T}")
    print(f"dense  L{i_dense}    : block vs numpy-fp64 Δ {d['par']:.2e} | fast==naive {d['fast_naive']:.2e}")
    print(f"moe    L{i_moe}    : block vs numpy-fp64 Δ {e['par']:.2e} | fast==naive {e['fast_naive']:.2e}"
          f" | sparse==dense {e['sparse_dense']:.2e}")
    print(f"             router set-match {e['router_set']} (w Δ {e['router_wdiff']:.2e}) | "
          f"indexer shapes ok {e['index_ok']}")

    _ck(d["par"] < 1e-4, f"dense block (L{i_dense}) != numpy ref: {d['par']:.2e}")
    _ck(d["fast_naive"] < 1e-4, f"dense block fast != naive: {d['fast_naive']:.2e}")
    _ck(e["par"] < 1e-4, f"moe block (L{i_moe}) != numpy ref: {e['par']:.2e}")
    _ck(e["fast_naive"] < 1e-4, f"moe block fast != naive: {e['fast_naive']:.2e}")
    _ck(e["sparse_dense"] < 1e-4, f"moe sparse gather_mm != dense oracle: {e['sparse_dense']:.2e}")
    _ck(e["router_set"], "moe router selection != numpy oracle (real gate/bias)")
    _ck(e["router_wdiff"] < 1e-5, f"moe router weights != numpy oracle: {e['router_wdiff']:.2e}")
    _ck(e["index_ok"], "trained sparse indexer tensors absent / wrong shape (rule-6 coverage)")

    print(f"PARITY-CHECKS: {_N}")
    print(f"PASS — MiniMax-M3-VL real L{i_dense}+L{i_moe} blocks match the numpy fp64 reference @ "
          f"hidden {cfg.hidden_size}; loader + real-shape/dtype wiring validated at scale.")


if __name__ == "__main__":
    run()
