"""Parity: DeepSeek-V4 Hyper-Connections (MLX) == a faithful numpy transcription of the reference.

No CUDA/torch reference runs on this box, so this gates :mod:`quanta.dsv4.hyper` against an
independent numpy reimplementation of the reference maths (``model.py`` ``hc_pre``/``hc_post``/
``hc_head`` + ``kernel.py`` ``hc_split_sinkhorn``), on both random params and the **real** layer-2
HC params loaded from the checkpoint. Also sanity-checks that ``comb`` is ~doubly-stochastic after
the Sinkhorn iterations.

    uv run --with numpy python -m parity.dsv4_hyper_test
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from quanta.dsv4 import hyper
from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.loader import DeepSeekV4SourceCheckpoint

ART = "/Users/pmrj/models/DeepSeek-V4-Flash"


# --- numpy reference (float64) ----------------------------------------------
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _np_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps):
    pre = _sig(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * _sig(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
    comb = comb.reshape(*mixes.shape[:-1], hc, hc)
    m = comb.max(-1, keepdims=True)
    e = np.exp(comb - m)
    comb = e / e.sum(-1, keepdims=True) + eps
    comb = comb / (comb.sum(-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + eps)
        comb = comb / (comb.sum(-2, keepdims=True) + eps)
    return pre, post, comb


def _np_hc_pre(x, fn, scale, base, hc, iters, neps, heps):
    shape = x.shape
    xf = x.reshape(*shape[:2], -1).astype(np.float64)
    rsqrt = 1.0 / np.sqrt((xf * xf).mean(-1, keepdims=True) + neps)
    mixes = (xf @ fn.T.astype(np.float64)) * rsqrt
    pre, post, comb = _np_sinkhorn(mixes, scale, base, hc, iters, heps)
    reduced = (pre[..., None] * xf.reshape(shape)).sum(2)
    return reduced, post, comb


def _np_hc_post(sub, res, post, comb):
    term1 = post[..., None] * sub[..., None, :]
    term2 = np.einsum("btjk,btjd->btkd", comb, res.astype(np.float64))
    return term1 + term2


def _np_hc_head(x, fn, scale, base, neps, heps):
    shape = x.shape
    xf = x.reshape(*shape[:2], -1).astype(np.float64)
    rsqrt = 1.0 / np.sqrt((xf * xf).mean(-1, keepdims=True) + neps)
    mixes = (xf @ fn.T.astype(np.float64)) * rsqrt
    pre = _sig(mixes * scale + base) + heps
    return (pre[..., None] * xf.reshape(shape)).sum(2)


def _d(a_np, b_mx):
    return float(np.max(np.abs(a_np - np.array(b_mx.astype(mx.float32)).astype(np.float64))))


def run() -> None:
    cfg = DeepSeekV4Config.from_pretrained(ART)
    hc, neps, heps, iters = cfg.hc_mult, cfg.norm_eps, cfg.hc_eps, cfg.hc_sinkhorn_iters
    ok = True
    rng = np.random.default_rng(0)

    def check(tag, cond, extra=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'OK' if cond else 'FAIL'}] {tag}{(' ' + extra) if extra else ''}")

    # --- random small case -----------------------------------------------------
    print("=== random (B=2,T=5,hc=4,d=64) ===")
    B, T, d = 2, 5, 64
    x = rng.standard_normal((B, T, hc, d)).astype(np.float32)
    fn = rng.standard_normal((cfg.mix_hc, hc * d)).astype(np.float32) * 0.02
    base = rng.standard_normal(cfg.mix_hc).astype(np.float32) * 0.1
    scale = rng.standard_normal(3).astype(np.float32)
    r_np, p_np, c_np = _np_hc_pre(x, fn, scale, base, hc, iters, neps, heps)
    r_mx, p_mx, c_mx = hyper.hc_pre(mx.array(x), mx.array(fn), mx.array(scale), mx.array(base),
                                    hc, iters, neps, heps)
    check("hc_pre reduced", _d(r_np, r_mx) < 1e-3, f"|Δ|={_d(r_np, r_mx):.1e}")
    check("hc_pre post", _d(p_np, p_mx) < 1e-3, f"|Δ|={_d(p_np, p_mx):.1e}")
    check("hc_pre comb", _d(c_np, c_mx) < 1e-3, f"|Δ|={_d(c_np, c_mx):.1e}")
    sub = rng.standard_normal((B, T, d)).astype(np.float32)
    o_np = _np_hc_post(sub, x, p_np, c_np)
    o_mx = hyper.hc_post(mx.array(sub), mx.array(x), p_mx, c_mx)
    check("hc_post expand", o_mx.shape == (B, T, hc, d) and _d(o_np, o_mx) < 1e-3,
          f"|Δ|={_d(o_np, o_mx):.1e}")
    # doubly-stochastic-ish comb
    rs = float(np.max(np.abs(c_np.sum(-1) - 1.0)))
    cs = float(np.max(np.abs(c_np.sum(-2) - 1.0)))
    check("comb ~doubly-stochastic", rs < 5e-2 and cs < 5e-2, f"row|Δ|={rs:.2e} col|Δ|={cs:.2e}")

    # --- real params (layer-2 hc_attn + final hc_head) -------------------------
    print("=== real checkpoint params ===")
    ck = DeepSeekV4SourceCheckpoint(ART, cfg)
    bhc = ck.block_hc(2)
    H = cfg.hidden_size
    xr = rng.standard_normal((1, 4, hc, H)).astype(np.float32) * 0.5
    fn = np.array(bhc["hc_attn_fn"].astype(mx.float32))
    base = np.array(bhc["hc_attn_base"].astype(mx.float32))
    scale = np.array(bhc["hc_attn_scale"].astype(mx.float32))
    r_np, p_np, c_np = _np_hc_pre(xr, fn, scale, base, hc, iters, neps, heps)
    r_mx, p_mx, c_mx = hyper.hc_pre(mx.array(xr), bhc["hc_attn_fn"], bhc["hc_attn_scale"],
                                    bhc["hc_attn_base"], hc, iters, neps, heps)
    check("real hc_pre (L2 attn)", _d(r_np, r_mx) < 2e-3 and _d(c_np, c_mx) < 2e-3,
          f"reduced|Δ|={_d(r_np, r_mx):.1e} comb|Δ|={_d(c_np, c_mx):.1e}")
    fhc = ck.final_hc()
    fn = np.array(fhc["fn"].astype(mx.float32))
    base = np.array(fhc["base"].astype(mx.float32))
    scale = np.array(fhc["scale"].astype(mx.float32))
    h_np = _np_hc_head(xr, fn, scale, base, neps, heps)
    h_mx = hyper.hc_head(mx.array(xr), fhc["fn"], fhc["scale"], fhc["base"], hc, neps, heps)
    check("real hc_head (final)", h_mx.shape == (1, 4, H) and _d(h_np, h_mx) < 2e-3,
          f"|Δ|={_d(h_np, h_mx):.1e}")
    ck.release()

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
