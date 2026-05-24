"""Parity: DeepSeek-V4 dense sliding-window attention (MLX) == a faithful numpy reference.

Gates :mod:`quanta.dsv4.attention` (ratio-0 regime: low-rank q/kv, weighted q_norm/kv_norm,
unweighted per-head q RMS, partial interleaved-complex RoPE + inverse-RoPE on the output, per-head
sink, grouped low-rank O) against an independent numpy transcription of ``model.py Attention.forward``
on the **real** layer-0 params, for a short causal sequence and a longer window-crossing one. No
torch/CUDA oracle exists for this model, so MLX-vs-numpy agreement + the structural checks are the
gate (e2e ppl in #74 is the final arbiter).

    uv run --with numpy python -m parity.dsv4_attention_test
"""

from __future__ import annotations

import math

import numpy as np

import mlx.core as mx

from quanta.dsv4 import attention as A
from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.loader import DeepSeekV4SourceCheckpoint

ART = "/Users/pmrj/models/DeepSeek-V4-Flash"


# --- numpy reference --------------------------------------------------------
def _np_cos_sin(dim, seqlen, orig, base, factor, bf, bs):
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    if orig > 0:
        def cd(nr):
            return dim * math.log(orig / (nr * 2 * math.pi)) / (2 * math.log(base))
        low, high = max(math.floor(cd(bf)), 0), min(math.ceil(cd(bs)), dim - 1)
        if low == high:
            high += 0.001
        ramp = np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    ang = np.arange(seqlen)[:, None] * freqs[None, :]
    return np.cos(ang), np.sin(ang)


def _np_rope(x, cos, sin, rd, inverse=False):
    head, tail = x[..., :-rd], x[..., -rd:]
    *lead, _ = tail.shape
    xr = tail.reshape(*lead, rd // 2, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    if tail.ndim == 4:
        c, s = cos[None, :, None, :], sin[None, :, None, :]
    else:
        c, s = cos[None, :, :], sin[None, :, :]
    if inverse:
        s = -s
    rot = np.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).reshape(*lead, rd)
    return np.concatenate([head, rot], -1)


def _rms(x, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def _np_attn(x, p, cfg):
    b, t, _ = x.shape
    nh, hd, rd, eps = cfg.num_attention_heads, cfg.head_dim, cfg.rope_head_dim, cfg.norm_eps
    orig, theta = cfg.attn_rope(0)
    cos, sin = _np_cos_sin(rd, t, orig, theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
    qr = _rms(x @ p["wq_a"].T, eps) * p["q_norm"]
    q = (qr @ p["wq_b"].T).reshape(b, t, nh, hd)
    q = _rms(q, eps)
    q = _np_rope(q, cos, sin, rd)
    kv = _rms(x @ p["wkv"].T, eps) * p["kv_norm"]
    kv = _np_rope(kv, cos, sin, rd)
    scores = np.einsum("bthd,bsd->bths", q, kv) * cfg.attn_scale
    qi, ki = np.arange(t)[:, None], np.arange(t)[None, :]
    allow = (ki <= qi) & (ki > qi - cfg.sliding_window)
    scores = scores + np.where(allow, 0.0, -1e9)[None, :, None, :]
    m = scores.max(-1, keepdims=True)
    ex = np.exp(scores - m)
    denom = ex.sum(-1) + np.exp(p["attn_sink"][None, None, :] - m[..., 0])
    o = np.einsum("bths,bsd->bthd", ex, kv) / denom[..., None]
    o = _np_rope(o, cos, sin, rd, inverse=True)
    ng, olr = cfg.o_groups, cfg.o_lora_rank
    og = o.reshape(b, t, ng, (nh * hd) // ng)
    proj = np.einsum("btgd,grd->btgr", og, p["wo_a"].reshape(ng, olr, -1)).reshape(b, t, ng * olr)
    return proj @ p["wo_b"].T


def run() -> None:
    cfg = DeepSeekV4Config.from_pretrained(ART)
    ck = DeepSeekV4SourceCheckpoint(ART, cfg)
    p_mx = ck.attention(0)                         # layer 0: pure sliding-window
    ck.release()
    p_np = {k: np.array(v.astype(mx.float32)).astype(np.float64) for k, v in p_mx.items()}
    p_f32 = {k: v.astype(mx.float32) for k, v in p_mx.items()}
    rng = np.random.default_rng(0)
    ok = True

    for t in (16, 160):                            # causal ; window-crossing (window=128)
        x = (rng.standard_normal((1, t, cfg.hidden_size)) * 0.5).astype(np.float32)
        ref = _np_attn(x.astype(np.float64), p_np, cfg)
        ours = A.attention_dense(mx.array(x), p_f32, cfg, 0)
        d = float(np.max(np.abs(ref - np.array(ours.astype(mx.float32)).astype(np.float64))))
        rel = d / float(np.max(np.abs(ref)))
        good = ours.shape == (1, t, cfg.hidden_size) and rel < 5e-3
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] T={t:3d} ({'causal' if t <= cfg.sliding_window else 'windowed'})"
              f"  |Δ|={d:.2e} rel={rel:.2e} absmax={float(np.abs(ref).max()):.3f}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
