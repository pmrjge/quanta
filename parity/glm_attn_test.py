"""Gate: GLM-5.1 MLA attention — model-free, tiny random weights, ~0 GB.

Parity-first (#83): for a tiny random :class:`quanta.glm.attention.MLAAttention` (small hidden / few
heads), diff the ``mlx.nn`` module against an **independent plain-``mlx.core`` transcription** of MLA
(low-rank q/kv, weighted q_a/kv_a RMSNorm, partial **interleaved** RoPE with NO YaRN, causal masked
softmax, ``o_proj``) — a divergence is a forward-math bug, not a mirrored one. Then prove the two
optimized paths are output-equivalent to the naive reference (rule 4):

* naive (explicit softmax) **==** the independent plain-mlx reference,
* fast (``mx.fast.rope`` + ``mx.fast.scaled_dot_product_attention``) **==** naive,
* incremental decode (``step``, naive AND fast) **==** the full-sequence prefill at the same positions.

Formula-vs-real-checkpoint correctness is the deferred torch/ppl oracle; here every tensor is a few KB.

    uv run --with numpy python -m parity.glm_attn_test
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from quanta.glm.attention import MLAAttention, _SUBNORM_EPS
from quanta.glm.config import GLMConfig

CFG = {
    "model_type": "glm_moe_dsa", "vocab_size": 40, "hidden_size": 16, "intermediate_size": 24,
    "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 2,
    "q_lora_rank": 8, "kv_lora_rank": 6, "qk_nope_head_dim": 4, "qk_rope_head_dim": 4,
    "qk_head_dim": 8, "v_head_dim": 8,
    "index_head_dim": 6, "index_n_heads": 2, "index_topk": 3,
    "n_routed_experts": 5, "num_experts_per_tok": 2, "n_shared_experts": 1, "moe_intermediate_size": 6,
    "first_k_dense_replace": 1, "num_nextn_predict_layers": 1,
    "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
    "eos_token_id": [39], "pad_token_id": 39, "tie_word_embeddings": False,
}


def _randomize(module: MLAAttention, rng: np.random.Generator) -> None:
    """Replace every leaf param with small random values (default init is fine, but a controlled,
    non-degenerate fill makes the diff meaningful and reproducible)."""
    flat = {}
    for k, v in module.parameters().items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = vv
        else:
            flat[k] = v
    upd = {k: mx.array((rng.standard_normal(v.shape) * 0.5).astype(np.float32)) for k, v in flat.items()}
    module.load_weights(list(upd.items()))


def _np_rms(x, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def _np_rope_interleaved(x, cos, sin):
    """Interleaved-complex RoPE on the last dim of x [B,H,T,rd]; cos/sin [T, rd/2]."""
    *lead, rd = x.shape
    xr = x.reshape(*lead, rd // 2, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    c, s = cos[None, None], sin[None, None]
    o0 = x0 * c - x1 * s
    o1 = x0 * s + x1 * c
    return np.stack([o0, o1], -1).reshape(*lead, rd)


def _np_mla(x, p, cfg: GLMConfig):
    """Independent plain-numpy MLA reference (float64). ``p`` holds the module's weights as numpy."""
    b, t, _ = x.shape
    h, nope, rope, vhd = (cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim,
                          cfg.v_head_dim)
    kv_lora, eps = cfg.kv_lora_rank, _SUBNORM_EPS
    # q low-rank: q_a_proj -> RMSNorm(weighted) -> q_b_proj
    q_lat = _np_rms(x @ p["q_a_proj"].T, eps) * p["q_a_layernorm"]
    q = (q_lat @ p["q_b_proj"].T).reshape(b, t, h, cfg.qk_head_dim).transpose(0, 2, 1, 3)  # [B,H,T,qk]
    q_nope, q_pe = q[..., :nope], q[..., nope:]
    # kv: kv_a_proj_with_mqa -> split c_kv/k_pe; c_kv RMSNorm(weighted); kv_b_proj
    ckv = x @ p["kv_a_proj_with_mqa"].T
    c_kv, k_pe = ckv[..., :kv_lora], ckv[..., kv_lora:]
    c_kv = _np_rms(c_kv, eps) * p["kv_a_layernorm"]
    k_pe = k_pe.reshape(b, t, 1, rope).transpose(0, 2, 1, 3)            # [B,1,T,rope]
    kv = (c_kv @ p["kv_b_proj"].T).reshape(b, t, h, nope + vhd).transpose(0, 2, 1, 3)
    k_nope, value = kv[..., :nope], kv[..., nope:]
    # interleaved RoPE (no YaRN)
    inv = 1.0 / (cfg.rope_theta ** (np.arange(0, rope, 2) / rope))
    ang = np.arange(t)[:, None] * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    q_pe = _np_rope_interleaved(q_pe, cos, sin)
    k_pe = _np_rope_interleaved(k_pe, cos, sin)
    k_pe = np.broadcast_to(k_pe, (b, h, t, rope))
    query = np.concatenate([q_nope, q_pe], -1)
    key = np.concatenate([k_nope, k_pe], -1)
    scores = np.einsum("bhtd,bhsd->bhts", query, key) * cfg.softmax_scale
    ti, si = np.arange(t)[:, None], np.arange(t)[None, :]
    scores = scores + np.where(si <= ti, 0.0, -np.inf)[None, None]
    w = np.exp(scores - scores.max(-1, keepdims=True))
    w = w / w.sum(-1, keepdims=True)
    out = np.einsum("bhts,bhsd->bhtd", w, value).transpose(0, 2, 1, 3).reshape(b, t, h * vhd)
    return out @ p["o_proj"].T


class _StubKV:
    """Minimal layer KV cache (the ``.update`` append protocol ``MLAAttention.step`` needs)."""

    def __init__(self) -> None:
        self.c_kv = None
        self.k_pe = None

    def update(self, c_new, k_new):
        self.c_kv = c_new if self.c_kv is None else mx.concatenate([self.c_kv, c_new], axis=1)
        self.k_pe = k_new if self.k_pe is None else mx.concatenate([self.k_pe, k_new], axis=2)
        return self.c_kv, self.k_pe


def run() -> None:
    cfg = GLMConfig.from_dict(CFG)
    rng = np.random.default_rng(0)
    attn = MLAAttention(cfg)
    _randomize(attn, rng)
    p_np = {}
    for k, v in attn.parameters().items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                p_np[f"{k}.{kk}".replace(".weight", "")] = np.array(vv).astype(np.float64)
        else:
            p_np[k.replace(".weight", "")] = np.array(v).astype(np.float64)

    B, T = 1, 6
    x = (rng.standard_normal((B, T, cfg.hidden_size)) * 0.5).astype(np.float32)
    pos = mx.arange(T)
    ok = True

    ref = _np_mla(x.astype(np.float64), p_np, cfg)
    o_naive = np.array(attn(mx.array(x), pos, use_fast=False).astype(mx.float32)).astype(np.float64)
    d = float(np.max(np.abs(ref - o_naive)))
    rel = d / float(np.max(np.abs(ref)))
    good = o_naive.shape == (B, T, cfg.hidden_size) and rel < 1e-4
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] naive MLA == independent plain-numpy ref  |Δ|={d:.2e} rel={rel:.2e}")

    o_fast = attn(mx.array(x), pos, use_fast=True)
    d = float(np.max(np.abs(o_naive - np.array(o_fast.astype(mx.float32)).astype(np.float64))))
    good = d < 1e-4
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] fast (mx.fast.rope + SDPA) == naive  |Δ|={d:.2e}")

    o_pref = attn(mx.array(x), pos, use_fast=False)
    for fast in (False, True):
        cache = _StubKV()
        cols = [attn.step(mx.array(x[:, t:t + 1]), cache, t, use_fast=fast) for t in range(T)]
        o_inc = mx.concatenate(cols, axis=1)
        d = float(np.max(np.abs(np.array((o_pref - o_inc).astype(mx.float32)))))
        good = d < 1e-4
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] incremental decode ({'fast' if fast else 'naive'}) == "
              f"prefill  |Δ|={d:.2e}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
