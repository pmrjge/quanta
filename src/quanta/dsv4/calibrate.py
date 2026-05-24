"""DeepSeek-V4 AWQ calibration: per-MoE-layer post-norm activations + routing for the bake.

Mirrors the Nemotron calibration (:mod:`quanta.nemotron.calibrate`) on the DSV4 Hyper-Connection
stack. A streamed, one-layer-resident forward advances the HC residual through every block; at each
layer it records the **FFN input** ``x`` ``[N, dim]`` (``rmsnorm(hc_pre(h))`` — the routed experts'
exact input) and the routing ``idx`` ``[N, topk]``. Per expert, AWQ then calibrates ``w1``/``w3`` on
its routed rows of ``x`` and ``w2`` on the SwiGLU intermediate of those rows (see
:mod:`quanta.dsv4.bake`). Memory-disciplined: one block's bf16 params resident at a time; experts are
loaded only to advance the stream (rule-8).

The capture inlines :func:`quanta.dsv4.model.dsv4_block` so each sub-block is computed exactly once —
the attention sub-block advances the residual, the FFN ``hc_pre``+norm gives the capture point, then
the MoE + ``hc_post`` finish the block to feed the next layer.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.dsv4.attention import _rms_w, attention_dense
from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.hyper import hc_expand, hc_post, hc_pre
from quanta.dsv4.indexer import attention_compressed
from quanta.dsv4.model import load_block_params
from quanta.dsv4.moe import dsv4_moe, dsv4_route


def capture_calibration(
    ck, cfg: DeepSeekV4Config, calib_ids: mx.array, n_layers: int | None = None,
) -> dict[int, tuple[mx.array, mx.array]]:
    """Per-MoE-layer ``{i: (x [N,dim] bf16, idx [N,topk] int32)}`` for AWQ calibration.

    ``calib_ids`` is ``[S]`` (or ``[1,S]``) token ids. Every decoder layer (0..n-1) is a MoE layer in
    DSV4, so every layer is captured. The forward is the bf16 reference (one block resident).
    """
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    ids = calib_ids.reshape(1, -1)
    eps, hc, iters, heps = cfg.norm_eps, cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps

    h = hc_expand(ck.embed()[ids].astype(mx.bfloat16), hc)   # [1,S,hc,dim]
    mx.eval(h)
    ck.release()

    caps: dict[int, tuple[mx.array, mx.array]] = {}
    for i in range(n):
        p = load_block_params(ck, cfg, i, mx.bfloat16)
        attn = attention_compressed if cfg.has_compressor(i) else attention_dense

        # attention sub-block (advances the residual; computed once)
        res = h
        x, post, comb = hc_pre(h, p["hc_attn_fn"], p["hc_attn_scale"], p["hc_attn_base"], hc, iters, eps, heps)
        x = _rms_w(x, p["attn_norm"], eps)
        x = attn(x, p["attn"], cfg, i)
        h = hc_post(x, res, post, comb)

        # FFN sub-block: capture the experts' input + routing, then advance
        res = h
        x, post, comb = hc_pre(h, p["hc_ffn_fn"], p["hc_ffn_scale"], p["hc_ffn_base"], hc, iters, eps, heps)
        x = _rms_w(x, p["ffn_norm"], eps)
        xf = x.reshape(-1, cfg.hidden_size)
        idx, _ = dsv4_route(xf.astype(mx.float32), p["router"], cfg, i, ids)
        mx.eval(xf, idx)
        caps[i] = (xf.astype(mx.bfloat16), idx.astype(mx.int32))

        if i < n - 1:  # advance the residual through the MoE to feed the next layer
            y = dsv4_moe(x, p["router"], p["experts"], p["shared"], cfg, i, ids)
            h = hc_post(y, res, post, comb)
            mx.eval(h)
        del p
        ck.release()
        mx.clear_cache()
    return caps
