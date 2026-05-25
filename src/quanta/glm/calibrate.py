"""GLM-5.1 (``glm_moe_dsa``) AWQ calibration: per-MoE-layer post-norm activations + routing.

Mirrors the DSV4 calibration (:mod:`quanta.dsv4.calibrate`) on GLM's **plain pre-norm residual**
stack (GLM has no Hyper-Connections — :mod:`quanta.glm.model`). A streamed, one-layer-resident forward
advances the residual through every block; at each **MoE** layer it records the routed experts' exact
input ``x`` ``[N, hidden]`` (``post_attention_layernorm(h + attn(input_layernorm(h)))`` — the same
``ln2`` the runtime feeds the experts) and the routing ``idx`` ``[N, topk]``. Per expert, AWQ then
calibrates ``gate_proj``/``up_proj`` on its routed rows of ``x`` and ``down_proj`` on the SwiGLU
intermediate of those rows (see :mod:`quanta.glm.bake`).

Memory-disciplined (rule 8): one block (:class:`quanta.glm.model.GLMDecoderLayer`) resident at a time
via :class:`quanta.glm.loader.GLMSourceCheckpoint` — built, used to advance the stream, and dropped
before the next; the per-layer expert stacks are the memory peak and are freed each iteration. The first
``first_k_dense_replace`` layers are dense FFN (no routed experts) so they are advanced but **not**
captured. The forward reuses :func:`quanta.glm.model.load_block` so it is the exact bf16 reference path.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.glm.config import GLMConfig
from quanta.glm.model import load_block


def capture_calibration(
    ck, cfg: GLMConfig, calib_ids: mx.array, *, n_layers: int | None = None,
    use_fast: bool = True, use_indexer: bool = True,
) -> dict[int, tuple[mx.array, mx.array]]:
    """Per-MoE-layer ``{i: (x [N,hidden] bf16, idx [N,topk] int32)}`` for AWQ calibration.

    ``calib_ids`` is ``[S]`` (or ``[1,S]``) token ids. Only MoE layers
    (``cfg.is_moe_layer(i)``) are captured; the dense ``first_k_dense_replace`` layers are advanced
    without a capture. The forward is the bf16 reference (one block resident at a time).
    """
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    ids = calib_ids.reshape(1, -1)
    positions = mx.arange(ids.shape[1])

    h = ck.embed()[ids].astype(mx.bfloat16)                          # [1,S,hidden]
    mx.eval(h)
    ck.release()

    caps: dict[int, tuple[mx.array, mx.array]] = {}
    for i in range(n):
        layer = load_block(ck, cfg, i, mx.bfloat16)
        last = i == n - 1

        if cfg.is_dense_layer(i):
            # no routed experts to calibrate; advance the residual (attention computed once).
            if not last:
                h = layer(h, positions, use_fast=use_fast, use_indexer=use_indexer)
                mx.eval(h)
            del layer
            ck.release()
            mx.clear_cache()
            continue

        # MoE layer: compute the attention sub-block ONCE, capture the experts' exact input (ln2) +
        # routing, then reuse the same intermediates to advance the residual (mirrors dsv4.calibrate —
        # no second attention pass per layer, rule 3).
        x = layer.input_layernorm(h)
        q_latent = layer.self_attn.q_a_layernorm(layer.self_attn.q_a_proj(x))
        mask = (layer.indexer.select_mask(x, q_latent, positions, use_fast=use_fast)
                if use_indexer else None)
        resid = h + layer.self_attn(x, positions, use_fast=use_fast, index_mask=mask)
        ln2 = layer.post_attention_layernorm(resid)
        xf = ln2.reshape(-1, cfg.hidden_size)
        idx, _ = layer.mlp.gate(xf)
        mx.eval(xf, idx)
        caps[i] = (xf.astype(mx.bfloat16), idx.astype(mx.int32))

        if not last:  # advance: residual + MoE(ln2), reusing resid/ln2 already computed above
            h = resid + layer.mlp(ln2)
            mx.eval(h)
        del layer
        ck.release()
        mx.clear_cache()
    return caps
