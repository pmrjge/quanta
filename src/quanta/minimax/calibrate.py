"""MiniMax-M2.7 GPTQ calibration: per-MoE-layer post-norm activations + routing for the bake.

Every layer is MoE (no dense L0 like Kimi), so we capture at **every** decoder layer. A streamed,
one-layer-resident forward (mirroring :mod:`quanta.bake.calibrate` / :mod:`quanta.nemotron.calibrate`)
advances the residual through each :class:`quanta.minimax.model.MiniMaxBlock`; at each layer it records
the routed experts' input ``ln2 = post_attention_layernorm(h + attn(input_layernorm(h)))``
``[N, hidden]`` and the routing ``idx`` ``[N, topk]`` (sigmoid ``noaux_tc`` selection). Per expert,
GPTQ's input ``X`` is then ``ln2[tokens routed to it]`` (via :func:`quanta.bake.calibrate.expert_rows`),
the activation set whose Hessian ``XᵀX`` GPTQ minimizes over.

Memory-disciplined: one block resident at a time (rule 8); the heavy real load is **deferred to a GPU
session** — this module never instantiates against real tensors at import/test time. The model-free
gate exercises the GPTQ/int8/manifest paths on tiny random tensors only.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.minimax.config import MiniMaxConfig
from quanta.minimax.model import MiniMaxBlock, load_block
from quanta.minimax.moe import minimax_route


def capture_calibration(
    ck, cfg: MiniMaxConfig, token_ids: mx.array, *, n_layers: int | None = None
) -> dict[int, tuple[mx.array, mx.array]]:
    """Per-layer ``{i: (ln2 [N,hidden] bf16, idx [N,topk] int32)}`` for GPTQ calibration.

    ``ck`` is a :class:`quanta.minimax.loader.MiniMaxSourceCheckpoint`-shaped reader. ``token_ids``
    is a 1-D id sequence ``[S]``. All ``n`` layers are MoE, so every layer yields a capture.
    """
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    ids = token_ids.reshape(-1)
    h = ck.embed()[ids][None].astype(mx.bfloat16)  # [1, S, hidden]
    mx.eval(h)
    ck.release()

    caps: dict[int, tuple[mx.array, mx.array]] = {}
    for i in range(n):
        block = MiniMaxBlock(cfg, i)
        load_block(block, ck, cfg, i)
        # Capture the routed experts' input (post-attn-norm) + routing BEFORE advancing the residual.
        resid1 = h + block.self_attn(block.input_layernorm(h), use_fast=True)
        ln2 = block.post_attention_layernorm(resid1)
        ln2f = ln2.reshape(-1, cfg.hidden_size)
        idx, _ = minimax_route(ln2f, block._router(), cfg)
        mx.eval(ln2f, idx)
        caps[i] = (ln2f.astype(mx.bfloat16), idx)

        if i < n - 1:  # advance the residual through this block's MoE to feed the next layer
            h = block(h, use_fast=True)
            mx.eval(h)
        del block, resid1, ln2
        ck.release()
        mx.clear_cache()
    return caps
