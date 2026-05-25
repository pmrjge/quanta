"""MiniMax-M2.7 sparse MoE — sigmoid ``noaux_tc`` routing, top-8 of 256, NO shared expert (MLX-native).

Routing (``scoring_func="sigmoid"``, ``use_routing_bias``):

* ``scores = sigmoid(x @ gate.T)`` over the 256 routed experts (per-expert sigmoid, **not** a
  softmax over experts);
* **selection**: pick the top-``num_experts_per_tok`` by ``scores + e_score_correction_bias`` (the
  ``noaux_tc`` correction bias steers *which* experts are chosen but is **not** part of the weight);
* **weights**: gathered from the bias-free ``scores`` at the selected indices, then (``norm_topk_prob``)
  normalized to sum 1 over the chosen 8, then ``* routed_scaling_factor``.

Each expert is a SwiGLU MLP ``down(silu(gate(x)) * up(x))`` with Mixtral naming ``w1``=gate,
``w3``=up, ``w2``=down. The routing weight is a per-token scalar applied after ``down`` (equivalent
to scaling pre-``down`` since ``down`` is linear). **There is no shared expert**
(``shared_intermediate_size == 0``) — unlike Kimi/DSV4 there is no always-on ``routed(x)+shared(x)``
branch; the MoE output is purely the routed sum (refuse to invent a shared expert, rule 6).

Dispatch is sparse ``mx.gather_mm`` over the stacked ``[E,*]`` expert weights — never a per-token /
per-expert python loop (rule 3) and never a dense ``tokens × experts × hidden`` intermediate
(rule 7). bf16/f32 here for the parity reference; the baked runtime swaps in ``mx.gather_qmm`` over
packed stacks. Gated sparse==dense + correct top-8-with-bias selection in
:mod:`parity.minimax_forward_test`.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.minimax.config import MiniMaxConfig


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def minimax_route(xf: mx.array, router: dict, cfg: MiniMaxConfig) -> tuple[mx.array, mx.array]:
    """Sigmoid ``noaux_tc`` routing. ``xf`` ``[N,dim]`` -> ``(idx [N,topk] int32, weights [N,topk] f32)``.

    ``router``: ``{"weight": [E,dim], "e_score_correction_bias": [E]}`` (Mixtral ``gate.weight`` +
    the routed-bias control tensor). Selection uses ``scores + bias``; weights are the **bias-free**
    sigmoid scores at the chosen indices.
    """
    topk = cfg.num_experts_per_tok
    logits = xf.astype(mx.float32) @ router["weight"].astype(mx.float32).T   # [N, E]
    if cfg.scoring_func != "sigmoid":
        raise ValueError(f"minimax_route only implements sigmoid scoring, got {cfg.scoring_func!r}")
    scores = mx.sigmoid(logits)                                              # [N, E], per-expert
    if cfg.use_routing_bias:
        choice = scores + router["e_score_correction_bias"].astype(mx.float32)[None]
    else:
        choice = scores
    idx = mx.argpartition(-choice, kth=topk - 1, axis=-1)[:, :topk].astype(mx.int32)  # [N, topk]
    w = mx.take_along_axis(scores, idx, axis=-1)                            # bias-free scores
    if topk > 1 and cfg.norm_topk_prob:
        w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
    return idx, w * cfg.routed_scaling_factor


def _swiglu_stack(xf: mx.array, idx_flat: mx.array, tok: mx.array, experts: dict) -> mx.array:
    """Routed SwiGLU over gathered experts: ``[mc, dim] -> [mc, dim]`` for ``mc`` (token,slot) pairs.

    ``experts``: ``{w1,w3:[E,inter,dim], w2:[E,dim,inter]}``. ``gather_mm`` selects expert ``idx_flat[i]``
    and token ``tok[i]`` per output row — no per-expert python loop, no dense expansion."""
    col = xf[:, :, None].astype(experts["w1"].dtype)                        # [N, dim, 1] (match expert dtype)
    g = mx.gather_mm(experts["w1"], col, lhs_indices=idx_flat, rhs_indices=tok)[:, :, 0]  # [mc, inter]
    u = mx.gather_mm(experts["w3"], col, lhs_indices=idx_flat, rhs_indices=tok)[:, :, 0]  # [mc, inter]
    a = (silu(g) * u)[:, :, None]                                           # [mc, inter, 1]
    mc = idx_flat.shape[0]
    d = mx.gather_mm(experts["w2"], a, lhs_indices=idx_flat, rhs_indices=mx.arange(mc, dtype=mx.int32))
    return d[:, :, 0]                                                       # [mc, dim]


def minimax_moe(x: mx.array, router: dict, experts: dict, cfg: MiniMaxConfig) -> mx.array:
    """Full MoE: top-``topk`` routed (sparse gather_mm), **no shared expert**. ``x`` ``[B,S,dim] -> [B,S,dim]``.

    ``router``: ``{weight:[E,dim], e_score_correction_bias:[E]}``;
    ``experts``: ``{w1,w3:[E,inter,dim], w2:[E,dim,inter]}``.
    """
    b, s, dim = x.shape
    n = b * s
    xf = x.reshape(n, dim).astype(mx.float32)
    topk = cfg.num_experts_per_tok
    idx, w = minimax_route(xf, router, cfg)
    idx_flat = idx.reshape(-1)
    tok = mx.repeat(mx.arange(n, dtype=mx.int32), topk)
    routed = _swiglu_stack(xf, idx_flat, tok, experts).reshape(n, topk, dim)
    routed = mx.sum(routed.astype(mx.float32) * w[:, :, None], axis=1)      # [N, dim] (f32 accum)
    return routed.astype(x.dtype).reshape(b, s, dim)
