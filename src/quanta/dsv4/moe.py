"""DeepSeek-V4 MoE — hash / sqrtsoftplus routing, top-6 routed + 1 shared expert (MLX-native).

Routing (``Gate``): ``scores = sqrt(softplus(x @ gate_weight.T))`` over 256 experts.
* **hash layers** (first ``n_hash_layers``): expert indices are a fixed per-token-id lookup
  (``tid2eid[input_ids]``); no learned bias.
* **score layers**: select top-``num_experts_per_tok`` by ``scores + bias`` (``noaux_tc``); the routing
  *weights* are gathered from the **bias-free** ``scores``, normalized to sum 1, then ``* routed_scaling_factor``.

Each expert is a SwiGLU MLP ``down(silu(clamp(gate, max=L)) * clamp(up, -L, L))`` (``L = swiglu_limit``);
the routing weight is a per-token scalar, applied after ``down`` (equivalent to the reference's
pre-``down`` scaling since ``down`` is linear). A single **shared** expert runs on every token (no
routing weight) and is added. Routed dispatch is sparse ``mx.gather_mm`` over stacked ``[E,*]`` weights
(no per-expert python loop — rule-3). Gated vs the authors' ``Gate``/``Expert`` in
``parity/dsv4_moe_test.py``.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.dsv4.config import DeepSeekV4Config


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def dsv4_route(xf: mx.array, router: dict, cfg: DeepSeekV4Config, layer_id: int,
               input_ids: mx.array | None) -> tuple[mx.array, mx.array]:
    """Return ``(idx [N,topk] int32, weights [N,topk] f32)`` for ``xf`` ``[N,dim]``."""
    topk = cfg.num_experts_per_tok
    logits = xf.astype(mx.float32) @ router["weight"].astype(mx.float32).T   # [N, E]
    if cfg.scoring_func == "sqrtsoftplus":
        scores = mx.sqrt(mx.logaddexp(mx.zeros_like(logits), logits))
    elif cfg.scoring_func == "sigmoid":
        scores = mx.sigmoid(logits)
    else:
        scores = mx.softmax(logits, axis=-1)
    if cfg.is_hash(layer_id):
        if input_ids is None:
            raise ValueError(f"hash layer {layer_id} needs input_ids for tid2eid routing")
        idx = router["tid2eid"][input_ids.reshape(-1)].astype(mx.int32)      # [N, topk]
    else:
        choice = scores + router["bias"].astype(mx.float32)[None]
        idx = mx.argpartition(-choice, kth=topk - 1, axis=-1)[:, :topk].astype(mx.int32)
    w = mx.take_along_axis(scores, idx, axis=-1)                             # bias-free scores
    if cfg.scoring_func != "softmax":
        w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
    return idx, w * cfg.routed_scaling_factor


def _swiglu_stack(xf: mx.array, idx_flat: mx.array, tok: mx.array, experts: dict,
                  limit: float) -> mx.array:
    """Routed SwiGLU over gathered experts: ``[mc, dim] -> [mc, dim]`` for ``mc`` (token,slot) pairs."""
    col = xf[:, :, None].astype(experts["w1"].dtype)                        # match expert dtype (bf16/f32)
    g = mx.gather_mm(experts["w1"], col, lhs_indices=idx_flat, rhs_indices=tok)[:, :, 0]
    u = mx.gather_mm(experts["w3"], col, lhs_indices=idx_flat, rhs_indices=tok)[:, :, 0]
    if limit > 0:
        g = mx.minimum(g, limit)
        u = mx.clip(u, -limit, limit)
    a = (silu(g) * u)[:, :, None]                                           # [mc, inter, 1]
    mc = idx_flat.shape[0]
    d = mx.gather_mm(experts["w2"], a, lhs_indices=idx_flat, rhs_indices=mx.arange(mc, dtype=mx.int32))
    return d[:, :, 0]                                                       # [mc, dim]


def _shared(xf: mx.array, shared: dict, limit: float) -> mx.array:
    xd = xf.astype(shared["w1"].dtype)
    g = xd @ shared["w1"].T
    u = xd @ shared["w3"].T
    if limit > 0:
        g = mx.minimum(g, limit)
        u = mx.clip(u, -limit, limit)
    return (silu(g) * u) @ shared["w2"].T


def dsv4_moe(x: mx.array, router: dict, experts: dict, shared: dict, cfg: DeepSeekV4Config,
             layer_id: int, input_ids: mx.array | None = None) -> mx.array:
    """Full MoE: top-``topk`` routed (gather_mm) + shared expert. ``x`` ``[B,S,dim] -> [B,S,dim]``.
    ``experts``: ``{w1,w3:[E,inter,dim], w2:[E,dim,inter]}``; ``shared``: ``{w1,w3:[inter,dim], w2:[dim,inter]}``."""
    b, s, dim = x.shape
    n = b * s
    xf = x.reshape(n, dim).astype(mx.float32)
    topk, limit = cfg.num_experts_per_tok, cfg.swiglu_limit
    idx, w = dsv4_route(xf, router, cfg, layer_id, input_ids)
    idx_flat = idx.reshape(-1)
    tok = mx.repeat(mx.arange(n, dtype=mx.int32), topk)
    routed = _swiglu_stack(xf, idx_flat, tok, experts, limit).reshape(n, topk, dim)
    routed = mx.sum(routed.astype(mx.float32) * w[:, :, None], axis=1)       # [N, dim] (f32 accum)
    y = routed + _shared(xf, shared, limit).astype(mx.float32)
    return y.astype(x.dtype).reshape(b, s, dim)
