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
               input_ids: mx.array | None,
               *, topk_override: int | None = None) -> tuple[mx.array, mx.array]:
    """Return ``(idx [N,topk] int32, weights [N,topk] f32)`` for ``xf`` ``[N,dim]``.

    ``topk_override`` (optional) reduces the number of routed slots per token below
    ``cfg.num_experts_per_tok``; for **hash** layers it slices the precomputed ``tid2eid`` map
    (taking the first ``k`` mapped experts per token) so the hash path also honors the override."""
    topk = topk_override if topk_override is not None else cfg.num_experts_per_tok
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
        idx = router["tid2eid"][input_ids.reshape(-1)].astype(mx.int32)      # [N, cfg.topk]
        if topk < idx.shape[-1]:                                             # honor lighter draft
            idx = idx[:, :topk]
    else:
        choice = scores + router["bias"].astype(mx.float32)[None]
        idx = mx.argpartition(-choice, kth=topk - 1, axis=-1)[:, :topk].astype(mx.int32)
    w = mx.take_along_axis(scores, idx, axis=-1)                             # bias-free scores
    if cfg.scoring_func != "softmax":
        w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
    return idx, w * cfg.routed_scaling_factor


def _swiglu_stack(xf: mx.array, idx_flat: mx.array, tok: mx.array, experts: dict,
                  limit: float) -> mx.array:
    """Routed SwiGLU over gathered experts: ``[mc, dim] -> [mc, dim]`` for ``mc`` (token,slot) pairs.

    Sorts the routed slots by expert id and passes ``sorted_indices=True`` to ``mx.gather_mm`` so
    the kernel can amortize each expert's weight load across all slots routed to it (mirrors
    :class:`quanta.nemotron.moe.NemotronQuantizedMoE`). Restores the original (token,slot) order
    on output so the caller's ``reshape(n, topk, dim)`` is unchanged."""
    mc = idx_flat.shape[0]
    order = mx.argsort(idx_flat)                                            # [mc] (stable for ties)
    inv = mx.argsort(order)
    idx_s, tok_s = idx_flat[order], tok[order]
    col = xf[:, :, None].astype(experts["w1"].dtype)                        # match expert dtype (bf16/f32)
    g = mx.gather_mm(experts["w1"], col, lhs_indices=idx_s, rhs_indices=tok_s, sorted_indices=True)[:, :, 0]
    u = mx.gather_mm(experts["w3"], col, lhs_indices=idx_s, rhs_indices=tok_s, sorted_indices=True)[:, :, 0]
    if limit > 0:
        g = mx.minimum(g, limit)
        u = mx.clip(u, -limit, limit)
    a = (silu(g) * u)[:, :, None]                                           # [mc, inter, 1] in sorted order
    d = mx.gather_mm(experts["w2"], a, lhs_indices=idx_s,
                     rhs_indices=mx.arange(mc, dtype=mx.int32), sorted_indices=True)
    return d[:, :, 0][inv]                                                  # [mc, dim] unpermuted


def _shared(xf: mx.array, shared: dict, limit: float) -> mx.array:
    xd = xf.astype(shared["w1"].dtype)
    g = xd @ shared["w1"].T
    u = xd @ shared["w3"].T
    if limit > 0:
        g = mx.minimum(g, limit)
        u = mx.clip(u, -limit, limit)
    return (silu(g) * u) @ shared["w2"].T


def _swiglu_stack_packed(xf: mx.array, idx_flat: mx.array, tok: mx.array, experts: dict,
                         limit: float) -> mx.array:
    """Packed routed SwiGLU: ``experts[proj]`` is ``{packed, scale, bias, awq_scale}`` instead of a
    bf16 ``[E,*,*]`` stack. Mirrors :func:`_swiglu_stack` but routes the three projections through
    ``mx.gather_qmm`` over the int4 codestream (the bandwidth-cheap decode path that #141 unblocks
    so the full 43-layer DSV4 fits the 490 GB ceiling). AWQ correction: the packed codes are of
    ``W·diag(s)``; recover ``W·x`` by dividing the gathered activation by the per-expert
    per-input-channel ``awq_scale`` *before* the qmm (``s=1`` for cold/RTN experts → identity, so
    a uniform path covers both warm and cold).

    Sorts routed slots by expert id and passes ``sorted_indices=True`` so the kernel amortizes each
    expert's int4 weight read across all slots routed to it (#143). Natural-language routing has
    more inter-token overlap than random, so this cuts bandwidth in spec verify (and large prefill
    batches) where multiple tokens land on shared experts."""
    mc = idx_flat.shape[0]
    order = mx.argsort(idx_flat)                                            # [mc]
    inv = mx.argsort(order)
    idx_s = idx_flat[order]                                                 # sorted expert ids
    rows = mx.arange(mc, dtype=mx.int32)
    x_in = xf[tok[order]]                                                   # [mc, dim] (f32, sorted)

    def _qmm(x_for_proj: mx.array, proj_dict: dict) -> mx.array:
        """gather_qmm convention (mirrors :class:`quanta.nemotron.moe.NemotronQuantizedMoE._qmm`):
        ``lhs_indices`` gathers rows of the first arg (``x``); ``rhs_indices`` gathers rows of the
        second arg (``packed`` weight stack along its ``E`` axis). We pre-gather ``x_for_proj`` per
        routed slot so the lhs gather is identity (``rows``); ``idx_s`` picks each slot's expert."""
        return mx.gather_qmm(
            x_for_proj[:, None, :].astype(mx.bfloat16),
            proj_dict["packed"], proj_dict["scale"], proj_dict["bias"],
            lhs_indices=rows, rhs_indices=idx_s, transpose=True, sorted_indices=True,
            group_size=int(proj_dict["group_size"]), bits=int(proj_dict["bits"]),
        )[:, 0, :]

    s_w1 = experts["w1"]["awq_scale"][idx_s].astype(mx.float32)             # [mc, dim] sorted
    s_w3 = experts["w3"]["awq_scale"][idx_s].astype(mx.float32)
    g = _qmm(x_in / s_w1, experts["w1"]).astype(mx.float32)                 # [mc, inter] sorted
    u = _qmm(x_in / s_w3, experts["w3"]).astype(mx.float32)
    if limit > 0:
        g = mx.minimum(g, limit)
        u = mx.clip(u, -limit, limit)
    h = silu(g) * u                                                         # [mc, inter] sorted

    s_w2 = experts["w2"]["awq_scale"][idx_s].astype(mx.float32)             # [mc, inter] sorted
    d = _qmm(h / s_w2, experts["w2"]).astype(mx.float32)                    # [mc, dim] sorted
    return d[inv]                                                           # restore (tok, slot) order


def dsv4_moe(x: mx.array, router: dict, experts: dict, shared: dict, cfg: DeepSeekV4Config,
             layer_id: int, input_ids: mx.array | None = None,
             *, topk_override: int | None = None) -> mx.array:
    """Full MoE: top-``topk`` routed + shared expert. ``x`` ``[B,S,dim] -> [B,S,dim]``.

    Two routed-expert modes (auto-detected by the shape of ``experts['w1']``):

    * **bf16 stacks** (``experts['w1']`` is an ``mx.array`` of shape ``[E, inter, dim]``):
      :func:`_swiglu_stack` over ``mx.gather_mm`` — the parity reference path.
    * **packed int4 stacks** (``experts['w1']`` is a dict ``{packed, scale, bias, awq_scale,
      group_size, bits}``): :func:`_swiglu_stack_packed` over ``mx.gather_qmm`` with per-expert
      AWQ rescaling — the bandwidth-cheap decode path (#141) that keeps the full 43-layer
      DSV4 under the 490 GB ceiling.

    ``shared``: ``{w1,w3:[inter,dim], w2:[dim,inter]}`` (always bf16; single-per-layer is cheap).

    ``topk_override`` (optional): route to only this many top experts instead of
    ``cfg.num_experts_per_tok``. Used by the MTP draft head (:class:`quanta.dsv4.mtp.MTPHead`)
    to run a *lighter* MoE during speculation — cuts the routed FFN cost ~k/topk_override-fold
    while keeping every other op identical. Forwards (verify) still use the full ``cfg.topk``
    so spec losslessness is preserved (the main model arbitrates every emitted token)."""
    b, s, dim = x.shape
    n = b * s
    xf = x.reshape(n, dim).astype(mx.float32)
    topk = topk_override if topk_override is not None else cfg.num_experts_per_tok
    if topk < 1 or topk > cfg.num_experts_per_tok:
        raise ValueError(f"topk_override {topk_override} must be in [1, {cfg.num_experts_per_tok}]")
    limit = cfg.swiglu_limit
    idx, w = dsv4_route(xf, router, cfg, layer_id, input_ids, topk_override=topk)
    idx_flat = idx.reshape(-1)
    tok = mx.repeat(mx.arange(n, dtype=mx.int32), topk)
    routed_fn = _swiglu_stack_packed if isinstance(experts["w1"], dict) else _swiglu_stack
    routed = routed_fn(xf, idx_flat, tok, experts, limit).reshape(n, topk, dim)
    routed = mx.sum(routed.astype(mx.float32) * w[:, :, None], axis=1)       # [N, dim] (f32 accum)
    y = routed + _shared(xf, shared, limit).astype(mx.float32)
    return y.astype(x.dtype).reshape(b, s, dim)
