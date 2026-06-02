"""Qwen3.5 sparse MoE (512 experts, top-10 softmax) + 1 shared expert, MLX-native.

* **Router** (``gate`` 4096 -> 512): **softmax** over all 512 experts; pick the top-10; the routing
  weights are the gathered softmax probabilities, re-normalized to sum 1 when ``norm_topk_prob``
  (so the top-10 form a proper convex combination). No DeepSeek ``noaux_tc`` sigmoid+bias scheme —
  Qwen3.5 ships only ``router_aux_loss_coef`` (a *training* aux loss, inert at inference).
* **Routed experts**: stored **pre-stacked 3D** and ``gather_qmm``-ready —
  ``experts.gate_up_proj`` ``[E, 2*moe_inter, hidden]`` (fused gate+up) and ``experts.down_proj``
  ``[E, hidden, moe_inter]``. Each expert is a SwiGLU ``down(silu(gate) * up)`` (width 1024).
  Dispatch is sparse :func:`mx.gather_mm` over the gathered (token, slot) rows — no per-expert
  python loop (rule-3), token-chunked for bounded long-context prefill. The post-bake resident
  runtime swaps ``gather_mm`` -> ``gather_qmm`` over the **packed** int4 stacks
  (:func:`_routed_sparse_packed`; same ``[E,out,in]`` layout) so the experts stay int4-resident
  (~4× lighter) instead of dequantized to bf16 — auto-detected when the expert stacks are packed
  triplet dicts.
* **Shared expert** (always-on, width 1024): a SwiGLU ``shared_down(silu(shared_gate)·shared_up)``
  whose *whole* output is scaled by a learned sigmoid scalar gate ``sigmoid(x @ shared_expert_gate)``
  (Qwen2-MoE shared-gate), then added to the routed sum.

Three numerically equivalent routed paths (gated in ``parity/qwen35_forward_test.py``): the sparse
bf16 ``gather_mm`` dispatch, its packed-int4 ``gather_qmm`` sibling (same codes, fused dequant), and
a dense reference that runs every expert and masks — they must all match.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.qwen35.config import Qwen35Config


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def qwen35_route(xf: mx.array, gate_w: mx.array, cfg: Qwen35Config,
                 *, topk_override: int | None = None) -> tuple[mx.array, mx.array]:
    """Top-k softmax routing. ``xf`` ``[N,hidden]`` -> ``(idx [N,topk] int32, w [N,topk] f32)``.

    Softmax over all experts, then select top-k and (optionally) re-normalize the gathered probs.

    ``topk_override`` (optional) reduces the number of routed slots per token below
    ``cfg.num_experts_per_tok`` — used by the MTP draft head to run a lighter MoE during
    speculation (lossless: the main model verifies every drafted token at ``cfg.num_experts_per_tok``).
    """
    topk = topk_override if topk_override is not None else cfg.num_experts_per_tok
    logits = xf.astype(mx.float32) @ gate_w.astype(mx.float32).T          # [N, E]
    if cfg.scoring_func == "sigmoid":
        scores = mx.sigmoid(logits)
    else:  # softmax (Qwen3.5 default)
        scores = mx.softmax(logits, axis=-1)
    idx = mx.argpartition(-scores, kth=topk - 1, axis=-1)[:, :topk].astype(mx.int32)
    w = mx.take_along_axis(scores, idx, axis=-1)                          # [N, topk]
    if cfg.norm_topk_prob:
        w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
    return idx, w


def _swiglu_gate_up(g_up: mx.array, inter: int) -> mx.array:
    """SwiGLU on a fused gate+up activation ``[..., 2*inter]`` -> ``[..., inter]``."""
    g = g_up[..., :inter]
    u = g_up[..., inter:]
    return silu(g) * u


def _routed_sparse(xf: mx.array, idx: mx.array, gate_up: mx.array, down: mx.array,
                   inter: int) -> mx.array:
    """Sparse routed SwiGLU via ``gather_mm`` over pre-stacked experts.

    ``xf`` ``[N,hidden]``; ``idx`` ``[N,topk]``; ``gate_up`` ``[E,2*inter,hidden]``;
    ``down`` ``[E,hidden,inter]``. Returns the per-(token,slot) expert outputs ``[N,topk,hidden]``.
    """
    n, hidden = xf.shape
    topk = idx.shape[1]
    mc = n * topk
    exp = idx.reshape(-1)                                                 # [mc] expert id per slot
    tok = mx.repeat(mx.arange(n, dtype=mx.int32), topk)                   # [mc] token per slot
    col = xf[:, :, None].astype(gate_up.dtype)                           # [N, hidden, 1]
    gu = mx.gather_mm(gate_up, col, lhs_indices=exp, rhs_indices=tok)[:, :, 0]   # [mc, 2*inter]
    h = _swiglu_gate_up(gu, inter)[:, :, None]                            # [mc, inter, 1]
    d = mx.gather_mm(down, h, lhs_indices=exp, rhs_indices=mx.arange(mc, dtype=mx.int32))[:, :, 0]
    return d.reshape(n, topk, hidden)                                     # [N, topk, hidden]


def _routed_sparse_packed(xf: mx.array, idx: mx.array, gate_up: dict, down: dict,
                          inter: int) -> mx.array:
    """Sparse routed SwiGLU via ``mx.gather_qmm`` over the **packed int4** expert stacks.

    The memory-lean sibling of :func:`_routed_sparse`: ``gate_up`` / ``down`` are packed affine
    triplets ``{"packed", "scale", "bias", "group_size", "bits"}`` (the ``[E, 2*inter, hidden]`` /
    ``[E, hidden, inter]`` int4 codestream held verbatim — never dequantized to a bf16 ``[E,*,*]``
    array), so the routed experts stay int4-resident (~4× lighter; the ~79→~30 GiB lever).
    ``mx.gather_qmm`` dequantizes each routed expert's codes inline. Plain **affine** int4 — no AWQ
    activation rescale (unlike DSV4's :func:`quanta.dsv4.moe._swiglu_stack_packed`) — and a **fused**
    ``gate_up`` ⇒ exactly **two** ``gather_qmm`` calls (gate_up, then down).

    Output-equivalent to :func:`_routed_sparse` on the SAME codes (greedy-exact — only the kernel
    differs: fused-quantized vs ``mx.dequantize`` + dense ``gather_mm``), and **batch-invariant**: it
    is the same per-(token,slot) **M=1 matvec** structure as ``gather_mm``, so it does NOT reorder
    accumulation across batch-M (the #153 bug stays fixed; the "MoE is exempt" finding holds).

    ``mx.gather_qmm`` arg order differs from ``gather_mm``: the **activation comes first**
    (``lhs_indices`` gathers its rows), the packed weight stack second (``rhs_indices`` gathers its
    ``E`` axis); ``transpose=True`` matches the ``[E, out, in]`` layout (``x[.,in] @ W[.,out,in].T``).
    Unsorted — a clean packed-vs-dequant diff at the identical dispatch order ``_routed_sparse`` uses
    (``sorted_indices`` is an optional bandwidth lever, deferred). No activation dtype cast: the qmm
    output follows the baked ``scale`` dtype (bf16 or fp32), so the model-free f32 gate is a clean
    ~ULP fused-vs-separate-dequant diff and the bf16 runtime stays bf16 (matches the mixer path).
    """
    n, hidden = xf.shape
    topk = idx.shape[1]
    mc = n * topk
    exp = idx.reshape(-1)                                                 # [mc] expert id per slot
    tok = mx.repeat(mx.arange(n, dtype=mx.int32), topk)                   # [mc] token per slot
    rows = mx.arange(mc, dtype=mx.int32)                                  # identity lhs gather
    x_in = xf[tok]                                                        # [mc, hidden] per-slot acts
    gu = mx.gather_qmm(x_in[:, None, :], gate_up["packed"], gate_up["scale"], gate_up["bias"],
                       lhs_indices=rows, rhs_indices=exp, transpose=True,
                       group_size=int(gate_up["group_size"]), bits=int(gate_up["bits"]))[:, 0, :]
    h = _swiglu_gate_up(gu, inter)                                        # [mc, inter]
    d = mx.gather_qmm(h[:, None, :], down["packed"], down["scale"], down["bias"],
                      lhs_indices=rows, rhs_indices=exp, transpose=True,
                      group_size=int(down["group_size"]), bits=int(down["bits"]))[:, 0, :]
    return d.reshape(n, topk, hidden)                                     # [N, topk, hidden]


def _routed_dense(xf: mx.array, idx: mx.array, w: mx.array, gate_up: mx.array, down: mx.array,
                  inter: int) -> mx.array:
    """Dense reference: run **every** expert on every token, then combine only the top-k.

    The parity oracle for :func:`_routed_sparse` — same math, no gather (small E only)."""
    n, hidden = xf.shape
    e = gate_up.shape[0]
    xd = xf.astype(gate_up.dtype)
    gu = mx.einsum("nh,eoh->neo", xd, gate_up)                            # [N, E, 2*inter]
    h = _swiglu_gate_up(gu, inter)                                        # [N, E, inter]
    d = mx.einsum("nei,ehi->neh", h, down)                               # [N, E, hidden]
    # scatter the per-slot routing weight onto the chosen experts -> [N, E]
    gates = mx.zeros((n, e), dtype=mx.float32)
    rows = mx.repeat(mx.arange(n, dtype=mx.int32), idx.shape[1])
    gates[rows, idx.reshape(-1)] = w.reshape(-1).astype(mx.float32)
    return mx.sum(d.astype(mx.float32) * gates[:, :, None], axis=1)       # [N, hidden]


def _shared(xf: mx.array, p: dict) -> mx.array:
    """Shared SwiGLU expert, scaled by its sigmoid scalar gate. ``xf`` ``[N,hidden]`` -> ``[N,hidden]``."""
    xd = xf.astype(p["shared_gate_proj"].dtype)
    h = silu(xd @ p["shared_gate_proj"].T) * (xd @ p["shared_up_proj"].T)
    out = h @ p["shared_down_proj"].T                                     # [N, hidden]
    sg = mx.sigmoid(xf.astype(mx.float32) @ p["shared_expert_gate"].astype(mx.float32).T)  # [N,1]
    return out.astype(mx.float32) * sg


def qwen35_moe(x: mx.array, p: dict, cfg: Qwen35Config, *, sparse: bool = True,
               token_chunk: int = 8192,
               topk_override: int | None = None) -> mx.array:
    """Full MoE: top-10 routed (gather_mm) + sigmoid-gated shared expert. ``x`` ``[B,S,h] -> [B,S,h]``.

    ``p``: ``{gate, experts_gate_up [E,2*inter,h], experts_down [E,h,inter], shared_gate_proj,
    shared_up_proj, shared_down_proj [.,h], shared_expert_gate [1,h]}``. ``sparse=False`` runs the
    dense reference (every expert) — the parity oracle, small-E only.

    ``topk_override`` (optional): route to only this many top experts instead of
    ``cfg.num_experts_per_tok``. Used by the MTP draft head (:class:`quanta.qwen35.mtp.MTPHead`)
    to run a *lighter* MoE during speculation — cuts the routed FFN cost ~k/topk_override-fold
    while keeping every other op identical. Forwards (verify) still use the full ``cfg.topk``
    so spec losslessness is preserved (the main model arbitrates every emitted token).
    """
    b, s, hidden = x.shape
    n = b * s
    inter = cfg.moe_intermediate_size
    topk = topk_override if topk_override is not None else cfg.num_experts_per_tok
    if topk < 1 or topk > cfg.num_experts_per_tok:
        raise ValueError(
            f"topk_override {topk_override} must be in [1, {cfg.num_experts_per_tok}]"
        )
    xf = x.reshape(n, hidden)
    idx, w = qwen35_route(xf, p["gate"], cfg, topk_override=topk)
    # Auto-detect packed int4 experts (a triplet dict) → gather_qmm; bf16 stacks → gather_mm. Same
    # signature, same matvec, greedy-exact (mirrors DSV4's dsv4_moe routed_fn auto-detect).
    packed = isinstance(p["experts_gate_up"], dict)
    if sparse:
        routed_fn = _routed_sparse_packed if packed else _routed_sparse
        chunk = token_chunk if token_chunk and token_chunk > 0 else n
        multi = n > chunk
        parts = []
        for c0 in range(0, n, chunk):  # bounded chunked-prefill loop; experts stay vectorized
            c1 = min(c0 + chunk, n)
            slots = routed_fn(xf[c0:c1], idx[c0:c1], p["experts_gate_up"],
                              p["experts_down"], inter)                   # [nc, topk, hidden]
            rc = mx.sum(slots.astype(mx.float32) * w[c0:c1][:, :, None], axis=1)  # [nc, hidden]
            parts.append(rc)
            if multi:
                mx.eval(rc)
        routed = parts[0] if not multi else mx.concatenate(parts, axis=0)
    elif packed:  # rule-6: the dense oracle runs mx.einsum over bf16 stacks — never a packed dict
        raise ValueError("qwen35_moe(sparse=False) is the bf16 dense oracle; packed int4 experts "
                         "require sparse=True (gather_qmm). Refusing to dequant silently.")
    else:
        routed = _routed_dense(xf, idx, w, p["experts_gate_up"], p["experts_down"], inter)
    y = routed.astype(mx.float32) + _shared(xf, p)
    return y.astype(x.dtype).reshape(b, s, hidden)
