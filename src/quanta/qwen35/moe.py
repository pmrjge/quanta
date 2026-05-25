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
  runtime swaps ``gather_mm`` -> ``gather_qmm`` over the packed stacks (same ``[E,out,in]`` layout).
* **Shared expert** (always-on, width 1024): a SwiGLU ``shared_down(silu(shared_gate)·shared_up)``
  whose *whole* output is scaled by a learned sigmoid scalar gate ``sigmoid(x @ shared_expert_gate)``
  (Qwen2-MoE shared-gate), then added to the routed sum.

Two numerically equivalent routed paths (gated in ``parity/qwen35_forward_test.py``): the sparse
``gather_mm`` dispatch and a dense reference that runs every expert and masks — they must match.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.qwen35.config import Qwen35Config


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def qwen35_route(xf: mx.array, gate_w: mx.array, cfg: Qwen35Config) -> tuple[mx.array, mx.array]:
    """Top-k softmax routing. ``xf`` ``[N,hidden]`` -> ``(idx [N,topk] int32, w [N,topk] f32)``.

    Softmax over all experts, then select top-k and (optionally) re-normalize the gathered probs.
    """
    topk = cfg.num_experts_per_tok
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
               token_chunk: int = 8192) -> mx.array:
    """Full MoE: top-10 routed (gather_mm) + sigmoid-gated shared expert. ``x`` ``[B,S,h] -> [B,S,h]``.

    ``p``: ``{gate, experts_gate_up [E,2*inter,h], experts_down [E,h,inter], shared_gate_proj,
    shared_up_proj, shared_down_proj [.,h], shared_expert_gate [1,h]}``. ``sparse=False`` runs the
    dense reference (every expert) — the parity oracle, small-E only.
    """
    b, s, hidden = x.shape
    n = b * s
    inter = cfg.moe_intermediate_size
    xf = x.reshape(n, hidden)
    idx, w = qwen35_route(xf, p["gate"], cfg)
    if sparse:
        chunk = token_chunk if token_chunk and token_chunk > 0 else n
        multi = n > chunk
        parts = []
        for c0 in range(0, n, chunk):  # bounded chunked-prefill loop; experts stay vectorized
            c1 = min(c0 + chunk, n)
            slots = _routed_sparse(xf[c0:c1], idx[c0:c1], p["experts_gate_up"],
                                   p["experts_down"], inter)              # [nc, topk, hidden]
            rc = mx.sum(slots.astype(mx.float32) * w[c0:c1][:, :, None], axis=1)  # [nc, hidden]
            parts.append(rc)
            if multi:
                mx.eval(rc)
        routed = parts[0] if not multi else mx.concatenate(parts, axis=0)
    else:
        routed = _routed_dense(xf, idx, w, p["experts_gate_up"], p["experts_down"], inter)
    y = routed.astype(mx.float32) + _shared(xf, p)
    return y.astype(x.dtype).reshape(b, s, hidden)
