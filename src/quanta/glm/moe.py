"""Sparse MoE for GLM-5.1 (``glm_moe_dsa``) — noaux_tc sigmoid routing, top-8 + 1 shared expert.

Mirrors :class:`quanta.modeling.moe.SparseMoE` (the Kimi DeepSeek-V3 MoE) and :mod:`quanta.dsv4.moe`,
grounded in :mod:`quanta.glm.config` / :mod:`quanta.glm.loader` (256 routed experts, top-8, 1 shared,
``e_score_correction_bias``, ``routed_scaling_factor=2.5``):

* **Router** (``noaux_tc``, ``sigmoid``): ``scores = sigmoid(x @ gate_weight.T)`` over the 256
  experts; select the top-``num_experts_per_tok`` by ``scores + e_score_correction_bias`` (the bias
  steers *selection* only). The routing **weights** are gathered from the bias-free ``scores``,
  normalized to sum 1 (``norm_topk_prob``), then ``* routed_scaling_factor``. With
  ``n_group == topk_group == 1`` the group machinery is a no-op (asserted).
* **Experts**: each is a SwiGLU MLP ``down(silu(gate(x)) * up(x))`` (no clamp — GLM has no
  ``swiglu_limit``). Stacks are ``gate/up: [E, moe_inter, hidden]``, ``down: [E, hidden, moe_inter]``
  (the loader's ``expert_stacks`` order).
* **Dispatch**: sparse top-k via :func:`mx.gather_mm` over the stacked expert weights — no per-token
  weight materialization, no Python loop over experts/tokens (rule 3/7). bf16 here for forward-path
  parity; the post-bake runtime swaps in :func:`mx.gather_qmm`.
* **Output**: ``Σ_topk w·expert(x) + shared(x)`` (the shared expert runs on every token, no routing
  weight; bf16, never quantized).

:meth:`SparseMoE.dense_reference` is the dead-simple parity oracle: it runs **every** expert densely on
**every** token and combines with a one-hot routing mask — obviously correct, ``O(N·E)``. The sparse
gather path must equal it to fp tolerance (the #85 ``sparse == dense`` gate). Both consume the same
router. Gated model-free (tiny random weights) in ``parity/glm_forward_test.py`` / ``parity/glm_moe_test.py``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.glm.config import GLMConfig


class MoEGate(nn.Module):
    """noaux_tc sigmoid router: select by ``sigmoid+bias``, weight by bias-free normalized sigmoid."""

    def __init__(self, cfg: GLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.weight = mx.zeros((cfg.n_routed_experts, cfg.hidden_size))
        self.e_score_correction_bias = mx.zeros((cfg.n_routed_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """``x`` ``[N,dim]`` → ``(idx [N,topk] int32, weights [N,topk] fp32)``."""
        assert self.cfg.n_group == 1 and self.cfg.topk_group == 1, "group routing unsupported"
        topk = self.cfg.num_experts_per_tok
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T   # [N,E]
        scores = mx.sigmoid(logits)
        choice = scores + self.e_score_correction_bias.astype(mx.float32)[None]
        idx = mx.argpartition(-choice, kth=topk - 1, axis=-1)[:, :topk].astype(mx.int32)
        weights = mx.take_along_axis(scores, idx, axis=-1)                # bias-free scores
        if topk > 1 and self.cfg.norm_topk_prob:
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        return idx, weights * self.cfg.routed_scaling_factor


class SparseMoE(nn.Module):
    """GLM-5.1 MoE block: top-k routed experts (sparse ``gather_mm``) + a single shared expert."""

    def __init__(self, cfg: GLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gate = MoEGate(cfg)
        e, inter, hidden = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size
        # shared expert width = moe_intermediate_size * n_shared_experts (1 → one expert's width)
        sh = inter * cfg.n_shared_experts
        self.shared_gate = nn.Linear(hidden, sh, bias=False)
        self.shared_up = nn.Linear(hidden, sh, bias=False)
        self.shared_down = nn.Linear(sh, hidden, bias=False)
        # routed expert stacks (loader expert_stacks order): gate/up [E,inter,hidden], down [E,hidden,inter]
        self.gate_stack = mx.zeros((e, inter, hidden))
        self.up_stack = mx.zeros((e, inter, hidden))
        self.down_stack = mx.zeros((e, hidden, inter))
        # Run tokens through the experts in chunks so the routed intermediate ([chunk*topk, inter])
        # stays bounded at long-context prefill. Per-token independent ⇒ output-equivalent; a no-op
        # (single chunk) when n <= token_chunk (decode / short prefill).
        self.token_chunk = 8192

    def set_experts(self, gate: mx.array, up: mx.array, down: mx.array) -> None:
        self.gate_stack, self.up_stack, self.down_stack = gate, up, down

    def _shared(self, x: mx.array) -> mx.array:
        return self.shared_down(nn.silu(self.shared_gate(x)) * self.shared_up(x))

    def _routed_chunk(self, xc: mx.array, idx_c: mx.array, w_c: mx.array) -> mx.array:
        """Routed top-k expert output for one token chunk ``xc`` ``[nc,hidden]`` → ``[nc,hidden]``."""
        nc = xc.shape[0]
        topk = self.cfg.num_experts_per_tok
        x_col = xc[:, :, None].astype(self.gate_stack.dtype)             # [nc,hidden,1]
        mc = nc * topk
        exp = idx_c.reshape(-1)                                          # [mc] expert per (token,slot)
        tok = mx.repeat(mx.arange(nc, dtype=mx.int32), topk)            # [mc] local token index
        g = mx.gather_mm(self.gate_stack, x_col, lhs_indices=exp, rhs_indices=tok)
        u = mx.gather_mm(self.up_stack, x_col, lhs_indices=exp, rhs_indices=tok)
        h = (nn.silu(g) * u)                                            # [mc,inter,1]
        d = mx.gather_mm(self.down_stack, h, lhs_indices=exp, rhs_indices=mx.arange(mc, dtype=mx.int32))
        d = d[:, :, 0].reshape(nc, topk, self.cfg.hidden_size)
        return mx.sum(d.astype(mx.float32) * w_c[:, :, None], axis=1).astype(xc.dtype)

    def __call__(self, x: mx.array) -> mx.array:
        """Sparse MoE forward. ``x`` ``[B,T,dim]`` → ``[B,T,dim]``."""
        b, t, hd = x.shape
        n = b * t
        xf = x.reshape(n, hd)
        idx, weights = self.gate(xf)                                    # [n,topk] int32, [n,topk] fp32
        chunk = self.token_chunk if self.token_chunk and self.token_chunk > 0 else n
        multi = n > chunk
        parts = []
        for c0 in range(0, n, chunk):                                   # coarse bounded chunk loop
            c1 = min(c0 + chunk, n)
            xc = xf[c0:c1]
            oc = self._routed_chunk(xc, idx[c0:c1], weights[c0:c1]) + self._shared(xc)
            parts.append(oc)
            if multi:
                mx.eval(oc)                                             # bound the per-chunk peak
        out = parts[0] if not multi else mx.concatenate(parts, axis=0)
        return out.reshape(b, t, hd)

    def dense_reference(self, x: mx.array) -> mx.array:
        """Parity oracle: run EVERY expert on EVERY token, combine by a one-hot routing mask. Obviously
        correct, ``O(N·E)`` — small-config only. Must equal :meth:`__call__` to fp tolerance."""
        b, t, hd = x.shape
        n, e = b * t, self.cfg.n_routed_experts
        xf = x.reshape(n, hd).astype(mx.float32)
        idx, weights = self.gate(xf)                                    # [n,topk]
        # per-token, per-expert routing weight (0 for unselected): scatter the top-k weights.
        wfull = mx.zeros((n, e), dtype=mx.float32)
        wfull = mx.put_along_axis(wfull, idx.astype(mx.int64), weights.astype(mx.float32), axis=-1)
        gw, uw, dw = self.gate_stack, self.up_stack, self.down_stack
        # dense per-expert SwiGLU: [n,E,inter] then [n,E,hidden]
        g = mx.einsum("nd,eid->nei", xf.astype(gw.dtype), gw)
        u = mx.einsum("nd,eid->nei", xf.astype(uw.dtype), uw)
        h = nn.silu(g) * u                                             # [n,E,inter]
        d = mx.einsum("nei,ehi->neh", h, dw).astype(mx.float32)        # [n,E,hidden]
        routed = mx.sum(d * wfull[:, :, None], axis=1)                 # [n,hidden]
        out = (routed + self._shared(xf).astype(mx.float32)).astype(x.dtype)
        return out.reshape(b, t, hd)
