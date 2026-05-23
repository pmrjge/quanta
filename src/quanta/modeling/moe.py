"""Sparse MoE block for Kimi-K2.6 (DeepSeek-V3 ``DeepseekV3MoE``), MLX-native.

* Router (``noaux_tc``, sigmoid): select top-k by ``sigmoid(logits) + correction_bias``
  but weight by the **raw** sigmoid scores (normalized, then ``* routed_scaling_factor``).
  With ``n_group == topk_group == 1`` the group machinery is a no-op (asserted).
* Dispatch: sparse top-k via :func:`mx.gather_mm` over stacked expert weights
  ``[E, out, in]`` (computes ``W @ x`` — no per-token weight materialization, no
  Python loop over experts). bf16 here for forward-path parity; the post-bake
  runtime swaps in :func:`mx.gather_qmm`.
* Output: ``Σ_topk w·expert(x) + shared(x)``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.config import KimiTextConfig
from quanta.modeling.mlp import DenseMLP


class MoEGate(nn.Module):
    def __init__(self, cfg: KimiTextConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.weight = mx.zeros((cfg.n_routed_experts, cfg.hidden_size))
        self.e_score_correction_bias = mx.zeros((cfg.n_routed_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        assert self.cfg.n_group == 1 and self.cfg.topk_group == 1, "group routing unsupported"
        topk = self.cfg.num_experts_per_tok
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        choice = scores + self.e_score_correction_bias.astype(mx.float32)[None]
        idx = mx.argpartition(-choice, kth=topk - 1, axis=-1)[:, :topk].astype(mx.int32)
        weights = mx.take_along_axis(scores, idx, axis=-1)
        if topk > 1 and self.cfg.norm_topk_prob:
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        weights = weights * self.cfg.routed_scaling_factor
        return idx, weights


class SparseMoE(nn.Module):
    def __init__(self, cfg: KimiTextConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gate = MoEGate(cfg)
        shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts
        self.shared_experts = DenseMLP(cfg, intermediate_size=shared_inter)
        e, inter, hidden = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size
        self.gate_stack = mx.zeros((e, inter, hidden))
        self.up_stack = mx.zeros((e, inter, hidden))
        self.down_stack = mx.zeros((e, hidden, inter))
        # Sort tokens by expert id so each expert's rows are contiguous (grouped GEMM).
        # Output-equivalent to the unsorted path; the real throughput win lands on the
        # post-bake gather_qmm path (mlx PR #2078). Off by default until parity-proven.
        self.sort_dispatch = False

    def set_experts(self, gate: mx.array, up: mx.array, down: mx.array) -> None:
        self.gate_stack, self.up_stack, self.down_stack = gate, up, down

    def __call__(self, x: mx.array, *, return_parts: bool = False):
        b, t, hd = x.shape
        n = b * t
        topk = self.cfg.num_experts_per_tok
        xf = x.reshape(n, hd)

        idx, weights = self.gate(xf)  # [n,topk] int32, [n,topk] fp32
        x_col = xf[:, :, None]  # [n, hidden, 1]
        m = n * topk
        exp = idx.reshape(-1)  # [m] expert per (token, slot) row
        tok = mx.repeat(mx.arange(n, dtype=mx.int32), topk)  # [m] token per row

        if self.sort_dispatch:
            order = mx.argsort(exp)
            inv = mx.argsort(order)
            exp, tok = exp[order], tok[order]
        srt = self.sort_dispatch

        g = mx.gather_mm(self.gate_stack, x_col, lhs_indices=exp, rhs_indices=tok, sorted_indices=srt)
        u = mx.gather_mm(self.up_stack, x_col, lhs_indices=exp, rhs_indices=tok, sorted_indices=srt)
        h = nn.silu(g) * u  # [m, inter, 1] (rows in dispatch order)
        d = mx.gather_mm(
            self.down_stack, h, lhs_indices=exp, rhs_indices=mx.arange(m, dtype=mx.int32),
            sorted_indices=srt,
        )
        d = d[:, :, 0]
        if self.sort_dispatch:
            d = d[inv]  # restore original (token, slot) order
        d = d.reshape(n, topk, hd)
        routed = mx.sum(d.astype(mx.float32) * weights[:, :, None], axis=1).astype(x.dtype)
        routed = routed.reshape(b, t, hd)
        shared = self.shared_experts(xf).reshape(b, t, hd)
        if return_parts:
            return routed + shared, routed, shared
        return routed + shared
