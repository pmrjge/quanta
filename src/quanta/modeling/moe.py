"""Sparse MoE block for Kimi-K2.6 (DeepSeek-V3 ``DeepseekV3MoE``), MLX-native.

* Router (``noaux_tc``, sigmoid): select top-k by ``sigmoid(logits) + correction_bias``
  but weight by the **raw** sigmoid scores (normalized, then ``* routed_scaling_factor``).
  With ``n_group == topk_group == 1`` the group machinery is a no-op (asserted).
* Dispatch: sparse top-k via :func:`mx.gather_mm` over stacked expert weights
  ``[E, out, in]`` (computes ``W @ x`` — no per-token weight materialization, no
  Python loop over experts). bf16 here for forward-path parity; the post-bake
  runtime swaps in :func:`mx.gather_qmm`.
* Output: ``Σ_topk w·expert(x) + shared(x)``.
* Long context: tokens are run through the experts in bounded chunks (``token_chunk``)
  so the routed intermediate ``[chunk·topk, hidden]`` (the dominant prefill transient)
  never scales with the full prompt; MoE is per-token independent ⇒ chunking is
  output-equivalent to processing all tokens at once.
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
        # Output-equivalent to the unsorted path (verified bit-identical in #11); the real
        # throughput win lands on the post-bake gather_qmm path (mlx PR #2078). Default-on per
        # the #133 optimization audit — bf16 reference passes still match unsorted to within
        # bf16 reorder ULPs.
        self.sort_dispatch = True
        # Run tokens through the experts in chunks of this many so the routed intermediate
        # ([chunk*topk, hidden]) stays bounded at long-context prefill. Per-token independent
        # ⇒ output-equivalent; a no-op (single chunk) when n <= token_chunk (decode / short
        # prefill), so decode latency and small-batch parity are unchanged.
        self.token_chunk = 8192

    def set_experts(self, gate: mx.array, up: mx.array, down: mx.array) -> None:
        self.gate_stack, self.up_stack, self.down_stack = gate, up, down

    def _routed_chunk(self, xc: mx.array, idx_c: mx.array, w_c: mx.array) -> mx.array:
        """Routed top-k expert output for one token chunk ``xc`` ``[nc, hidden]`` → ``[nc, hidden]``."""
        nc = xc.shape[0]
        topk = self.cfg.num_experts_per_tok
        x_col = xc[:, :, None]  # [nc, hidden, 1]
        mc = nc * topk
        exp = idx_c.reshape(-1)  # [mc] expert per (token, slot)
        tok = mx.repeat(mx.arange(nc, dtype=mx.int32), topk)  # [mc] LOCAL token index within the chunk
        srt = self.sort_dispatch
        if srt:
            order = mx.argsort(exp)
            inv = mx.argsort(order)
            exp, tok = exp[order], tok[order]
        g = mx.gather_mm(self.gate_stack, x_col, lhs_indices=exp, rhs_indices=tok, sorted_indices=srt)
        u = mx.gather_mm(self.up_stack, x_col, lhs_indices=exp, rhs_indices=tok, sorted_indices=srt)
        h = nn.silu(g) * u  # [mc, inter, 1]
        d = mx.gather_mm(
            self.down_stack, h, lhs_indices=exp, rhs_indices=mx.arange(mc, dtype=mx.int32),
            sorted_indices=srt,
        )
        d = d[:, :, 0]
        if srt:
            d = d[inv]  # restore (token, slot) order
        d = d.reshape(nc, topk, self.cfg.hidden_size)
        return mx.sum(d.astype(mx.float32) * w_c[:, :, None], axis=1).astype(xc.dtype)

    def __call__(self, x: mx.array, *, return_parts: bool = False):
        b, t, hd = x.shape
        n = b * t
        xf = x.reshape(n, hd)
        idx, weights = self.gate(xf)  # [n,topk] int32, [n,topk] fp32

        chunk = self.token_chunk if self.token_chunk and self.token_chunk > 0 else n
        multi = n > chunk  # only split (and eval per chunk) when it actually chunks
        routed_parts, shared_parts, out_parts = [], [], []
        for c0 in range(0, n, chunk):  # coarse chunked-prefill loop; experts stay vectorized
            c1 = min(c0 + chunk, n)
            xc = xf[c0:c1]
            rc = self._routed_chunk(xc, idx[c0:c1], weights[c0:c1])  # [nc, hidden]
            sc = self.shared_experts(xc)  # [nc, hidden]
            if return_parts:
                routed_parts.append(rc)
                shared_parts.append(sc)
                if multi:
                    mx.eval(rc, sc)
            else:
                oc = rc + sc
                out_parts.append(oc)
                if multi:
                    mx.eval(oc)  # bound peak: free this chunk's expert intermediates

        if return_parts:
            routed = (routed_parts[0] if not multi else mx.concatenate(routed_parts, axis=0)).reshape(b, t, hd)
            shared = (shared_parts[0] if not multi else mx.concatenate(shared_parts, axis=0)).reshape(b, t, hd)
            return routed + shared, routed, shared
        out = out_parts[0] if not multi else mx.concatenate(out_parts, axis=0)
        return out.reshape(b, t, hd)
