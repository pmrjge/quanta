"""Routed sparse MoE for MiMo-V2.5 (the 47 MoE layers L1-47), MLX-native.

Wiring (verified against the checkpoint + ``MiMoV2MoE``):

* Router (``noaux_tc`` sigmoid, top-8 of 256) on the hidden state: select experts by
  ``sigmoid(logits) + e_score_correction_bias``, weight by the **raw** sigmoid (normalized, then
  ``* routed_scaling_factor`` = 1.0). ``n_group == topk_group == 1`` ⇒ no group machinery.
* Each routed expert is a SwiGLU MLP on the hidden state: ``down(silu(gate(x)) * up(x))``
  (``moe_intermediate_size`` = 2048). **No shared expert** (``n_shared_experts`` is null in V2.5),
  so the layer output is the routed sum alone.

Dispatch is sparse ``mx.gather_mm`` over stacked ``[E,*]`` weights (no per-token weight
materialization, no python loop over experts; rule-3); token-chunked for bounded long-context
prefill. bf16 here for parity vs ``MiMoV2MoE``; the baked runtime swaps in ``mx.gather_qmm`` (#62).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.mimo.config import MiMoV2Config


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class MiMoMoE(nn.Module):
    def __init__(self, cfg: MiMoV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        e, h, inter = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.gate_weight = mx.zeros((e, h))            # router weight [E, hidden]
        self.e_score_correction_bias = mx.zeros((e,))
        self.gate_stack = mx.zeros((e, inter, h))      # expert gate_proj [E, inter, hidden]
        self.up_stack = mx.zeros((e, inter, h))        # expert up_proj
        self.down_stack = mx.zeros((e, h, inter))      # expert down_proj [E, hidden, inter]
        self.token_chunk = 8192

    def set_experts(self, gate: mx.array, up: mx.array, down: mx.array) -> None:
        self.gate_stack, self.up_stack, self.down_stack = gate, up, down

    def _route(self, xf: mx.array) -> tuple[mx.array, mx.array]:
        assert self.cfg.n_group == 1 and self.cfg.topk_group == 1, "group routing unsupported"
        topk = self.cfg.num_experts_per_tok
        logits = xf.astype(mx.float32) @ self.gate_weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        choice = scores + self.e_score_correction_bias.astype(mx.float32)[None]
        idx = mx.argpartition(-choice, kth=topk - 1, axis=-1)[:, :topk].astype(mx.int32)
        w = mx.take_along_axis(scores, idx, axis=-1)
        if topk > 1 and self.cfg.norm_topk_prob:
            w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
        return idx, w * self.cfg.routed_scaling_factor

    def _routed_chunk(self, xc: mx.array, idx_c: mx.array, w_c: mx.array) -> mx.array:
        """Top-k routed SwiGLU output for a hidden chunk ``[nc, hidden]`` → ``[nc, hidden]``."""
        nc, h = xc.shape[0], self.cfg.hidden_size
        topk = self.cfg.num_experts_per_tok
        col = xc[:, :, None]                                   # [nc, hidden, 1]
        mc = nc * topk
        exp = idx_c.reshape(-1)
        tok = mx.repeat(mx.arange(nc, dtype=mx.int32), topk)
        g = mx.gather_mm(self.gate_stack, col, lhs_indices=exp, rhs_indices=tok)  # [mc, inter, 1]
        u = mx.gather_mm(self.up_stack, col, lhs_indices=exp, rhs_indices=tok)
        a = silu(g) * u                                        # [mc, inter, 1]
        d = mx.gather_mm(self.down_stack, a, lhs_indices=exp, rhs_indices=mx.arange(mc, dtype=mx.int32))
        d = d[:, :, 0].reshape(nc, topk, h)                    # [nc, topk, hidden]
        return mx.sum(d.astype(mx.float32) * w_c[:, :, None], axis=1).astype(xc.dtype)

    def __call__(self, x: mx.array) -> mx.array:
        b, t, hd = x.shape
        n = b * t
        xf = x.reshape(n, hd)
        idx, w = self._route(xf)
        chunk = self.token_chunk if self.token_chunk and self.token_chunk > 0 else n
        multi = n > chunk
        parts = []
        for c0 in range(0, n, chunk):  # bounded chunked-prefill loop; experts stay vectorized
            c1 = min(c0 + chunk, n)
            rc = self._routed_chunk(xf[c0:c1], idx[c0:c1], w[c0:c1])
            parts.append(rc)
            if multi:
                mx.eval(rc)
        out = parts[0] if not multi else mx.concatenate(parts, axis=0)
        return out.reshape(b, t, hd)
