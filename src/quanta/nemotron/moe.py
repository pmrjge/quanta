"""Latent relu^2 sparse MoE for Nemotron-H (the 40 ``E`` layers), MLX-native.

Wiring (verified against the checkpoint shapes):

* Router (``noaux_tc`` sigmoid, top-22 of 512) on the **hidden** state: select by
  ``sigmoid(logits) + correction_bias``, weight by the **raw** sigmoid (normalized, then
  ``* routed_scaling_factor``). ``n_group == topk_group == 1`` ⇒ no group machinery.
* Experts on a low-rank **latent**: ``hidden --fc1--> latent(1024)``; each routed expert is
  ``down(relu^2(up(latent)))`` (2 matrices, no SwiGLU gate — relu^2 is Nemotron's activation);
  combine the top-k in latent, then ``--fc2--> hidden``.
* Shared expert (always-on) is a relu^2 MLP on the **hidden** state.

Dispatch is sparse ``mx.gather_mm`` over stacked ``[E,*]`` weights (no per-token weight
materialization, no python loop over experts); token-chunked for bounded long-context
prefill. bf16 here for parity; the baked runtime swaps in ``mx.gather_qmm``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.nemotron.config import NemotronHConfig


def relu2(x: mx.array) -> mx.array:
    """Squared ReLU (Nemotron ``relu2`` activation)."""
    return mx.square(mx.maximum(x, 0))


class NemotronLatentMoE(nn.Module):
    def __init__(self, cfg: NemotronHConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h, lat, inter = cfg.hidden_size, cfg.moe_latent_size, cfg.moe_intermediate_size
        si, e = cfg.moe_shared_expert_intermediate_size, cfg.n_routed_experts
        self.gate_weight = mx.zeros((e, h))
        self.e_score_correction_bias = mx.zeros((e,))
        self.fc1_latent_proj = nn.Linear(h, lat, bias=False)
        self.fc2_latent_proj = nn.Linear(lat, h, bias=False)
        self.up_stack = mx.zeros((e, inter, lat))    # [E, inter, latent]
        self.down_stack = mx.zeros((e, lat, inter))  # [E, latent, inter]
        self.shared_up = nn.Linear(h, si, bias=False)
        self.shared_down = nn.Linear(si, h, bias=False)
        self.token_chunk = 8192

    def set_experts(self, up: mx.array, down: mx.array) -> None:
        self.up_stack, self.down_stack = up, down

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

    def _routed_chunk(self, lat_c: mx.array, idx_c: mx.array, w_c: mx.array) -> mx.array:
        """Top-k routed expert output for a latent chunk ``[nc, latent]`` → ``[nc, latent]``."""
        nc = lat_c.shape[0]
        topk, lat = self.cfg.num_experts_per_tok, self.cfg.moe_latent_size
        col = lat_c[:, :, None]  # [nc, latent, 1]
        mc = nc * topk
        exp = idx_c.reshape(-1)
        tok = mx.repeat(mx.arange(nc, dtype=mx.int32), topk)
        up = mx.gather_mm(self.up_stack, col, lhs_indices=exp, rhs_indices=tok)  # [mc, inter, 1]
        h = relu2(up)
        d = mx.gather_mm(self.down_stack, h, lhs_indices=exp, rhs_indices=mx.arange(mc, dtype=mx.int32))
        d = d[:, :, 0].reshape(nc, topk, lat)
        return mx.sum(d.astype(mx.float32) * w_c[:, :, None], axis=1).astype(lat_c.dtype)

    def __call__(self, x: mx.array) -> mx.array:
        b, t, hd = x.shape
        n = b * t
        xf = x.reshape(n, hd)
        idx, w = self._route(xf)
        lat = self.fc1_latent_proj(xf)
        chunk = self.token_chunk if self.token_chunk and self.token_chunk > 0 else n
        multi = n > chunk
        parts = []
        for c0 in range(0, n, chunk):  # bounded chunked-prefill loop; experts stay vectorized
            c1 = min(c0 + chunk, n)
            rc = self._routed_chunk(lat[c0:c1], idx[c0:c1], w[c0:c1])
            parts.append(rc)
            if multi:
                mx.eval(rc)
        routed_lat = parts[0] if not multi else mx.concatenate(parts, axis=0)
        routed = self.fc2_latent_proj(routed_lat)
        shared = self.shared_down(relu2(self.shared_up(xf)))
        return (routed + shared).reshape(b, t, hd)
