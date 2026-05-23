"""Resident quantized runtime modules — decode from packed weights, no dequant-to-bf16.

``QuantizedSparseMoE`` mirrors :class:`SparseMoE` but routes through ``mx.gather_qmm`` over
packed expert stacks (the post-bake decode path). ``gather_qmm`` takes one ``bits`` per call,
so the DP's **dynamic mixed int3/int4** layer is stored as one stack per width; each global
expert carries its width and its local slot within that width's stack. At runtime every
present width is gathered (with per-width remapped indices — wrong-width rows use a harmless
dummy slot 0), then the per-(token,slot) result is selected by the expert's width. Router and
shared expert stay bf16 (always-on path). Output-equivalent to the bf16 path on the same
(dequantized) weights.

(Both widths are computed over all routed rows then selected — exact and simple; at decode
``m = topk`` so the ~2× is negligible. A row-partition variant computing each row once is the
prefill throughput refinement.)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.config import KimiTextConfig
from quanta.modeling.mlp import DenseMLP
from quanta.modeling.moe import MoEGate

_PROJ = ("gate", "up", "down")


class QuantizedSparseMoE(nn.Module):
    def __init__(self, cfg: KimiTextConfig, group_size: int = 128) -> None:
        super().__init__()
        self.cfg = cfg
        self.group_size = group_size
        self.gate = MoEGate(cfg)
        self.shared_experts = DenseMLP(cfg, intermediate_size=cfg.moe_intermediate_size * cfg.n_shared_experts)
        self._stacks: dict[int, dict[str, mx.array]] = {}  # bits -> {proj_packed/_scale/_bias}
        self._rmap: dict[int, mx.array] = {}  # bits -> [E] local slot (0 for other-width experts)
        self.expert_bits: mx.array | None = None  # [E]

    def set_experts(self, stacks: dict[int, dict[str, mx.array]], expert_bits: mx.array, slots: mx.array) -> None:
        """``stacks``: per-width packed stacks. ``expert_bits``/``slots``: each expert's width and
        its local index within that width's stack."""
        self._stacks = stacks
        self.expert_bits = expert_bits.astype(mx.int32)
        zero = mx.array(0, mx.int32)
        self._rmap = {bits: mx.where(self.expert_bits == bits, slots.astype(mx.int32), zero) for bits in stacks}

    def _qmm(self, x: mx.array, bits: int, proj: str, rhs: mx.array, lhs: mx.array) -> mx.array:
        s = self._stacks[bits]
        out = mx.gather_qmm(  # x as batched [.,1,in] row-matrices, gathered by lhs; w by rhs
            x[:, None, :], s[f"{proj}_packed"], s[f"{proj}_scale"], s[f"{proj}_bias"],
            lhs_indices=lhs, rhs_indices=rhs, transpose=True, group_size=self.group_size, bits=bits,
        )
        return out[:, 0, :]

    def __call__(self, x: mx.array) -> mx.array:
        b, t, hd = x.shape
        n = b * t
        topk = self.cfg.num_experts_per_tok
        xf = x.reshape(n, hd)
        idx, weights = self.gate(xf)  # [n,topk]
        exp = idx.reshape(-1)  # [m]
        tok = mx.repeat(mx.arange(n, dtype=mx.int32), topk)
        m = n * topk
        rows = mx.arange(m, dtype=mx.int32)

        d_out: mx.array | None = None
        wexp = self.expert_bits[exp]  # [m] width of each routed slot's expert
        for bits in sorted(self._stacks):
            rhs = self._rmap[bits][exp]  # local slot in this width's stack (dummy 0 for others)
            g = self._qmm(xf, bits, "gate", rhs, tok)
            u = self._qmm(xf, bits, "up", rhs, tok)
            h = nn.silu(g) * u
            d = self._qmm(h, bits, "down", rhs, rows)  # [m, hidden]
            d_out = d if d_out is None else mx.where((wexp == bits)[:, None], d, d_out)

        d_out = d_out.reshape(n, topk, hd)
        routed = mx.sum(d_out.astype(mx.float32) * weights[:, :, None], axis=1).astype(x.dtype)
        return routed.reshape(b, t, hd) + self.shared_experts(xf).reshape(b, t, hd)
