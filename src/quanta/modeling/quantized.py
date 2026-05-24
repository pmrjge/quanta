"""Resident quantized runtime modules — decode from packed weights, no dequant-to-bf16.

``QuantizedSparseMoE`` mirrors :class:`SparseMoE` but routes through ``mx.gather_qmm`` over
packed expert stacks (the post-bake decode path). ``gather_qmm`` takes one ``bits`` per call,
and the bake's DP allocates width **per (expert, projection)** — one expert can carry
``gate=int3`` but ``down=int4`` (the canonical "gate/up int3 + down int4" is itself
per-projection). So weights are stored as one stack per ``(projection, width)``; each expert
carries, **per projection**, its width and its local slot within that width's stack. At runtime
each projection dispatches its own widths: every present width is gathered (wrong-width rows use
a harmless dummy slot 0), then the per-row result is selected by that row-expert's width for the
projection. Router and shared expert stay bf16 (always-on path). Output-equivalent to the bf16
path on the same (dequantized) weights.

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


class QuantizedSparseMoE(nn.Module):
    def __init__(self, cfg: KimiTextConfig, group_size: int = 128) -> None:
        super().__init__()
        self.cfg = cfg
        self.group_size = group_size
        self.gate = MoEGate(cfg)
        self.shared_experts = DenseMLP(cfg, intermediate_size=cfg.moe_intermediate_size * cfg.n_shared_experts)
        self._stacks: dict[str, dict[int, dict[str, mx.array]]] = {}  # proj -> bits -> {packed,scale,bias}
        self._pbits: dict[str, mx.array] = {}  # proj -> [E] each expert's width for this projection
        self._pslot: dict[str, mx.array] = {}  # proj -> [E] each expert's slot in its width's stack
        self._rmap: dict[str, dict[int, mx.array]] = {}  # proj -> bits -> [E] slot (0 for other widths)

    def set_experts(self, stacks: dict[str, dict[int, dict[str, mx.array]]],
                    pbits: dict[str, mx.array], pslot: dict[str, mx.array]) -> None:
        """``stacks[proj][bits]``: packed expert stack for that projection+width. ``pbits``/``pslot``:
        per projection, each expert's width and its local slot in that width's stack. Bits are per
        (expert, projection) — the DP can give one expert ``gate=int3`` but ``down=int4`` — so each
        projection dispatches its own widths independently."""
        self._stacks = stacks
        self._pbits = {p: pbits[p].astype(mx.int32) for p in pbits}
        self._pslot = {p: pslot[p].astype(mx.int32) for p in pslot}
        zero = mx.array(0, mx.int32)
        self._rmap = {p: {bits: mx.where(self._pbits[p] == bits, self._pslot[p], zero) for bits in stacks[p]}
                      for p in stacks}

    def _proj_out(self, x: mx.array, proj: str, exp: mx.array, lhs: mx.array) -> mx.array:
        """One projection's output over routed rows ``[m, out]``: ``gather_qmm`` per present width,
        select each row by its expert's width for this projection. ``x`` is gathered by ``lhs``
        (token rows for gate/up; identity rows for down on the already-routed hidden)."""
        pb = self._pbits[proj][exp]  # [m] this projection's width for each routed row's expert
        out: mx.array | None = None
        for bits in sorted(self._stacks[proj]):
            s = self._stacks[proj][bits]
            rhs = self._rmap[proj][bits][exp]  # local slot in this width's stack (dummy 0 for others)
            o = mx.gather_qmm(  # x as batched [.,1,in] row-matrices, gathered by lhs; w by rhs
                x[:, None, :], s["packed"], s["scale"], s["bias"],
                lhs_indices=lhs, rhs_indices=rhs, transpose=True, group_size=self.group_size, bits=bits,
            )[:, 0, :]
            out = o if out is None else mx.where((pb == bits)[:, None], o, out)
        return out

    def __call__(self, x: mx.array) -> mx.array:
        b, t, hd = x.shape
        n = b * t
        topk = self.cfg.num_experts_per_tok
        xf = x.reshape(n, hd)
        idx, weights = self.gate(xf)  # [n,topk]
        exp = idx.reshape(-1)  # [m]
        tok = mx.repeat(mx.arange(n, dtype=mx.int32), topk)
        rows = mx.arange(n * topk, dtype=mx.int32)
        g = self._proj_out(xf, "gate", exp, tok)
        u = self._proj_out(xf, "up", exp, tok)
        d = self._proj_out(nn.silu(g) * u, "down", exp, rows)  # [m, hidden]
        d_out = d.reshape(n, topk, hd)
        routed = mx.sum(d_out.astype(mx.float32) * weights[:, :, None], axis=1).astype(x.dtype)
        return routed.reshape(b, t, hd) + self.shared_experts(xf).reshape(b, t, hd)
