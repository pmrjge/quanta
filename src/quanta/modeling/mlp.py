"""Dense SwiGLU MLP (DeepSeek-V3 ``DeepseekV3MLP``) — used by the L0 dense layer."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.config import KimiTextConfig


class DenseMLP(nn.Module):
    def __init__(self, cfg: KimiTextConfig, intermediate_size: int | None = None) -> None:
        super().__init__()
        inter = cfg.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(cfg.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
