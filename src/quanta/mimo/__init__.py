"""MiMo-V2.5 (XiaomiMiMo/MiMo-V2.5) — multimodal sparse-MoE runtime (parity-first bootstrap).

DeepSeek-V3-style block-fp8 source; hybrid full/SWA attention with fused qkv, partial RoPE, and
SWA attention sinks; 256-expert top-8 MoE (no shared expert); native MTP; Qwen2.5-VL vision tower
+ speech/audio encoder; 1M context. See :mod:`quanta.mimo.config` and :mod:`quanta.mimo.fp8`.
"""

from __future__ import annotations

from quanta.mimo.config import MiMoV2Config

__all__ = ["MiMoV2Config"]
