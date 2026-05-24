"""DeepSeek-V4-Flash (deepseek_v4 / DeepseekV4ForCausalLM) — sparse-MoE runtime (parity-first).

A research-grade architecture: **Hyper-Connections** (4x Sinkhorn-mixed residual stream), per-layer
**KV compression** + a **Lightning Indexer** (DeepSeek Sparse Attention) over sliding-window
attention with per-head sinks, **hash + sqrtsoftplus** MoE routing (256 experts top-6 + 1 shared),
grouped low-rank attention, and a native MTP head. Source weights are **fp8-e4m3** (non-experts)
and **fp4-e2m1** (experts), both with **e8m0/MX** scales. See :mod:`quanta.dsv4.config` and
:mod:`quanta.dsv4.fp`.
"""

from __future__ import annotations

from quanta.dsv4.config import DeepSeekV4Config

__all__ = ["DeepSeekV4Config"]
