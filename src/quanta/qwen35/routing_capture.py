"""Qwen3.5 hierarchical-routing data capture.
Phase 1 of the hierarchical MoE routing design (``docs/hierarchical_moe_routing.md``).

Purpose: produce, for every MoE layer (Qwen3.5 has MoE on every layer, no
hash layers), tuples ``(hidden_input, top_k_selection)`` for offline meta-
router training. The meta-router pre-selects a ``K_meta``-subset of the 512
experts; the existing :func:`quanta.qwen35.moe.qwen35_route` picks its top-10
inside the subset.

This is a thin wrapper over :func:`quanta.qwen35.calibrate.capture_calibration`
(which already captures ``(x [N, hidden] bf16, idx [N, topk] int32)``). The
capture is the post-attention-norm hidden state — *exactly* the input the
meta-router will see at inference time. No re-derivation needed.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from quanta.qwen35.config import Qwen35Config


def capture_routing(
    capture_fn,
    ck,
    cfg: Qwen35Config,
    input_ids: mx.array,
    output_dir: str | Path,
    *,
    n_layers: int | None = None,
) -> dict[int, str]:
    """Capture routing data for Qwen3.5's MoE layers (every decoder layer is MoE).

    ``capture_fn`` is :func:`quanta.qwen35.calibrate.capture_calibration`
    (passed in to keep this module loader-agnostic / ``transformers``-free).
    Returns ``{layer_id: shard_path}``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = cfg.num_hidden_layers if n_layers is None else n_layers
    if not 1 <= n <= cfg.num_hidden_layers:
        raise ValueError(f"n_layers={n} out of range [1, {cfg.num_hidden_layers}]")

    caps = capture_fn(ck, cfg, input_ids, n_layers=n)

    written: dict[int, str] = {}
    for layer_id, (x_arr, idx_arr) in caps.items():
        if x_arr.ndim != 2 or x_arr.shape[1] != cfg.hidden_size:
            raise ValueError(
                f"layer {layer_id} x shape {x_arr.shape} != [N, {cfg.hidden_size}]"
            )
        if idx_arr.ndim != 2 or idx_arr.shape[1] != cfg.num_experts_per_tok:
            raise ValueError(
                f"layer {layer_id} idx shape {idx_arr.shape} != "
                f"[N, {cfg.num_experts_per_tok}]"
            )
        if x_arr.shape[0] != idx_arr.shape[0]:
            raise ValueError(
                f"layer {layer_id} N mismatch: x={x_arr.shape[0]} idx={idx_arr.shape[0]}"
            )
        idx_np = np.asarray(idx_arr).astype(np.int32)
        if int(idx_np.min()) < 0 or int(idx_np.max()) >= cfg.num_experts:
            raise ValueError(
                f"layer {layer_id} idx out of range [0, {cfg.num_experts}): "
                f"min={int(idx_np.min())}, max={int(idx_np.max())}"
            )

        x_np = np.asarray(x_arr.astype(mx.float16))
        shard_path = output_dir / f"qwen35_routing_L{layer_id:03d}.npz"
        np.savez(
            shard_path,
            x=x_np,
            idx=idx_np,
            layer_id=np.int32(layer_id),
            hash_skip=np.int32(0),
        )
        written[layer_id] = str(shard_path)
    return written


def load_routing_shard(path: str | Path) -> tuple[mx.array, mx.array, int]:
    """Read a routing shard back as ``(x [N, hidden] bf16, idx [N, topk] int32, layer_id int)``."""
    data = np.load(path)
    x = mx.array(data["x"]).astype(mx.bfloat16)
    idx = mx.array(data["idx"]).astype(mx.int32)
    layer_id = int(data["layer_id"])
    return x, idx, layer_id
