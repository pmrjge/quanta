"""Nemotron-H hierarchical-routing data capture.
Phase 1 of the hierarchical MoE routing design (``docs/hierarchical_moe_routing.md``).

Purpose: produce, for every MoE layer (Nemotron has no hash layers), tuples
``(hidden_input, top_k_selection)`` for offline meta-router training. The
meta-router pre-selects a ``K_meta``-subset of experts; the existing
:func:`quanta.nemotron.moe.NemotronLatentMoE._route` picks its top-22 inside
the subset.

This wraps :func:`quanta.nemotron.calibrate.capture_calibration`, but with one
substitution: the existing calibration captures the **latent**
``fc1_latent_proj(x)`` (because Nemotron's experts read latents, and AWQ is
calibrated on latent inputs). The meta-router, however, reads **hidden**
(it pre-selects experts *before* the latent projection — gating on latent is
not the right signal: latent is shared across experts and would force the
meta-router to learn fc1's inverse). We therefore re-run the streamed forward
inline and capture ``x [N, hidden_size]`` (the input to the post-attention
RMSNorm + router) instead.

The capture is loader-agnostic (``capture_fn`` is injected at the call site,
keeping this module free of ``transformers``/safetensors imports).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from quanta.nemotron.config import NemotronHConfig


def capture_routing(
    capture_fn,
    ck,
    cfg: NemotronHConfig,
    input_ids: mx.array,
    output_dir: str | Path,
    *,
    n_layers: int | None = None,
) -> dict[int, str]:
    """Capture routing data for Nemotron's MoE layers.

    ``capture_fn`` must return, per MoE layer ``i``, a tuple
    ``(hidden [N, hidden_size] bf16, idx [N, topk] int32)`` — i.e. the
    **hidden** input (the routing signal) and the top-k selection. The
    Phase 2 orchestrator builds this from a wrapped version of
    :func:`quanta.nemotron.calibrate.capture_calibration` that swaps the
    latent capture for a hidden capture (one extra reshape — no extra forward
    work). The injection here keeps this module loader-agnostic.

    Returns ``{layer_id: shard_path}`` for layers that were written.

    Each shard is an ``npz`` with the same keys as
    :func:`quanta.dsv4.routing_capture.capture_routing`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = cfg.num_hidden_layers if n_layers is None else n_layers
    if not 1 <= n <= cfg.num_hidden_layers:
        raise ValueError(f"n_layers={n} out of range [1, {cfg.num_hidden_layers}]")

    caps = capture_fn(ck, cfg, input_ids, n_layers=n)

    written: dict[int, str] = {}
    for layer_id, (x_arr, idx_arr) in caps.items():
        # Validate shapes loudly (rule-6).
        if x_arr.ndim != 2 or x_arr.shape[1] != cfg.hidden_size:
            raise ValueError(
                f"layer {layer_id} x shape {x_arr.shape} != [N, {cfg.hidden_size}] "
                f"(must capture HIDDEN not LATENT — see module docstring)"
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
        if int(idx_np.min()) < 0 or int(idx_np.max()) >= cfg.n_routed_experts:
            raise ValueError(
                f"layer {layer_id} idx out of range [0, {cfg.n_routed_experts}): "
                f"min={int(idx_np.min())}, max={int(idx_np.max())}"
            )

        x_np = np.asarray(x_arr.astype(mx.float16))
        shard_path = output_dir / f"nemotron_routing_L{layer_id:03d}.npz"
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
