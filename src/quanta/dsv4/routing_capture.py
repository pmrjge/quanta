"""DSV4 hierarchical-routing data capture.
Phase 1 of the hierarchical MoE routing design (``docs/hierarchical_moe_routing.md``).

Purpose: produce, for the **score** (non-hash) MoE layers, `(hidden_input,
top_k_selection)` pairs suitable for offline training of a *meta-router* —
a cheap sigmoid-linear that pre-selects a `K_meta`-subset of experts before
:func:`quanta.dsv4.moe.dsv4_route` makes its top-k pick within the subset.

This is a thin wrapper over :func:`quanta.dsv4.calibrate.capture_calibration`
(which already records `(x [N, hidden] bf16, idx [N, topk] int32)` per layer
for AWQ). The wrapper:

* selects only **score** layers (skipping ``cfg.is_hash(layer_id)`` layers —
  the hash table already enumerates ``topk`` experts deterministically; no
  meta-router needed);
* validates shapes loudly (rule-6: no silent failure);
* writes each layer to a separate ``.npz`` shard under ``output_dir`` for
  layer-local meta-router training.

Memory-disciplined: re-uses the streamed one-layer-resident capture (rule-8).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from quanta.dsv4.config import DeepSeekV4Config


def capture_routing(
    capture_fn,
    ck,
    cfg: DeepSeekV4Config,
    input_ids: mx.array,
    output_dir: str | Path,
    *,
    n_layers: int | None = None,
) -> dict[int, str]:
    """Capture routing data for DSV4's **score** MoE layers and write per-layer shards.

    ``capture_fn`` is :func:`quanta.dsv4.calibrate.capture_calibration` (passed
    in to keep this module ``transformers``-free / loader-agnostic at import).
    ``ck`` is the source checkpoint reader the orchestrator (Phase 2) passes
    in. ``input_ids`` is the ``[S]`` (or ``[1,S]``) calibration token-id
    sequence.

    Returns ``{layer_id: shard_path}`` for the score layers that were written.

    Each shard is an ``npz`` with keys:
        ``x``         : ``[N, hidden]`` bfloat16 (cast to float16 for storage —
                        bf16 is not natively numpy-storable). Caller's training
                        loop casts back via ``mx.array(x).astype(mx.bfloat16)``.
        ``idx``       : ``[N, topk]`` int32, the existing top-k selection
                        — the supervision signal for meta-router BCE.
        ``layer_id``  : int32 scalar, recorded for sanity.
        ``hash_skip`` : int32 scalar; 0 for score layers (Always; hash layers
                        are skipped before write).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = cfg.num_hidden_layers if n_layers is None else n_layers
    if not 1 <= n <= cfg.num_hidden_layers:
        raise ValueError(f"n_layers={n} out of range [1, {cfg.num_hidden_layers}]")

    caps = capture_fn(ck, cfg, input_ids, n_layers=n)

    written: dict[int, str] = {}
    for layer_id, (x_arr, idx_arr) in caps.items():
        if cfg.is_hash(layer_id):
            # Hash layers route by fixed tid2eid table — meta-router would be a
            # no-op identity. Skip the write so the trainer doesn't see them.
            continue
        # Validate shapes loudly (rule-6: never silently mis-store)
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
        # validate idx range (loud — catches a routing bug in capture)
        idx_np = np.asarray(idx_arr).astype(np.int32)
        if int(idx_np.min()) < 0 or int(idx_np.max()) >= cfg.n_routed_experts:
            raise ValueError(
                f"layer {layer_id} idx out of range [0, {cfg.n_routed_experts}): "
                f"min={int(idx_np.min())}, max={int(idx_np.max())}"
            )

        # Store x as float16 (numpy lacks bf16); the training loader recasts.
        # Precision drop is fine for routing supervision (only used for the
        # meta-router's linear; the existing top-k labels are exact).
        x_np = np.asarray(x_arr.astype(mx.float16))
        shard_path = output_dir / f"dsv4_routing_L{layer_id:03d}.npz"
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
    """Read a routing shard back as ``(x [N, hidden] bf16, idx [N, topk] int32, layer_id int)``.

    Convenience for the Phase 2 trainer; symmetric with :func:`capture_routing`.
    """
    data = np.load(path)
    x = mx.array(data["x"]).astype(mx.bfloat16)
    idx = mx.array(data["idx"]).astype(mx.int32)
    layer_id = int(data["layer_id"])
    return x, idx, layer_id
