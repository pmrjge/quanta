"""Nemotron-H AWQ calibration: per-MoE-layer latent activations + routing for the bake.

Mirrors the Kimi calibration (:mod:`quanta.bake.calibrate`) on Nemotron's hybrid stack. A
streamed, one-layer-resident forward advances the residual through every block
(mamba/attention/moe); at each MoE layer it records the **latent** ``fc1_latent_proj(norm(x))``
``[N, latent]`` (the routed experts' input) and the routing ``idx`` ``[N, topk]``. Per expert,
AWQ then calibrates ``up`` on its routed latent rows and ``down`` on ``relu2(up·latent)`` of
those rows (see :mod:`quanta.nemotron.bake`). Memory-disciplined: one block resident at a time;
experts are loaded only to advance the stream (rule-8).
"""

from __future__ import annotations

import mlx.core as mx

from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.loader import NemotronSourceCheckpoint
from quanta.nemotron.model import NemotronBlock, load_block

EMBED = "backbone.embeddings.weight"


def capture_calibration(
    ck: NemotronSourceCheckpoint, cfg: NemotronHConfig, token_ids: mx.array,
    *, n_layers: int | None = None,
) -> dict[int, tuple[mx.array, mx.array]]:
    """Per-MoE-layer ``{i: (latent [N,lat] bf16, idx [N,topk] int32)}`` for AWQ calibration."""
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    h = ck.read(EMBED)[token_ids][None].astype(mx.bfloat16)
    mx.eval(h)
    ck.release()
    caps: dict[int, tuple[mx.array, mx.array]] = {}
    for i in range(n):
        block = NemotronBlock(cfg, cfg.layer_kind(i))
        load_block(block, ck, cfg, i)
        if block.kind == "moe":  # capture the experts' input (latent) + routing before advancing
            hf = block.norm(h).reshape(-1, cfg.hidden_size)
            idx, _ = block.mixer._route(hf)
            latent = block.mixer.fc1_latent_proj(hf)
            mx.eval(latent, idx)
            caps[i] = (latent.astype(mx.bfloat16), idx)
        if i < n - 1:  # advance the residual to feed the next layer (stateless prefill)
            h, _, _ = block(h)
            mx.eval(h)
        del block
        ck.release()
        mx.clear_cache()
    return caps
