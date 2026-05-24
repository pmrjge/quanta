"""Oracle gate: MLX MoE == the authors' real Gate + Expert (inference/model.py on CPU).

Validates :func:`quanta.dsv4.moe.dsv4_moe` against the authors' MoE forward (sqrtsoftplus routing,
hash vs scored selection, bias-free normalized weights, swiglu-limit clamp, shared expert) for a hash
layer (L0) and a scored layer (L3). The reference loads only routed (hit) experts on demand; the MLX
side runs the full gather_mm dispatch over all 256 expert stacks. Checks both the routing (selected
expert set) and the final output.

    uv run --with torch --with safetensors --with numpy python -m parity.dsv4_moe_test
"""

from __future__ import annotations

import numpy as np
import torch

import mlx.core as mx

from quanta.dsv4 import moe as MOE
from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.loader import DeepSeekV4SourceCheckpoint
from parity import dsv4_torch_ref as ref

ART = "/Users/pmrj/models/DeepSeek-V4-Flash"


def _f32(d):
    return {k: (_f32(v) if isinstance(v, dict) else v.astype(mx.float32)) for k, v in d.items()}


def run() -> None:
    cfg = DeepSeekV4Config.from_pretrained(ART)
    M, args = ref.load_model_module(cfg, max_seq_len=64)
    ck = DeepSeekV4SourceCheckpoint(ART, cfg)
    rng = np.random.default_rng(0)
    n, dim = 16, cfg.hidden_size
    x = (rng.standard_normal((1, n, dim)) * 0.5).astype(np.float32)
    ids = rng.integers(0, cfg.vocab_size, size=(1, n)).astype(np.int64)
    ok = True

    for L in (0, 3):                                            # hash ; scored
        router = _f32(ck.moe_router(L))
        experts = _f32(ck.expert_stacks(L))                    # full [256,*] stacks (f32)
        shared = _f32(ck.shared_expert(L))
        ck.release()
        w_ref, idx_ref, y_ref = ref.moe_reference(
            M, args, cfg, L, torch.from_numpy(x.reshape(n, dim)), torch.from_numpy(ids.reshape(n)))
        y_mx = MOE.dsv4_moe(mx.array(x), router, experts, shared, cfg, L, mx.array(ids))
        y = np.array(y_mx.reshape(n, dim).astype(mx.float32)).astype(np.float64)
        rel = float(np.max(np.abs(y_ref.numpy().astype(np.float64) - y))) / float(np.max(np.abs(y_ref.numpy())))

        # routing: selected expert set per token must match
        idx_mx, _ = MOE.dsv4_route(mx.array(x).reshape(n, dim).astype(mx.float32), router, cfg, L, mx.array(ids))
        idx_mx = np.array(idx_mx)
        idx_rf = idx_ref.numpy()
        set_ok = all(set(idx_mx[t]) == set(idx_rf[t]) for t in range(n))

        good = y.shape == (n, dim) and rel < 1e-3 and set_ok
        ok = ok and good
        kind = "hash " if cfg.is_hash(L) else "score"
        print(f"  [{'OK' if good else 'FAIL'}] L{L} ({kind}) routing_set_match={set_ok}  "
              f"output rel={rel:.2e} absmax={float(np.abs(y_ref.numpy()).max()):.3f}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
