"""Oracle gate: MLX dense attention == the authors' real Attention (inference/model.py on CPU).

Upgrades the #70 gate from "MLX vs my numpy transcription" to "MLX vs the authors' actual code" via
the torch CPU reference (:mod:`parity.dsv4_torch_ref`). Same (bit-exact, per #67) weights feed both
sides, so any divergence is a forward-math bug. Layer 0 = pure sliding-window; tested causal and
window-crossing.

    uv run --with torch --with safetensors --with numpy python -m parity.dsv4_attn_oracle_test
"""

from __future__ import annotations

import numpy as np
import torch

import mlx.core as mx

from quanta.dsv4 import attention as A
from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.loader import DeepSeekV4SourceCheckpoint
from parity import dsv4_torch_ref as ref

ART = "/Users/pmrj/models/DeepSeek-V4-Flash"


def run() -> None:
    cfg = DeepSeekV4Config.from_pretrained(ART)
    M, args = ref.load_model_module(cfg, max_seq_len=256)
    attn_ref = ref.load_attention(M, args, cfg, 0)

    ck = DeepSeekV4SourceCheckpoint(ART, cfg)
    p_f32 = {k: v.astype(mx.float32) for k, v in ck.attention(0).items()}
    ck.release()

    rng = np.random.default_rng(0)
    ok = True
    for t in (16, 160):
        x = (rng.standard_normal((1, t, cfg.hidden_size)) * 0.5).astype(np.float32)
        with torch.no_grad():
            o_ref = attn_ref(torch.from_numpy(x), 0).numpy().astype(np.float64)
        o_mx = np.array(A.attention_dense(mx.array(x), p_f32, cfg, 0).astype(mx.float32)).astype(np.float64)
        d = float(np.max(np.abs(o_ref - o_mx)))
        rel = d / float(np.max(np.abs(o_ref)))
        good = o_mx.shape == o_ref.shape and rel < 1e-3
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] T={t:3d} ({'causal' if t <= cfg.sliding_window else 'windowed'})"
              f"  vs authors' Attention  |Δ|={d:.2e} rel={rel:.2e} absmax={float(np.abs(o_ref).max()):.3f}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
