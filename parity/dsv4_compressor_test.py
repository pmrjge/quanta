"""Oracle gate: MLX KV Compressor (prefill) == the authors' real Compressor (model.py on CPU).

Gates :func:`quanta.dsv4.compressor.compressor_prefill` against the authors' ``Compressor.forward``
(start_pos=0) for both regimes: L2 (ratio 4, overlapping windows, coff=2) and L3 (ratio 128). Same
checkpoint weights feed both sides. Sequence lengths chosen to exercise the ``remainder`` split.

    uv run --with torch --with safetensors --with numpy python -m parity.dsv4_compressor_test
"""

from __future__ import annotations

import numpy as np
import torch

import mlx.core as mx

from quanta.dsv4 import attention as A
from quanta.dsv4 import compressor as C
from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.loader import DeepSeekV4SourceCheckpoint
from parity import dsv4_torch_ref as ref

ART = "/Users/pmrj/models/DeepSeek-V4-Flash"


def run() -> None:
    cfg = DeepSeekV4Config.from_pretrained(ART)
    M, args = ref.load_model_module(cfg, max_seq_len=512)
    ck = DeepSeekV4SourceCheckpoint(ART, cfg)
    rng = np.random.default_rng(0)
    ok = True

    for L, T in ((2, 158), (3, 300)):                       # ratio 4 (overlap) ; ratio 128
        ratio = cfg.compress_ratio(L)
        attn_ref = ref.load_attention(M, args, cfg, L)
        cp = ck.attention(L)["compressor"]
        ck.release()
        x = (rng.standard_normal((1, T, cfg.hidden_size)) * 0.5).astype(np.float32)
        with torch.no_grad():
            ref_kv = attn_ref.compressor(torch.from_numpy(x), 0).numpy().astype(np.float64)
        orig, theta = cfg.attn_rope(L)
        cos, sin = A.rope_cos_sin(cfg.rope_head_dim, T, orig, theta, cfg.rope_factor,
                                  cfg.beta_fast, cfg.beta_slow)
        my_kv = C.compressor_prefill(
            mx.array(x), cp["ape"].astype(mx.float32), cp["norm"].astype(mx.float32),
            cp["wkv"].astype(mx.float32), cp["wgate"].astype(mx.float32),
            ratio=ratio, head_dim=cfg.head_dim, rope_head_dim=cfg.rope_head_dim, eps=cfg.norm_eps,
            cos=cos, sin=sin)
        my = np.array(my_kv.astype(mx.float32)).astype(np.float64)
        d = float(np.max(np.abs(ref_kv - my)))
        rel = d / float(np.max(np.abs(ref_kv)))
        good = my.shape == ref_kv.shape and rel < 1e-3
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] L{L} ratio={ratio:3d} T={T} -> {my.shape}  "
              f"|Δ|={d:.2e} rel={rel:.2e} absmax={float(np.abs(ref_kv).max()):.3f}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
