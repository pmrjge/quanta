"""Parity: MiMo-V2.5 blockwise-fp8 dequant (MLX) == torch native ``float8_e4m3fn`` (the bug gate).

MiMo's source is DeepSeek-style block-fp8 (e4m3, [128,128]); loading it wrong is the classic
"loads fine, emits garbage" trap. This gates :mod:`quanta.mimo.fp8` against torch's authoritative
e4m3 decode on the *real* checkpoint tensors — especially the full-attention **fused qkv**, whose
scale grid has 2 trailing padding rows (out=13568 padded to 13824 → 108 rows vs 106 real blocks).
Also cross-checks the fused-qkv split offsets (vLLM #42803 class): the config's per-layer-type
``(q,k,v)`` must sum to the real stored ``out`` dim, and ``o_proj`` in-features must match.

    uv run --with torch --with safetensors --with numpy python -m parity.mimo_fp8_dequant_test
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import torch
from safetensors import safe_open

from quanta.mimo.config import MiMoV2Config
from quanta.mimo.fp8 import dequant_block_fp8, e4m3_to_float

ART = "/Users/pmrj/models/MiMo-V2.5"
BLOCK = 128


def _torch_ref_dequant(t_fp8: torch.Tensor, s: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Authoritative blockwise dequant (DeepSeek/transformers convention)."""
    out, inn = t_fp8.shape
    sf = s.repeat_interleave(block, 0).repeat_interleave(block, 1)[:out, :inn]
    return t_fp8.float() * sf


def _max_abs_diff(a_np, b_mx: mx.array) -> float:
    return float(mx.max(mx.abs(mx.array(a_np) - b_mx)).item())


def run() -> None:
    d = Path(ART)
    wmap = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = MiMoV2Config.from_pretrained(ART)

    def loc(key):  # -> (fp8_uint8_mx, scale_f32_mx, torch_fp8, torch_scale)
        with safe_open(str(d / wmap[key]), framework="pt") as f:
            t = f.get_tensor(key)                                  # float8_e4m3fn [out,in]
            s = f.get_tensor(key + "_scale_inv").float()           # f32 grid
        u8 = mx.array(t.view(torch.uint8).numpy())                 # raw e4m3 bytes
        return u8, mx.array(s.numpy()), t, s

    cases = [
        ("L0 full-attn qkv (108-row anomaly)", "model.layers.0.self_attn.qkv_proj.weight"),
        ("L1 SWA qkv (116, aligned)",          "model.layers.1.self_attn.qkv_proj.weight"),
        ("L0 dense gate_proj",                  "model.layers.0.mlp.gate_proj.weight"),
        ("L0 dense down_proj",                  "model.layers.0.mlp.down_proj.weight"),
        ("L1 expert0 gate_proj",               "model.layers.1.mlp.experts.0.gate_proj.weight"),
        ("L1 expert0 down_proj",               "model.layers.1.mlp.experts.0.down_proj.weight"),
    ]

    ok = True
    print("=== fp8 e4m3 decode + blockwise dequant: MLX vs torch ===")
    for tag, key in cases:
        u8, s_mx, t, s = loc(key)
        dec_diff = _max_abs_diff(t.float().numpy(), e4m3_to_float(u8))       # raw decode
        ours = dequant_block_fp8(u8, s_mx, dtype=mx.float32)
        deq_diff = _max_abs_diff(_torch_ref_dequant(t, s).numpy(), ours)     # full dequant
        good = dec_diff == 0.0 and deq_diff == 0.0
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] {tag:38s} out={t.shape[0]:5d} "
              f"scale={tuple(s.shape)}  decode|Δ|={dec_diff:.1e}  dequant|Δ|={deq_diff:.1e}")

    # --- bug #2: fused-qkv split offsets must match the real stored shapes -----------------
    print("=== fused-qkv split offsets (vLLM #42803 guard) ===")
    for li in (0, 1):
        swa = cfg.is_swa(li)
        q, k, v = cfg.qkv_sizes(swa)
        with safe_open(str(d / wmap[f"model.layers.{li}.self_attn.qkv_proj.weight"]), framework="pt") as f:
            qkv_out = f.get_slice(f"model.layers.{li}.self_attn.qkv_proj.weight").get_shape()[0]
            o_in = f.get_slice(f"model.layers.{li}.self_attn.o_proj.weight").get_shape()[1]
        sum_ok = (q + k + v) == qkv_out and cfg.o_in_features(swa) == o_in
        ok = ok and sum_ok
        print(f"  [{'OK' if sum_ok else 'FAIL'}] L{li} {'SWA ' if swa else 'full'}: "
              f"q+k+v={q}+{k}+{v}={q + k + v} vs qkv_out={qkv_out} | "
              f"o_in cfg={cfg.o_in_features(swa)} vs real={o_in} | rope_dim={cfg.rope_dim(swa)}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
