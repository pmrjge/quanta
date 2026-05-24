"""Parity: DeepSeek-V4 source-quant dequant (MLX) == torch's native fp8/fp4/e8m0 decode (the bug gate).

DSV4 ships fp8-e4m3 (non-experts) + packed fp4-e2m1 (experts), both with **e8m0/MX** power-of-two
scales — a new flavour of the "loads fine, emits garbage" trap. ``mx.load`` refuses ``F8_E8M0``, so
:mod:`quanta.dsv4.fp` reads raw bytes and decodes with hand-built LUTs. This gate validates those
LUTs **bit-exactly against torch 2.12's native dtypes** (``float8_e4m3fn`` / ``float8_e8m0fnu`` /
``float4_e2m1fn_x2``), which safetensors reads directly — an *independent* authority (torch is
offline-only, allowed for source-checkpoint loading per project rule 5). Also checks config geometry.

    uv run --with torch --with safetensors --with numpy python -m parity.dsv4_dequant_test
"""

from __future__ import annotations

import json
import struct

import numpy as np
import torch
from safetensors import safe_open

import mlx.core as mx

from quanta.dsv4 import fp
from quanta.dsv4.config import DeepSeekV4Config

ART = "/Users/pmrj/models/DeepSeek-V4-Flash"
_wmap = json.loads(open(f"{ART}/model.safetensors.index.json").read())["weight_map"]


def _raw(key):
    """Raw bytes of a tensor as (dtype_str, shape, np.uint8 1-D) — fed to the code under test."""
    with open(f"{ART}/{_wmap[key]}", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        m = json.loads(f.read(n))[key]
        b, e = m["data_offsets"]
        f.seek(8 + n + b)
        buf = f.read(e - b)
    return m["dtype"], tuple(m["shape"]), np.frombuffer(buf, dtype=np.uint8)


def _torch(key):
    """Authoritative tensor via torch (native fp8/fp4/e8m0 decode)."""
    with safe_open(f"{ART}/{_wmap[key]}", framework="pt") as f:
        return f.get_tensor(key)


def _maxdiff(a_np, b_mx):
    return float(np.max(np.abs(a_np.astype(np.float64)
                               - np.array(b_mx.astype(mx.float32)).astype(np.float64))))


def run() -> None:
    cfg = DeepSeekV4Config.from_pretrained(ART)
    ok = True

    # --- fp8 e4m3 + e8m0 block-128 (non-experts) vs torch ----------------------
    print("=== block-fp8 (e4m3 + e8m0): MLX LUTs vs torch native ===")
    for key in ["layers.2.attn.wq_a", "layers.2.attn.wq_b", "layers.2.attn.wkv",
                "layers.2.attn.wo_a", "layers.2.attn.wo_b", "layers.2.ffn.shared_experts.w1",
                "layers.2.attn.indexer.wq_b", "mtp.0.e_proj"]:
        _, shw, wb = _raw(key + ".weight")
        _, shs, sb = _raw(key + ".scale")
        tw, ts = _torch(key + ".weight"), _torch(key + ".scale")
        w_ref, s_ref = tw.float().numpy(), ts.float().numpy()            # torch authoritative
        dec = _maxdiff(w_ref, fp.e4m3_to_float(mx.array(wb.reshape(shw))))
        sdec = _maxdiff(s_ref, fp.e8m0_to_float(mx.array(sb.reshape(shs))))
        out, inn = shw
        ref = w_ref * np.repeat(np.repeat(s_ref, 128, 0), 128, 1)[:out, :inn]
        ours = fp.dequant_block_fp8(mx.array(wb.reshape(shw)), mx.array(sb.reshape(shs)), dtype=mx.float32)
        deq = _maxdiff(ref, ours)
        good = dec == 0.0 and sdec == 0.0 and deq == 0.0
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] {key:34s} e4m3|Δ|={dec:.1e} e8m0|Δ|={sdec:.1e} "
              f"dequant|Δ|={deq:.1e}")

    # --- fp4 e2m1 + e8m0 group-32 (experts) ------------------------------------
    # torch 2.12 has no CPU copy-kernel for float4_e2m1fn_x2 (can't .float() it), so the oracle is
    # DeepSeek's own FP4_TABLE from inference/convert.py (the authors' canonical e2m1 decode),
    # applied with independent torch ops on the raw bytes. e8m0 scale stays torch-native.
    print("=== group-fp4 (e2m1 packed + e8m0): MLX LUTs vs DeepSeek FP4_TABLE (torch) ===")
    FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                              0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
    for key in ["layers.2.ffn.experts.0.w1", "layers.2.ffn.experts.0.w2",
                "layers.2.ffn.experts.0.w3", "layers.42.ffn.experts.255.w2"]:
        _, shw, wb = _raw(key + ".weight")
        _, shs, sb = _raw(key + ".scale")
        u8 = _torch(key + ".weight").view(torch.uint8).to(torch.int64)   # int8 storage -> raw bytes
        lo, hi = FP4_TABLE[u8 & 0xF], FP4_TABLE[(u8 >> 4) & 0xF]
        w_ref = torch.stack([lo, hi], -1).flatten(-2).numpy()            # [out, in], low nibble first
        s_ref = _torch(key + ".scale").float().numpy()                   # e8m0 torch-native
        udec = _maxdiff(w_ref, fp.unpack_fp4(mx.array(wb.reshape(shw))))
        ref = w_ref * np.repeat(s_ref, 32, 1)[:, :w_ref.shape[1]]
        ours = fp.dequant_group_fp4(mx.array(wb.reshape(shw)), mx.array(sb.reshape(shs)), dtype=mx.float32)
        deq = _maxdiff(ref, ours)
        good = udec == 0.0 and deq == 0.0 and w_ref.shape[1] == shw[1] * 2
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] {key:32s} unpack|Δ|={udec:.1e} dequant|Δ|={deq:.1e} "
              f"-> {w_ref.shape}")

    # --- bf16 passthrough vs torch ---------------------------------------------
    print("=== bf16 decode vs torch native ===")
    for key in ["layers.2.ffn.gate.weight", "norm.weight", "layers.2.attn.q_norm.weight"]:
        _, sh, ub = _raw(key)
        ref = _torch(key).float().numpy()
        ours = fp.decode_buffer("BF16", sh, ub.tobytes())
        diff = _maxdiff(ref, ours)
        good = diff == 0.0
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] {key:34s} {sh}  |Δ|={diff:.1e}")

    # --- config geometry sanity ------------------------------------------------
    print("=== config geometry ===")
    geo = [
        ("compress_ratios len == n_layers+n_mtp",
         len(cfg.compress_ratios) == cfg.num_hidden_layers + cfg.n_mtp_layers),
        ("layers 0,1 pure-SW (ratio 0)", cfg.compress_ratio(0) == 0 and cfg.compress_ratio(1) == 0),
        ("layer 2 has indexer (ratio 4)", cfg.has_indexer(2) and cfg.compress_ratio(2) == 4),
        ("layer 3 compressor no indexer (ratio 128)",
         cfg.has_compressor(3) and not cfg.has_indexer(3)),
        ("layers 0,1,2 hash; 3 scored", cfg.is_hash(0) and cfg.is_hash(2) and not cfg.is_hash(3)),
        ("MTP block ratio 0", cfg.compress_ratio(cfg.num_hidden_layers) == 0),
        ("compressed layers use YaRN theta",
         cfg.attn_rope(2) == (cfg.original_seq_len, cfg.compress_rope_theta)),
        ("pure-SW layers use base theta", cfg.attn_rope(0) == (0, cfg.rope_theta)),
        ("nope+rope == head_dim", cfg.nope_head_dim + cfg.rope_head_dim == cfg.head_dim),
        ("mix_hc == (2+hc)*hc", cfg.mix_hc == (2 + cfg.hc_mult) * cfg.hc_mult),
    ]
    for tag, good in geo:
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] {tag}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
