"""Gate: MiniMax-M2.7 streamed fp8->bf16 loader — model-free, ~0 GB (also the fp8 block-dequant gate).

Two parts, neither of which loads any real checkpoint tensor (per the host-OOM safety rule):

(a) **fp8 block-dequant round-trip on tiny random tensors.** Build small bf16 weights, derive a
    per-``[128,128]``-block fp32 ``scale_inv`` (block-absmax / E4M3_MAX), encode to e4m3 bytes with
    the native authority ``mx.to_fp8``, then dequant with the loader's
    :func:`quanta.minimax.loader.dequant_block_fp8_f32` and assert the relative error sits within the
    e4m3 (3-mantissa-bit) bound. Cross-checks the loader's e4m3fn LUT decode against native
    ``mx.from_fp8`` on the produced bytes (NaN bytes excluded) and exercises a **partial** trailing
    block. This validates the dequant math entirely model-free.

(b) **Key / dtype / shape schema vs the REAL index** (safetensors **header** parse only — never a
    tensor byte). For a sample layer, assert every key the loader will request exists in
    ``weight_map`` with the right dtype (``F8_E4M3`` for the fp8 q/k/v/o + experts and ``F32`` for
    their ``.weight_scale_inv``; ``BF16`` for norms/embed/lm_head; ``F32`` for the unquantized gate /
    ``e_score_correction_bias``), that the unquantized modules carry **no** scale, and that fp8
    weight/scale shapes are block-128 consistent with the config.

    uv run --with numpy python -m parity.minimax_loader_test
"""

from __future__ import annotations

import json
import math
import struct

import mlx.core as mx

from quanta.dsv4 import fp
from quanta.minimax.config import MiniMaxConfig
from quanta.minimax.loader import MiniMaxSourceCheckpoint, dequant_block_fp8_f32

MODEL_DIR = "/Users/pmrj/models/MiniMax-M2.7"
E4M3_MAX = 448.0          # e4m3fn max finite magnitude
E4M3_HALF_ULP = 1.0 / 16  # 3 mantissa bits -> worst-case relative round = 2^-4


# --- (a) fp8 block-dequant round-trip (tiny, model-free) ---------------------
def _block_absmax(w: mx.array, br: int, bc: int) -> mx.array:
    """Per-``[br,bc]``-block absmax as an fp32 grid ``[ceil(out/br), ceil(in/bc)]`` (tiny; padded)."""
    out, inn = int(w.shape[0]), int(w.shape[1])
    nbo, nbi = (out + br - 1) // br, (inn + bc - 1) // bc
    wf = mx.abs(w.astype(mx.float32))
    pad = mx.zeros((nbo * br, nbi * bc), dtype=mx.float32)
    pad[:out, :inn] = wf
    tiled = pad.reshape(nbo, br, nbi, bc)
    return tiled.max(axis=(1, 3))            # [nbo, nbi]


def _roundtrip_case(out: int, inn: int, br: int, bc: int) -> tuple[bool, float, float, int]:
    """Encode a random bf16 weight to e4m3 + fp32 block-scale, dequant, return (ok, mean_rel, max_rel,
    nan_bytes). Scales to ~0.9*E4M3_MAX so the encoder never emits the e4m3fn NaN bytes."""
    w = (mx.random.normal((out, inn)) * 0.1).astype(mx.bfloat16)
    amax = _block_absmax(w, br, bc)
    scale_inv = amax / (0.9 * E4M3_MAX) + 1e-12          # fp32 grid; block-max -> ~0.9*max
    s_full = mx.repeat(mx.repeat(scale_inv, br, axis=0), bc, axis=1)[:out, :inn]
    q = mx.to_fp8(w.astype(mx.float32) / s_full)         # uint8 e4m3 (native encode authority)

    # loader-LUT decode must equal native decode on the produced bytes (NaN bytes excluded)
    nan_bytes = int(((q == 0x7F) | (q == 0xFF)).sum().item())
    lut = fp.e4m3_to_float(q)
    nat = mx.from_fp8(q, dtype=mx.float32)
    lut_eq_native = bool(mx.all(lut == nat).item())      # no NaNs since nan_bytes asserted 0 below

    deq = dequant_block_fp8_f32(q, scale_inv, block=(br, bc), dtype=mx.float32)
    wf = w.astype(mx.float32)
    mask = mx.abs(wf) > (0.05 * float(mx.max(mx.abs(wf)).item()))   # ignore near-zero (huge rel)
    rel = mx.abs(deq - wf) / (mx.abs(wf) + 1e-9)
    mean_rel = float((rel * mask).sum().item() / max(int(mask.sum().item()), 1))
    max_rel = float(mx.max(mx.where(mask, rel, mx.zeros_like(rel))).item())
    shape_ok = tuple(deq.shape) == (out, inn)
    ok = (shape_ok and nan_bytes == 0 and lut_eq_native
          and mean_rel <= E4M3_HALF_ULP and max_rel <= E4M3_HALF_ULP + 1e-6)
    return ok, mean_rel, max_rel, nan_bytes


def part_a() -> bool:
    mx.random.seed(0)
    ok = True
    print("=== (a) fp8 block-dequant round-trip (tiny, model-free) ===")
    for out, inn, br, bc in [(256, 256, 128, 128), (200, 200, 128, 128), (384, 128, 128, 128)]:
        good, mean_rel, max_rel, nb = _roundtrip_case(out, inn, br, bc)
        ok = ok and good
        partial = "" if (out % br == 0 and inn % bc == 0) else " (partial block)"
        print(f"  [{'OK' if good else 'FAIL'}] [{out},{inn}] mean_rel={mean_rel:.4f} "
              f"max_rel={max_rel:.4f} nan_bytes={nb}{partial}")

    # fail-loud: scale grid too small for the weight (rule 6)
    try:
        dequant_block_fp8_f32(mx.zeros((256, 256), dtype=mx.uint8), mx.ones((1, 1)), block=(128, 128))
        guard_ok = False
    except ValueError:
        guard_ok = True
    ok = ok and guard_ok
    print(f"  [{'OK' if guard_ok else 'FAIL'}] undersized scale grid -> ValueError")
    return ok


# --- (b) key/dtype/shape schema vs the real index (HEADER parse only) --------
def _index() -> dict[str, str]:
    return json.loads(open(f"{MODEL_DIR}/model.safetensors.index.json").read())["weight_map"]


_HDR_CACHE: dict[str, dict] = {}


def _meta(wm: dict[str, str], key: str) -> tuple[str, tuple[int, ...]]:
    """``(dtype, shape)`` from the safetensors **header** of the key's shard — no tensor bytes read."""
    fn = wm[key]
    hdr = _HDR_CACHE.get(fn)
    if hdr is None:
        with open(f"{MODEL_DIR}/{fn}", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        hdr.pop("__metadata__", None)
        _HDR_CACHE[fn] = hdr
    m = hdr[key]
    return m["dtype"], tuple(m["shape"])


def _check(label: str, cond: bool) -> bool:
    print(f"  [{'OK' if cond else 'FAIL'}] {label}")
    return cond


def part_b() -> bool:
    import os
    if not os.path.isdir(MODEL_DIR):
        print(f"=== (b) SKIPPED: {MODEL_DIR} absent (run part (a) only) ===")
        return True

    cfg = MiniMaxConfig.from_pretrained(MODEL_DIR)
    wm = _index()
    ck = MiniMaxSourceCheckpoint(MODEL_DIR, cfg=cfg)   # JSON index only; no tensor load
    H = cfg.hidden_size
    br, bc = cfg.weight_block_size
    ok = True
    print(f"=== (b) schema vs real index ({len(wm)} keys; header parse only) ===")

    # top-level bf16
    top_ok = True
    for key, sh in [("model.embed_tokens.weight", (cfg.vocab_size, H)),
                    ("model.norm.weight", (H,)),
                    ("lm_head.weight", (cfg.vocab_size, H))]:
        if key not in wm:
            top_ok = False
            continue
        dt, got = _meta(wm, key)
        top_ok = top_ok and dt == "BF16" and got == sh
    ok = _check("top-level embed/norm/lm_head: BF16 + shapes", top_ok) and ok

    i = 1
    p = f"model.layers.{i}."

    # bf16 norms (incl. per-layer QK norm)
    norms_ok = True
    for key in ["input_layernorm.weight", "post_attention_layernorm.weight",
                "self_attn.q_norm.weight", "self_attn.k_norm.weight"]:
        k = p + key
        norms_ok = norms_ok and k in wm and _meta(wm, k)[0] == "BF16"
    ok = _check("layer norms + q/k_norm: present & BF16", norms_ok) and ok

    # fp8 attention proj + F32 scale_inv, block-128 consistent, NO scale on the bf16 norms
    attn_ok = True
    for proj, wsh in [("q_proj", (cfg.q_dim, H)), ("k_proj", (cfg.kv_dim, H)),
                      ("v_proj", (cfg.kv_dim, H)), ("o_proj", (H, cfg.q_dim))]:
        wk = p + f"self_attn.{proj}.weight"
        sk = p + f"self_attn.{proj}.weight_scale_inv"
        if wk not in wm or sk not in wm:
            attn_ok = False
            continue
        wdt, wgot = _meta(wm, wk)
        sdt, sgot = _meta(wm, sk)
        want_scale = (math.ceil(wgot[0] / br), math.ceil(wgot[1] / bc))
        attn_ok = (attn_ok and wdt == "F8_E4M3" and wgot == wsh
                   and sdt == "F32" and sgot == want_scale)
    attn_ok = attn_ok and (p + "self_attn.q_norm.weight_scale_inv") not in wm
    ok = _check("attn q/k/v/o: F8_E4M3 + F32 block-128 scale (norms unscaled)", attn_ok) and ok

    # router gate + bias: F32, unquantized (no scale)
    gk = p + "block_sparse_moe.gate.weight"
    bk = p + "block_sparse_moe.e_score_correction_bias"
    router_ok = (gk in wm and _meta(wm, gk) == ("F32", (cfg.num_local_experts, H))
                 and (gk + "_scale_inv") not in wm
                 and bk in wm and _meta(wm, bk) == ("F32", (cfg.num_local_experts,))
                 and (bk + "_scale_inv") not in wm)
    ok = _check("router gate + e_score_correction_bias: F32, no scale", router_ok) and ok

    # routed experts w1/w2/w3 (Mixtral naming) fp8 + F32 block scale; check first + last + count
    MI = cfg.moe_intermediate_size
    exp_shapes = {"w1": (MI, H), "w2": (H, MI), "w3": (MI, H)}
    experts_ok = True
    for e in (0, cfg.num_local_experts - 1):
        for proj, wsh in exp_shapes.items():
            wk = p + f"block_sparse_moe.experts.{e}.{proj}.weight"
            sk = wk + "_scale_inv"
            if wk not in wm or sk not in wm:
                experts_ok = False
                continue
            wdt, wgot = _meta(wm, wk)
            sdt, sgot = _meta(wm, sk)
            want_scale = (math.ceil(wgot[0] / br), math.ceil(wgot[1] / bc))
            experts_ok = (experts_ok and wdt == "F8_E4M3" and wgot == wsh
                          and sdt == "F32" and sgot == want_scale)
    # exactly num_local_experts experts, no off-by-one over-run
    over = p + f"block_sparse_moe.experts.{cfg.num_local_experts}.w1.weight"
    experts_ok = experts_ok and over not in wm
    ok = _check(f"experts.{{0,{cfg.num_local_experts - 1}}} w1/w2/w3: F8_E4M3 + F32 scale; "
                f"count=={cfg.num_local_experts}", experts_ok) and ok

    # no shared expert, no MTP weights (verified structural absence — loader refuses to invent them)
    absent_ok = (not cfg.has_shared_expert
                 and not any(("shared_expert" in k) for k in wm)
                 and not any(("nextn" in k.lower() or "eh_proj" in k.lower()
                              or ".mtp" in k.lower()) for k in wm))
    ok = _check("no shared-expert / MTP weights in index", absent_ok) and ok

    # loader.has() agrees with the index; missing key fails loud (rule 6)
    fail_loud = True
    if not ck.has(p + "self_attn.q_proj.weight"):
        fail_loud = False
    try:
        ck.shape("model.layers.0.does.not.exist")
        fail_loud = False
    except KeyError:
        pass
    ok = _check("loader.has() matches index; missing key -> KeyError", fail_loud) and ok
    return ok


def run() -> None:
    a = part_a()
    b = part_b()
    print("PASS" if (a and b) else "FAIL")


if __name__ == "__main__":
    run()
