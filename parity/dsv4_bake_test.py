"""Model-free gate for the DeepSeek-V4 bake (task #76) — tiny synthetic tensors only.

Exercises the bake's three quant paths on a few-KB of random data (NO checkpoint, NO GPU, no big
allocations): (a) the int4 AWQ expert path reconstructs a small stacked SwiGLU expert within a loose
bound; (b) the int8 affine non-expert path reconstructs a small weight tightly; (c) a tiny
:class:`ArtifactWriter` bake emits exactly the manifest formats/suffixes the Unit-1 ``DSV4Artifact``
runtime reads (``awq_packed`` + ``.awq_scale``; ``affine_packed`` + ``.weight_packed/_scale/_bias``;
``group_size``/``bits`` present). Uses ``group_size=32`` so tiny ``in`` dims (32/64) are divisible —
the real bake uses 128.

    uv run --with numpy python -m parity.dsv4_bake_test

deferred (run later on GPU, task #76): the real bake + e2e teacher-forced ppl, e.g.
    bake_dsv4("/Users/pmrj/models/DeepSeek-V4-Flash", "/Users/pmrj/models/DeepSeek-V4-Flash-quanta_int4",
              calib_ids, group_size=128, expert_method="awq", scale_dtype=mx.bfloat16)
    # then teacher_forced_ppl over the resident int4/int8 runtime vs the bf16 reference.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx

from quanta.bake.artifact import ArtifactWriter
from quanta.bake.awq import awq_quantize
from quanta.bake.quant import quantize_affine
from quanta.dsv4.bake import _bake_expert, _swiglu_inter, _write_int8

GS = 32  # tiny-tensor group size (real bake uses 128); divides the synthetic in-dims 32/64


def _rel(num: mx.array, den: mx.array) -> float:
    return float((mx.linalg.norm(num.astype(mx.float32)) / (mx.linalg.norm(den.astype(mx.float32)) + 1e-12)).item())


def _awq_expert_path() -> tuple[bool, float, float]:
    """(a) int4 AWQ on a small stacked SwiGLU expert. Reconstruct w1 (gate) and the SwiGLU-fed w2
    (down) on routed calib rows; assert recon relative error within a loose int4 bound."""
    rng = mx.random.key(0)
    inter, dim, n = 64, 32, 24  # one expert's projections (the bake stacks E of these)
    k1, k2, k3, kx = mx.random.split(rng, 4)
    w1 = mx.random.normal((inter, dim), key=k1) * 0.1   # gate [inter, dim]
    w3 = mx.random.normal((inter, dim), key=k2) * 0.1   # up   [inter, dim]
    w2 = mx.random.normal((dim, inter), key=k3) * 0.1   # down [dim, inter]
    xe = mx.random.normal((n, dim), key=kx)             # routed post-norm rows [n, dim]

    # w1 (input = xe): runtime computes (xe / s) @ dequant(w1·diag s)ᵀ ≈ xe @ w1ᵀ
    s, p, sc, b = awq_quantize(w1, xe, 4, GS)
    wq = mx.dequantize(p, sc, b, group_size=GS, bits=4).astype(mx.float32)
    y1 = (xe.astype(mx.float32) / s.astype(mx.float32)[None]) @ wq.T
    err1 = _rel(y1 - xe.astype(mx.float32) @ w1.astype(mx.float32).T, xe.astype(mx.float32) @ w1.astype(mx.float32).T)

    # w2 (input = SwiGLU intermediate): exercise the bake's exact down-proj calibration input
    inter_in = _swiglu_inter(xe, w1, w3, limit=0.0)     # [n, inter]
    s2, p2, sc2, b2 = awq_quantize(w2, inter_in, 4, GS)
    wq2 = mx.dequantize(p2, sc2, b2, group_size=GS, bits=4).astype(mx.float32)
    y2 = (inter_in / s2.astype(mx.float32)[None]) @ wq2.T
    err2 = _rel(y2 - inter_in @ w2.astype(mx.float32).T, inter_in @ w2.astype(mx.float32).T)

    ok = err1 < 0.15 and err2 < 0.15
    return ok, err1, err2


def _int8_nonexpert_path() -> tuple[bool, float]:
    """(b) int8 affine on a small non-expert weight; recon relative error must be tight."""
    w = mx.random.normal((48, 64), key=mx.random.key(1))
    p, sc, b = quantize_affine(w, 8, GS)
    wd = mx.dequantize(p, sc, b, group_size=GS, bits=8)
    err = _rel(w - wd, w)
    return err < 0.02, err


def _manifest_contract() -> tuple[bool, dict]:
    """(c) Tiny bake into a tempdir; assert the manifest entries use the exact formats + companion
    suffixes the Unit-1 runtime reads, with bits/group_size present."""
    out = tempfile.mkdtemp(suffix="_dsv4_bake_gate")
    try:
        (Path(out) / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
        writer = ArtifactWriter(out, Path(out) / "config.json")

        rng = mx.random.key(2)
        ka, kw, k1, k2, k3, kx = mx.random.split(rng, 6)
        # one int8 non-expert (e.g. an attention projection), one dense control tensor (a norm),
        # and one AWQ int4 expert (warm) + one RTN int4 expert (cold) via the bake helpers.
        _write_int8(writer, "layers.0.attn.wq_a", mx.random.normal((48, 64), key=ka), GS, mx.bfloat16)
        writer.add_dense("layers.0.attn_norm.weight", mx.random.normal((64,), key=kw).astype(mx.bfloat16))
        w1 = mx.random.normal((64, 32), key=k1) * 0.1
        w3 = mx.random.normal((64, 32), key=k2) * 0.1
        w2 = mx.random.normal((32, 64), key=k3) * 0.1
        xe = mx.random.normal((20, 32), key=kx)
        _bake_expert(writer, "layers.0.ffn.experts.0", w1, w3, w2, xe, GS, "awq", 0.0, mx.bfloat16)
        _bake_expert(writer, "layers.0.ffn.experts.1", w1, w3, w2, None, GS, "rtn", 0.0, mx.bfloat16)
        writer.finalize({"experts": "int4 awq g32", "non_experts": "int8 g32"})

        man = json.loads((Path(out) / "manifest.json").read_text())["tensors"]
        wmap = json.loads((Path(out) / "model.safetensors.index.json").read_text())["weight_map"]

        int8 = man["layers.0.attn.wq_a"]
        expert = man["layers.0.ffn.experts.0.w1"]
        cold = man["layers.0.ffn.experts.1.w2"]
        norm = man["layers.0.attn_norm.weight"]
        suffix_ok = all(f"layers.0.attn.wq_a{s}" in wmap for s in (".weight_packed", ".weight_scale", ".weight_bias")) \
            and all(f"layers.0.ffn.experts.0.w1{s}" in wmap
                    for s in (".weight_packed", ".weight_scale", ".weight_bias", ".awq_scale"))
        ok = (
            int8["format"] == "affine_packed" and int8["bits"] == 8 and int8["group_size"] == GS
            and expert["format"] == "awq_packed" and expert["bits"] == 4 and expert["group_size"] == GS
            and cold["format"] == "awq_packed" and cold["bits"] == 4  # cold expert is still awq_packed (s=1)
            and norm["format"] == "dense"
            and "layers.0.attn.wq_a.awq_scale" not in wmap  # int8 path must NOT emit an awq_scale
            and suffix_ok
        )
        return ok, {"int8": int8, "expert": expert, "cold": cold, "norm": norm}
    finally:
        shutil.rmtree(out, ignore_errors=True)


def run() -> None:
    awq_ok, e1, e2 = _awq_expert_path()
    int8_ok, e8 = _int8_nonexpert_path()
    man_ok, formats = _manifest_contract()

    print("\n=== DSV4 bake gate (model-free, tiny synthetic) ===")
    print(f"(a) int4 AWQ expert recon   : w1 {e1:.4f}<0.15  w2 {e2:.4f}<0.15 -> {awq_ok}")
    print(f"(b) int8 affine non-expert  : {e8:.5f}<0.02 -> {int8_ok}")
    print(f"(c) manifest formats/suffix : {man_ok}  {formats}")
    ok = awq_ok and int8_ok and man_ok
    assert ok, "DSV4 bake gate FAILED"
    print("PASS — DSV4 bake paths reconstruct and the manifest matches the Unit-1 runtime contract")


if __name__ == "__main__":
    run()
