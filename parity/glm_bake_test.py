"""Model-free gate for the GLM-5.1 bake (task #88) — tiny synthetic tensors only.

Exercises the bake's quant paths on a few-KB of random data (NO checkpoint, NO GPU, no big
allocations) and round-trips the **contract pair** (the :mod:`quanta.glm.bake` writer ⇄ the
:class:`quanta.glm.artifact.GLMArtifact` reader):

(a) the **int4 AWQ g64** routed-expert path reconstructs a small stacked SwiGLU expert
    (``gate_proj``/``down_proj`` on routed calib rows) within a loose int4 bound;
(b) the **int8 affine** non-expert path reconstructs a small weight tightly;
(c) a tiny bake over synthetic loader-shaped dicts (1 dense layer + 1 MoE layer, via the real
    :mod:`quanta.glm.bake` helpers) emits exactly the manifest formats / companion suffixes the
    :class:`GLMArtifact` runtime reads, and that reader dequantizes back to the bf16 weights, returns
    the per-kind loader dicts with the confirmed shapes/keys, and reports the correct bits/group/method.

Uses ``group_size=64`` (the real GLM expert recipe) so tiny ``in`` dims (64/128) are divisible; the
int8 non-experts also use g64 here (the real bake uses the same ``group_size`` for both).

    uv run --with numpy python -m parity.glm_bake_test

deferred (run later on GPU, task #88): the real bake + e2e teacher-forced ppl, e.g.
    bake_glm("/Users/pmrj/models/GLM-5.1", "/Users/pmrj/models/GLM-5.1-quanta_int4",
             calib_ids, group_size=64, expert_method="awq", scale_dtype=mx.bfloat16)
    # then quanta.glm.model.glm_teacher_forced_ppl over the resident int4/int8 runtime vs the bf16 ref.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx

from quanta.bake.awq import awq_quantize
from quanta.bake.quant import quantize_affine
from quanta.glm.artifact import GLMArtifact
from quanta.glm.bake import (
    _bake_attention,
    _bake_dense_mlp,
    _bake_expert,
    _bake_router,
    _bake_shared,
    _swiglu_inter,
    _write_int8,
)
from quanta.glm.config import GLMConfig

GS = 64  # GLM expert recipe group size; divides the synthetic in-dims (64/128)

# Tiny GLM config with the real key schema (mirrors parity/glm_loader_test.py), in-dims multiples of GS.
CFG = {
    "model_type": "glm_moe_dsa", "vocab_size": 10, "hidden_size": 64, "intermediate_size": 128,
    "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
    "q_lora_rank": 64, "kv_lora_rank": 64, "qk_nope_head_dim": 4, "qk_rope_head_dim": 2,
    "qk_head_dim": 6, "v_head_dim": 32,
    "index_head_dim": 8, "index_n_heads": 2, "index_topk": 4,
    "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1, "moe_intermediate_size": 64,
    "first_k_dense_replace": 1, "num_nextn_predict_layers": 1,
    "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
    "eos_token_id": [9], "pad_token_id": 9, "tie_word_embeddings": False,
}


def _rel(num: mx.array, den: mx.array) -> float:
    return float((mx.linalg.norm(num.astype(mx.float32))
                  / (mx.linalg.norm(den.astype(mx.float32)) + 1e-12)).item())


def _awq_expert_path() -> tuple[bool, float, float]:
    """(a) int4 AWQ g64 on a small stacked SwiGLU expert. Reconstruct gate_proj (input = routed rows)
    and the SwiGLU-fed down_proj on routed calib rows; assert recon relative error within an int4 bound.

    Exercises the runtime decode identity ``(x / s) @ dequant(W·diag s)ᵀ ≈ x @ Wᵀ`` and the bake's
    exact down-proj calibration input (:func:`quanta.glm.bake._swiglu_inter`)."""
    rng = mx.random.key(0)
    inter, hidden, n = 64, 64, 40  # one expert's projections (the bake stacks E of these)
    k1, k2, k3, kx = mx.random.split(rng, 4)
    w_gate = mx.random.normal((inter, hidden), key=k1) * 0.1   # [inter, hidden]
    w_up = mx.random.normal((inter, hidden), key=k2) * 0.1     # [inter, hidden]
    w_down = mx.random.normal((hidden, inter), key=k3) * 0.1   # [hidden, inter]
    xe = mx.random.normal((n, hidden), key=kx)                 # routed post-attn-norm rows [n, hidden]

    s, p, sc, b = awq_quantize(w_gate, xe, 4, GS)
    wq = mx.dequantize(p, sc, b, group_size=GS, bits=4).astype(mx.float32)
    yg = (xe.astype(mx.float32) / s.astype(mx.float32)[None]) @ wq.T
    ref_g = xe.astype(mx.float32) @ w_gate.astype(mx.float32).T
    err_gate = _rel(yg - ref_g, ref_g)

    inter_in = _swiglu_inter(xe, w_gate, w_up)                 # [n, inter] (silu(g)*u, no clamp)
    s2, p2, sc2, b2 = awq_quantize(w_down, inter_in, 4, GS)
    wq2 = mx.dequantize(p2, sc2, b2, group_size=GS, bits=4).astype(mx.float32)
    yd = (inter_in / s2.astype(mx.float32)[None]) @ wq2.T
    ref_d = inter_in @ w_down.astype(mx.float32).T
    err_down = _rel(yd - ref_d, ref_d)

    ok = err_gate < 0.15 and err_down < 0.15
    return ok, err_gate, err_down


def _int8_nonexpert_path() -> tuple[bool, float]:
    """(b) int8 affine g64 on a small non-expert weight; recon relative error must be tight."""
    w = mx.random.normal((64, 128), key=mx.random.key(1))
    p, sc, b = quantize_affine(w, 8, GS)
    wd = mx.dequantize(p, sc, b, group_size=GS, bits=8)
    err = _rel(w - wd, w)
    return err < 0.02, err


# ---- (c) full bake-path contract round-trip ---------------------------------------------------------
def _rand(*shape, key, scale=1.0):
    return mx.random.normal(shape, key=key) * scale


def _synth_attention(c: GLMConfig, key) -> dict:
    """An attention sub-dict shaped exactly like GLMSourceCheckpoint.attention (incl. nested indexer)."""
    nh, h = c.num_attention_heads, c.hidden_size
    ks = mx.random.split(key, 12)
    return {
        "q_a_proj": _rand(c.q_lora_rank, h, key=ks[0]),
        "q_a_layernorm": _rand(c.q_lora_rank, key=ks[1]).astype(mx.bfloat16),
        "q_b_proj": _rand(nh * c.qk_head_dim, c.q_lora_rank, key=ks[2]),
        "kv_a_proj_with_mqa": _rand(c.kv_lora_rank + c.qk_rope_head_dim, h, key=ks[3]),
        "kv_a_layernorm": _rand(c.kv_lora_rank, key=ks[4]).astype(mx.bfloat16),
        "kv_b_proj": _rand(nh * (c.qk_nope_head_dim + c.v_head_dim), c.kv_lora_rank, key=ks[5]),
        "o_proj": _rand(h, nh * c.v_head_dim, key=ks[6]),
        "indexer": {
            "wq_b": _rand(c.index_n_heads * c.index_head_dim, c.q_lora_rank, key=ks[7]),
            "wk": _rand(c.index_head_dim, h, key=ks[8]),
            "weights_proj": _rand(c.index_n_heads, h, key=ks[9]),
            "k_norm_weight": _rand(c.index_head_dim, key=ks[10]).astype(mx.bfloat16),
            "k_norm_bias": _rand(c.index_head_dim, key=ks[11]).astype(mx.bfloat16),
        },
    }


def _shapes_match(c: GLMConfig, art: GLMArtifact) -> bool:
    """The reader's per-kind dicts match the confirmed GLMSourceCheckpoint shapes/keys."""
    nh, h, mi, E = c.num_attention_heads, c.hidden_size, c.moe_intermediate_size, c.n_routed_experts
    a = art.attention(1)
    ix = a["indexer"]
    es = art.expert_stacks(1)
    dm = art.dense_mlp(0)
    sh = art.shared_expert(1)
    r = art.moe_router(1)
    bn = art.block_norms(1)
    return (
        tuple(a["q_b_proj"].shape) == (nh * c.qk_head_dim, c.q_lora_rank)
        and tuple(a["kv_a_proj_with_mqa"].shape) == (c.kv_lora_rank + c.qk_rope_head_dim, h)
        and tuple(a["kv_b_proj"].shape) == (nh * (c.qk_nope_head_dim + c.v_head_dim), c.kv_lora_rank)
        and tuple(a["o_proj"].shape) == (h, nh * c.v_head_dim)
        and tuple(ix["wq_b"].shape) == (c.index_n_heads * c.index_head_dim, c.q_lora_rank)
        and tuple(ix["wk"].shape) == (c.index_head_dim, h)
        and tuple(ix["k_norm_bias"].shape) == (c.index_head_dim,)
        and set(es) == {"gate_proj", "up_proj", "down_proj"}
        and tuple(es["gate_proj"].shape) == (E, mi, h)
        and tuple(es["down_proj"].shape) == (E, h, mi)
        and tuple(dm["gate_proj"].shape) == (c.intermediate_size, h)
        and tuple(dm["down_proj"].shape) == (h, c.intermediate_size)
        and tuple(sh["gate_proj"].shape) == (mi, h) and tuple(sh["down_proj"].shape) == (h, mi)
        and tuple(r["weight"].shape) == (E, h) and tuple(r["e_score_correction_bias"].shape) == (E,)
        and tuple(bn["input_layernorm"].shape) == (h,)
    )


def _manifest_contract() -> tuple[bool, dict]:
    """(c) Tiny bake (real glm.bake helpers) into a tempdir, then read back via GLMArtifact: assert the
    manifest formats/suffixes + per-kind dequant + shapes match the runtime contract."""
    from quanta.bake.artifact import ArtifactWriter

    out = tempfile.mkdtemp(suffix="_glm_bake_gate")
    try:
        c = GLMConfig.from_dict(CFG)
        h, mi, E = c.hidden_size, c.moe_intermediate_size, c.n_routed_experts
        (Path(out) / "config.json").write_text(json.dumps(CFG))
        writer = ArtifactWriter(out, Path(out) / "config.json")

        rng = mx.random.key(2)
        ks = mx.random.split(rng, 16)
        # top-level: embed/final-norm dense, lm_head int8 (untied)
        writer.add_dense("model.embed_tokens.weight", _rand(c.vocab_size, h, key=ks[0]).astype(mx.bfloat16))
        writer.add_dense("model.norm.weight", _rand(h, key=ks[1]).astype(mx.bfloat16))
        _write_int8(writer, "lm_head", _rand(c.vocab_size, h, key=ks[2]), GS, mx.bfloat16)

        # L0 dense layer: attention (int8 matmuls + dense norms) + dense MLP (int8)
        _bake_attention(writer, "layers.0.self_attn.", _synth_attention(c, ks[3]), GS, mx.bfloat16)
        writer.add_dense("layers.0.input_layernorm.weight", _rand(h, key=ks[4]).astype(mx.bfloat16))
        writer.add_dense("layers.0.post_attention_layernorm.weight", _rand(h, key=ks[5]).astype(mx.bfloat16))
        _bake_dense_mlp(writer, "layers.0.mlp.", {
            "gate_proj": _rand(c.intermediate_size, h, key=ks[6]),
            "up_proj": _rand(c.intermediate_size, h, key=ks[7]),
            "down_proj": _rand(h, c.intermediate_size, key=ks[8]),
        }, GS, mx.bfloat16)

        # L1 MoE layer: attention + norms + router (dense) + shared (int8) + routed experts (int4 AWQ/RTN)
        _bake_attention(writer, "layers.1.self_attn.", _synth_attention(c, ks[9]), GS, mx.bfloat16)
        writer.add_dense("layers.1.input_layernorm.weight", _rand(h, key=ks[10]).astype(mx.bfloat16))
        writer.add_dense("layers.1.post_attention_layernorm.weight", _rand(h, key=ks[11]).astype(mx.bfloat16))
        _bake_router(writer, "layers.1.mlp.gate.", {
            "weight": _rand(E, h, key=ks[12]).astype(mx.bfloat16),
            "e_score_correction_bias": mx.zeros((E,), dtype=mx.float32),
        })
        _bake_shared(writer, "layers.1.mlp.shared_experts.", {
            "gate_proj": _rand(mi, h, key=ks[13]), "up_proj": _rand(mi, h, key=ks[14]),
            "down_proj": _rand(h, mi, key=ks[15]),
        }, GS, mx.bfloat16)

        # one warm AWQ expert + the rest RTN cold; keep a reference for the dequant round-trip check.
        ke = mx.random.split(mx.random.key(7), 4 * E + 1)
        w_ref: dict[str, mx.array] = {}
        for e in range(E):
            w_gate = _rand(mi, h, key=ke[4 * e + 0], scale=0.1)
            w_up = _rand(mi, h, key=ke[4 * e + 1], scale=0.1)
            w_down = _rand(h, mi, key=ke[4 * e + 2], scale=0.1)
            xe = _rand(24, h, key=ke[4 * e + 3]) if e == 0 else None
            method = "awq" if e == 0 else "rtn"
            _bake_expert(writer, f"layers.1.mlp.experts.{e}", w_gate, w_up, w_down, xe, GS, method, mx.bfloat16)
            if e in (0, 1):  # check a warm (AWQ) and a cold (RTN) expert
                w_ref[f"layers.1.mlp.experts.{e}.gate_proj.weight"] = w_gate
                w_ref[f"layers.1.mlp.experts.{e}.down_proj.weight"] = w_down

        writer.finalize({"experts": "int4 awq g64", "non_experts": "int8 g64"})

        # ---- read back through the runtime reader ----
        man = json.loads((Path(out) / "manifest.json").read_text())["tensors"]
        wmap = json.loads((Path(out) / "model.safetensors.index.json").read_text())["weight_map"]
        int8 = man["layers.0.self_attn.q_a_proj"]
        idx_mm = man["layers.0.self_attn.indexer.wq_b"]
        head = man["lm_head"]
        warm = man["layers.1.mlp.experts.0.gate_proj"]
        cold = man["layers.1.mlp.experts.1.down_proj"]
        norm = man["layers.0.input_layernorm.weight"]
        router = man["layers.1.mlp.gate.weight"]
        bias = man["layers.1.mlp.gate.e_score_correction_bias"]

        suffix_ok = (
            all(f"layers.0.self_attn.q_a_proj{s}" in wmap
                for s in (".weight_packed", ".weight_scale", ".weight_bias"))
            and "layers.0.self_attn.q_a_proj.awq_scale" not in wmap   # int8 path emits NO awq_scale
            and all(f"layers.1.mlp.experts.0.gate_proj{s}" in wmap
                    for s in (".weight_packed", ".weight_scale", ".weight_bias", ".awq_scale"))
        )
        formats_ok = (
            int8["format"] == "affine_packed" and int8["bits"] == 8 and int8["group_size"] == GS
            and idx_mm["format"] == "affine_packed" and idx_mm["bits"] == 8
            and head["format"] == "affine_packed" and head["bits"] == 8
            and warm["format"] == "awq_packed" and warm["bits"] == 4 and warm["group_size"] == GS
            and cold["format"] == "awq_packed" and cold["bits"] == 4  # cold expert still awq_packed (s=1)
            and norm["format"] == "dense" and router["format"] == "dense"
            and bias["format"] == "dense" and bias["dtype"] == "float32"  # correction bias kept f32
        )

        art = GLMArtifact(out)
        # dequant round-trips the original bf16 weights within the int paths' bounds.
        recon = {}
        recon["int8 lm_head"] = _rel(art.read("lm_head.weight") - _rand(c.vocab_size, h, key=ks[2]),
                                     _rand(c.vocab_size, h, key=ks[2]))
        recon["awq expert gate"] = _rel(
            art.read("layers.1.mlp.experts.0.gate_proj.weight") - w_ref["layers.1.mlp.experts.0.gate_proj.weight"],
            w_ref["layers.1.mlp.experts.0.gate_proj.weight"])
        recon["rtn expert down"] = _rel(
            art.read("layers.1.mlp.experts.1.down_proj.weight") - w_ref["layers.1.mlp.experts.1.down_proj.weight"],
            w_ref["layers.1.mlp.experts.1.down_proj.weight"])
        # correction bias survives as f32 verbatim (control tensor, no cast/dequant)
        rb = art.moe_router(1)["e_score_correction_bias"]
        bias_ok = rb.dtype == mx.float32 and float(mx.abs(rb).max().item()) == 0.0
        # raw() returns packed codes for the gather_qmm path; fails loud on a dense key.
        raw_ok = art.raw("layers.1.mlp.experts.0.gate_proj.weight").dtype == mx.uint32
        try:
            art.raw("layers.0.input_layernorm.weight")
            raw_ok = False  # must have raised (dense has no packed codes)
        except ValueError:
            pass
        # fail-loud on a missing key
        try:
            art.read("layers.0.self_attn.does_not_exist.weight")
            miss_ok = False
        except KeyError:
            miss_ok = True

        recon_ok = recon["int8 lm_head"] < 0.02 and recon["awq expert gate"] < 0.15 and recon["rtn expert down"] < 0.15
        ok = (formats_ok and suffix_ok and recon_ok and _shapes_match(c, art)
              and bias_ok and raw_ok and miss_ok)
        return ok, {"formats_ok": formats_ok, "suffix_ok": suffix_ok, "shapes_ok": _shapes_match(c, art),
                    "bias_ok": bias_ok, "raw_ok": raw_ok, "miss_ok": miss_ok, "recon": recon}
    finally:
        shutil.rmtree(out, ignore_errors=True)


def run() -> None:
    awq_ok, e_g, e_d = _awq_expert_path()
    int8_ok, e8 = _int8_nonexpert_path()
    man_ok, detail = _manifest_contract()

    print("\n=== GLM-5.1 bake gate (model-free, tiny synthetic) ===")
    print(f"(a) int4 AWQ g64 expert recon : gate {e_g:.4f}<0.15  down {e_d:.4f}<0.15 -> {awq_ok}")
    print(f"(b) int8 affine non-expert    : {e8:.5f}<0.02 -> {int8_ok}")
    print(f"(c) bake⇄artifact round-trip  : {man_ok}")
    for k, v in detail.items():
        print(f"      {k}: {v}")
    ok = awq_ok and int8_ok and man_ok
    assert ok, "GLM bake gate FAILED"
    print("PASS — GLM bake paths reconstruct and the artifact matches the runtime reader contract")


if __name__ == "__main__":
    run()
