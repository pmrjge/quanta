"""Model-free gate for the Qwen3.5-397B-A17B bake (#115) — tiny synthetic tensors only.

Exercises the bake's quant paths on a few-KB of random data (NO checkpoint, NO GPU, no big
allocations):

  (a) int4 g64 reconstructs a small **pre-stacked** SwiGLU expert (3-D ``[E, out, in]``) within a
      loose int4 bound — the exact gather_qmm-ready layout the bake quantizes in one shot;
  (b) int8 affine reconstructs a small dense non-expert weight tightly;
  (c) a tiny ``ArtifactWriter`` bake via the real bake helpers emits exactly the manifest the
      ``Qwen35Artifact`` runtime reads, and — the #115 invariant — the GatedDeltaNet **SSM control**
      tensors (``A_log`` / ``dt_bias`` / ``conv1d`` / ``norm``) are carried **bf16 dense** (NOT
      quantized) in the manifest, while the routed-expert stacks are ``affine_packed`` int4 g64 and
      the projections are ``affine_packed`` int8;
  (d) the tiny artifact round-trips through ``Qwen35Artifact`` (dense verbatim + 3-D expert dequant +
      2-D int8 dequant) and its baked ``config.json`` carries the dynamic-YaRN 1M policy fields.

Uses ``group_size=32`` so tiny ``in`` dims (32/64) are divisible — the real bake uses 64.

    uv run --with numpy python -m parity.qwen35_bake_test

deferred (run later on GPU, #115): the real bake + e2e teacher-forced ppl — see the
``quanta.qwen35.bake`` module docstring for the (NOT run here) heavy invocation.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx

from quanta.bake.artifact import ArtifactWriter
from quanta.qwen35.artifact import Qwen35Artifact
from quanta.qwen35.bake import (
    _LINEAR_BF16,
    _LINEAR_INT8,
    _bake_long_context,
    _write_expert_stack,
    _write_int8,
    _write_suffix_sub,
)
from quanta.qwen35.config import Qwen35Config

GS = 32  # tiny-tensor group size (real bake uses 64); divides the synthetic in-dims 32/64
SSM_CONTROL = ("conv1d.weight", "A_log", "dt_bias", "norm.weight")  # must stay bf16 (NOT quantized)


def _rel(num: mx.array, den: mx.array) -> float:
    return float((mx.linalg.norm(num.astype(mx.float32))
                  / (mx.linalg.norm(den.astype(mx.float32)) + 1e-12)).item())


def _tiny_config_dir() -> str:
    """A tiny but ``from_pretrained``-valid Qwen3.5 source config.json in a fresh tempdir."""
    d = tempfile.mkdtemp(suffix="_qwen35_bake_gate")
    tc = {
        "vocab_size": 64, "hidden_size": 64, "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"], "full_attention_interval": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "attn_output_gate": True, "partial_rotary_factor": 0.25,
        "rope_parameters": {"rope_theta": 1e7, "mrope_section": [], "partial_rotary_factor": 0.25},
        "use_qk_norm": True, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "linear_key_head_dim": 8, "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32", "num_experts": 4, "num_experts_per_tok": 2,
        "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64, "scoring_func": "softmax",
        "norm_topk_prob": True, "router_aux_loss_coef": 0.001, "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False, "hidden_act": "silu", "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096, "tie_word_embeddings": False,
    }
    conf = {"model_type": "qwen3_5_moe", "text_config": tc, "tie_word_embeddings": False}
    (Path(d) / "config.json").write_text(json.dumps(conf))
    return d


def _expert_stack_path() -> tuple[bool, float]:
    """(a) int4 g64 on a small **pre-stacked** SwiGLU expert ``[E, 2*inter, in]`` (one-shot 3-D)."""
    rng = mx.random.key(0)
    e, inter, dim = 4, 16, 64
    gate_up = mx.random.normal((e, 2 * inter, dim), key=rng) * 0.1   # fused gate+up stack
    p, sc, b = mx.quantize(gate_up.astype(mx.bfloat16), group_size=GS, bits=4)
    wd = mx.dequantize(p, sc, b, group_size=GS, bits=4)
    err = _rel(gate_up - wd, gate_up)
    return err < 0.15, err


def _int8_nonexpert_path() -> tuple[bool, float]:
    """(b) int8 affine on a small non-expert weight; recon relative error must be tight."""
    w = mx.random.normal((48, 64), key=mx.random.key(1))
    p, sc, b = mx.quantize(w, group_size=GS, bits=8)
    wd = mx.dequantize(p, sc, b, group_size=GS, bits=8)
    err = _rel(w - wd, w)
    return err < 0.02, err


def _tiny_bake() -> str:
    """Bake a tiny artifact via the REAL bake helpers into a fresh tempdir; return its path.

    Covers one linear-attention layer's projections (int8) + SSM control (bf16 dense), one MoE block
    (router/shared-gate bf16, shared expert int8, pre-stacked routed experts int4 g64), and the
    dynamic-YaRN 1M policy baked into config.json.
    """
    d = _tiny_config_dir()
    cfg = Qwen35Config.from_pretrained(d)
    writer = ArtifactWriter(d, Path(d) / "config.json")
    rng = mx.random.key(2)
    keys = mx.random.split(rng, 16)
    h, inter, e = cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts

    # --- a linear-attention layer: int8 projections + bf16 SSM control (via the real helper) ---
    lp = "model.language_model.layers.0."
    writer.add_dense(lp + "input_layernorm.weight", mx.random.normal((h,), key=keys[0]).astype(mx.bfloat16))
    writer.add_dense(lp + "post_attention_layernorm.weight",
                     mx.random.normal((h,), key=keys[1]).astype(mx.bfloat16))
    conv_dim = cfg.linear_qkv_dim
    lin = {
        "in_proj_qkv.weight": mx.random.normal((conv_dim, h), key=keys[2]) * 0.1,
        "in_proj_a.weight": mx.random.normal((cfg.linear_num_value_heads, h), key=keys[3]) * 0.1,
        "in_proj_b.weight": mx.random.normal((cfg.linear_num_value_heads, h), key=keys[4]) * 0.1,
        "in_proj_z.weight": mx.random.normal((cfg.linear_v_dim, h), key=keys[5]) * 0.1,
        "out_proj.weight": mx.random.normal((h, cfg.linear_v_dim), key=keys[6]) * 0.1,
        # SSM control (must NOT be quantized): conv1d (C,1,K), A_log/dt_bias [Hv] f32, norm [Dv] f32
        "conv1d.weight": mx.random.normal((conv_dim, 1, cfg.linear_conv_kernel_dim), key=keys[7]) * 0.2,
        "A_log": (mx.random.normal((cfg.linear_num_value_heads,), key=keys[8]) * 0.5).astype(mx.float32),
        "dt_bias": (mx.random.normal((cfg.linear_num_value_heads,), key=keys[9]) * 0.1).astype(mx.float32),
        "norm.weight": mx.random.uniform(0.5, 1.5, (cfg.linear_value_head_dim,), key=keys[10]).astype(mx.float32),
    }
    _write_suffix_sub(writer, lp + "linear_attn.", lin, _LINEAR_INT8, _LINEAR_BF16, GS, mx.bfloat16)

    # --- the MoE block: router/shared-gate bf16, shared expert int8, routed experts int4 g64 ---
    mp = lp + "mlp."
    writer.add_dense(mp + "gate.weight", mx.random.normal((e, h), key=keys[11]).astype(mx.bfloat16))
    writer.add_dense(mp + "shared_expert_gate.weight",
                     mx.random.normal((1, h), key=keys[12]).astype(mx.bfloat16))
    for proj in ("gate_proj", "up_proj"):
        _write_int8(writer, f"{mp}shared_expert.{proj}", mx.random.normal((inter, h), key=keys[13]) * 0.1,
                    GS, mx.bfloat16)
    _write_int8(writer, f"{mp}shared_expert.down_proj", mx.random.normal((h, inter), key=keys[14]) * 0.1,
                GS, mx.bfloat16)
    gate_up = mx.random.normal((e, 2 * inter, h), key=keys[15]) * 0.1
    down = mx.random.normal((e, h, inter), key=mx.random.split(keys[15], 1)[0]) * 0.1
    _write_expert_stack(writer, mp + "experts.gate_up_proj", gate_up, GS, mx.bfloat16)
    _write_expert_stack(writer, mp + "experts.down_proj", down, GS, mx.bfloat16)

    writer.finalize({"experts": "int4 affine g32", "non_experts": "int8 affine g32"})
    _bake_long_context(Path(d), cfg)
    return d


def _manifest_and_ssm() -> tuple[bool, dict]:
    """(c) Manifest formats + the #115 SSM-bf16 invariant, off a tiny real-helper bake."""
    d = _tiny_bake()
    try:
        man = json.loads((Path(d) / "manifest.json").read_text())["tensors"]
        wmap = json.loads((Path(d) / "model.safetensors.index.json").read_text())["weight_map"]
        lp = "model.language_model.layers.0."

        # SSM control carried bf16 dense, NOT quantized (the precedent this bake must keep)
        ssm_dense = all(man[lp + "linear_attn." + s]["format"] == "dense" for s in SSM_CONTROL)
        ssm_no_packed = all((lp + "linear_attn." + s + ".weight_packed") not in wmap
                            and (lp + "linear_attn." + s) not in
                            {k for k, v in man.items() if v.get("format") == "affine_packed"}
                            for s in SSM_CONTROL)
        # A_log / dt_bias / DeltaNet norm preserve f32 (never silently downcast)
        f32_ok = all(man[lp + "linear_attn." + s]["dtype"] == "float32"
                     for s in ("A_log", "dt_bias", "norm.weight"))

        # int8 projections
        proj = man[lp + "linear_attn.in_proj_qkv"]
        proj_ok = proj["format"] == "affine_packed" and proj["bits"] == 8 and proj["group_size"] == GS
        # int4 g64 pre-stacked routed experts (+ companion suffixes)
        eg = man[lp + "mlp.experts.gate_up_proj"]
        ed = man[lp + "mlp.experts.down_proj"]
        expert_ok = (eg["format"] == "affine_packed" and eg["bits"] == 4 and eg["group_size"] == GS
                     and ed["format"] == "affine_packed" and ed["bits"] == 4)
        suffix_ok = all(lp + "mlp.experts.gate_up_proj" + s in wmap
                        for s in (".weight_packed", ".weight_scale", ".weight_bias"))
        # router gate + shared_expert_gate bf16 dense; shared expert int8
        gate_dense = man[lp + "mlp.gate.weight"]["format"] == "dense"
        sgate_dense = man[lp + "mlp.shared_expert_gate.weight"]["format"] == "dense"
        shared_int8 = man[lp + "mlp.shared_expert.gate_proj"]["format"] == "affine_packed" \
            and man[lp + "mlp.shared_expert.gate_proj"]["bits"] == 8

        ok = all([ssm_dense, ssm_no_packed, f32_ok, proj_ok, expert_ok, suffix_ok, gate_dense,
                  sgate_dense, shared_int8])
        info = {"ssm_dense": ssm_dense, "ssm_no_packed": ssm_no_packed, "ssm_f32": f32_ok,
                "proj_int8": proj_ok, "expert_int4": expert_ok, "gate_bf16": gate_dense,
                "shared_int8": shared_int8}
        return ok, info
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _roundtrip_and_yarn() -> tuple[bool, dict]:
    """(d) Round-trip a tiny artifact through ``Qwen35Artifact`` + assert the baked 1M YaRN config."""
    d = _tiny_bake()
    try:
        art = Qwen35Artifact(d)
        lp = "model.language_model.layers.0."

        # dense control round-trips; int8 proj + 3-D int4 expert stack dequantize to expected shapes
        a_log = art.read(lp + "linear_attn.A_log")
        norm = art.read(lp + "linear_attn.norm.weight")
        proj = art.read(lp + "linear_attn.in_proj_qkv.weight")
        moe = art.moe(0)
        eg, ed = moe["experts_gate_up"], moe["experts_down"]
        cfg = art.cfg
        shapes_ok = (
            a_log.shape == (cfg.linear_num_value_heads,)
            and norm.shape == (cfg.linear_value_head_dim,)
            and proj.shape == (cfg.linear_qkv_dim, cfg.hidden_size)
            and eg.shape == (cfg.num_experts, cfg.moe_gate_up_out, cfg.hidden_size)
            and ed.shape == (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size)
            and bool(mx.all(mx.isfinite(eg)).item()) and bool(mx.all(mx.isfinite(proj)).item())
        )
        # raw() returns packed codes for the expert stack; refuses a dense key
        raw_ok = art.raw(lp + "mlp.experts.gate_up_proj").ndim == 3
        try:
            art.raw(lp + "linear_attn.A_log")
            dense_refuse = False
        except ValueError:
            dense_refuse = True

        # the baked dynamic-YaRN 1M policy in config.json. NEW contract (N0): the artifact config must
        # DECLARE the 1M served window as a first-class field — max_position_embeddings is RAISED to
        # max_context + a standard YaRN block is written — while the dynamic-YaRN BASELINE
        # (yarn_original_max) stays native, read DECOUPLED from rope.original_max_position_embeddings,
        # so short sequences still pay no tax and dynamic YaRN survives the raised window.
        conf = json.loads((Path(d) / "config.json").read_text())
        qlc = conf.get("quanta_long_context", {})
        tc = conf.get("text_config", conf)
        native = 4096
        served_raised = (tc.get("max_position_embeddings") == 1_010_000
                         and conf.get("max_position_embeddings") == 1_010_000)
        rope = tc.get("rope_parameters", {})
        yarn_block_ok = (rope.get("rope_type") == "yarn" and rope.get("factor") == 4.0
                         and rope.get("original_max_position_embeddings") == native
                         and tc.get("rope_scaling", {}).get("rope_type") == "yarn")
        yarn_ok = (qlc.get("max_context") == 1_010_000 and qlc.get("yarn_factor") == 4.0
                   and qlc.get("yarn_dynamic") is True and qlc.get("yarn_original_max") == native
                   and served_raised and yarn_block_ok)
        # regression guard: the reloaded artifact cfg keeps the BASELINE native (short ctx pays no tax)
        # yet scales beyond it and reaches the 1M target — dynamic YaRN survives the raised window.
        dyn_ok = (cfg.yarn_original_max == native and cfg.max_position_embeddings == 1_010_000
                  and cfg.effective_yarn_factor(native) == 1.0
                  and cfg.effective_yarn_factor(native * 4) > 1.0
                  and cfg.yarn_original_max * cfg.yarn_factor >= 1_000_000 / 256)

        ok = shapes_ok and raw_ok and dense_refuse and yarn_ok and dyn_ok
        return ok, {"shapes": shapes_ok, "raw_codes": raw_ok, "dense_refuse": dense_refuse,
                    "yarn_1M": yarn_ok, "dyn_yarn": dyn_ok, "qlc": qlc}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run() -> None:
    exp_ok, ee = _expert_stack_path()
    int8_ok, e8 = _int8_nonexpert_path()
    man_ok, man_info = _manifest_and_ssm()
    rt_ok, rt_info = _roundtrip_and_yarn()

    print("\n=== Qwen3.5 bake gate (model-free, tiny synthetic) ===")
    print(f"(a) int4 g64 stacked expert recon : {ee:.4f}<0.15 -> {exp_ok}")
    print(f"(b) int8 affine non-expert recon  : {e8:.5f}<0.02 -> {int8_ok}")
    print(f"(c) manifest + SSM bf16 invariant : {man_ok}  {man_info}")
    print(f"(d) artifact round-trip + 1M YaRN : {rt_ok}  {rt_info}")
    ok = exp_ok and int8_ok and man_ok and rt_ok
    assert ok, "Qwen3.5 bake gate FAILED"
    print("PASS — Qwen3.5 bake: int4-g64 stacked experts + int8 non-experts reconstruct; SSM control "
          "is bf16; the 1M dynamic-YaRN config is baked; the artifact round-trips through Qwen35Artifact")


if __name__ == "__main__":
    run()
