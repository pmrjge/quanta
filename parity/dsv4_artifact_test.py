"""Validate the DSV4 baked-artifact reader: dispatch, dequant round-trip, raw codes, fail-loud.

Model-free and tiny (a few KB of random weights) — synthesizes a minimal quanta artifact in a
tempdir (one affine_packed weight + one awq_packed weight + one dense bf16 tensor + a manifest /
index / config.json), then constructs :class:`quanta.dsv4.artifact.DSV4Artifact` over it and checks:

  * ``read(key)`` round-trips an affine weight close to the original (loose tol);
  * ``raw(key)`` returns the packed codes verbatim (== the stored ``.weight_packed``);
  * an AWQ weight (codes of ``W·diag(s)``) is recovered via ``dequant / s`` to within tol;
  * a dense key reads back as bf16;
  * an unknown manifest format fails loud (no silent fallback / wrong-bits dequant).

No checkpoint / resident / bake / ppl run — safe to run while the GPU is busy.

    uv run --with numpy python -m parity.dsv4_artifact_test
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlx.core as mx

from quanta.bake.quant import quantize_affine
from quanta.dsv4.artifact import DSV4Artifact


def _min_config() -> dict:
    """A minimal DSV4 config.json that ``DeepSeekV4Config.from_pretrained`` accepts (geometry only;
    no weights involved). ``compress_ratios`` must have length ``num_hidden_layers + n_mtp``."""
    n_layers, n_mtp = 2, 1
    return {
        "vocab_size": 64,
        "hidden_size": 32,
        "num_hidden_layers": n_layers,
        "moe_intermediate_size": 16,
        "num_attention_heads": 4,
        "head_dim": 8,
        "qk_rope_head_dim": 4,
        "q_lora_rank": 8,
        "o_lora_rank": 8,
        "o_groups": 1,
        "sliding_window": 16,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 4,
        "compress_ratios": [0, 4] + [0] * n_mtp,
        "compress_rope_theta": 10000.0,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "num_hash_layers": 0,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 0.0,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 2,
        "hc_eps": 1e-6,
        "num_nextn_predict_layers": n_mtp,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "rope_scaling": {"factor": 1.0, "beta_fast": 32, "beta_slow": 1,
                         "original_max_position_embeddings": 4096, "type": "yarn"},
        "max_position_embeddings": 4096,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "tie_word_embeddings": False,
    }


def _build_artifact(out: Path) -> dict:
    """Write a tiny self-contained quanta artifact and return the reference tensors for checks."""
    out.mkdir(parents=True, exist_ok=True)
    bits, gs = 4, 32  # group_size must divide `in`

    # affine_packed weight at base "layers.0.attn.wq_a"
    aff_w = mx.random.normal((16, 32))
    aff_packed, aff_scale, aff_bias = quantize_affine(aff_w, bits, gs)
    aff_ref = mx.dequantize(aff_packed, aff_scale, aff_bias, group_size=gs, bits=bits)

    # awq_packed weight at base "layers.0.ffn.experts.0.w1": codes are of W·diag(s); read() recovers W.
    awq_w = mx.random.normal((16, 32))
    awq_s = (mx.random.uniform(shape=(32,)) + 0.5).astype(mx.bfloat16)  # per-input-channel scale
    scaled = (awq_w * awq_s[None, :]).astype(mx.bfloat16)
    awq_packed, awq_scale, awq_bias = quantize_affine(scaled, bits, gs)
    awq_recovered_ref = mx.dequantize(awq_packed, awq_scale, awq_bias,
                                      group_size=gs, bits=bits) / awq_s[None, :]

    # dense bf16 tensor stored verbatim at its full key
    dense_key = "layers.0.attn.q_norm.weight"
    dense_w = mx.random.normal((32,)).astype(mx.bfloat16)

    shard = {
        "layers.0.attn.wq_a.weight_packed": aff_packed,
        "layers.0.attn.wq_a.weight_scale": aff_scale,
        "layers.0.attn.wq_a.weight_bias": aff_bias,
        "layers.0.ffn.experts.0.w1.weight_packed": awq_packed,
        "layers.0.ffn.experts.0.w1.weight_scale": awq_scale,
        "layers.0.ffn.experts.0.w1.weight_bias": awq_bias,
        "layers.0.ffn.experts.0.w1.awq_scale": awq_s,
        dense_key: dense_w,
        # a tensor whose manifest entry has a bogus format → must fail loud
        "layers.0.bogus.weight_packed": aff_packed,
    }
    fn = "model-00000.safetensors"
    mx.save_safetensors(str(out / fn), shard)

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: fn for k in shard}}, indent=0)
    )
    (out / "config.json").write_text(json.dumps(_min_config(), indent=2))
    manifest = {
        "format": "quanta",
        "tensors": {
            "layers.0.attn.wq_a": {"format": "affine_packed", "bits": bits, "group_size": gs},
            "layers.0.ffn.experts.0.w1": {"format": "awq_packed", "bits": bits, "group_size": gs},
            dense_key: {"format": "dense", "dtype": "bfloat16"},
            "layers.0.bogus": {"format": "nonsense_packed", "bits": bits, "group_size": gs},
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=0))

    return {"aff_packed": aff_packed, "aff_ref": aff_ref,
            "awq_packed": awq_packed, "awq_ref": awq_recovered_ref,
            "dense_key": dense_key, "dense_w": dense_w}


def _maxabs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def run() -> None:
    out = Path(tempfile.mkdtemp()) / "dsv4-quanta_test"
    ref = _build_artifact(out)
    art = DSV4Artifact(out)
    ok = True

    print("\n=== DSV4 artifact reader ===")

    # cfg loaded self-contained
    cfg_ok = art.cfg.num_hidden_layers == 2 and art.cfg.n_routed_experts == 4
    ok = ok and cfg_ok
    print(f"  [{'OK' if cfg_ok else 'FAIL'}] config self-contained "
          f"(layers={art.cfg.num_hidden_layers} experts={art.cfg.n_routed_experts})")

    # affine read() round-trips to the source weight (loose tol — int4 g32)
    aff = art.read("layers.0.attn.wq_a.weight")
    aff_drift = _maxabs(aff, ref["aff_ref"])
    aff_ok = aff.dtype == mx.bfloat16 and aff_drift < 5e-2
    ok = ok and aff_ok
    print(f"  [{'OK' if aff_ok else 'FAIL'}] read(affine) round-trip  dtype={aff.dtype} "
          f"max|Δ|={aff_drift:.2e}")

    # raw() returns the packed codes verbatim
    raw_codes = art.raw("layers.0.attn.wq_a.weight")
    raw_ok = raw_codes.dtype == ref["aff_packed"].dtype and \
        _maxabs(raw_codes.astype(mx.float32), ref["aff_packed"].astype(mx.float32)) == 0.0
    ok = ok and raw_ok
    print(f"  [{'OK' if raw_ok else 'FAIL'}] raw() packed codes verbatim  shape={tuple(raw_codes.shape)} "
          f"dtype={raw_codes.dtype}")

    # awq read() recovers W via /s
    awq = art.read("layers.0.ffn.experts.0.w1.weight")
    awq_drift = _maxabs(awq, ref["awq_ref"])
    awq_ok = awq.dtype == mx.bfloat16 and awq_drift < 1e-2
    ok = ok and awq_ok
    print(f"  [{'OK' if awq_ok else 'FAIL'}] read(awq) recover W=deq/s  max|Δ|={awq_drift:.2e}")

    # dense read() returns bf16 verbatim
    dense = art.read(ref["dense_key"])
    dense_ok = dense.dtype == mx.bfloat16 and _maxabs(dense, ref["dense_w"]) == 0.0
    ok = ok and dense_ok
    print(f"  [{'OK' if dense_ok else 'FAIL'}] read(dense) bf16 verbatim  dtype={dense.dtype}")

    # get() raw shard access on the dense key
    get_ok = _maxabs(art.get(ref["dense_key"]), ref["dense_w"]) == 0.0
    ok = ok and get_ok
    print(f"  [{'OK' if get_ok else 'FAIL'}] get() exact-key shard access")

    # fail-loud on an unknown manifest format (no silent fallback / wrong-bits dequant)
    try:
        art.read("layers.0.bogus.weight")
        fail_loud_ok = False
    except ValueError:
        fail_loud_ok = True
    ok = ok and fail_loud_ok
    print(f"  [{'OK' if fail_loud_ok else 'FAIL'}] read(unknown format) raises (no silent fallback)")

    # raw() on a dense key fails loud (dense has no packed codes)
    try:
        art.raw(ref["dense_key"])
        raw_dense_ok = False
    except ValueError:
        raw_dense_ok = True
    ok = ok and raw_dense_ok
    print(f"  [{'OK' if raw_dense_ok else 'FAIL'}] raw(dense) raises (no packed codes)")

    # missing key fails loud
    try:
        art.read("layers.9.does.not.exist.weight")
        missing_ok = False
    except KeyError:
        missing_ok = True
    ok = ok and missing_ok
    print(f"  [{'OK' if missing_ok else 'FAIL'}] read(missing key) raises")

    art.release()
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    run()
