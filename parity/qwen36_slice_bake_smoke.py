"""Bounded slice-bake smoke for Qwen3.6-35B-A3B: real loader→bake→artifact round-trip, ~few GB.

NOT a parity gate — a de-risking smoke before the multi-hour full bake. Bakes a small prefix
(layers 0-2 linear + layer 3 full) + the native MTP block + head from the REAL bf16 checkpoint, then
re-opens the artifact and dequantizes one layer of each kind. Exercises, on real tensors:

  * the streamed loader (incl. BF16 ``A_log``/``dt_bias``/``norm`` + ``conv1d`` ``[C,1,K]``),
  * 3-D pre-stacked expert int4 quant + 2-D int8 quant (``quantize_affine`` / ``mx.quantize``),
  * the FIXED MTP path (fused pre-stacked experts, baked via ``_bake_moe_block``),
  * ``Qwen35Artifact`` dequant read-back + the baked dynamic-YaRN config.json policy.

One model only, bounded (n_layers=4 + MTP + head), throwaway artifact under /tmp. GPU must be free.

    uv run python -u -m parity.qwen36_slice_bake_smoke
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx

from quanta.qwen35.artifact import Qwen35Artifact
from quanta.qwen35.bake import bake_qwen35

SRC = "/Users/pmrj/models/Qwen3.6-35B-A3B"
OUT = "/tmp/qwen36_slice_bake"
N_LAYERS = 4  # 0,1,2 linear + 3 full — both mixer kinds


def run() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    print(f"slice-baking {N_LAYERS} layers + MTP + head from {SRC} -> {OUT}", flush=True)
    summary = bake_qwen35(SRC, OUT, mx.array([0, 1, 2, 3]),  # dummy calib (capture_acts=False ⇒ unused)
                          n_layers=N_LAYERS, include_head=True, include_mtp=True,
                          group_size=64, scale_dtype=mx.bfloat16)
    print("bake summary:", json.dumps(summary, default=int), flush=True)

    # --- re-open + dequantize one layer of each kind (proves the artifact round-trips) ---
    art = Qwen35Artifact(OUT)
    cfg = art.cfg
    print(f"reopened artifact: layers(cfg)={cfg.num_hidden_layers} experts={cfg.num_experts} "
          f"top-{cfg.num_experts_per_tok}", flush=True)

    lin = art.linear_attn(0)      # BF16 SSM control + int8 projections, dequantized
    full = art.full_attn(3)       # int8 q/k/v/o + bf16 q/k norm
    moe0 = art.moe(0)             # fused pre-stacked experts dequantized
    mtp = art.mtp(0)             # THE FIX: fused pre-stacked MTP experts

    checks = {
        "linear in_proj_qkv": tuple(lin["in_proj_qkv.weight"].shape) == (cfg.linear_qkv_dim, cfg.hidden_size),
        "linear A_log finite": bool(mx.all(mx.isfinite(lin["A_log"])).item()),
        "linear conv1d ndim": lin["conv1d.weight"].ndim in (2, 3),
        "full q_proj": tuple(full["q_proj.weight"].shape) == (cfg.q_proj_out, cfg.hidden_size),
        "moe experts_gate_up": tuple(moe0["experts_gate_up"].shape) == (cfg.num_experts, cfg.moe_gate_up_out, cfg.hidden_size),
        "moe experts_down": tuple(moe0["experts_down"].shape) == (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
        "moe gate_up finite": bool(mx.all(mx.isfinite(moe0["experts_gate_up"])).item()),
        "mtp fused experts_gate_up": tuple(mtp["moe"]["experts_gate_up"].shape) == (cfg.num_experts, cfg.moe_gate_up_out, cfg.hidden_size),
        "mtp fused experts_down": tuple(mtp["moe"]["experts_down"].shape) == (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
        "mtp fc shape": tuple(mtp["fc"].shape) == (cfg.hidden_size, 2 * cfg.hidden_size),
    }
    # baked dynamic-YaRN policy landed in config.json
    conf = json.loads((Path(OUT) / "config.json").read_text())
    pol = conf.get("quanta_long_context", {})
    checks["yarn policy baked"] = (pol.get("yarn_dynamic") is True
                                   and pol.get("yarn_original_max") == cfg.yarn_original_max)
    # the two-eos stop set survives into the artifact (gen-eos <|im_end|> 248046 + <|endoftext|> 248044)
    checks["two-eos preserved"] = (cfg.eos_token_id == 248046
                                   and set(cfg.eos_token_ids) == {248046, 248044})
    # self-contained: tokenizer + generation metadata copied in
    checks["metadata sidecars copied"] = all(
        (Path(OUT) / f).exists()
        for f in ("generation_config.json", "tokenizer_config.json", "tokenizer.json",
                  "vocab.json", "merges.txt"))

    for k, v in checks.items():
        print(f"  [{'OK' if v else 'XX'}] {k}", flush=True)
    ok = all(checks.values())
    print("SMOKE PASS" if ok else "SMOKE FAIL", flush=True)
    shutil.rmtree(OUT, ignore_errors=True)  # throwaway
    assert ok


if __name__ == "__main__":
    run()
