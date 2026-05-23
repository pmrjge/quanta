"""Validate the bake orchestration end-to-end on a tiny slice (bounded: L0+L1, 2 experts).

Bakes a slice and checks the artifact: routed experts are affine_packed int3/int4, attention
is int8, norms/router/shared are dense bf16, refs are relative, and a quantized expert
dequantizes to finite values. Proves calibration -> sensitivity -> DP -> GPTQ -> pack ->
artifact wiring; the full bake is the same call over all layers/experts.

    uv run --with tiktoken python -m parity.bake_test
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlx.core as mx

from quanta.bake.calibrate import capture_calibration
from quanta.bake.orchestrate import bake
from quanta.config import KimiTextConfig
from quanta.loader import SourceCheckpoint
from quanta.tokenizer import KimiTokenizer

MODEL = "/Users/pmrj/models/Kimi-K2.6"
PROSE = (
    "Photosynthesis is the process by which green plants, algae, and some bacteria convert "
    "light energy into chemical energy stored in sugars, releasing oxygen as a byproduct."
)
P = "language_model.model.layers."


def run() -> None:
    cfg = KimiTextConfig.from_pretrained(MODEL)
    tok = KimiTokenizer(MODEL, bos_id=cfg.bos_token_id)
    ids = mx.array(tok.encode(PROSE, add_bos=True))
    out = Path(tempfile.mkdtemp()) / "kimi-quanta_int3"

    # pick one routed expert (GPTQ path) + one cold expert (RTN fallback path)
    _, idx = capture_calibration(SourceCheckpoint(MODEL), cfg, ids, n_layers=2)[0]
    counts = mx.sum((idx[..., None] == mx.arange(cfg.n_routed_experts)[None, None, :]), axis=(0, 1))
    routed_e = int(mx.argmax(counts).item())
    cold_e = int(mx.argmin(counts).item())  # a 0-row expert
    print(f"routed expert {routed_e} ({int(counts[routed_e].item())} rows), cold expert "
          f"{cold_e} ({int(counts[cold_e].item())} rows)")

    stats = bake(MODEL, out, ids, n_layers=2, expert_subset=[routed_e, cold_e], include_head=False)
    print(f"=== bake slice (L0+L1, experts {routed_e},{cold_e}) ===\n{stats}")

    idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    man = json.loads((out / "manifest.json").read_text())["tensors"]

    e_gate = f"{P}1.mlp.experts.{routed_e}.gate_proj"
    attn = f"{P}0.self_attn.o_proj"
    norm = f"{P}1.input_layernorm.weight"
    shared = f"{P}1.mlp.shared_experts.gate_proj.weight"

    expert_q = man[e_gate]["format"] == "affine_packed" and man[e_gate]["bits"] in (3, 4)
    attn_int8 = man[attn]["format"] == "affine_packed" and man[attn]["bits"] == 8
    norm_bf16 = man[norm]["format"] == "dense"
    shared_bf16 = man[shared]["format"] == "dense"
    rel_only = all(("/" not in v and ":" not in v) for v in idx.values())

    shard = mx.load(str(out / idx[e_gate + ".weight_packed"]))
    recon = mx.dequantize(shard[e_gate + ".weight_packed"], shard[e_gate + ".weight_scale"],
                          shard[e_gate + ".weight_bias"], group_size=128, bits=man[e_gate]["bits"])
    finite = bool(mx.all(mx.isfinite(recon)).item())

    print(f"expert GPTQ int3/4   : {expert_q} (bits={man[e_gate]['bits']})")
    print(f"attention int8       : {attn_int8}")
    print(f"norm / shared bf16   : {norm_bf16} / {shared_bf16}")
    print(f"relative refs only   : {rel_only}")
    print(f"expert dequant finite: {finite}")
    assert all([expert_q, attn_int8, norm_bf16, shared_bf16, rel_only, finite])
    print(f"bake pipeline OK -> {out}")


if __name__ == "__main__":
    run()
