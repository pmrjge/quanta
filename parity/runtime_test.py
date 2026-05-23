"""Validate the resident quantized runtime layer vs bf16 (bounded: bake L0, no experts).

Bakes L0 only (dense → int8 attention + int8 MLP, fast, no GPTQ), loads it as a resident
quantized layer, and compares its forward to the bf16 L0. The gap is int8 quantization error
(small), proving the artifact loader + nn.QuantizedLinear assembly + resident layer forward.
(QuantizedSparseMoE is validated separately and exact.)

    uv run --with tiktoken python -m parity.runtime_test
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx

from quanta.bake.orchestrate import bake
from quanta.config import KimiTextConfig
from quanta.loader import SourceCheckpoint
from quanta.model import build_runtime_layer, load_layer_raw
from quanta.runtime import ResidentArtifact, build_resident_layer
from quanta.tokenizer import KimiTokenizer

MODEL = "/Users/pmrj/models/Kimi-K2.6"
PROSE = "Photosynthesis converts light into chemical energy stored in sugars."


def run() -> None:
    cfg = KimiTextConfig.from_pretrained(MODEL)
    tok = KimiTokenizer(MODEL, bos_id=cfg.bos_token_id)
    ids = mx.array(tok.encode(PROSE, add_bos=True))
    out = Path(tempfile.mkdtemp()) / "kimi-quanta_l0"

    bake(MODEL, out, ids, n_layers=1, include_head=False)  # L0 dense only: int8, no experts

    rlayer = build_resident_layer(ResidentArtifact(out), 0)  # resident quantized L0
    ck = SourceCheckpoint(MODEL)
    blayer = build_runtime_layer(cfg, load_layer_raw(ck, cfg, 0, mx.bfloat16))  # bf16 L0

    h = mx.random.normal((1, 8, cfg.hidden_size)).astype(mx.bfloat16)
    pos = mx.arange(8)
    yb = blayer(h, pos, use_fast=True)
    yr = rlayer(h, pos, use_fast=True)
    rel = (mx.linalg.norm((yb - yr).astype(mx.float32)) / mx.linalg.norm(yb.astype(mx.float32))).item()

    print("\n=== resident quantized L0 vs bf16 ===")
    print(f"resident int8 layer vs bf16 L0: rel {rel:.4%}  (int8 quant error)")
    assert rel < 0.05, "resident quantized layer should match bf16 within int8 error"
    print("resident runtime layer (loader + QuantizedLinear + forward) OK")


if __name__ == "__main__":
    run()
