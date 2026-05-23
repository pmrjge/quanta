"""De-risk the full pipeline: bake a small COMPLETE model, load resident, compare to bf16.

Bakes L0+L1+head with ALL experts (real expert scale + the embed/lm_head int8 path), loads
it as a ResidentModel, and compares a 2-layer forward to the bf16 KimiModel. The gap is
quantization error (int8 non-experts + int3/int4 experts). Proves bake -> artifact ->
resident -> forward end to end before the full multi-hour bake.

    uv run --with tiktoken python -m parity.resident_full_test
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import mlx.core as mx

from quanta.bake.orchestrate import bake
from quanta.config import KimiTextConfig
from quanta.loader import SourceCheckpoint
from quanta.model import KimiModel
from quanta.runtime import ResidentModel
from quanta.tokenizer import KimiTokenizer

MODEL = "/Users/pmrj/models/Kimi-K2.6"
PROSE = (
    "Photosynthesis is the process by which green plants, algae, and some bacteria convert "
    "light energy into chemical energy stored in sugars, releasing oxygen as a byproduct."
)


def run() -> None:
    cfg = KimiTextConfig.from_pretrained(MODEL)
    tok = KimiTokenizer(MODEL, bos_id=cfg.bos_token_id)
    ids = mx.array(tok.encode(PROSE, add_bos=True))
    out = Path(tempfile.mkdtemp()) / "kimi-quanta_small"

    t0 = time.perf_counter()
    stats = bake(MODEL, out, ids, n_layers=2, include_head=True)  # L0+L1+head, ALL experts
    print(f"\nbake(L0+L1+head, all experts) {time.perf_counter() - t0:.0f}s  {stats}")

    rm = ResidentModel(out, n_layers=2)
    yq = rm(ids, n_layers=2, sparse=None)[0].astype(mx.float32)  # resident quantized logits

    bm = KimiModel(cfg, SourceCheckpoint(MODEL), mx.bfloat16)
    yb = bm(ids, n_layers=2, use_fast=True, sparse=None)[0].astype(mx.float32)  # bf16 logits

    rel = (mx.linalg.norm(yq - yb) / mx.linalg.norm(yb)).item()
    top1 = (mx.argmax(yq, -1) == mx.argmax(yb, -1)).astype(mx.float32).mean().item()
    print("\n=== resident quantized vs bf16 (2 layers) ===")
    print(f"relative logit error : {rel:.4%}")
    print(f"top-1 agreement      : {top1:.3f}")
    assert rel < 0.25, "resident quantized model should track bf16"
    print(f"full pipeline OK (bake -> resident -> forward) at {out}")


if __name__ == "__main__":
    run()
