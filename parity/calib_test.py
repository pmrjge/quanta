"""Validate calibration capture + the DP sensitivity metric (bounded, ~1 expert loaded).

Captures L1's (ln2, idx) via the no-experts path (L1 is the last requested layer, so the
residual is never advanced through the 34 GB experts), checks shapes/routing, and computes
the activation-weighted int3-vs-int4 error on one real expert — exercising the exact metric
the DP allocator will use. Also surfaces how under-covered experts are (n << in), which is
why GPTQ uses the Woodbury inverse.

    uv run --with tiktoken python -m parity.calib_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.bake.calibrate import activation_weighted_error, capture_calibration, expert_rows
from quanta.compressed_int4 import dequantize_packed_int4
from quanta.config import KimiTextConfig
from quanta.loader import SourceCheckpoint
from quanta.tokenizer import KimiTokenizer

MODEL = "/Users/pmrj/models/Kimi-K2.6"
PROSE = (
    "Photosynthesis is the process by which green plants, algae, and some bacteria convert "
    "light energy into chemical energy stored in sugars. Inside the chloroplasts, chlorophyll "
    "absorbs sunlight, driving the splitting of water into oxygen, protons, and electrons."
)


def run() -> None:
    cfg = KimiTextConfig.from_pretrained(MODEL)
    ck = SourceCheckpoint(MODEL)
    tok = KimiTokenizer(MODEL, bos_id=cfg.bos_token_id)
    ids = mx.array(tok.encode(PROSE, add_bos=True))

    caps = capture_calibration(ck, cfg, ids, n_layers=2)  # L0 dense + L1 capture (no experts)
    ln2, idx = caps[0]
    n_tok, hidden = ln2.shape
    print(f"\n=== calibration capture @ L1 (tokens={n_tok}) ===")
    print(f"ln2 shape {tuple(ln2.shape)}  idx shape {tuple(idx.shape)}  (topk={idx.shape[1]})")

    e_range = mx.arange(cfg.n_routed_experts)
    counts = mx.sum((idx[..., None] == e_range[None, None, :]), axis=(0, 1))  # [n_experts]
    e = int(mx.argmax(counts).item())
    xe = expert_rows(ln2, idx, e)
    print(f"routing: {int(mx.sum(counts > 0).item())}/{cfg.n_routed_experts} experts hit; "
          f"max rows/expert={int(mx.max(counts).item())} (in={hidden} ⇒ n<<in, Woodbury)")

    base = f"language_model.model.layers.1.mlp.experts.{e}.gate_proj."
    w = dequantize_packed_int4(
        ck.read(base + "weight_packed"), ck.read(base + "weight_scale"), 2048, hidden, 32, mx.float32
    )
    err3 = activation_weighted_error(w, xe, 3)
    err4 = activation_weighted_error(w, xe, 4)
    print(f"\nexpert {e} gate_proj, X={tuple(xe.shape)}  (activation-weighted RTN proxy):")
    print(f"  int3 err {err3:.4%}   int4 err {err4:.4%}   (int4 < int3 expected; GPTQ lowers both)")
    assert err4 < err3, "sensitivity metric should rank int4 below int3"
    print("calibration capture + sensitivity metric OK")


if __name__ == "__main__":
    run()
