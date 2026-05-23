"""L0 parity gate: plain-mlx.core reference vs the mlx.nn runtime.

Runs identical token ids through both and reports per-op max abs / max rel error,
in fp32 (tight math gate) and bf16 (realistic floor). Also checks the runtime's
flash ``mx.fast`` path (use_fast=True) against the naive reference end-to-end.

    uv run python -m parity.layer0
"""

from __future__ import annotations

import mlx.core as mx

from parity.reference import reference_layer0
from quanta.config import KimiTextConfig
from quanta.loader import SourceCheckpoint
from quanta.modeling.decoder import DenseDecoderLayer

MODEL = "/Users/pmrj/models/Kimi-K2.6"

# Fixed token ids (BOS + 15 arbitrary valid ids) for reproducibility.
TOKEN_IDS = [163584, 100, 500, 1024, 2048, 4096, 8192, 16000, 32000, 64000, 100000, 120000, 150000, 42, 7, 9001]

BOUNDARIES = ["ln1", "attn", "resid1", "ln2", "mlp", "hout"]


def _diff(a: mx.array, b: mx.array) -> tuple[float, float]:
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    abs_err = mx.max(mx.abs(a - b)).item()
    denom = mx.maximum(mx.abs(a), mx.abs(b))
    rel_err = mx.max(mx.abs(a - b) / mx.where(denom > 0, denom, mx.array(1.0))).item()
    return abs_err, rel_err


def _runtime_intermediates(layer: DenseDecoderLayer, h: mx.array, pos: mx.array) -> dict[str, mx.array]:
    ln1 = layer.input_layernorm(h)
    attn = layer.self_attn(ln1, pos, use_fast=False)
    resid1 = h + attn
    ln2 = layer.post_attention_layernorm(resid1)
    mlp = layer.mlp(ln2)
    return {"ln1": ln1, "attn": attn, "resid1": resid1, "ln2": ln2, "mlp": mlp, "hout": resid1 + mlp}


def run(dtype: mx.Dtype) -> None:
    cfg = KimiTextConfig.from_pretrained(MODEL)
    ck = SourceCheckpoint(MODEL)
    weights = ck.load_dense_layer(0)

    ids = mx.array(TOKEN_IDS)
    h = ck.embed_tokens(ids)[None].astype(dtype)  # [1, T, hidden]
    pos = mx.arange(h.shape[1])

    ref = reference_layer0(h, weights, cfg, pos, dtype=dtype)

    layer = DenseDecoderLayer(cfg)
    layer.load_weights([(k, v.astype(dtype)) for k, v in weights.items()])

    rt = _runtime_intermediates(layer, h, pos)
    hout_naive = layer(h, pos, use_fast=False)
    hout_fast = layer(h, pos, use_fast=True)
    layer.self_attn.absorbed = True
    hout_abs = layer(h, pos, use_fast=False)
    layer.self_attn.absorbed = False
    mx.eval(list(ref.values()), list(rt.values()), hout_naive, hout_fast, hout_abs)

    name = {mx.float32: "fp32", mx.bfloat16: "bf16"}[dtype]
    print(f"\n=== L0 parity ({name}) — reference (plain mlx.core) vs runtime (mlx.nn) ===")
    print(f"{'op':<10}{'max_abs':>14}{'max_rel':>14}")
    for k in BOUNDARIES:
        a, r = _diff(ref[k], rt[k])
        print(f"{k:<10}{a:>14.3e}{r:>14.3e}")

    a, r = _diff(ref["hout"], hout_fast)
    print(f"{'hout/fast':<10}{a:>14.3e}{r:>14.3e}   (mx.fast rope+sdpa vs naive reference)")
    a, r = _diff(ref["hout"], hout_abs)
    print(f"{'hout/absrb':<10}{a:>14.3e}{r:>14.3e}   (#3 absorbed MLA vs naive reference)")
    a, r = _diff(hout_naive, rt["hout"])
    print(f"full-call naive vs stepwise: max_abs {a:.3e}")


if __name__ == "__main__":
    run(mx.float32)
    run(mx.bfloat16)
