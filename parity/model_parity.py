"""End-to-end full-model parity: streamed runtime vs plain-mlx.core reference.

Streams each decoder layer once (one resident at a time), advances an independent
reference residual stream and runtime residual stream through it, and reports the
per-layer drift. After the final norm + lm_head it reports logits error and the
top-1 next-token agreement — the end-to-end arbiter.

    uv run python -m parity.model_parity        # all 61 layers
    uv run python -m parity.model_parity 4      # first 4 layers (fast)
"""

from __future__ import annotations

import sys

import mlx.core as mx

from parity.reference import _rms, reference_layer0, reference_moe_layer
from quanta.config import KimiTextConfig
from quanta.loader import SourceCheckpoint
from quanta.model import FINAL_NORM_KEY, LM_HEAD_KEY, build_runtime_layer, load_layer_raw

MODEL = "/Users/pmrj/models/Kimi-K2.6"
TOKEN_IDS = [163584, 100, 500, 1024, 2048, 4096, 8192, 16000, 32000, 64000, 100000, 120000, 150000, 42, 7, 9001]


def _diff(a: mx.array, b: mx.array) -> tuple[float, float]:
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    abs_err = mx.max(mx.abs(a - b)).item()
    denom = mx.maximum(mx.abs(a), mx.abs(b))
    rel = mx.max(mx.abs(a - b) / mx.where(denom > 0, denom, mx.array(1.0))).item()
    return abs_err, rel


def run(n_layers: int | None = None, dtype: mx.Dtype = mx.bfloat16) -> None:
    cfg = KimiTextConfig.from_pretrained(MODEL)
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    ck = SourceCheckpoint(MODEL)

    ids = mx.array(TOKEN_IDS)
    h0 = ck.embed_tokens(ids)[None].astype(dtype)
    pos = mx.arange(h0.shape[1])
    h_ref = h0
    h_rt = h0
    max_drift = 0.0

    for i in range(n):
        raw = load_layer_raw(ck, cfg, i, dtype)
        if raw["kind"] == "dense":
            h_ref = reference_layer0(h_ref, raw["weights"], cfg, pos, dtype=dtype)["hout"]
        else:
            h_ref = reference_moe_layer(h_ref, raw["weights"], raw["experts"], cfg, pos, dtype=dtype)["hout"]
        layer = build_runtime_layer(cfg, raw)
        h_rt = layer(h_rt, pos)
        mx.eval(h_ref, h_rt)
        a, _ = _diff(h_ref, h_rt)
        max_drift = max(max_drift, a)
        if i < 3 or i == n - 1 or a > 1e-2:
            print(f"layer {i:2d} [{raw['kind']:5}] hout max_abs {a:.3e}")
        del layer, raw
        ck.release()

    norm_w = ck.read(FINAL_NORM_KEY).astype(dtype)
    lm_head = ck.read(LM_HEAD_KEY).astype(dtype)
    ref_logits = _rms(h_ref, norm_w, cfg.rms_norm_eps) @ lm_head.T
    rt_logits = mx.fast.rms_norm(h_rt, norm_w, cfg.rms_norm_eps) @ lm_head.T
    mx.eval(ref_logits, rt_logits)

    a, r = _diff(ref_logits, rt_logits)
    top1 = (mx.argmax(ref_logits, -1) == mx.argmax(rt_logits, -1)).astype(mx.float32).mean().item()
    print(f"\nlayers={n} dtype=bf16")
    print(f"max per-layer hout drift : {max_drift:.3e}")
    print(f"final logits             : max_abs {a:.3e}  max_rel {r:.3e}")
    print(f"top-1 next-token agree   : {top1:.4f}")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else None)
