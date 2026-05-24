"""int8 MLACache: round-trip, decode==prefill (per-token), and save/load — no model load.

The int8 latent must (a) reconstruct c_kv within int8 tolerance, (b) be bit-identical whether
filled in one prefill or one token at a time (per-token quant → decode==prefill exactly), (c)
leave the bf16 path untouched, and (d) round-trip through save/load.

    uv run python -m parity.mla_int8_test
"""

from __future__ import annotations

import tempfile

import mlx.core as mx

from quanta.cache import MLACache, load_caches, save_caches

B, S, LORA, ROPE = 1, 40, 512, 64


def run() -> None:
    mx.random.seed(0)
    ckv = mx.random.normal((B, S, LORA)).astype(mx.bfloat16)
    kpe = mx.random.normal((B, 1, S, ROPE)).astype(mx.bfloat16)

    def md(a, b):
        return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))

    # (a) bulk prefill, int8 round-trip
    c = MLACache(quantized=True)
    deq, _ = c.update(ckv, kpe)
    rel = md(deq, ckv) / (float(mx.max(mx.abs(ckv.astype(mx.float32)))) + 1e-6)
    rt_ok = rel < 2e-2 and c.offset == S

    # (b) per-token (decode) == bulk prefill, exactly
    c2 = MLACache(quantized=True)
    d2 = None
    for t in range(S):
        d2, _ = c2.update(ckv[:, t : t + 1], kpe[:, :, t : t + 1])
    decode_ok = md(d2, deq) == 0.0 and c2.offset == S

    # (c) bf16 path unchanged (exact)
    c3 = MLACache()
    d3, _ = c3.update(ckv, kpe)
    bf16_ok = md(d3, ckv) == 0.0

    # (d) save/load round-trip (int8)
    p = tempfile.mktemp(suffix=".safetensors")
    save_caches(p, [c])
    loaded = load_caches(p)
    sl_ok = loaded[0].quantized and md(loaded[0]._dequant(), deq) == 0.0

    print("\n=== int8 MLACache ===")
    print(f"int8 round-trip (rel {rel:.2e} < 2e-2)   : {rt_ok}")
    print(f"per-token decode == bulk prefill (exact) : {decode_ok}")
    print(f"bf16 path unchanged (exact)              : {bf16_ok}")
    print(f"save/load round-trip (int8)              : {sl_ok}")
    assert all([rt_ok, decode_ok, bf16_ok, sl_ok])
    print("int8 MLACache OK (round-trip; decode==prefill; bf16 intact; persists)")


if __name__ == "__main__":
    run()
