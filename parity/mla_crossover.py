"""Expand-vs-absorb MLA decode crossover (bounded: one layer's attention, no experts).

At decode (one new query over a cache of length S) the expanded path reconstructs per-head
K/V from the latent each step (materializes [B,H,S,256] and a kv_b_proj over S), while the
absorbed path folds W_UK into the query and attends the compressed c_kv directly. This times
a single decode step both ways across growing S to find where absorbed starts winning, so the
decode loop can pick the cheaper path per cache length. Output-equivalence is already proven
(#12); this is purely timing/memory.

    uv run python -m parity.mla_crossover
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.cache import MLACache
from quanta.config import KimiTextConfig
from quanta.loader import SourceCheckpoint
from quanta.model import load_module_weights
from quanta.modeling.attention import MLAAttention

MODEL = "/Users/pmrj/models/Kimi-K2.6"
SEQS = (512, 2048, 8192, 32768, 131072, 262144)


def _time(step, iters: int = 3) -> float:
    mx.eval(step())  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(step())
    return (time.perf_counter() - t0) / iters * 1000  # ms


def run() -> None:
    cfg = KimiTextConfig.from_pretrained(MODEL)
    ck = SourceCheckpoint(MODEL)
    ne = ck.load_moe_nonexpert(1)  # attention + router + shared (no 34GB experts)
    attn = MLAAttention(cfg)
    aw = {k[len("self_attn."):]: v for k, v in ne.items() if k.startswith("self_attn.")}
    load_module_weights(attn, aw)
    ck.release()

    kv_lora, rope, hidden = cfg.kv_lora_rank, cfg.qk_rope_head_dim, cfg.hidden_size
    x = mx.random.normal((1, 1, hidden)).astype(mx.bfloat16)

    print("\n=== MLA decode step: expanded vs absorbed (ms) ===")
    print(f"{'cache S':>9} {'expanded':>10} {'absorbed':>10} {'winner':>9} {'speedup':>8}")
    for s in SEQS:
        c_kv = mx.random.normal((1, s, kv_lora)).astype(mx.bfloat16)
        k_pe = mx.random.normal((1, 1, s, rope)).astype(mx.bfloat16)
        pos = mx.array([s])

        def step(absorbed: bool) -> mx.array:
            attn.absorbed = absorbed
            cache = MLACache()
            cache.c_kv, cache.k_pe = c_kv, k_pe
            return attn(x, pos, use_fast=True, cache=cache)

        exp_ms = _time(lambda: step(False))
        abs_ms = _time(lambda: step(True))
        win = "absorbed" if abs_ms < exp_ms else "expanded"
        print(f"{s:>9} {exp_ms:>10.2f} {abs_ms:>10.2f} {win:>9} {max(exp_ms, abs_ms) / min(exp_ms, abs_ms):>7.2f}x")


if __name__ == "__main__":
    run()
