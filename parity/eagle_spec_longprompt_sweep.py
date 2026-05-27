"""Long-prompt EAGLE spec-decode sweep against the stabilized int3g128 drafter (task #124 follow-up).

The short-prompt sweep (``parity/eagle_spec_sweep.py``, 32-token prompt / 128-token output) measures
the break-even regime — useful for "is the drafter worth shipping" but unrepresentative of agentic
serving, where decode reads multi-thousand-token KV caches and spec-decode wins multiply. This sweep
re-runs the same configs at realistic prompt lengths, with **decode-only tok/s separated from
prefill** so the comparison is apples-to-apples on the steady-state serving number.

Within each prompt length, all configs see the SAME prompt (a strict prefix of the features
``in_tokens``), so config-vs-config differences are real. The prefill probe (``generate(max_new=1)``)
runs once per length; ``decode_time = total_time - prefill_time`` for each subsequent run.

    uv run python -m parity.eagle_spec_longprompt_sweep

NOTE: ~410 GB resident — run only with the memory free, never alongside another big-resident job.
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.eagle.capture import load_features
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.spec import LAYERS, spec_generate
from quanta.eagle.train import load_drafter, load_frozen_embed_head
from quanta.generate import generate
from quanta.runtime import ResidentModel

ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int3g128"
DRAFTER = "/Users/pmrj/models/kimi_eagle/drafter_int3g128_stable.safetensors"
FEATURES = "/Users/pmrj/models/kimi_eagle/features_int3g128/feat_0000.safetensors"

PROMPT_LENGTHS = (32, 2048, 8192, 16384)
MAX_TOKENS = (1024,)  # standard reference for LLM speed evals; decode-dominated at all prompt lengths

# (k, absorbed) — same set as the short sweep so numbers stay directly comparable
CONFIGS: list[tuple[int, bool]] = [
    (4, False),
    (3, False),
    (2, False),
    (4, True),
    (2, True),
]


def run() -> None:
    mx.set_wired_limit(int(480 * 1024**3))
    t_load = time.perf_counter()
    model = ResidentModel(ART)
    embed, head = load_frozen_embed_head(ART)
    drafter = load_drafter(DRAFTER, EagleDrafter(
        hidden=embed.shape[1], n_heads=56, head_dim=128, intermediate=14336, rope_base=50000.0))
    mx.eval(drafter.parameters())
    full_in = load_features(FEATURES)["in_tokens"]
    print(f"loaded resident + drafter in {(time.perf_counter() - t_load) / 60:.1f} min "
          f"| drafter {DRAFTER}", flush=True)

    for L in PROMPT_LENGTHS:
        if L > full_in.shape[0]:
            print(f"\n!! skip L={L}: features only has {full_in.shape[0]} tokens", flush=True)
            continue
        prompt = [int(x) for x in full_in[:L].tolist()]

        # Warm every config at this length once (prefill + first few decodes compile/cache JIT shapes
        # for both the baseline path and every spec config). One warmup per L covers all MAX_TOKENS.
        generate(model, prompt, max_new_tokens=8, temperature=0.0)
        for k, absorbed in CONFIGS:
            spec_generate(model, drafter, embed, head, prompt, max_new=k + 1, k=k,
                          layers=LAYERS, absorbed=absorbed)

        # Prefill probe (once per L): baseline generate with max_new=1 → time ≈ prefill + 1 decode step.
        # Decode time per timed run = total - prefill_probe. Same prefill cost is subtracted from every
        # config, so the speedup ratio is exact.
        t0 = time.perf_counter()
        _ = generate(model, prompt, max_new_tokens=1, temperature=0.0)
        t_prefill = time.perf_counter() - t0

        for MAXN in MAX_TOKENS:
            print(f"\n=== prompt L={L} | max_new={MAXN} | prefill {t_prefill:.2f}s ===", flush=True)

            # Baseline (greedy)
            t0 = time.perf_counter()
            base = generate(model, prompt, max_new_tokens=MAXN, temperature=0.0)
            base_total = time.perf_counter() - t0
            base_decode = max(base_total - t_prefill, 1e-3)
            base_decode_tps = (len(base) - 1) / base_decode  # subtract the probe's 1 decode step
            print(f"baseline      : {len(base):>4} tok  total {base_total:5.1f}s  "
                  f"decode {base_decode:5.1f}s  {base_decode_tps:5.2f} tok/s (decode-only)",
                  flush=True)

            # Spec-decode configs
            print(f"{'k':>2} {'absorb':>6} {'tok':>4} {'total':>6} {'decode':>6} {'tok/s':>6} "
                  f"{'mean_accept':>11} {'decode_speedup':>14}", flush=True)
            for k, absorbed in CONFIGS:
                t0 = time.perf_counter()
                spec, stats = spec_generate(model, drafter, embed, head, prompt, max_new=MAXN, k=k,
                                            layers=LAYERS, absorbed=absorbed)
                total = time.perf_counter() - t0
                decode = max(total - t_prefill, 1e-3)
                tps = len(spec) / decode
                speedup = tps / base_decode_tps
                print(f"{k:>2} {str(absorbed):>6} {len(spec):>4} {total:>6.1f} {decode:>6.1f} "
                      f"{tps:>6.2f} {stats['mean_accept']:>11.2f} {speedup:>13.2f}x", flush=True)


if __name__ == "__main__":
    run()
