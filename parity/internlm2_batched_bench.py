"""InternLM2.5-7B batched decode throughput — fused (#176 default) vs per-stream loop.

Benchmarks the fused ``step_batch`` default (``InternLM2ResidentModel.decode_batched``: per-stream
``mx.fast.rope`` + one padded ``mx.fast.scaled_dot_product_attention`` + batched ``_qmm`` across ``B``
streams) against the retained ``_step_batch_looped`` reference (``B`` independent single-stream
forwards) on the resident int8-g64 InternLM2.5-7B bake. This is the **first real-model exercise** of
the packed ``_PackedModel.decode_batched`` path — its model-free parity is gated by
``parity/internlm2_batched_attention_test.py``; here we confirm it on the actual baked weights.

Two parts:
  A. **parity** — fused vs looped from identical caches:
       A1. ``B=1`` (no batched kernels) must be **bit-exact** — proves a faithful port; any Δ is a
           real bug. (This caught the bf16 RoPE drift: a hand-rolled fp32-then-cast batched rotate-half
           matched in fp32 but its bf16 ULPs compounded over 32 layers and flipped greedy tokens. The
           fix loops the runtime's own ``mx.fast.rope`` per stream → ``B=1`` is now 0.0e0.)
       A2. ``B=4`` ragged offsets must be **greedy-exact** — once batched matmul + padded SDPA engage,
           ``|Δlogit|`` is non-zero (per-row-independent reduction-order ULP, input-dependent), but
           argmax-stable: the equivalence class the project accepts for every batched/tiled path.
  B. **throughput** — ``B in {1,2,4,8,16,32}``, uniform 1024-tok prefill, ``GEN=64`` decode/stream:
     per-stream + aggregate tok/s for the fused and looped paths, and the fused/looped speedup.

One model only (int8-g64 ≈ 8.3 GB on disk, ~6.2 GB resident) — safe to run solo on the M3 Ultra.

    uv run --with tokenizers python -u -m parity.internlm2_batched_bench
"""

from __future__ import annotations

import resource
import sys
import time

import mlx.core as mx

from quanta.internlm2.batched_runtime import InternLM2BatchedResidentModel

ART = "/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64"
WARMUP_PROMPT_LEN = 1024
GEN = 64                          # timed decode tokens per stream
WARMUP_STEPS = 4                  # steady-state ramp-up (JIT + KV warm) — not timed
BATCH_SIZES = (1, 2, 4, 8, 16, 32)


def _peak_rss_gib() -> float:
    """Peak resident set in GiB (macOS ru_maxrss is bytes; Linux KiB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2)


def _prompt(n: int, bos: int) -> mx.array:
    """A deterministic ``bos + ramp`` prompt of length ``n`` (uniform across streams ⇒ even KV load)."""
    return mx.array([bos] + list(range(1, n)))


def _seed(model: InternLM2BatchedResidentModel, prompts: list[mx.array]) -> tuple[list, list[int]]:
    """Prefill each prompt into its own fresh decode cache; return (caches, per-stream offsets)."""
    caches, offsets = [], []
    for p in prompts:
        c = model.new_cache()
        mx.eval(model.prefill(p, c))      # grows the cache in place (last_only logits discarded)
        caches.append(c)
        offsets.append(c.offset)          # == len(p): abs position of the first decode token
    return caches, offsets


def _decode_compare(model: InternLM2BatchedResidentModel, prompts: list[mx.array], steps: int
                    ) -> tuple[float, bool]:
    """Lock-step decode ``steps`` tokens through the fused ``step_batch`` and the looped reference from
    two identically-seeded cache sets; return (worst ``|Δlogit|``, all-steps-greedy-match)."""
    f_caches, f_off = _seed(model, prompts)
    l_caches, l_off = _seed(model, prompts)
    f_tok = [mx.array([int(p[-1].item())]) for p in prompts]
    l_tok = [mx.array([int(p[-1].item())]) for p in prompts]
    worst, match = 0.0, True
    for _ in range(steps):
        fused = model.step_batch(f_tok, f_caches, offsets=list(f_off))          # fused default
        loope = model._step_batch_looped(l_tok, l_caches, list(l_off))          # retained reference
        mx.eval(fused, loope)
        nf, nl = [], []
        for s in range(len(prompts)):
            fo, lo = fused[s][0, -1], loope[s][0, -1]
            worst = max(worst, float(mx.max(mx.abs(fo - lo))))
            ft, lt = int(mx.argmax(fo).item()), int(mx.argmax(lo).item())
            match = match and (ft == lt)
            nf.append(mx.array([ft]))
            nl.append(mx.array([lt]))
        f_tok, l_tok = nf, nl
        f_off = [o + 1 for o in f_off]
        l_off = [o + 1 for o in l_off]
    return worst, match


def _parity(model: InternLM2BatchedResidentModel) -> None:
    """A: real-model fused step_batch parity (see module docstring). A1 B=1 must be **bit-exact** (a
    real bug otherwise); A2 B=4 ragged must be **greedy-exact** (batched-kernel ULP, argmax-stable)."""
    bos = model.cfg.bos_token_id

    # A1: B=1 — no batched matmul / no padding ⇒ decode_batched must equal the loop bit-for-bit.
    w1, m1 = _decode_compare(model, [_prompt(11, bos)], steps=4)
    print(f"  [{'OK' if (w1 == 0.0 and m1) else 'XX'}] B=1 bit-exact vs loop      "
          f"|Δlogit|={w1:.2e}  (faithful port — no batched kernels engaged)", flush=True)
    assert w1 == 0.0, (f"B=1 decode_batched is NOT bit-exact vs the loop (|Δlogit|={w1:.2e}) — a real "
                       "forward bug, not batched-kernel ULP reorder")

    # A2: B=4 ragged — batched _qmm + padded SDPA engage ⇒ greedy-exact (the accepted batched class).
    lengths = [9, 13, 7, 11]                          # heterogeneous prefill ⇒ ragged RoPE offsets
    worst, match = _decode_compare(model, [_prompt(n, bos) for n in lengths], steps=4)
    print(f"  [{'OK' if match else 'XX'}] B=4 ragged greedy-exact    |Δlogit|={worst:.2e}  "
          f"offsets={lengths} greedy_match={match}  (batched-kernel reduction ULP — argmax-stable)",
          flush=True)
    assert match, "real-model fused greedy tokens diverged from the per-stream loop (B=4 ragged)"


def _time_path(model: InternLM2BatchedResidentModel, prompt_ids: mx.array, b: int,
               fused: bool) -> tuple[float, float]:
    """Time GEN steady-state decode steps at batch ``b`` on the fused or looped path; (per-stream, agg)."""
    caches, offsets = _seed(model, [prompt_ids] * b)
    cur = [mx.array([int(prompt_ids[-1].item())]) for _ in range(b)]
    off = list(offsets)

    def _step(ids: list, cs: list, os: list) -> list:
        if fused:
            return model.step_batch(ids, cs, offsets=os)
        return model._step_batch_looped(ids, cs, os)

    for _ in range(WARMUP_STEPS):
        out = _step(cur, caches, off)
        mx.eval(out)
        cur = [mx.array([int(mx.argmax(out[s][0, -1]).item())]) for s in range(b)]
        off = [o + 1 for o in off]

    t0 = time.perf_counter()
    for _ in range(GEN):
        out = _step(cur, caches, off)
        mx.eval(out)
        cur = [mx.array([int(mx.argmax(out[s][0, -1]).item())]) for s in range(b)]
        off = [o + 1 for o in off]
    dt = time.perf_counter() - t0
    return GEN / dt, b * GEN / dt


def run() -> None:
    mx.set_wired_limit(int(24 * 1024 ** 3))           # ~6.2 GB resident + KV + transient — bounded
    model = InternLM2BatchedResidentModel(ART, max_batch=max(BATCH_SIZES))
    assert model._fused, "batched runtime must default to the fused decode path"
    prompt_ids = _prompt(WARMUP_PROMPT_LEN, model.cfg.bos_token_id)

    print("\nA. real-model parity (fused step_batch == per-stream loop):", flush=True)
    _parity(model)

    print(f"\nB. throughput (int8-g64, prompt {WARMUP_PROMPT_LEN} tok, {GEN} gen/stream, "
          f"fused = #176 default):", flush=True)
    print(f"{'B':>4}  {'fused per/agg tok/s':>24}  {'looped per/agg tok/s':>24}  "
          f"{'agg speedup':>12}  {'peak GiB':>9}", flush=True)
    for b in BATCH_SIZES:
        f_per, f_agg = _time_path(model, prompt_ids, b, fused=True)
        l_per, l_agg = _time_path(model, prompt_ids, b, fused=False)
        spd = f_agg / l_agg if l_agg else float("nan")
        print(f"{b:>4}  {f_per:>10.2f} /{f_agg:>11.2f}  {l_per:>10.2f} /{l_agg:>11.2f}  "
              f"{spd:>10.2f}x  {_peak_rss_gib():>9.2f}", flush=True)


if __name__ == "__main__":
    run()
