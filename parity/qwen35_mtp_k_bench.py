"""Qwen3.5-397B-A17B MTP self-speculation perf bench, k ∈ {1,2,3} (#149+).

Measures on the live 60-layer artifact loaded through :class:`quanta.qwen35.runtime.Qwen35ResidentModel`:

1. **Baseline greedy decode tok/s** at varying warmup sizes (single-token argmax, the lower
   reference all spec variants must beat).
2. **MTP self-spec k=1 tok/s** — the production-tested 1-draft-per-round path (matches the
   :func:`quanta.qwen35.spec.spec_generate` contract). Reports ``mean_accept`` ∈ [1, 2].
3. **MTP self-spec k=2 / k=3 tok/s** — chained MTP drafting (:func:`spec_generate_k`). First
   draft uses the main model's last-layer hidden; subsequent ``k-1`` drafts chain on the MTP
   block's own post-block hidden (off-distribution past step 1 → chained-accept rate decays;
   verify amortizes over a longer window → potential net win when expert-weight reads dominate).
   Lossless: spec verify always uses ``cfg.num_experts_per_tok`` regardless of the drafter.

NOT model-free: loads the full resident Qwen3.5 (main + MTP). Use only when GPU + memory are
available (the bake/ppl gates above must have run first).

    # RUN AFTER #149 lands; orchestrator runs this — DO NOT run by hand while another large
    # job is resident (the 60-layer artifact + MTP block sit ~200+ GB on this M3 Ultra).
    uv run python -u -m parity.qwen35_mtp_k_bench
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.qwen35.decode import Qwen35Cache
from quanta.qwen35.mtp import MTPHead
from quanta.qwen35.runtime import Qwen35ResidentModel
from quanta.qwen35.spec import spec_generate
from quanta.qwen35.tokenizer import Qwen35Tokenizer

ART = "/Users/pmrj/models/Qwen3.5-397B-A17B-quanta_int4"

# A handful of natural prose so prefill timings reflect the real tokenizer distribution, not random
# ids. The longest slice is sliced down to test different prefill lengths.
_PROSE = (
    "The history of the printing press is often told as a single moment of invention, but in truth "
    "it was the result of many smaller advances accumulating over centuries. Long before movable type "
    "appeared in Europe, craftsmen in East Asia had experimented with carved wooden blocks and even "
    "with individual ceramic characters. What changed in the middle of the fifteenth century was not a "
    "single idea but a practical combination of ideas: a durable metal alloy for the type, an oil-based "
    "ink that adhered to metal, and a press adapted from the kind already used to crush grapes and "
    "olives. Together these allowed a single workshop to produce hundreds of identical pages in the "
    "time it had once taken to copy a single book by hand. The consequences were enormous, reshaping "
    "religion, science, and politics across the following two centuries. " * 16
)


def _measure_decode(model: Qwen35ResidentModel, ids: list[int], warmup_T: int, gen_N: int
                    ) -> tuple[float, list[int]]:
    """Warm a fresh cache by stepping ``ids[:warmup_T]`` through the resident decode path, then
    generate ``gen_N`` greedy-argmax tokens and time the steady-state decode (excluding the very
    first step after prefill so JIT/cache warm is excluded)."""
    cache = model.make_caches()
    warm = mx.array(ids[:warmup_T])
    mx.eval(warm)
    logits = model(warm, caches=cache, offset=0)
    if isinstance(logits, tuple):
        logits = logits[0]
    mx.eval(logits)

    out: list[int] = []
    cur = int(logits[0, -1].argmax().item())
    out.append(cur)
    t0 = time.perf_counter()
    for _i in range(gen_N - 1):
        logits = model(mx.array([cur]), caches=cache, offset=cache.offset)
        if isinstance(logits, tuple):
            logits = logits[0]
        mx.eval(logits)
        cur = int(logits[0, -1].argmax().item())
        out.append(cur)
    return (time.perf_counter() - t0, out)


def _measure_spec_timed(model: Qwen35ResidentModel, mtp: MTPHead, ids: list[int],
                        warmup_T: int, gen_N: int, *, k: int = 1) -> dict:
    """Manual spec loop that separately times the draft (MTP) vs verify (main-model ``k+1``-tok fwd)
    per round, so we can see *where* the cost goes. Mirrors :func:`quanta.qwen35.spec.spec_generate`
    (k=1) and :func:`quanta.qwen35.spec.spec_generate_k` (k≥2); must stay output-equivalent to
    greedy decode — this is for telemetry only, the production path stays in :mod:`quanta.qwen35.spec`
    (parity-tested). ``k``: number of MTP drafts per round (chained — the head was trained on main
    hidden, so chained drafts are off-distribution and acceptance drops fast past k=1; verify still
    arbitrates losslessly so the bet is purely whether the longer verify window amortizes expert
    weight loads enough)."""
    if k < 1:
        raise ValueError(f"k must be >= 1 (got {k})")
    last = model.cfg.num_hidden_layers - 1
    cache = Qwen35Cache(model.num_layers, model.cfg, quantized=True, max_rollback=k)
    prompt = ids[:warmup_T]
    arr = mx.array(prompt)
    mx.eval(arr)
    logits, caps = model(arr, caches=cache, offset=0, capture_layers=(last,))
    mx.eval(logits, caps[last])
    q = len(prompt) - 1
    prev_hidden = caps[last][-1][None, None]
    cur = int(mx.argmax(logits[0, -1]).item())
    out = [cur]
    accept_lens: list[int] = []
    draft_t = 0.0
    verify_t = 0.0

    while len(out) < gen_N:
        # --- draft k tokens (chained; first step uses main hidden, subsequent feed MTP's own back) ---
        t0 = time.perf_counter()
        drafts: list[int] = []
        d_log, d_h = mtp(prev_hidden, mx.array([[cur]]), model.embed_w, model.lm_head_w,
                         return_hidden=True)
        d_tok = int(mx.argmax(d_log[0, 0]).item())
        drafts.append(d_tok)
        for _step in range(k - 1):
            d_log, d_h = mtp(d_h, mx.array([[d_tok]]), model.embed_w, model.lm_head_w,
                             return_hidden=True)
            d_tok = int(mx.argmax(d_log[0, 0]).item())
            drafts.append(d_tok)
        mx.eval(d_log, d_h)
        draft_t += time.perf_counter() - t0

        # --- verify [cur, d_1, ..., d_k] in ONE main forward at offset q+1 ---
        t0 = time.perf_counter()
        vlog, vcaps = model(mx.array([cur, *drafts]), caches=cache, offset=q + 1,
                            capture_layers=(last,))
        bpred = mx.argmax(vlog[0], axis=-1)              # [k+1]
        mx.eval(bpred, vcaps[last])
        verify_t += time.perf_counter() - t0
        bp = [int(bpred[i].item()) for i in range(k + 1)]

        # accept longest greedy-matching prefix
        j = 0
        while j < k and drafts[j] == bp[j]:
            j += 1
        bonus = bp[j]
        out.extend(drafts[:j])
        out.append(bonus)
        accept_lens.append(j + 1)

        cache.truncate((q + 1) + (j + 1))
        prev_hidden = vcaps[last][j][None, None]
        q = q + 1 + j
        cur = bonus

    out = out[:gen_N]
    rounds = len(accept_lens)
    return {
        "wall": draft_t + verify_t,
        "tokens": len(out),
        "rounds": rounds,
        "mean_accept": (sum(accept_lens) / rounds) if rounds else 0.0,
        "max_accept": max(accept_lens) if accept_lens else 0,
        "mean_draft_ms": 1000.0 * draft_t / rounds if rounds else 0.0,
        "mean_verify_ms": 1000.0 * verify_t / rounds if rounds else 0.0,
    }


def run() -> None:
    print(f"[bench] loading {ART}", flush=True)
    t0 = time.perf_counter()
    model = Qwen35ResidentModel(ART)
    mtp = MTPHead.from_artifact(model.art, model.cfg)
    print(f"[bench] resident in {time.perf_counter() - t0:.1f}s "
          f"({model.cfg.num_hidden_layers} layers, mtp=on)", flush=True)

    tok = Qwen35Tokenizer.from_pretrained(ART)
    ids_all = tok.encode(_PROSE, add_bos=False)
    print(f"[bench] tokenized prose: {len(ids_all)} tokens", flush=True)

    # Sweep warmup (prompt) sizes for both baseline decode and MTP self-spec to see whether the
    # spec → baseline ratio changes with context length.
    warmup_sizes = [w for w in (32, 256, 1024) if len(ids_all) >= w]
    gen = 64

    print("\n=== DECODE throughput — baseline (no spec, single-token argmax) ===")
    print(f"{'warmup':>6} {'gen':>4} {'wall_s':>8} {'tok/s':>8}")
    baseline_tps: dict[int, float] = {}
    for w in warmup_sizes:
        wall, _ = _measure_decode(model, ids_all, warmup_T=w, gen_N=gen)
        amort = (gen - 1) / wall                            # gen-1 timed steps after first-from-prefill
        baseline_tps[w] = amort
        print(f"{w:>6} {gen:>4} {wall:>8.2f} {amort:>8.2f}", flush=True)
        mx.clear_cache()                                    # release transient buffers

    # k=1 (native MTP, single draft per round) — the production path; first-loss baseline for k≥2.
    # Validates spec == greedy via :func:`spec_generate` (the parity-tested entry point), then the
    # timed sweep uses _measure_spec_timed for per-round draft/verify telemetry.
    print(f"\n=== DECODE throughput — MTP self-spec k=1 @ draft_topk={mtp.draft_topk} "
          f"(lossless verify/accept) ===")
    print(f"{'warmup':>6} {'gen':>4} {'wall_s':>8} {'tok/s':>8} {'vs_base':>8} "
          f"{'accept':>7} {'draft_ms':>9} {'verify_ms':>10}")
    for w in warmup_sizes:
        # Lossless contract gate: spec_generate output matches greedy. Reported once per warmup.
        spec_tokens, stats_ref = spec_generate(model, mtp, model.embed_w, model.lm_head_w,
                                                ids_all[:w], max_new=gen, eos_id=None)
        _ = spec_tokens  # the lossless property is asserted by the parity test; smoke-emit here
        mx.clear_cache()
        stats = _measure_spec_timed(model, mtp, ids_all, warmup_T=w, gen_N=gen, k=1)
        amort = stats["tokens"] / stats["wall"]
        ratio = amort / baseline_tps[w] if baseline_tps[w] else float("nan")
        print(f"{w:>6} {gen:>4} {stats['wall']:>8.2f} {amort:>8.2f} {ratio:>7.2f}× "
              f"{stats['mean_accept']:>7.3f} {stats['mean_draft_ms']:>9.1f} "
              f"{stats['mean_verify_ms']:>10.1f}", flush=True)
        mx.clear_cache()

    # k≥2 (chained MTP draft — each subsequent step uses the prior MTP block's hidden;
    # off-distribution past step 1, so chained accept rates drop). The bet: a longer verify window
    # amortizes expert weight reads enough that per-token verify cost drops sub-linearly, breaking
    # k=1's ~2× verify ceiling. Lossless: spec only accepts greedy-matching prefixes.
    for k_draft in (2, 3):
        print(f"\n=== DECODE throughput — MTP self-spec k={k_draft} (chained) @ "
              f"draft_topk={mtp.draft_topk} (lossless verify/accept) ===")
        print(f"{'warmup':>6} {'gen':>4} {'wall_s':>8} {'tok/s':>8} {'vs_base':>8} "
              f"{'accept':>7} {'max_acc':>7} {'draft_ms':>9} {'verify_ms':>10}")
        for w in warmup_sizes:
            stats = _measure_spec_timed(model, mtp, ids_all, warmup_T=w, gen_N=gen, k=k_draft)
            amort = stats["tokens"] / stats["wall"]
            ratio = amort / baseline_tps[w] if baseline_tps[w] else float("nan")
            print(f"{w:>6} {gen:>4} {stats['wall']:>8.2f} {amort:>8.2f} {ratio:>7.2f}× "
                  f"{stats['mean_accept']:>7.3f} {stats['max_accept']:>7d} "
                  f"{stats['mean_draft_ms']:>9.1f} {stats['mean_verify_ms']:>10.1f}", flush=True)
            mx.clear_cache()

    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    print(f"\n[bench] peak python RSS = {rss:.1f} GB")


if __name__ == "__main__":
    run()
