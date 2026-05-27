"""DSV4 int4-g64 perf bench on the real baked artifact (#76 → ppl=5.08).

Measures on the live 43-layer artifact loaded through the packed-weight resident runtime
(#141, mx.gather_qmm path):

1. **Load time** — wall-clock from `DSV4ResidentModel(art_dir)` start to all layers materialized
   (one layer at a time per rule-8) + final embed/norm/head pinned, **plus** the j=0 native MTP
   block (~4 GB extra resident for the draft head).
2. **Prefill tok/s** at varying prompt lengths.
3. **Decode tok/s — baseline** (greedy argmax, single-token decode steady-state).
4. **Decode tok/s — MTP self-speculation** via :func:`quanta.dsv4.spec.spec_generate`: each round
   the MTP head drafts 1 token, the main model verifies both ``[cur, draft]`` in one forward,
   accepts on greedy match. Reports ``mean_accept`` ∈ [1, 2] (1 = MTP never helps, 2 = always
   accepted) and the resulting amortized tok/s — that's the actual end-user decode speed.

NOT model-free: loads the full ~176 GB resident model (main + MTP). Use only when the GPU +
memory are available (the bake/ppl gates above must have run first).

    uv run python -u -m parity.dsv4_int4_bench
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.dsv4 import decode as D
from quanta.dsv4.runtime import DSV4ResidentModel
from quanta.dsv4.spec import spec_generate
from quanta.dsv4.tokenizer import DeepSeekV4Tokenizer

ART = "/Users/pmrj/models/DeepSeek-V4-Flash-quanta_int4g64"

# A handful of natural prose (drawn from parity/dsv4_ppl.PROSE) so prefill timings reflect the
# real tokenizer distribution, not random ids. The longest slice is sliced down to test
# different prefill lengths.
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


def _measure_prefill(model: DSV4ResidentModel, ids: list[int], T: int) -> tuple[float, mx.array]:
    """Run a fresh prefill over ``ids[:T]`` (no cache); return (wall_sec, last_logits)."""
    cut = ids[:T]
    arr = mx.array(cut)
    mx.eval(arr)
    t0 = time.perf_counter()
    logits = model(arr)        # [1, T, V]
    mx.eval(logits)
    return (time.perf_counter() - t0, logits)


def _measure_decode(model: DSV4ResidentModel, ids: list[int], warmup_T: int, gen_N: int
                    ) -> tuple[float, list[int]]:
    """Warm a fresh KV cache by stepping ``ids[:warmup_T]`` through the resident decode path,
    then generate ``gen_N`` greedy-argmax tokens and time the steady-state decode."""
    cache = D.DSV4Cache(model.num_layers)
    # Seed the cache by stepping the warmup prompt one token at a time (the same path the spec
    # decoder + generate.py use; output-equivalent to a batched prefill but lets us reuse the
    # cache directly afterwards for fair single-token decode timing).
    warm = mx.array(ids[:warmup_T])
    mx.eval(warm)
    _ = model(warm, caches=cache, offset=0)
    mx.eval(_)

    out: list[int] = []
    cur = int(_[0, -1].argmax().item())
    out.append(cur)
    t0 = time.perf_counter()
    for _i in range(gen_N - 1):
        logits = model(mx.array([cur]), caches=cache, offset=cache.offset)
        mx.eval(logits)
        cur = int(logits[0, -1].argmax().item())
        out.append(cur)
    return (time.perf_counter() - t0, out)


def _measure_spec(model: DSV4ResidentModel, ids: list[int], warmup_T: int, gen_N: int
                  ) -> tuple[float, list[int], dict]:
    """MTP self-speculation: prefill ``ids[:warmup_T]``, then spec_generate ``gen_N`` new tokens
    (the spec loop drafts 1 token via MTP each round and the main model verifies both in one
    forward). Returns (wall_sec, tokens, stats) where stats['mean_accept'] is the average tokens
    emitted per main-model forward (1 = MTP never helps; 2 = every draft is accepted)."""
    assert model.mtp is not None, "spec_generate requires model.mtp (load_mtp=True)"
    prompt = ids[:warmup_T]
    t0 = time.perf_counter()
    tokens, stats = spec_generate(model, model.mtp, model.embed_w, model.lm_head_w,
                                  prompt, max_new=gen_N, eos_id=None)
    return (time.perf_counter() - t0, tokens, stats)


def _measure_spec_timed(model: DSV4ResidentModel, ids: list[int], warmup_T: int, gen_N: int,
                        *, k: int = 1) -> dict:
    """Manual spec loop that separately times the draft (MTP) vs verify (main-model ``k+1``-tok fwd)
    per round, so we can see *where* the cost goes. Mirrors :func:`quanta.dsv4.spec.spec_generate`
    (k=1) and :func:`quanta.dsv4.spec.spec_generate_k` (k≥2); must stay output-equivalent to greedy
    decode — this is for telemetry only, the production path stays in `spec.py` (parity-tested).
    ``k``: number of MTP drafts per round (chained — the head was trained on main hidden, so chained
    drafts are off-distribution and acceptance drops fast past k=1; verify still arbitrates losslessly
    so the bet is purely whether the longer verify window amortizes expert weight loads enough)."""
    assert model.mtp is not None, "MTP self-spec requires model.mtp"
    if k < 1:
        raise ValueError(f"k must be >= 1 (got {k})")
    last = model.cfg.num_hidden_layers - 1
    cache = D.DSV4Cache(model.num_layers, max_rollback=k)   # ring sized for k-token suffix rollback
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
        d_log, d_h = model.mtp(prev_hidden, mx.array([[cur]]), model.embed_w, model.lm_head_w,
                                return_hidden=True)
        d_tok = int(mx.argmax(d_log[0, 0]).item())
        drafts.append(d_tok)
        for _step in range(k - 1):
            d_log, d_h = model.mtp(d_h, mx.array([[d_tok]]), model.embed_w, model.lm_head_w,
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
    model = DSV4ResidentModel(ART)
    print(f"[bench] resident in {time.perf_counter() - t0:.1f}s "
          f"({model.cfg.num_hidden_layers} layers, packed_experts={model.packed_experts}, "
          f"mtp={'on' if model.mtp is not None else 'off'})", flush=True)

    tok = DeepSeekV4Tokenizer.from_pretrained(ART)
    ids_all = tok.encode(_PROSE, add_bos=True)
    print(f"[bench] tokenized prose: {len(ids_all)} tokens (BOS-first={ids_all[0] == tok.bos_id})",
          flush=True)

    print("\n=== PREFILL throughput (packed int4 experts + int8 KV + bf16 attn) ===")
    print(f"{'T':>6} {'wall_s':>8} {'tok/s':>8}")
    for T in (32, 256, 1024, 4096):
        if len(ids_all) < T:
            break
        _measure_prefill(model, ids_all, min(T, 32))    # warm kernels once per T
        wall, _ = _measure_prefill(model, ids_all, T)
        print(f"{T:>6} {wall:>8.2f} {T / wall:>8.1f}", flush=True)

    # Sweep warmup (prompt) sizes for both baseline decode and MTP self-spec to see whether the
    # spec → baseline ratio changes with context length (longer KV cache → costlier attention per
    # step → potentially shifts the verify-vs-baseline ratio that's currently 2.0×). Capped at 1024
    # to stay well under the 490 GB working-set ceiling (the cache + indexer state at 2048+ pushes
    # us into paging on this 169 GB-resident artifact).
    warmup_sizes = [w for w in (32, 256, 1024) if len(ids_all) >= w]

    print("\n=== DECODE throughput — baseline (no spec, single-token argmax) ===")
    print(f"{'warmup':>6} {'gen':>4} {'wall_s':>8} {'tok/s':>8}")
    baseline_tps: dict[int, float] = {}
    for w in warmup_sizes:
        wall, _ = _measure_decode(model, ids_all, warmup_T=w, gen_N=64)
        amort = 63 / wall                                  # 63 timed steps after the first-from-prefill
        baseline_tps[w] = amort
        print(f"{w:>6} {64:>4} {wall:>8.2f} {amort:>8.2f}", flush=True)
        mx.clear_cache()                                   # release transient buffers (per-iter caches)

    if model.mtp is not None:
        # k=1 (native MTP, single draft per round) — the production path; first-loss baseline for k≥2.
        print(f"\n=== DECODE throughput — MTP self-spec k=1 @ draft_topk={model.mtp.draft_topk} "
              f"(lossless verify/accept) ===")
        print("(spec sweeps the prompt size; main verify stays at cfg.topk=6, drafter at draft_topk)")
        print(f"{'warmup':>6} {'gen':>4} {'wall_s':>8} {'tok/s':>8} {'vs_base':>8} "
              f"{'accept':>7} {'draft_ms':>9} {'verify_ms':>10}")
        for w in warmup_sizes:
            stats = _measure_spec_timed(model, ids_all, warmup_T=w, gen_N=64, k=1)
            amort = stats["tokens"] / stats["wall"]
            ratio = amort / baseline_tps[w] if baseline_tps[w] else float("nan")
            print(f"{w:>6} {64:>4} {stats['wall']:>8.2f} {amort:>8.2f} {ratio:>7.2f}× "
                  f"{stats['mean_accept']:>7.3f} {stats['mean_draft_ms']:>9.1f} "
                  f"{stats['mean_verify_ms']:>10.1f}", flush=True)
            mx.clear_cache()

        # k≥2 (chained MTP draft — each subsequent step uses the prior MTP block's hidden;
        # off-distribution past step 1, so chained accept rates drop). The bet: a longer verify
        # window amortizes expert weight reads enough that per-token verify cost drops sub-linearly,
        # breaking k=1's ~2× verify ceiling. The prefill curve (T=32→1024 = 30× drop in ms/tok)
        # is the upper bound on that amortization — at small verify lengths (T=3, T=4) we sit on the
        # transition. Lossless: spec only accepts greedy-matching prefixes; output is bit-equivalent.
        for k_draft in (2, 3):
            print(f"\n=== DECODE throughput — MTP self-spec k={k_draft} (chained) @ draft_topk="
                  f"{model.mtp.draft_topk} (lossless verify/accept) ===")
            print(f"{'warmup':>6} {'gen':>4} {'wall_s':>8} {'tok/s':>8} {'vs_base':>8} "
                  f"{'accept':>7} {'max_acc':>7} {'draft_ms':>9} {'verify_ms':>10}")
            for w in warmup_sizes:
                stats = _measure_spec_timed(model, ids_all, warmup_T=w, gen_N=64, k=k_draft)
                amort = stats["tokens"] / stats["wall"]
                ratio = amort / baseline_tps[w] if baseline_tps[w] else float("nan")
                print(f"{w:>6} {64:>4} {stats['wall']:>8.2f} {amort:>8.2f} {ratio:>7.2f}× "
                      f"{stats['mean_accept']:>7.3f} {stats['max_accept']:>7d} "
                      f"{stats['mean_draft_ms']:>9.1f} {stats['mean_verify_ms']:>10.1f}", flush=True)
                mx.clear_cache()
    else:
        print("\n=== MTP spec skipped (model.mtp is None — re-bake with full MTP block params) ===")

    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    print(f"\n[bench] peak python RSS = {rss:.1f} GB")


if __name__ == "__main__":
    run()
