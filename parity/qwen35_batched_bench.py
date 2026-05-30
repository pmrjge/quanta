"""Qwen3.6 batched (B>1) UNPAGED decode throughput bench — the #153 hybrid loop-kill on the real model.

HEAVY: loads the resident int4-g64 Qwen3.6-35B-A3B bake (~19 GiB). RUN SOLO — no other model
resident (one-model-at-a-time; an over-subscribed load OOM-reboots the host). Standalone:

    uv run python -m parity.qwen35_batched_bench              # B in {1,4,8,16,32}
    uv run python -m parity.qwen35_batched_bench 1,4          # cheap harness validation
    uv run python -m parity.qwen35_batched_bench 4            # only the prod operating point

(Drives raw token-id lists straight into the runtime — no tokenizer, no extra deps.)

Drives :meth:`quanta.qwen35.batched_runtime.Qwen35BatchedResidentModel.step_batch` directly over ``B``
concurrent streams with DISTINCT prompts (each stream keeps its OWN GDN recurrent state + GQA KV — a
distinct leading token defeats any trivial identical-prompt amortization). ``step_batch`` IS the prod
decode hot path: Qwen serving is UNPAGED (``shim/omlx`` forces ``paged_kv=False``) and the
``_Qwen35BatchedSession`` wrapper just orchestrates admit/release around this same call, so timing it
directly isolates the loop-kill cleanly. For each B it times TWO decode paths on the SAME resident
weights, flipping ONLY ``model._loopkill`` (the #153 :data:`QWEN35_BATCHED_LOOPKILL_DEFAULT` flag)
between them:

  * **loop**     — ``_loopkill=False``: the per-stream mixer loop (``for s in range(B)``) inside
    :func:`~quanta.qwen35.batched_runtime.batched_decode_step` — every stream runs its GDN recurrence /
    GQA attention through its own cache one at a time (the proven pre-#153 path);
  * **loopkill** — ``_loopkill=True``: that B-stream mixer loop replaced by ONE batched mixer per layer —
    GQA via :meth:`~quanta.qwen35.attention.Qwen35Attention.decode_step_batched` (batched q/k/v/o proj +
    per-stream RoPE + shared fused padded SDPA, M1) and Gated-DeltaNet via
    :func:`~quanta.qwen35.batched_runtime._gdn_step_batched` (gather B streams' (conv,recurrent) state +
    ONE recurrence + scatter, M2 — the bigger lever: 45 of 60 layers are GDN).

Both paths run the IDENTICAL batched MoE sub-block (the routed/shared expert reads already amortize over
B via the existing ``qwen35_moe`` ``[B,1,h]→[N,h]`` reshape) — only the MIXER step differs — so
``loopkill/loop`` aggregate tok/s isolates the #153 mixer-loop-kill win. Qwen is the HYBRID case: the
loop-kill trims the per-stream Python loop on ALL 60 layers (45 GDN + 15 GQA), amortizing each mixer's
in/out projection + conv / q/k/v/o weight read ONCE across B (the bandwidth win, mirroring the MoE
expert-read amortization). Qwen's batched operating point is **B=32** (re-pinned from the original B=4
to match the cohort cap), so the win should approach InternLM2.5's (3.20×@B32) / Nemotron's (1.15×@B32);
B=1 anchors bit-exactness and the sweep shows the B-trend.

This is ALSO the real-model correctness gate for the hybrid loop-kill at the TRUE dims/dtypes (the
model-free ``qwen35_batched_loopkill_test`` used a tiny random-weights config). ``run()`` asserts EVERY
stream's greedy token trace is IDENTICAL loop-vs-loopkill — B=1 bit-exact (the b==1 paths are strict
passthroughs: GDN no-concat, GQA all-zero pad mask == mask=None) and B>=2 greedy-exact. With the CURRENT
dequantized runtime B>=2 FAILS that bar — the dense-bf16 projection-GEMM batch-M accumulation reorder is
NOT argmax-stable: it compounds over depth and flips greedy tokens (worst |Δlogit|≈1.3, three orders over
tol — NOT the [[feedback-batched-rope-bf16]] ULP-noise class). That is the real loop-kill blocker option B
fixes, surfaced loud (rule 6). Throughput is reported, not asserted (hardware-variable).

Memory is read with MLX's own counters (``get_active_memory`` / ``get_peak_memory``); ``clear_cache`` +
``reset_peak_memory`` run between every path/B so a peak is that configuration's true transient.

Geometry: prompt = 128 tok/stream (distinct per stream; single-stream Design-A prefill, loopkill-off),
GEN = 64 decoded tok/stream (steady state; WARMUP steps JIT-warm and are not timed), no EOS stop (every
stream decodes exactly WARMUP+GEN).
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.qwen35.batched_runtime import Qwen35BatchedResidentModel

ART = "/Users/pmrj/models/Qwen3.6-35B-A3B-quanta_int4g64"
PROMPT_LEN = 128                  # distinct prompt tokens per stream (own GDN state + GQA KV)
GEN = 64                          # decoded tokens per stream (timed)
WARMUP_STEPS = 4                  # JIT warm (not timed)
BATCH_SIZES = (1, 4, 8, 16, 32)   # 1 = bit-exact anchor; 4 = prod operating point; 8/16/32 = trend


def _gib(nbytes: int) -> float:
    return nbytes / (1024 ** 3)


def _distinct_prompt(b: int, vocab: int, n: int, bos: int | None) -> list[int]:
    """A length-``n`` prompt whose leading token differs across streams (so no two streams share a
    GDN recurrent trajectory / GQA KV prefix — each stream gets its own full state, the real B-stream
    load). A large per-stream stride spreads the ids; clamp into ``[1, vocab)`` (the token VALUES only
    need to be valid + stream-distinct; decode tok/s is matmul-bandwidth dominated, ~context-independent,
    and the bench never stops on EOS so an EOS landing in a prompt is just another token)."""
    stride = (b + 1) * 1009
    head = [int(bos)] if bos is not None else []
    body = [1 + ((stride + j * 7) % (vocab - 2)) for j in range(n - len(head))]
    return head + body


def _time_path(model: Qwen35BatchedResidentModel, B: int, prompts: list[list[int]], *,
               loopkill: bool) -> dict:
    """Time GEN steady-state decode steps at batch B on one path (``loopkill`` False=loop / True=loop-
    kill). Builds FRESH per-stream caches, prefills each prompt (single-stream Design-A prefill, always
    loopkill-off), warms WARMUP_STEPS, then times GEN. Returns aggregate/per-stream tok/s, the per-stream
    greedy token traces (WARMUP+GEN), and active/peak GiB. Flipping ONLY ``model._loopkill`` is the sole
    difference between the two paths — the prefilled caches are bit-identical (deterministic prompts +
    loopkill-off prefill), so the post-prefill greedy traces are directly comparable."""
    model._loopkill = bool(loopkill)          # the ONLY thing that differs between paths
    mx.clear_cache()
    mx.reset_peak_memory()

    caches = model.make_batch_caches(B)
    for i, p in enumerate(prompts):
        model.prefill(p, caches[i])           # single-stream prefill (loopkill pinned off inside)
        mx.eval(caches[i].offset)             # land the cache write before the next stream's prefill
    offsets = [c.offset for c in caches]
    toks = [prompts[i][-1] for i in range(B)]  # last prompt token feeds the first decode step
    traces: list[list[int]] = [[] for _ in range(B)]

    def _step() -> None:
        nonlocal toks, offsets
        per_stream = model.step_batch(toks, caches, offsets)
        mx.eval(per_stream)
        toks = [int(mx.argmax(lg[0, -1]).item()) for lg in per_stream]
        offsets = [o + 1 for o in offsets]
        for s in range(B):
            traces[s].append(toks[s])

    for _ in range(WARMUP_STEPS):             # JIT warm (captured into traces; not timed)
        _step()

    t0 = time.perf_counter()
    for _ in range(GEN):
        _step()
    dt = time.perf_counter() - t0

    return {"per_stream": GEN / dt, "aggregate": B * GEN / dt, "traces": traces,
            "active_gib": _gib(mx.get_active_memory()), "peak_gib": _gib(mx.get_peak_memory())}


def _first_divergence(loop_traces: list[list[int]], lk_traces: list[list[int]]) -> tuple | None:
    """First (stream, step, loop_tok, loopkill_tok) where the two paths' greedy traces differ, or None
    if every stream is token-identical (rule 6 — surface the exact divergence, don't just say DIFF)."""
    for s in range(len(loop_traces)):
        ref, got = loop_traces[s], lk_traces[s]
        for k in range(min(len(ref), len(got))):
            if ref[k] != got[k]:
                return (s, k, ref[k], got[k])
    return None


def run(batch_sizes: tuple[int, ...] = BATCH_SIZES) -> None:
    max_batch = max(batch_sizes)
    # Pin the resident weight set (Qwen3.6-35B-A3B int4-g64 ≈ 19 GiB + per-stream GDN/KV state + transient).
    mx.set_wired_limit(int(80 * 1024 ** 3))
    model = Qwen35BatchedResidentModel(ART, max_batch=max_batch)

    cfg = model.cfg
    vocab = int(cfg.vocab_size)
    bos = getattr(cfg, "bos_token_id", None)
    n_lin = sum(cfg.is_linear_attention(i) for i in range(model.num_layers))
    n_full = model.num_layers - n_lin

    print(f"\n=== Qwen3.6-35B-A3B int4-g64 UNPAGED batched decode (prompt {PROMPT_LEN} tok, {GEN} gen/stream, "
          f"{model.num_layers} layers = {n_lin} GDN + {n_full} GQA): loop (per-stream mixer) vs loopkill "
          f"(#153: ONE batched mixer/layer) ===")
    print("aggregate tok/s (per-stream = aggregate / B). loopkill/loop = the #153 hybrid mixer-loop-kill "
          "win. GiB = loopkill-path active/peak. tok = loop==loopkill greedy-exact (real-model correctness).")
    print(f"{'B':>4}  {'loop':>9}  {'loopkill':>9}  {'loopkill/loop':>13}  "
          f"{'loopkill GiB a/p':>17}  {'tok':>4}")
    all_tok_ok = True
    ratios: dict[int, float] = {}
    for B in batch_sizes:
        prompts = [_distinct_prompt(b, vocab, PROMPT_LEN, bos) for b in range(B)]
        lp = _time_path(model, B, prompts, loopkill=False)   # per-stream mixer loop
        lk = _time_path(model, B, prompts, loopkill=True)    # #153 hybrid loop-kill
        ratio = lk["aggregate"] / lp["aggregate"] if lp["aggregate"] else float("nan")
        ratios[B] = ratio
        div = _first_divergence(lp["traces"], lk["traces"])
        tok_ok = div is None
        all_tok_ok = all_tok_ok and tok_ok
        print(f"{B:>4}  {lp['aggregate']:>9.1f}  {lk['aggregate']:>9.1f}  {ratio:>12.2f}x  "
              f"{lk['active_gib']:>7.1f}/{lk['peak_gib']:>7.1f}  {'ok' if tok_ok else 'DIFF':>4}")
        if div is not None:
            s, k, ref, got = div
            print(f"    [DIFF] loopkill diverges from loop at stream {s} step {k}: "
                  f"loop={ref} loopkill={got}")

    # Honest verdict (rule 6): the loop-kill MUST be token-identical to the per-stream loop (B=1 bit-
    # exact, B>=2 greedy-exact — same batched MoE, only the mixer step differs). A divergence is a real
    # #153 bug; fail loud. Throughput is reported, not asserted (hardware-variable). Graduation is judged
    # at B=32 (Qwen's re-pinned operating point), with the rest of the sweep as supporting trend.
    if not all_tok_ok:
        print("FAIL — the #153 hybrid loop-kill diverged from the per-stream mixer loop (see [DIFF] above)")
        raise SystemExit(1)
    op = 32 if 32 in ratios else max(ratios)          # prod operating point (fallback: largest swept B)
    best = max(ratios.values())
    verdict = f"PASS — loop == loopkill greedy-exact on the real model; B={op} loopkill/loop = {ratios[op]:.2f}x"
    if best > ratios[op]:
        best_b = max(ratios, key=ratios.get)
        verdict += f" (best {best:.2f}x @ B={best_b})"
    verdict += "; " + ("a win — graduate the default" if ratios[op] >= 1.0 else
                       "NOT a win at the operating point — keep default OFF")
    print(verdict)


if __name__ == "__main__":
    import sys

    bs = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else BATCH_SIZES
    run(bs)
