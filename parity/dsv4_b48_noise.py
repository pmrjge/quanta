"""Is the B=48 arena decode *within model noise* of the per-stream reference? (the #18 follow-on).

The throughput bench ``parity/dsv4_batched_bench.py`` demands a STRICT bar — looped == batched == arena
greedy-exact — and at B=48 it trips: one token flips on the fluent passage (``looped=97 batched=106``),
because batched decode pads B streams to a rectangular ``[B, L_max]`` and the fp32 softmax reduces in a
different ORDER than the per-stream loop (``decode.py`` documents this: B=1 bit-exact, B>=2 greedy-exact
only). fp addition is non-associative, so a few-ULP delta is unavoidable even though attention is already
fp32 — and at a token whose top-2 logit gap is below that delta, the argmax flips. Greedy-exactness is
the WRONG arbiter for that: a flip at an fp tie is not an error, it is two equally-valid continuations.

This gate measures the RIGHT thing (CLAUDE.md methodology #4 — teacher-forced ppl + top-1 agreement):
drive the SAME fixed prose through B=48 streams TEACHER-FORCED (feed the known next token, never the
argmax — so the two paths never feed themselves different tokens and there is no divergence cascade to
confound the per-position comparison) and, at every position, diff the per-stream-loop reference
distribution against the arena distribution. A flip is "within noise" iff the two tokens are a genuine
tie under the model's own distribution:

  * **rank** of the arena token in the reference distribution (a benign flip is rank 2 — the runner-up);
  * **Δlogit** = ``ref[ref_top] - ref[arena_top]`` (the gap the noise crossed; ~0 ⇒ a tie);
  * **sampling odds** = ``exp(Δlogit / T)`` at the serving temperature — what a sampler's odds are of
    preferring the reference token. ~1 ⇒ a sampler cannot tell the two paths apart (the user's question);
  * **KL(ref ‖ arena)** per position — the belief states are identical even where the hard argmax flips;
  * reference **top-2 gap** at flips vs non-flips — flips must concentrate where the MODEL is uncertain
    (small gap), not where the arena is wrong. The ``repeat`` streams (low-entropy, large gaps) should
    show ~no flips; the ``prose`` streams' flips should all sit on small-gap ties.

If the flips are all rank-2 sampling-indistinguishable ties, KL is ~1e-4, and teacher-forced ppl is
unchanged, then B=48 is **e2e-equivalent within fp noise** — a strictly stronger correctness claim than
greedy-exactness, and the real bar to put behind the B=48 default.

Reference = ``_fused=False`` looped: per-stream attention + per-stream KV, which is per-stream
INDEPENDENT (so looped@B=48 stream b is bit-exact to a B=1 forward of window b) — the parity-correct
Design-A path the arena is gated against. Test = ``_fused=True`` arena (``make_cache()``, the #18 M4
default). Two passes, one path resident at a time (the bench's known-safe per-path memory profile).

HEAVY: loads the resident int4-g64 DSV4-Flash bake (~180 GiB); arena pass peaks ~192 GiB at B=48. RUN
SOLO — no other model resident (one-model-at-a-time; an over-subscribed load OOM-reboots the host,
[[feedback-memory-safety]]). The metric math is validated model-free first:

    uv run python -m parity.dsv4_b48_noise --selftest    # model-free: validate the metric math (cheap)
    uv run --with tokenizers python -m parity.dsv4_b48_noise   # the real B=48 gate (SOLO, ~180 GiB)
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

from parity.dsv4_ppl import PROSE
from quanta.dsv4.batched_runtime import DSV4BatchedResidentModel
from quanta.dsv4.decode import DSV4Cache
from quanta.dsv4.tokenizer import DeepSeekV4Tokenizer

ART = "/Users/pmrj/models/DeepSeek-V4-Flash-quanta_int4g64"
B = 48                              # the operating point under question (BEST_BATCH deepseek_v4 = 48)
PROMPT = 16                         # per-stream prefill seed (short — decode noise is what we probe)
GEN = 48                            # teacher-forced steps scored per stream (past the bench's step-22 flip)
TEMP = 0.7                          # serving temperature for the sampling-odds metric
WIRED_GIB = 220                     # pin the resident weight set (DSV4-Flash int4-g64 ≈ 180 GiB)

# A low-entropy probe: a correct forward is near-certain at each position (large top-2 gaps ⇒ no ties ⇒
# no flips), the contrast that proves flips track MODEL uncertainty, not arena error.
REPEAT = ("The quick brown fox jumps over the lazy dog. " * 12).strip()

# "Within noise" verdict thresholds (informational — the numbers are printed regardless).
ODDS_TOL = 3.0                     # a flip is a tie if a sampler favors either token <= 3:1
KL_TOL = 1e-2                      # MEAN per-position KL(ref || arena) — distribution-wide closeness.
#                                    NOT max: at a genuine tie the two VALID distributions differ by
#                                    ~0.1 nats by construction (the tie-break the odds gate already
#                                    certifies benign), so a max-KL gate would double-penalize ties.
DPPL_TOL = 1e-2                    # |ppl_arena - ppl_ref| (absolute) and 1e-2 relative


def _gib(nbytes: int) -> float:
    return nbytes / (1024 ** 3)


def _logsoftmax(x: mx.array) -> mx.array:
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def _compare(ref: mx.array, ar: mx.array, tgt: mx.array, temp: float) -> dict:
    """Per-position noise metrics. ``ref``/``ar``: ``[N, V]`` fp32 logits (reference vs arena); ``tgt``:
    ``[N]`` teacher-forced next tokens. Returns a dict of ``[N]`` arrays (one row per stream-position).
    Pure — no model — so it is validated model-free in :func:`_selftest`."""
    n, v = ref.shape
    ref_ls, ar_ls = _logsoftmax(ref), _logsoftmax(ar)
    ref_p, ar_p = mx.exp(ref_ls), mx.exp(ar_ls)
    ref_top, ar_top = mx.argmax(ref, axis=-1), mx.argmax(ar, axis=-1)
    agree = ref_top == ar_top

    ref_top_val = mx.max(ref, axis=-1)                                   # ref logit at ITS argmax
    ref_at_artop = mx.take_along_axis(ref, ar_top[:, None], axis=-1)[:, 0]   # ref logit at ARENA's argmax
    dlogit = ref_top_val - ref_at_artop                                  # >= 0; 0 when agree
    rank = mx.sum((ref > ref_at_artop[:, None]).astype(mx.int32), axis=-1) + 1   # 1 when agree, 2 = runner-up

    kl = mx.sum(ref_p * (ref_ls - ar_ls), axis=-1)                       # KL(ref || arena) >= 0
    max_dp = mx.max(mx.abs(ref_p - ar_p), axis=-1)                       # max prob delta (TV-ish)

    masked = mx.where(mx.arange(v)[None, :] == ref_top[:, None], mx.array(float("-inf")), ref)
    gap2 = ref_top_val - mx.max(masked, axis=-1)                         # ref top-1 minus top-2 (certainty)

    odds = mx.exp(mx.minimum(dlogit / temp, mx.array(60.0)))             # sampler's ref:arena odds (cap exp)
    nll_ref = -mx.take_along_axis(ref_ls, tgt[:, None], axis=-1)[:, 0]
    nll_ar = -mx.take_along_axis(ar_ls, tgt[:, None], axis=-1)[:, 0]
    return {
        "agree": agree, "rank": rank, "dlogit": dlogit, "kl": kl, "max_dp": max_dp, "gap2": gap2,
        "odds": odds, "nll_ref": nll_ref, "nll_ar": nll_ar,
        "t1_ref": ref_top == tgt, "t1_ar": ar_top == tgt,
    }


def _stats(xs: list[float]) -> tuple[float, float, float]:
    """(mean, p50, max) of a non-empty list; (nan, nan, nan) if empty."""
    if not xs:
        return float("nan"), float("nan"), float("nan")
    s = sorted(xs)
    mean = sum(s) / len(s)
    return mean, s[len(s) // 2], s[-1]


# --------------------------------------------------------------------------- model-free metric self-test
def _selftest() -> None:
    """Validate the metric math on hand-built logits — no model load."""
    print("=== dsv4_b48_noise metric self-test (model-free) ===")
    V = 32
    base = mx.arange(V, dtype=mx.float32)[None, :]                        # one clear winner: token V-1

    # (1) identical logits ⇒ perfect agreement, zero divergence.
    m = _compare(base, base, mx.array([V - 1]), TEMP)
    mx.eval(*m.values())
    assert bool(m["agree"][0]) and int(m["rank"][0]) == 1
    assert abs(float(m["kl"][0])) < 1e-6 and abs(float(m["dlogit"][0])) < 1e-6
    assert abs(float(m["nll_ref"][0]) - float(m["nll_ar"][0])) < 1e-6
    print("  [OK] identical ref==arena: agree, rank 1, KL 0, Δlogit 0, Δnll 0")

    # (2) a genuine TIE: two top logits ~equal, arena picks the runner-up ⇒ rank 2, tiny Δlogit, odds ~1.
    tie = mx.array([[0.0] * (V - 2) + [10.000, 10.002]])                  # tokens V-2, V-1 within 2e-3
    arena_tie = mx.array([[0.0] * (V - 2) + [10.002, 10.000]])            # arena prefers V-2 instead
    m = _compare(tie, arena_tie, mx.array([V - 1]), TEMP)
    mx.eval(*m.values())
    assert not bool(m["agree"][0]) and int(m["rank"][0]) == 2
    assert float(m["dlogit"][0]) < 0.01 and float(m["odds"][0]) < 1.05    # sampler can't tell them apart
    assert float(m["kl"][0]) < 1e-3
    print(f"  [OK] tie flip: rank 2, Δlogit={float(m['dlogit'][0]):.1e}, "
          f"odds={float(m['odds'][0]):.3f}, KL={float(m['kl'][0]):.1e}")

    # (3) a REAL disagreement: arena picks a clearly-worse token ⇒ large Δlogit, large odds (NOT noise).
    bad = mx.array([[0.0] * (V - 1) + [20.0]])                            # token V-1 dominates by 20 logits
    arena_bad = mx.array([[20.0] + [0.0] * (V - 1)])                      # arena picks token 0
    m = _compare(bad, arena_bad, mx.array([V - 1]), TEMP)
    mx.eval(*m.values())
    assert float(m["dlogit"][0]) > 19.0 and float(m["odds"][0]) > 100.0   # the metric flags it loudly
    print(f"  [OK] real flip flagged: Δlogit={float(m['dlogit'][0]):.1f}, odds={float(m['odds'][0]):.1e}")
    print("PASS (metric math)")


# --------------------------------------------------------------------------- window construction
def _windows_from(ids: list[int], n: int, length: int) -> list[list[int]]:
    """``n`` windows of ``length`` tokens spread across ``ids`` (distinct start positions)."""
    maxstart = len(ids) - length
    if maxstart < 0:
        raise ValueError(f"passage too short: {len(ids)} tok < window {length} "
                         f"(need PROMPT+GEN+1 = {PROMPT}+{GEN}+1)")
    if n == 1:
        starts = [0]
    else:
        starts = [round(i * maxstart / (n - 1)) for i in range(n)]
    return [ids[s:s + length] for s in starts]


def _build_windows(tok: DeepSeekV4Tokenizer) -> tuple[list[list[int]], list[int]]:
    """``B`` teacher-forcing windows (each ``PROMPT+GEN+1`` tokens), half fluent prose, half the
    low-entropy repeat probe; ``tags`` marks each stream (0=prose, 1=repeat)."""
    length = PROMPT + GEN + 1
    prose_ids = [int(t) for t in tok.encode(PROSE, add_bos=True)]
    repeat_ids = [int(t) for t in tok.encode(REPEAT, add_bos=True)]
    n_rep = B // 2
    n_prose = B - n_rep
    windows = _windows_from(prose_ids, n_prose, length) + _windows_from(repeat_ids, n_rep, length)
    tags = [0] * n_prose + [1] * n_rep
    return windows, tags


# --------------------------------------------------------------------------- decode drivers
def _prefill(model: DSV4BatchedResidentModel, windows: list[list[int]], *, arena: bool) -> list:
    """One fresh cache per stream (arena handle or discrete), prefilled with ``window[:PROMPT]``."""
    caches = []
    for w in windows:
        cache = model.make_cache() if arena else DSV4Cache(model.num_layers)
        mx.eval(model.prefill(mx.array(w[:PROMPT]), cache))
        caches.append(cache)
    return caches


def _teacher_force(model: DSV4BatchedResidentModel, windows: list[list[int]], caches: list) -> list[mx.array]:
    """Teacher-forced decode: feed ``window[PROMPT+t]`` at offset ``PROMPT+t`` for ``t in [0, GEN)``;
    return per-step stacked next-token logits ``[B, V]`` (one array per step). ``model._fused`` /
    ``caches`` type select the path."""
    offsets = [PROMPT] * len(windows)
    steps: list[mx.array] = []
    for t in range(GEN):
        ids_in = [mx.array([w[PROMPT + t]]) for w in windows]
        out = model.step_batch(ids_in, caches, offsets)
        step_logits = mx.stack([out[b][0, -1] for b in range(len(windows))]).astype(mx.float32)
        mx.eval(step_logits)
        steps.append(step_logits)
        offsets = [o + 1 for o in offsets]
    return steps


# --------------------------------------------------------------------------- report
def _report(parts: list[dict], tags: list[int], arena_peak_gib: float) -> None:
    def col(name: str) -> list[float]:
        return [float(v) for p in parts for v in p[name].tolist()]

    agree = col("agree")
    rank = col("rank")
    dlogit = col("dlogit")
    kl = col("kl")
    max_dp = col("max_dp")
    gap2 = col("gap2")
    odds = col("odds")
    nll_ref = col("nll_ref")
    nll_ar = col("nll_ar")
    t1_ref = col("t1_ref")
    t1_ar = col("t1_ar")
    tag_flat = [tags[i % B] for _ in range(len(parts)) for i in range(B)]

    n = len(agree)
    n_flip = sum(1 for a in agree if a < 0.5)
    import math
    ppl_ref = math.exp(sum(nll_ref) / n)
    ppl_ar = math.exp(sum(nll_ar) / n)
    dppl = ppl_ar - ppl_ref

    flip_idx = [i for i in range(n) if agree[i] < 0.5]
    f_rank = [rank[i] for i in flip_idx]
    f_dlogit = [dlogit[i] for i in flip_idx]
    f_odds = [odds[i] for i in flip_idx]
    f_gap2 = [gap2[i] for i in flip_idx]
    f_kl = [kl[i] for i in flip_idx]
    nonflip_gap2 = [gap2[i] for i in range(n) if agree[i] >= 0.5]

    print(f"\n=== B={B} arena vs per-stream loop — teacher-forced noise band "
          f"({len(parts)} steps × {B} streams = {n} positions; T={TEMP}) ===")
    print(f"arena-pass peak {arena_peak_gib:.1f} GiB")
    print(f"top-1 agreement (argmax matches reference) : {100 * sum(agree) / n:.3f}%  "
          f"({n - n_flip}/{n})   flips: {n_flip}")
    print(f"teacher-forced ppl   ref {ppl_ref:.5f}   arena {ppl_ar:.5f}   Δ {dppl:+.2e}")
    print(f"top-1 vs target      ref {100 * sum(t1_ref) / n:.2f}%   arena {100 * sum(t1_ar) / n:.2f}%")
    print(f"KL(ref‖arena)/pos    mean {sum(kl) / n:.2e}   max {max(kl):.2e}      "
          f"max|Δp| {max(max_dp):.2e}   (ALL positions — distributions match even where argmax flips)")

    if n_flip:
        rk = {2: f_rank.count(2), 3: f_rank.count(3)}
        rk_hi = n_flip - rk[2] - rk[3]
        n_tie = sum(1 for o in f_odds if o <= ODDS_TOL)
        dl = _stats(f_dlogit)
        od = _stats(f_odds)
        g = _stats(f_gap2)
        print(f"\nflips ({n_flip}) — are they ties?")
        print(f"  rank of arena token in ref dist : rank2 {rk[2]}   rank3 {rk[3]}   rank>3 {rk_hi}")
        print(f"  Δlogit (gap noise crossed)       : mean {dl[0]:.3e}  p50 {dl[1]:.3e}  max {dl[2]:.3e}")
        print(f"  sampling odds exp(Δlogit/T)      : p50 {od[1]:.3f}  max {od[2]:.3f}   "
              f"<= {ODDS_TOL}:1 → {n_tie}/{n_flip} sampling-indistinguishable")
        print(f"  KL at flips                      : max {max(f_kl):.2e}")
        print(f"  ref top-2 gap AT flips           : mean {g[0]:.3e}  p50 {g[1]:.3e}  max {g[2]:.3e}")
    ng = _stats(nonflip_gap2)
    print(f"  ref top-2 gap at NON-flips        : mean {ng[0]:.3e}  p50 {ng[1]:.3e}   "
          f"(flips concentrate where the gap — model certainty — is small)")

    # per-passage bucket: low-entropy repeat should be ~flip-free; prose flips should be the ties.
    for tg, label in ((0, "prose "), (1, "repeat")):
        idx = [i for i in range(n) if tag_flat[i] == tg]
        if not idx:
            continue
        fl = sum(1 for i in idx if agree[i] < 0.5)
        g = _stats([gap2[i] for i in idx])
        print(f"  [{label}] {len(idx)} pos   flips {fl}   agree {100 * sum(agree[i] for i in idx) / len(idx):.2f}%"
              f"   median ref top-2 gap {g[1]:.3e}")

    within = (
        abs(dppl) <= DPPL_TOL and abs(dppl) <= DPPL_TOL * max(ppl_ref, 1.0)
        and sum(kl) / n <= KL_TOL                      # mean KL — see KL_TOL note (max is a tie artifact)
        and (n_flip == 0 or max(f_odds) <= ODDS_TOL)
    )
    print("\nWITHIN NOISE ✓ — every B=48 flip is a sampling-indistinguishable tie; ppl + distributions "
          "unchanged.\n→ B=48 arena is e2e-equivalent to the per-stream loop within fp noise."
          if within else
          "\nREVIEW — at least one flip exceeds the tie thresholds (odds/KL/Δppl); inspect the rows above.")


def run() -> None:
    mx.set_wired_limit(int(WIRED_GIB * 1024 ** 3))
    model = DSV4BatchedResidentModel(ART, max_batch=B, packed_experts=True)
    tok = DeepSeekV4Tokenizer.from_pretrained(ART)
    windows, tags = _build_windows(tok)
    print(f"built {B} windows ({PROMPT}+{GEN}+1 tok): {tags.count(0)} prose + {tags.count(1)} repeat")

    # PASS 1 — per-stream-loop reference (fused=False, discrete cache): bit-exact per-stream canonical.
    model._fused = False
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    ref_caches = _prefill(model, windows, arena=False)
    ref_steps = _teacher_force(model, windows, ref_caches)
    del ref_caches
    print(f"reference (looped) pass: {time.perf_counter() - t0:.1f}s")

    # PASS 2 — arena (fused=True, leased rows): the #18 M4 default serving path.
    model._fused = True
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    ar_caches = _prefill(model, windows, arena=True)
    try:
        offsets = [PROMPT] * B
        parts: list[dict] = []
        for t in range(GEN):
            ids_in = [mx.array([w[PROMPT + t]]) for w in windows]
            out = model.step_batch(ids_in, ar_caches, offsets)
            ar_logits = mx.stack([out[b][0, -1] for b in range(B)]).astype(mx.float32)
            tgt = mx.array([w[PROMPT + t + 1] for w in windows])
            m = _compare(ref_steps[t], ar_logits, tgt, TEMP)
            mx.eval(*m.values())
            parts.append(m)
            offsets = [o + 1 for o in offsets]
    finally:
        for c in ar_caches:
            model.free_cache(c)
    arena_peak = _gib(mx.get_peak_memory())
    print(f"arena pass: {time.perf_counter() - t0:.1f}s")

    _report(parts, tags, arena_peak)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        run()
