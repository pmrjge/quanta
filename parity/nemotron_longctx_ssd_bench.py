"""Nemotron-H 1M long-context validation + Mamba-SSD prefill-scaling benchmark (#41).

Nemotron-H is a hybrid: 40 Mamba-2 layers carry an **O(1)-state recurrence** (length-agnostic)
and only the 8 ``*`` attention layers grow a KV cache. So the architecture's long-context cost is
dominated by the chunked SSD prefill — which is **O(T)** by construction (segment-sum decay +
batched matmuls; see :mod:`quanta.nemotron.mamba_ssd`) — while the handful of attention layers are
the only O(T^2) term (each materializes a TxT score block in the naive path; the fast SDPA path is
tiled). This harness validates that picture and quantifies it.

Two modes:

* ``--selftest`` (MODEL-FREE, runnable now — the gate). Exercises the harness's own accounting on a
  tiny in-process stub: (a) the teacher-forced ppl computation matches the closed form on known
  logits (uniform -> ppl=V, one-hot-peaked -> ppl~=1), (b) the throughput/per-length timing
  records are internally consistent (tokens/sec = tokens/elapsed; aggregate matches sum), and
  (c) the O(T^2) attention-score alloc guard raises *before* allocating when a requested context
  exceeds a memory-safe bound. Prints PASS/FAIL. **No checkpoint load, no GPU big ops.**

* default / ``--run`` (DEFERRED real sweep — DO NOT run alongside any other big GPU job). Loads the
  RAM-resident int4 model and sweeps the context length, reporting per-length teacher-forced
  perplexity on long real text, prefill tokens/sec, the Mamba-SSD prefill throughput vs T (to show
  the O(T) SSM scaling against the few O(T^2) attention layers), and an optional needle-in-haystack
  retrieval check.

------------------------------------------------------------------------------------------------
SAFETY — the real sweep loads the resident model (~70 GB int4) and pushes it to long context, a
heavy GPU + memory job. Run it ALONE: never concurrently with another large model load / GPU
capture (a prior host OOM hard-rebooted this machine). The exact heavy invocation is::

    # ONLY when the GPU is otherwise idle and no other big job is queued:
    uv run --with tokenizers python -m parity.nemotron_longctx_ssd_bench --run

``--selftest`` is the only thing safe to run on a busy host::

    uv run --with numpy python -m parity.nemotron_longctx_ssd_bench --selftest
------------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass

import mlx.core as mx

ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"

# Context-length sweep for the real run, toward 1M. Each length is screened at runtime by the
# KV-cache guard (the runtime's true O(T) allocation; tiled SDPA never builds the naive [H,T,T]).
SWEEP_LENGTHS = (4096, 16384, 65536, 262144, 524288, 1048576)

# O(T^2) attention-score alloc guard. The 8 ``*`` layers are the only quadratic term; we estimate
# the peak bytes of a single attention layer's score-related tensors at the naive (non-tiled) path
# and refuse to proceed if it would blow past a safe fraction of the working set. The resident
# runtime uses *tiled* fast SDPA (never materializes TxT) for the real forward, but a long-context
# sweep must still fail LOUD here rather than risk an OOM (memory-safety rule).
WORKING_SET_GIB = 490.4          # M3 Ultra recommended max working set (CLAUDE.md)
SAFE_SCORE_FRACTION = 0.25       # refuse a single-layer naive [H,T,T] score block bigger than this

# The real runtime uses tiled SDPA (never materializes [H,T,T]); its only length-growing allocation
# is the KV cache, which is O(T): 8 attention layers x (k,v) x n_kv_heads x head_dim x T. At 1M that
# is ~8 GiB bf16 (see attention.py). The real sweep is screened on this true allocation against a
# safe slice of the headroom left by the ~70 GiB resident weights — so it can reach toward 1M.
RESIDENT_WEIGHTS_GIB = 70.0      # ~int4 resident weight set (decode-bandwidth bake)
SAFE_KV_FRACTION = 0.5           # refuse a KV cache bigger than this fraction of the free headroom


@dataclass(frozen=True)
class LengthRecord:
    """One context length's measured accounting (real run) or stubbed values (selftest)."""

    length: int
    ppl: float
    prefill_s: float          # wall time of the full prefill forward
    ssd_s: float              # wall time attributable to the Mamba-SSD prefill portion

    @property
    def prefill_toks_per_s(self) -> float:
        if self.prefill_s <= 0:
            raise ValueError(f"non-positive prefill_s={self.prefill_s} for length {self.length}")
        return self.length / self.prefill_s

    @property
    def ssd_toks_per_s(self) -> float:
        if self.ssd_s <= 0:
            raise ValueError(f"non-positive ssd_s={self.ssd_s} for length {self.length}")
        return self.length / self.ssd_s


# --------------------------------------------------------------------------------------------
# Accounting primitives — shared by both modes; unit-tested model-free by --selftest.
# --------------------------------------------------------------------------------------------
def teacher_forced_ppl(logits: mx.array, targets: mx.array) -> tuple[float, float]:
    """Teacher-forced perplexity + top-1 next-token accuracy from already-shifted logits/targets.

    ``logits`` ``[t, vocab]`` (fp32-comparable) are the predictions for ``targets`` ``[t]`` (the
    next token at each position). Identical CE math to the existing ppl harnesses so numbers are
    directly comparable: ``ce = logsumexp(logits) - logits[target]``; ``ppl = exp(mean ce)``.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected logits [t, vocab], got shape {tuple(logits.shape)}")
    if targets.shape[0] != logits.shape[0]:
        raise ValueError(f"targets len {targets.shape[0]} != logits rows {logits.shape[0]}")
    lg = logits.astype(mx.float32)
    ce = mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, targets[:, None], axis=-1)[:, 0]
    ppl = mx.exp(ce.mean()).item()
    acc = (mx.argmax(lg, axis=-1) == targets).astype(mx.float32).mean().item()
    return ppl, acc


def attn_score_bytes(length: int, n_attn_layers: int, n_heads: int, *, bytes_per_elem: int = 4) -> int:
    """Peak bytes of one O(T^2) attention-score block (naive path) — the quadratic-term estimate.

    A single ``*`` layer's score matrix is ``[heads, T, T]`` (fp32 softmax). The 8 attention
    layers run one at a time within the forward, so the *peak* is one layer's block, not the sum;
    ``n_attn_layers`` is accepted for documentation/over-estimate callers but the peak uses one.
    """
    if length < 0 or n_attn_layers < 0 or n_heads <= 0:
        raise ValueError(f"bad dims length={length} n_attn={n_attn_layers} n_heads={n_heads}")
    return n_heads * length * length * bytes_per_elem


def guard_attention_alloc(length: int, n_attn_layers: int, n_heads: int) -> None:
    """Fail LOUD before allocating if a length's O(T^2) attention score block is unsafe.

    Raises ``MemoryError`` when one attention layer's ``[heads, T, T]`` score block would exceed
    ``SAFE_SCORE_FRACTION`` of the working set — *before* anything is allocated (memory-safety
    rule: guard O(T^2) allocations to fail loud, never OOM). The real runtime uses tiled SDPA and
    would not actually materialize this, but the sweep refuses lengths it cannot prove safe.
    """
    gib = attn_score_bytes(length, n_attn_layers, n_heads) / 1024**3
    cap = SAFE_SCORE_FRACTION * WORKING_SET_GIB
    if gib > cap:
        raise MemoryError(
            f"attention score block [heads={n_heads}, T={length}, T={length}] ~= {gib:.1f} GiB "
            f"exceeds safe cap {cap:.1f} GiB ({SAFE_SCORE_FRACTION:.0%} of {WORKING_SET_GIB} GiB "
            f"working set). Refusing to allocate; lower the context length or use tiled SDPA only."
        )


def kv_cache_bytes(length: int, n_attn_layers: int, n_kv_heads: int, head_dim: int,
                   *, bytes_per_elem: int = 2) -> int:
    """Bytes of the resident KV cache at context ``length`` — the real runtime's O(T) allocation.

    ``n_attn_layers`` x (k and v) x ``n_kv_heads`` x ``head_dim`` x ``length`` x bytes. This is the
    only length-growing allocation in the tiled-SDPA forward (the 40 mamba layers keep an O(1)
    state); for Nemotron-H it is ~8 GiB bf16 at 1M.
    """
    if length < 0 or n_attn_layers < 0 or n_kv_heads <= 0 or head_dim <= 0:
        raise ValueError(f"bad dims length={length} n_attn={n_attn_layers} "
                         f"n_kv={n_kv_heads} hd={head_dim}")
    return n_attn_layers * 2 * n_kv_heads * head_dim * length * bytes_per_elem


def guard_kv_alloc(length: int, n_attn_layers: int, n_kv_heads: int, head_dim: int) -> None:
    """Fail LOUD before the real forward if the O(T) KV cache won't fit the safe headroom.

    The headroom is the working set minus the resident weights; we refuse a KV cache larger than
    ``SAFE_KV_FRACTION`` of it (memory-safety rule: raise before allocating, never OOM).
    """
    gib = kv_cache_bytes(length, n_attn_layers, n_kv_heads, head_dim) / 1024**3
    cap = SAFE_KV_FRACTION * (WORKING_SET_GIB - RESIDENT_WEIGHTS_GIB)
    if gib > cap:
        raise MemoryError(
            f"KV cache at T={length} ~= {gib:.1f} GiB exceeds safe cap {cap:.1f} GiB "
            f"({SAFE_KV_FRACTION:.0%} of {WORKING_SET_GIB - RESIDENT_WEIGHTS_GIB:.0f} GiB free "
            f"headroom). Refusing to allocate; lower the context length."
        )


def aggregate_throughput(records: list[LengthRecord]) -> tuple[int, float, float]:
    """Total tokens, total prefill seconds, aggregate prefill tokens/sec across all records."""
    total_tok = sum(r.length for r in records)
    total_s = sum(r.prefill_s for r in records)
    if total_s <= 0:
        raise ValueError("aggregate prefill time is non-positive")
    return total_tok, total_s, total_tok / total_s


# --------------------------------------------------------------------------------------------
# MODE 1: --selftest (model-free). The gate.
# --------------------------------------------------------------------------------------------
def selftest() -> bool:
    """Validate ppl math, timing accounting, and the O(T^2) alloc guard — no model. Prints rows."""
    ok = True
    print("\n=== Nemotron-H longctx/SSD harness self-test (MODEL-FREE) ===")

    # (a) ppl on known logits: uniform over V -> ppl = V exactly; one-hot-peaked -> ppl ~= 1.
    v = 50
    t = 16
    uniform = mx.zeros((t, v))                                   # equal logits -> uniform softmax
    targets = mx.arange(t) % v
    ppl_u, _acc_u = teacher_forced_ppl(uniform, targets)
    uniform_ok = abs(ppl_u - v) < 1e-3
    ok &= uniform_ok
    print(f"uniform ppl == V     : {ppl_u:.4f} (expect {v})            "
          f"{'PASS' if uniform_ok else 'FAIL'}")

    peaked = mx.zeros((t, v))
    big = mx.full((t,), 30.0)                                    # one logit huge at the target
    peaked = mx.put_along_axis(peaked, targets[:, None], big[:, None], axis=-1)
    ppl_p, acc_p = teacher_forced_ppl(peaked, targets)
    peaked_ok = abs(ppl_p - 1.0) < 1e-2 and abs(acc_p - 1.0) < 1e-6
    ok &= peaked_ok
    print(f"peaked ppl ~= 1, acc1: ppl {ppl_p:.4f} acc {acc_p:.3f}        "
          f"{'PASS' if peaked_ok else 'FAIL'}")

    # closed-form cross-check: ppl = exp(mean CE) for a hand-built two-logit case.
    z = mx.array([[2.0, 0.0]])                                   # target=0
    tgt = mx.array([0])
    ppl_cf, _ = teacher_forced_ppl(z, tgt)
    expect_cf = math.exp(math.log(math.exp(2.0) + math.exp(0.0)) - 2.0)
    cf_ok = abs(ppl_cf - expect_cf) < 1e-4
    ok &= cf_ok
    print(f"closed-form CE match : {ppl_cf:.5f} (expect {expect_cf:.5f})  "
          f"{'PASS' if cf_ok else 'FAIL'}")

    # (b) throughput/timing accounting consistency on stub records.
    recs = [
        LengthRecord(length=4096, ppl=5.5, prefill_s=2.0, ssd_s=1.0),
        LengthRecord(length=8192, ppl=5.6, prefill_s=4.0, ssd_s=2.0),
    ]
    per_ok = (abs(recs[0].prefill_toks_per_s - 2048.0) < 1e-6
              and abs(recs[1].ssd_toks_per_s - 4096.0) < 1e-6)
    ok &= per_ok
    total_tok, total_s, agg = aggregate_throughput(recs)
    agg_ok = total_tok == 12288 and abs(total_s - 6.0) < 1e-9 and abs(agg - 12288 / 6.0) < 1e-6
    ok &= agg_ok
    print(f"per-length tok/s     : {recs[0].prefill_toks_per_s:.0f}/{recs[1].ssd_toks_per_s:.0f}"
          f" (expect 2048/4096)   {'PASS' if per_ok else 'FAIL'}")
    print(f"aggregate tok/s      : {agg:.0f} over {total_tok} tok / {total_s:.1f}s   "
          f"{'PASS' if agg_ok else 'FAIL'}")

    # non-positive elapsed must fail loud (no silent div-by-zero).
    neg_guard = False
    try:
        _ = LengthRecord(length=10, ppl=1.0, prefill_s=0.0, ssd_s=1.0).prefill_toks_per_s
    except ValueError:
        neg_guard = True
    ok &= neg_guard
    print(f"zero-time fails loud : {neg_guard}                          "
          f"{'PASS' if neg_guard else 'FAIL'}")

    # (c) O(T^2) naive-attention-score guard: safe at small T, raises at long T, before allocating.
    n_attn, n_heads, n_kv, head_dim = 8, 32, 2, 128       # Nemotron-H attention dims
    small_ok = _length_is_safe(4096, n_attn, n_heads)        # ~2 GiB fp32 -> safe
    ok &= small_ok
    big_fires = not _length_is_safe(1_048_576, n_attn, n_heads)   # ~128 TiB fp32 -> must raise
    ok &= big_fires
    print(f"O(T^2) guard @4096   : {small_ok}                          "
          f"{'PASS' if small_ok else 'FAIL'}")
    print(f"O(T^2) guard @1M     : {big_fires} (raise before alloc)      "
          f"{'PASS' if big_fires else 'FAIL'}")

    # (c2) O(T) KV-cache guard (the real-sweep screen): 1M fits (~8 GiB), absurd T is refused.
    kv_1m_ok = _kv_length_is_safe(1_048_576, n_attn, n_kv, head_dim)   # ~8 GiB bf16 -> safe
    ok &= kv_1m_ok
    kv_fires = not _kv_length_is_safe(1 << 28, n_attn, n_kv, head_dim)  # ~2 TiB -> must raise
    ok &= kv_fires
    kv_bytes_1m = kv_cache_bytes(1_048_576, n_attn, n_kv, head_dim)
    print(f"KV guard @1M (~8GiB) : {kv_1m_ok} ({kv_bytes_1m / 1024**3:.1f} GiB)            "
          f"{'PASS' if kv_1m_ok else 'FAIL'}")
    print(f"KV guard @256M       : {kv_fires} (raise before alloc)      "
          f"{'PASS' if kv_fires else 'FAIL'}")

    # the real sweep is screened by the KV guard; with 1M-class lengths it should admit them all.
    admitted = [n for n in SWEEP_LENGTHS if _kv_length_is_safe(n, n_attn, n_kv, head_dim)]
    sweep_ok = len(admitted) == len(SWEEP_LENGTHS)
    ok &= sweep_ok
    print(f"sweep KV-screened    : {len(admitted)}/{len(SWEEP_LENGTHS)} lengths admitted (incl 1M) "
          f"  {'PASS' if sweep_ok else 'FAIL'}")

    print("\n" + ("PASS" if ok else "FAIL"))
    return ok


def _length_is_safe(length: int, n_attn_layers: int, n_heads: int) -> bool:
    """True iff ``length`` passes the O(T^2) naive-attention-score guard (no raise)."""
    try:
        guard_attention_alloc(length, n_attn_layers, n_heads)
        return True
    except MemoryError:
        return False


def _kv_length_is_safe(length: int, n_attn_layers: int, n_kv_heads: int, head_dim: int) -> bool:
    """True iff ``length``'s O(T) KV cache passes the runtime allocation guard (no raise)."""
    try:
        guard_kv_alloc(length, n_attn_layers, n_kv_heads, head_dim)
        return True
    except MemoryError:
        return False


# --------------------------------------------------------------------------------------------
# MODE 2: --run (DEFERRED real sweep). MUST NOT execute under --selftest. Loads the model ALONE.
# --------------------------------------------------------------------------------------------
# Real long text for the teacher-forced sweep: a public-domain passage repeated/extended to reach
# each sweep length (the per-length forward truncates to that many tokens). Kept long enough that
# even the largest in-budget length is exercised on genuine prose, not padding.
_LONG_TEXT = (
    "The history of science is the study of the development of science and scientific knowledge, "
    "including both the natural and social sciences. Science is a body of empirical, theoretical, "
    "and practical knowledge about the natural world, produced by scientists who emphasize the "
    "observation, explanation, and prediction of real-world phenomena. Historiography of science, "
    "in contrast, studies the methods employed by historians of science. The English word "
    "scientist is relatively recent, first coined by William Whewell in the nineteenth century. "
    "Earlier, investigators of nature called themselves natural philosophers. While empirical "
    "investigations of the natural world have been described since classical antiquity, and the "
    "scientific method has been employed since the Middle Ages, the dawn of modern science is "
    "often traced back to the early modern period, during what is known as the Scientific "
    "Revolution that took place in sixteenth and seventeenth century Europe. Scientific methods "
    "are considered to be so fundamental to modern science that some consider earlier inquiries "
    "into nature to be pre-scientific. "
)


def _needle_text(filler_tokens: int, secret: str = "The hidden access code is 7Q4Z9.") -> str:
    """A long filler with one factual needle near the middle — for the optional retrieval check."""
    half = "filler context. " * max(1, filler_tokens // 2)
    return half + secret + " " + half + " Question: what is the hidden access code? Answer:"


def run_real_sweep(art_dir: str = ART, *, do_needle: bool = False) -> None:
    """DEFERRED heavy path. Loads ``NemotronResidentModel`` ALONE and runs the sweep.

    Reports per length: teacher-forced ppl on long prose, prefill tok/s, and the Mamba-SSD prefill
    tok/s (timed by re-running the mamba mixers' chunked prefill in isolation, contrasting the O(T)
    SSM scaling with the few O(T^2) attention layers). Uses ONLY the real runtime API.

    WARNING: run with no other big GPU job present (host-OOM safety).
    """
    # Imports are local so --selftest never imports the runtime / tokenizer / model graph.
    from quanta.nemotron.config import NemotronHConfig
    from quanta.nemotron.generate import attn_caches
    from quanta.nemotron.runtime import NemotronResidentModel
    from quanta.nemotron.tokenizer import NemotronTokenizer

    cfg = NemotronHConfig.from_pretrained(art_dir)
    n_attn = cfg.count("attention")
    n_mamba = cfg.count("mamba")
    n_heads = cfg.num_attention_heads
    n_kv, head_dim = cfg.num_key_value_heads, cfg.head_dim

    # Screen lengths against the runtime's TRUE O(T) allocation — the KV cache (tiled SDPA never
    # builds the naive [H,T,T] block) — so the sweep can reach toward 1M. Fail loud before loading.
    lengths = [n for n in SWEEP_LENGTHS if _kv_length_is_safe(n, n_attn, n_kv, head_dim)]
    skipped = [n for n in SWEEP_LENGTHS if n not in lengths]
    if not lengths:
        raise MemoryError("every configured sweep length is refused by the KV-cache guard")

    mx.set_wired_limit(int(120 * 1024**3))
    t_load = time.perf_counter()
    model = NemotronResidentModel(art_dir)
    tok = NemotronTokenizer(art_dir)
    load_min = (time.perf_counter() - t_load) / 60

    # Build a token stream long enough for the largest in-budget length.
    max_len = max(lengths)
    base_ids = tok.encode(_LONG_TEXT, add_bos=False)
    reps = -(-max_len // max(1, len(base_ids)))            # ceil
    full_ids = (base_ids * reps)[:max_len]

    print("\n=== Nemotron-H int4 RESIDENT long-context sweep ===")
    print(f"layers: {n_mamba} mamba (O(1) state) | {n_attn} attention (O(T^2)) | heads {n_heads}")
    print(f"load {load_min:.1f} min | artifact {art_dir}")
    if skipped:
        print(f"skipped (guard): {skipped}")
    print(f"\n{'T':>9} {'ppl':>9} {'prefill s':>10} {'prefill tok/s':>14} "
          f"{'SSD tok/s':>11} {'SSD/T const?':>13}")

    records: list[LengthRecord] = []
    prev_ssd_per_tok = None
    for length in lengths:
        # Fail loud before allocating the KV cache for this length (memory-safety rule).
        guard_kv_alloc(length, n_attn, n_kv, head_dim)
        ids = mx.array(full_ids[:length])

        # Full prefill forward (real runtime; tiled fast SDPA inside — never materializes a TxT
        # score matrix, so the only O(T) growth is the KV cache; the 40 mamba layers are O(1)).
        caches = attn_caches(model)
        mx.eval(ids)
        t0 = time.perf_counter()
        logits, _, _ = model(ids, caches=caches)
        mx.eval(logits)
        prefill_s = time.perf_counter() - t0

        # Teacher-forced ppl on this length.
        lg = logits[0, :-1].astype(mx.float32)
        ppl, _acc = teacher_forced_ppl(lg, ids[1:])

        # Isolate the Mamba-SSD prefill cost: time only the mamba layers' mixer on the same input
        # depth. Re-embed once, then run each mamba mixer's chunked prefill (the O(T) SSD path).
        ssd_s = _time_ssd_prefill(model, ids)

        rec = LengthRecord(length=length, ppl=ppl, prefill_s=prefill_s, ssd_s=ssd_s)
        records.append(rec)

        # O(T): per-token SSD time should be ~flat as T grows (vs the attention O(T^2) growth).
        ssd_per_tok = ssd_s / length
        const_flag = "—" if prev_ssd_per_tok is None else f"{ssd_per_tok / prev_ssd_per_tok:.2f}x"
        prev_ssd_per_tok = ssd_per_tok
        print(f"{length:>9} {ppl:>9.3f} {prefill_s:>10.2f} {rec.prefill_toks_per_s:>14.0f} "
              f"{rec.ssd_toks_per_s:>11.0f} {const_flag:>13}")

    total_tok, total_s, agg = aggregate_throughput(records)
    print(f"\naggregate prefill    : {agg:.0f} tok/s over {total_tok} tok / {total_s:.1f}s")
    print("SSD tok/s ~constant across T ⇒ O(T) SSM prefill (the 8 attention layers are the only "
          "O(T^2) term).")

    if do_needle:
        _run_needle(model, tok)


def _time_ssd_prefill(model, ids: mx.array) -> float:
    """Wall time of just the Mamba-SSD prefill portion at this depth (the O(T) contrast).

    Embeds once and runs each mamba mixer's chunked prefill on the (per-layer-normed) hidden
    stream, exactly as the resident forward does for mamba layers. Uses the real block API
    (``blk.norm`` + ``blk.mixer(...)``); attention/moe layers are skipped so this isolates the
    SSM scaling. Returns seconds (after eval, so the time is real compute, not lazy graph build).
    """
    h = model.embed_w[ids][None].astype(mx.bfloat16)
    mx.eval(h)
    t0 = time.perf_counter()
    for blk in model.layers:
        if blk.kind == "mamba":
            y, _ssm, _conv = blk.mixer(blk.norm(h), state=None, conv_state=None)
            h = h + y
    mx.eval(h)
    return time.perf_counter() - t0


def _run_needle(model, tok, filler_tokens: int = 4096) -> None:
    """Optional needle-in-haystack: place a fact mid-context and check it's recovered (top-1)."""
    from quanta.nemotron.generate import attn_caches

    text = _needle_text(filler_tokens)
    ids = mx.array(tok.encode(text, add_bos=False))
    caches = attn_caches(model)
    logits, _, _ = model(ids, caches=caches)
    nxt = int(mx.argmax(logits[0, -1]).item())
    decoded = tok.decode([nxt]) if hasattr(tok, "decode") else str(nxt)
    print(f"\nneedle-in-haystack   : T={ids.shape[0]} next-token after 'Answer:' -> {decoded!r}")


_HEAVY_NOTICE = (
    "Refusing to run the heavy long-context sweep without the explicit --run flag.\n"
    "  The real sweep loads NemotronResidentModel (~70 GB int4) at long context — a heavy GPU +\n"
    "  memory job that must run ALONE (never alongside another big model load / GPU capture; a\n"
    "  prior host OOM hard-rebooted this machine). When the GPU is otherwise idle, run:\n"
    "      uv run --with tokenizers python -m parity.nemotron_longctx_ssd_bench --run\n"
    "  Safe anytime (model-free):\n"
    "      uv run --with numpy python -m parity.nemotron_longctx_ssd_bench --selftest"
)


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return 0 if selftest() else 1
    if "--run" not in argv:
        # Guard the heavy path: a bare / accidental invocation never loads the model (host-OOM
        # safety). The deferred real sweep requires the explicit --run opt-in.
        print(_HEAVY_NOTICE)
        return 2
    run_real_sweep(do_needle="--needle" in argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
