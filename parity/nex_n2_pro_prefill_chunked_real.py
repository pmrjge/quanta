"""Nex-N2-Pro N3-3 — long-context chunked prefill re-gate @ 397B (SOLO).

Real-weight gate for the chunked-prefill substrate (the 1M-window feasibility lever, model-free
gated in ``parity/qwen35_prefill_chunked_test.py``): the per-token serving prefill is O(T) full
forwards (~10 tok/s ⇒ a 32K prompt would take ~an hour; 1M is infeasible), so long context runs
:func:`quanta.qwen35.runtime.chunked_prefill` — one bounded forward per ``chunk_tokens`` block,
with the 45 Gated-DeltaNet layers on the chunk-parallel WY/UT kernel (``gdn_chunked_wy``) and the
15 full-attention layers extending their int8 KV under a bottom-right-aligned causal SDPA.

ONE resident load (``Qwen35ResidentModel``, packed mixer + packed int4 experts — the served
kernels), three gates:

  1. **Chunked == per-token (greedy-exact).** A real ~1K-token prose prompt: the per-token
     decode-path prefill (the proven serving semantics — bit-identical to ``generate``'s seeding)
     vs ``prefill_chunked`` (WY arm AND sequential arm), then a greedy continuation from each
     cache. All three greedy traces must MATCH token-for-token; the last-prefill-position
     |Δlogit| is reported INFO (the packed ``mx.quantized_matmul`` runs at batch-M=chunk vs M=1,
     the documented greedy-token-stable ULP class — same codes, reordered accumulation).
  2. **Needle retrieval @ depth (chunked-only).** A ~LONG_TOK-token synthetic haystack (varied
     filler, within the native 262K window — the >262K YaRN arbiter is the separate 1M gate) with
     a needle planted at 50% depth; chunked prefill + greedy decode must reproduce the needle's
     payload digits. Proves the chunk-carried state (conv window + GDN recurrent + KV) supports
     real retrieval at lengths the per-token path cannot reach.
  3. **Throughput (reported, not asserted).** Prefill tok/s for both runs — the feasibility
     headline (per-token serving prefill is ~10 tok/s for comparison).

One model resident — **RUN SOLO** (~215 GiB + chunk transients; well under the 490.4 ceiling).

    uv run python -u -m parity.nex_n2_pro_prefill_chunked_real            # full (1K parity + 32K needle)
    uv run python -u -m parity.nex_n2_pro_prefill_chunked_real 256 2048   # cheap smoke (tiny lengths)

# parity-gate: real-weight
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

from parity.nex_n2_pro_ppl import PROSE
from quanta.qwen35.runtime import Qwen35ResidentModel
from quanta.qwen35.tokenizer import Qwen35Tokenizer

ART = "/Users/pmrj/models/Nex-N2-Pro-quanta_int4g64"   # the SHIPPED int4-g64 artifact

PARITY_TOK = 1024        # gate-1 prompt length (per-token reference arm is O(T) — keep bounded)
LONG_TOK = 32768         # gate-2 haystack length (within native 262144; >262K = the 1M gate)
CHUNK = 2048             # prefill chunk size (multiple of the GDN sub-chunk 64)
GEN = 24                 # greedy continuation tokens (gate 1)
NEEDLE_GEN = 16          # greedy answer tokens (gate 2)
WIRED_GIB = 300

NEEDLE_CODE = "739214"
NEEDLE = f"The secret quanta access code is {NEEDLE_CODE}."
QUESTION = "\n\nWhat is the secret quanta access code? The secret quanta access code is"

_TOPICS = ("harbor cranes", "alpine meteorology", "tide tables", "freight manifests",
           "orchard grafting", "kiln firing schedules", "glacier surveys", "archival inks",
           "telegraph relays", "canal locks", "loom patterns", "foundry alloys")


def _haystack(tok: Qwen35Tokenizer, total_tokens: int, depth: float = 0.5) -> list[int]:
    """A ``total_tokens``-token synthetic long document: varied filler paragraphs (no literal
    repetition — reasoning models degenerate on pure repeats) with the needle sentence planted at
    ``depth``, and the retrieval question appended. Returns token ids."""
    paras = []
    i = 0
    ids_len = 0
    # generously over-produce paragraphs, then trim by tokens
    while ids_len < total_tokens * 4:  # chars ≈ 4×tokens heuristic
        t = _TOPICS[i % len(_TOPICS)]
        paras.append(f"Entry {i}: the survey of {t} continued through week {i % 52}, recording "
                     f"{(i * 37) % 1000} observations and {(i * 13) % 100} anomalies for the "
                     f"quarterly ledger.")
        ids_len += len(paras[-1])
        i += 1
    cut = max(1, int(len(paras) * depth))
    text = " ".join(paras[:cut]) + " " + NEEDLE + " " + " ".join(paras[cut:])
    ids = tok.encode(text)
    body = total_tokens - len(tok.encode(QUESTION))
    if len(ids) < body:
        raise RuntimeError(f"haystack underproduced: {len(ids)} < {body} tokens")
    ids = ids[:body]
    # rule 6: the trimmed haystack must still CONTAIN the needle (depth must survive the cut)
    txt = tok.decode(ids)
    if NEEDLE_CODE not in txt:
        raise RuntimeError("needle fell outside the trimmed haystack — adjust depth/length")
    return ids + tok.encode(QUESTION)


def _greedy_from(model: Qwen35ResidentModel, cache, first_logits: mx.array, offset: int,
                 n: int) -> list[int]:
    """Greedy ``n``-token continuation through the per-token decode path (serving semantics)."""
    tok_id = int(mx.argmax(first_logits[0, -1]).item())
    out = [tok_id]
    for _ in range(n - 1):
        lg = model([tok_id], caches=cache, offset=offset)
        mx.eval(lg)
        tok_id = int(mx.argmax(lg[0, -1]).item())
        out.append(tok_id)
        offset += 1
    return out


def run(parity_tok: int = PARITY_TOK, long_tok: int = LONG_TOK, chunk: int = CHUNK) -> None:
    mx.set_cache_limit(8 * 1024 ** 3)
    mx.set_wired_limit(int(WIRED_GIB * 1024 ** 3))
    tok = Qwen35Tokenizer.from_pretrained(ART)
    print("=== Nex-N2-Pro N3-3 — long-context chunked prefill re-gate @ 397B (SOLO) ===")
    print(f"  artifact={ART}")
    print(f"  parity_tok={parity_tok}  long_tok={long_tok}  chunk={chunk}  gen={GEN}", flush=True)

    t0 = time.perf_counter()
    model = Qwen35ResidentModel(ART)
    print(f"  loaded resident {mx.get_active_memory() / 1024**3:.1f} GiB in "
          f"{(time.perf_counter() - t0) / 60:.1f} min (packed mixer + packed int4 experts)",
          flush=True)

    # --- gate 1: chunked == per-token, greedy-exact (real ~1K prompt) ------------------------------
    base = tok.encode(PROSE)
    ids = (base * (parity_tok // len(base) + 1))[:parity_tok]
    print(f"\n  [gate 1] chunked == per-token prefill ({len(ids)} tok) + {GEN}-tok greedy:",
          flush=True)

    t0 = time.perf_counter()
    cache_ref = model.make_caches()
    lg_ref = model(ids, caches=cache_ref, offset=0)     # per-token decode-path prefill (serving)
    mx.eval(lg_ref)
    sec_ref = time.perf_counter() - t0
    lg_ref_last = lg_ref[:, -1:]
    trace_ref = _greedy_from(model, cache_ref, lg_ref_last, len(ids), GEN)
    del cache_ref, lg_ref
    mx.clear_cache()

    arms = {}
    for wy in (True, False):
        t0 = time.perf_counter()
        cache_c = model.make_caches()
        lg_c = model.prefill_chunked(ids, cache_c, chunk_tokens=chunk, wy=wy)
        mx.eval(lg_c)
        sec_c = time.perf_counter() - t0
        trace_c = _greedy_from(model, cache_c, lg_c, len(ids), GEN)
        dlog = float(mx.max(mx.abs(lg_c.astype(mx.float32) - lg_ref_last.astype(mx.float32))))
        arms[wy] = (trace_c, dlog, sec_c)
        del cache_c, lg_c
        mx.clear_cache()

    ok1 = arms[True][0] == trace_ref and arms[False][0] == trace_ref
    print(f"    per-token ref : {sec_ref:6.1f}s ({len(ids) / sec_ref:6.1f} tok/s)  "
          f"trace={trace_ref[:8]}…", flush=True)
    for wy in (True, False):
        tr, dl, sc = arms[wy]
        match = tr == trace_ref
        print(f"    chunked wy={str(wy):<5}: {sc:6.1f}s ({len(ids) / sc:6.1f} tok/s, "
              f"{sec_ref / sc:5.1f}x)  greedy=={'OK' if match else 'FAIL'}  "
              f"|Δlogit|={dl:.3f} (INFO: batch-M ULP class)", flush=True)
    if not ok1:
        print("FAIL — chunked prefill diverged from the per-token serving prefill (greedy trace)",
              flush=True)
        raise SystemExit(1)

    # --- gate 2: needle retrieval at depth, chunked-only -------------------------------------------
    hay = _haystack(tok, long_tok)
    print(f"\n  [gate 2] needle @ 50% depth in {len(hay)}-tok haystack (chunked-only, wy=True):",
          flush=True)
    t0 = time.perf_counter()
    cache_n = model.make_caches()
    lg_n = model.prefill_chunked(hay, cache_n, chunk_tokens=chunk, wy=True)
    mx.eval(lg_n)
    sec_n = time.perf_counter() - t0
    ans_ids = _greedy_from(model, cache_n, lg_n, len(hay), NEEDLE_GEN)
    answer = tok.decode(ans_ids)
    ok2 = NEEDLE_CODE in answer
    print(f"    prefill {len(hay)} tok in {sec_n:6.1f}s ({len(hay) / sec_n:6.1f} tok/s)  "
          f"peak {mx.get_peak_memory() / 1024**3:.1f} GiB", flush=True)
    print(f"    answer: {answer!r}  needle={'FOUND' if ok2 else 'MISSING'}", flush=True)
    if not ok2:
        print(f"FAIL — needle {NEEDLE_CODE!r} not retrieved from the {len(hay)}-tok haystack",
              flush=True)
        raise SystemExit(1)

    print(f"\nVERDICT: PASS — chunked prefill (WY + sequential) greedy-exact vs the per-token "
          f"serving prefill at 397B; needle retrieved at 50% depth of {len(hay)} tok; chunked "
          f"prefill {len(hay) / sec_n:.0f} tok/s (per-token ref {len(ids) / sec_ref:.0f} tok/s ⇒ "
          f"{(len(hay) / sec_n) / (len(ids) / sec_ref):.0f}x).", flush=True)
    print("PARITY-CHECKS: 2", flush=True)   # (1) chunked==per-token greedy, (2) needle retrieval


if __name__ == "__main__":
    pt = int(sys.argv[1]) if len(sys.argv) > 1 else PARITY_TOK
    lt = int(sys.argv[2]) if len(sys.argv) > 2 else LONG_TOK
    ck = int(sys.argv[3]) if len(sys.argv) > 3 else CHUNK
    run(pt, lt, ck)
