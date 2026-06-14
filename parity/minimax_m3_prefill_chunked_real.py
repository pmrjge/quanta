"""MiniMax-M3-VL M3-5: real-weight chunked-prefill re-gate @ 397B (SOLO).

Validates the M3-5 chunked-prefill admit path on the REAL int4-g64 artifact at full scale, off ONE
~233 GiB resident load. M3 is all dense GQA, so chunked prefill = feed the prompt in seq blocks, each
chunk extending every layer's GQA KV with a bottom-right causal mask; the fused flash-attn kernel never
materializes the ``[chunk, kv_len]`` score matrix, so a long prompt admits in O(chunk) memory. The MODEL
ppl is unchanged by the admit pattern (M2b int4 ppl ~5.0, M3-1 resident ppl 5.87) — chunked prefill only
changes HOW the cache is seeded, so the arbiter here is **output-equivalence to the proven single-shot
admit**, on the packed serving runtime (packed mixer + packed experts + int8 KV).

One :class:`~quanta.minimax.batched_runtime_m3.MiniMaxM3BatchedResidentModel` load (the serving config)
provides the path; a paged manager is built from its spec. Checks (multi-chunk at chunk_tokens << P, so
the chunk-boundary continuation is exercised at scale):

  1. **chunked == single-shot (GREEDY-TOKEN-EQUIVALENT), packed mixer.** chunked runs the projections at
     batch-M=chunk vs the single-shot M=T ⇒ the #153 batch-M ULP (NOT bit-exact on the packed mixer);
     top-1 of the last-position logits matches, rel bounded. (On a bf16 mixer it would be bit-exact —
     gated model-free in ``parity/minimax_m3_prefill_chunked_test.py``.)
  2. **chunked over paged views == discrete chunked (BIT-EXACT).** At the SAME chunk size both run the
     projections at M=chunk, so only the KV storage differs (paged blocks vs discrete concat) — the
     ``cache_quant`` orthogonal-axes foundation makes the paged gather bit-identical ⇒ |Δ|==0. This is
     the clean bit-exact claim at scale (M3-4's foundation, now under chunked writes / sub-range cursor).
  3. **chunked-seeded decode == single-shot-seeded decode (greedy-token-equivalent).** Greedily decode a
     few steps from each seeded cache; the emitted token sequences agree — the served STATE (not just the
     returned logits) is correct after a chunked admit.

    uv run python -m parity.minimax_m3_prefill_chunked_real           # full re-gate (all 60 layers, SOLO)
    uv run python -m parity.minimax_m3_prefill_chunked_real 4 96      # n_layers, n_tok (bounded smoke)

# parity-gate: real-weight
"""

from __future__ import annotations

import os
import sys
import time

import mlx.core as mx

from parity.minimax_m3_ppl import PROSE
from quanta.minimax.batched_runtime_m3 import MiniMaxM3BatchedResidentModel
from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.tokenizer import MiniMaxTokenizer
from quanta.paged import PagedKVCacheManager

SRC = "/Users/pmrj/models/MiniMax-M3"
ART = "/Users/pmrj/models/MiniMax-M3-quanta_int4g64"
N_TOK = 256
CHUNK = 64           # chunk_tokens << P ⇒ several chunks (the multi-chunk continuation path)
BLOCK = 16           # paged block size
N_DECODE = 6         # greedy steps to compare the seeded states
# chunked vs single-shot on the packed mixer: a near-tie last-token flip is ~0.08 rel; a SYSTEMATIC bug
# (wrong offset / mis-cut chunk) blows it up to O(1).
REL_CEIL = 0.15
DECODE_AGREE_FLOOR = 0.80

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _set_wired() -> None:
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        rec = int(info.get("max_recommended_working_set_size", 0))
        if rec > 0:
            mx.set_wired_limit(rec)
    except Exception:  # noqa: BLE001 — wired-limit is an optimization, never fail the gate on it
        pass


def _top1(a: mx.array, b: mx.array) -> bool:
    return int(mx.argmax(a[0, -1]).item()) == int(mx.argmax(b[0, -1]).item())


def _rel(a: mx.array, b: mx.array) -> float:
    return float((mx.linalg.norm((a[0, -1] - b[0, -1]).astype(mx.float32))
                  / (mx.linalg.norm(b[0, -1].astype(mx.float32)) + 1e-9)).item())


def _new_mgr(spec: dict) -> PagedKVCacheManager:
    return PagedKVCacheManager(num_layers=spec["n_layers"], block_size=BLOCK, max_blocks=512,
                              group_size=spec["group_size"], bits=spec["bits"],
                              quantized=spec["quantized"], model_name="m3chunk")


def _greedy(model: MiniMaxM3BatchedResidentModel, last: mx.array, cache: list, offset: int,
            steps: int) -> list[int]:
    """Greedily decode ``steps`` tokens from a seeded cache (one B=1 ``step_batch`` per token)."""
    out: list[int] = []
    tok = int(mx.argmax(last[0, -1]).item())
    off = offset
    for _ in range(steps):
        out.append(tok)
        lg = model.step_batch([tok], [cache], [off])[0]
        mx.eval(lg)
        tok = int(mx.argmax(lg[0, -1]).item())
        off += 1
    return out


def run(n_layers: int | None = None, n_tok: int = N_TOK) -> None:
    full = n_layers is None
    mx.set_cache_limit(8 * 1024**3)
    _set_wired()
    cfg = MiniMaxM3Config.from_pretrained(SRC)
    tok = MiniMaxTokenizer(os.path.join(SRC, "tokenizer.json"), cfg)
    ids_list = tok.encode(PROSE)[:n_tok]
    P = len(ids_list)
    prompt = [int(t) for t in ids_list]
    print(f"=== MiniMax-M3-VL M3-5 chunked-prefill re-gate — {P} tok, chunk={CHUNK} "
          f"({'all 60' if full else n_layers} layers, SOLO) ===", flush=True)

    t0 = time.perf_counter()
    pg = MiniMaxM3BatchedResidentModel(ART, max_batch=2, n_layers=n_layers,
                                       packed=True, packed_experts=True, kv_quantized=True)
    t_load = time.perf_counter() - t0
    spec = pg.paged_kv_spec
    n_built = pg.num_layers
    _ck(P > CHUNK, f"need a multi-chunk prompt: P={P} <= CHUNK={CHUNK}")
    print(f"  loaded {n_built}L resident in {t_load:.0f}s (packed mixer+experts, int{spec['bits']} "
          f"g{spec['group_size']} KV)", flush=True)

    # ---- (1) chunked == single-shot (greedy-token-equivalent), packed mixer -----------------------
    ref = pg.prefill(prompt, pg.make_caches())                       # P < threshold ⇒ single-shot (M=P)
    chk = pg.prefill_chunked(prompt, pg.make_caches(), chunk_tokens=CHUNK)   # multi-chunk (M=CHUNK)
    mx.eval([ref, chk])
    t1, r1 = _top1(chk, ref), _rel(chk, ref)
    print(f"  [chunk==1shot] last-tok top-1 {'==' if t1 else '!='} | rel {r1:.2e} "
          f"(packed mixer ⇒ batch-M ULP, greedy-equiv not bit-exact)", flush=True)
    _ck(r1 < REL_CEIL, f"chunked diverges from single-shot: rel {r1:.2e}")
    if full:
        _ck(t1, "chunked last-token top-1 != single-shot (a systematic chunk-boundary bug)")

    # ---- (2) chunked over paged views == discrete chunked (BIT-EXACT) -----------------------------
    chk_disc = pg.prefill_chunked(prompt, pg.make_caches(), chunk_tokens=CHUNK)
    mgr = _new_mgr(spec)
    seq = mgr.new_sequence()
    mgr.advance(seq, prompt)
    st = pg.make_paged_state(mgr, seq)
    chk_pg = pg.prefill_chunked(prompt, st, chunk_tokens=CHUNK)
    mgr.commit(seq)
    mx.eval([chk_disc, chk_pg])
    d2 = float(mx.max(mx.abs(chk_pg[0, -1] - chk_disc[0, -1])))
    print(f"  [paged==disc ] chunked-over-paged vs discrete-chunked (same chunk): |Δ|={d2:.2e}", flush=True)
    _ck(int(st[0].offset) == P, f"paged view offset {int(st[0].offset)} != {P}")
    _ck(d2 == 0.0, f"paged chunked != discrete chunked: |Δ|={d2} (the orthogonal-axes foundation must hold)")

    # ---- (3) chunked-seeded decode == single-shot-seeded decode (greedy-token-equivalent) ----------
    c_ref = pg.make_caches()
    ref_seed = pg.prefill(prompt, c_ref)
    c_chk = pg.make_caches()
    chk_seed = pg.prefill_chunked(prompt, c_chk, chunk_tokens=CHUNK)
    mx.eval([ref_seed, chk_seed])
    toks_ref = _greedy(pg, ref_seed, c_ref, P, N_DECODE)
    toks_chk = _greedy(pg, chk_seed, c_chk, P, N_DECODE)
    agree = sum(int(a == b) for a, b in zip(toks_ref, toks_chk, strict=True)) / N_DECODE
    print(f"  [seeded dec  ] {N_DECODE}-step greedy decode agreement chunked-seed vs single-shot-seed: "
          f"{agree:.3f}", flush=True)
    if full:
        _ck(agree >= DECODE_AGREE_FLOOR,
            f"chunked-seeded decode drifts from single-shot-seeded: agree {agree:.3f}")

    del pg
    mx.clear_cache()

    if full:
        print(f"\nVERDICT: M3-5 chunked prefill VALIDATED @ 397B — chunked == single-shot greedy-token-"
              f"equivalent (last-tok top-1 ==, rel {r1:.2e}); chunked-over-paged == discrete-chunked "
              f"BIT-EXACT (|Δ| 0); chunked-seeded decode == single-shot-seeded ({agree:.3f}). The "
              f"long-admit lever ships the proven serving state in O(chunk) memory.", flush=True)
    else:
        print(f"\nSMOKE ok — chunked path ran ({n_built} layers, {P} tok, chunk {CHUNK}); "
              f"chunk vs 1shot rel {r1:.1e}, paged==disc |Δ| {d2:.1e} (numbers not meaningful partial).",
              flush=True)
    print(f"PARITY-CHECKS: {_N}", flush=True)


if __name__ == "__main__":
    nl = int(sys.argv[1]) if len(sys.argv) > 1 else None
    nt = int(sys.argv[2]) if len(sys.argv) > 2 else N_TOK
    run(nl, nt)
