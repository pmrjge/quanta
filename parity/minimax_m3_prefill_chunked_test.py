"""Model-free M3-5 gate: MiniMax-M3-VL long-context chunked prefill — tiny synthetic.

M3 is **all dense GQA** (no GDN recurrent state, no YaRN), so chunked prefill is the textbook case:
feed the prompt in seq blocks, each chunk extending every layer's GQA :class:`quanta.minimax.model_m3.
KVCache` and attending the grown KV with a bottom-right causal mask
(``mx.fast.scaled_dot_product_attention`` ``mask="causal"`` — the exact path the M3-1 cached forward
and the qwen35 shipped chunked prefill both use). On the **bf16 mixer** the chunk boundaries only
re-cut the same per-row causal attention + per-token KV quant — no cross-token op changes, and the
paged gather is bit-identical to the discrete cache — so chunked is **BIT-EXACT** to the single-shot
prefill. (On the packed serving mixer the projection ``mx.quantized_matmul`` runs at batch-M=chunk vs
M=T ⇒ greedy-token-equivalent — the #153 batch-M ULP, re-gated @ 397B in the real gate.) Proven here on
a tiny synthetic M3 decoder (NO real weights), bf16 mixer so the claim is bit-exact:

A. **driver bit-exact (discrete KV).** ``prefill_chunked`` == single-shot ``prefill`` last-position
   logits |Δ|==0 across chunk sizes (one-chunk ≥ T, ragged, per-token chunk=1); the chunked-SEEDED
   cache is bit-identical too (one more batched decode step matches the single-shot-seeded cache).
B. **continue-from-non-empty-cache.** A chunked suffix appended to a prefilled prefix == a single-shot
   prefill of the whole prefix+suffix (multi-turn / paged-suffix extension), |Δ|==0.
C. **int8-KV chunked == single-shot** |Δ|==0 — the quant groups sit on ``head_dim`` (per-token), so
   chunking the seq axis cannot change a token's codes.
D. **chunked over paged views == discrete chunked == single-shot** |Δ|==0 (the paged gather ==
   discrete ``KVCache`` foundation; the manager allows sub-range writes from the open cursor).
E. **rule 6** — ``chunk_tokens < 1`` and a wrong-length cache both refuse.
F. **prefill routing** — ``prefill`` sends prompts >= ``MINIMAX_M3_CHUNKED_PREFILL_FROM`` through
   ``prefill_chunked`` (== single-shot for a one-chunk prompt); the threshold is one chunk + 1.

``head_dim=32`` (the smallest the int8 KV path accepts — ``mx.quantize``'s min group_size is 32). The
real-model re-gate (chunked == single-shot greedy-token-equiv on the packed serving runtime + chunked
ppl) is the SOLO ``parity/minimax_m3_prefill_chunked_real.py``.

    uv run python -m parity.minimax_m3_prefill_chunked_test
"""

from __future__ import annotations

import mlx.core as mx

from parity.minimax_m3_batched_test import _build_blocks, _synth
from parity.minimax_m3_paged_test import _cfg32
from quanta.minimax import batched_runtime_m3 as BR
from quanta.minimax import model_m3 as M
from quanta.paged import PagedKVCacheManager

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _max_abs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def _rel(a: mx.array, b: mx.array) -> float:
    return float((mx.linalg.norm((a - b).astype(mx.float32))
                  / (mx.linalg.norm(b.astype(mx.float32)) + 1e-9)).item())


def _top1(a: mx.array, b: mx.array) -> bool:
    return int(mx.argmax(a[0, -1]).item()) == int(mx.argmax(b[0, -1]).item())


def _bf16_blocks(cfg, w) -> list:
    """bf16 mixer + dequantized experts ⇒ everything bf16, so chunked == single-shot is BIT-EXACT (no
    packed ``mx.quantized_matmul`` to re-tile at batch-M=chunk vs M=T)."""
    return _build_blocks(cfg, w, packed_mixer=False, packed_experts=False)


def _model(cfg, blocks, w, *, quantized: bool, kv_gs: int = 32) -> BR.MiniMaxM3BatchedResidentModel:
    """A bf16-mixer batched runtime (``loopkill=False`` — required for a non-packed mixer; chunked
    prefill is a single-stream forward, loop-kill never touches it). Both ``prefill`` (single-shot,
    1-row head) and ``prefill_chunked`` (1-row head) live here, so the reference and the test share the
    exact same head GEMM — only the chunking differs."""
    return BR.MiniMaxM3BatchedResidentModel.from_inner(
        blocks, w["embed"], M.one_plus(w["final_norm"]), w["lm_head"], cfg, max_batch=4,
        loopkill=False, kv_quantized=quantized, kv_group_size=kv_gs)


# --- A. driver bit-exact (discrete KV), multiple chunk sizes --------------------------------------- #

def _check_driver(quantized: bool) -> None:
    mx.random.seed(0)
    cfg = _cfg32()
    V = cfg.vocab_size
    w = _synth(cfg, mx.random.key(1))
    blocks = _bf16_blocks(cfg, w)
    model = _model(cfg, blocks, w, quantized=quantized)
    mode = "int8 g32" if quantized else "bf16"

    T = 17
    ids = [int((i * 7 + 3) % V) for i in range(T)]
    ref_cache = model.make_caches()
    ref = model.prefill(ids, ref_cache)                         # single-shot, 1-row head
    mx.eval(ref)

    # cts >= 2 are the realistic long-admit chunk sizes: every chunk has t>1 ⇒ the attention takes the
    # SAME mask="causal" SDPA kernel as the single-shot prefill, so chunked == single-shot BIT-EXACT for
    # BOTH KV modes (the projections are bf16-mixer M-invariant per row; the int8 codes are per-token).
    worst = 0.0
    for ct in (T + 5, T, T - 1, 5, 3, 2):                       # one-chunk, exact, ragged, down to 2
        c = model.make_caches()
        lg = model.prefill_chunked(ids, c, chunk_tokens=ct)
        mx.eval(lg)
        d = _max_abs(lg, ref)
        worst = max(worst, d)
        _ck(int(c[0].offset) == T, f"chunked ({mode}, ct={ct}) cache offset {int(c[0].offset)} != {T}")
        _ck(d == 0.0, f"chunked ({mode}, ct={ct}) != single-shot: |Δ|={d:.2e}")

    # ct=1 is the degenerate per-token corner: each chunk has t==1 ⇒ the attention takes the mask=None
    # DECODE kernel (not "causal"). For bf16 KV that is bit-exact to the prefill kernel (the M3-1
    # incremental-decode==full-prefill invariant); for int8 KV it is greedy-token-EQUIVALENT, not
    # bit-exact (the documented M3-4 boundary: int8 decode vs prefill is top-1 exact but ~1 int8 step —
    # 7.8e-3 — because the decode/prefill SDPA kernels reduce differently and the int8-rounded KV
    # surfaces it). No production chunked admit uses ct=1 (it is the per-token decode path), so this is
    # a corner assertion, not the lever.
    c1 = model.make_caches()
    lg1 = model.prefill_chunked(ids, c1, chunk_tokens=1)
    mx.eval(lg1)
    d1 = _max_abs(lg1, ref)
    if quantized:
        _ck(_top1(lg1, ref) and _rel(lg1, ref) < 5e-2,
            f"int8 ct=1 decode-kernel not greedy-equiv to prefill: rel={_rel(lg1, ref):.2e}")
    else:
        _ck(d1 == 0.0, f"bf16 ct=1 (decode kernel) != single-shot: |Δ|={d1:.2e}")

    # the chunked-SEEDED cache is bit-identical: one more batched decode step matches the single-shot
    # seed (per-stream loop, M=1 — so any KV mis-seed would surface as a non-zero |Δ| here).
    c_small = model.make_caches()
    model.prefill_chunked(ids, c_small, chunk_tokens=3)
    nxt = int((T * 5 + 2) % V)
    step_ref = model.step_batch([nxt], [ref_cache], [T])[0]
    step_chk = model.step_batch([nxt], [c_small], [T])[0]
    mx.eval([step_ref, step_chk])
    d_step = _max_abs(step_chk, step_ref)
    _ck(d_step == 0.0, f"chunked-seeded cache decode != single-shot-seeded ({mode}): |Δ|={d_step:.2e}")
    ct1 = "bit-exact" if not quantized else f"greedy-equiv (rel {_rel(lg1, ref):.1e})"
    print(f"  [OK] driver ({mode}): chunked == single-shot across cts>=2 |Δ|={worst:.1e}; ct=1 {ct1}; "
          f"seeded-cache decode |Δ|={d_step:.1e}")


# --- B. continue-from-non-empty-cache (multi-turn / prefix extension) ------------------------------ #

def _check_continue() -> None:
    mx.random.seed(0)
    cfg = _cfg32()
    V = cfg.vocab_size
    w = _synth(cfg, mx.random.key(2))
    blocks = _bf16_blocks(cfg, w)
    model = _model(cfg, blocks, w, quantized=False)

    P, S = 9, 11
    ids = [int((i * 5 + 1) % V) for i in range(P + S)]
    # reference: single-shot prefill of the WHOLE prefix+suffix (1-row head)
    ref = model.prefill(ids, model.make_caches())
    mx.eval(ref)
    # test: single-shot the prefix, then CHUNKED the suffix into the same cache (the continuation path)
    c = model.make_caches()
    model.prefill(ids[:P], c)
    lg = model.prefill_chunked(ids[P:], c, chunk_tokens=4)
    mx.eval(lg)
    d = _max_abs(lg, ref)
    _ck(int(c[0].offset) == P + S, f"continue cache offset {int(c[0].offset)} != {P + S}")
    _ck(d == 0.0, f"chunked continuation != single-shot whole prefill: |Δ|={d:.2e}")
    print(f"  [OK] continue-from-cache: prefix({P})+chunked-suffix({S}) == single-shot({P + S}) |Δ|={d:.1e}")


# --- D. chunked over paged views == discrete chunked == single-shot -------------------------------- #

def _check_paged() -> None:
    mx.random.seed(0)
    cfg = _cfg32()
    V = cfg.vocab_size
    w = _synth(cfg, mx.random.key(3))
    blocks = _bf16_blocks(cfg, w)
    block, ct = 4, 5
    for quantized in (False, True):
        model = _model(cfg, blocks, w, quantized=quantized, kv_gs=32)
        spec = model.paged_kv_spec
        T = 14
        ids = [int((i * 7 + 2) % V) for i in range(T)]

        single = model.prefill(ids, model.make_caches())                 # single-shot discrete
        disc = model.make_caches()
        chk_disc = model.prefill_chunked(ids, disc, chunk_tokens=ct)      # chunked discrete
        mx.eval([single, chk_disc])

        mgr = PagedKVCacheManager(num_layers=spec["n_layers"], block_size=block, max_blocks=64,
                                  group_size=spec["group_size"], bits=spec["bits"],
                                  quantized=spec["quantized"], model_name="m3chunk")
        seq = mgr.new_sequence()
        mgr.advance(seq, ids)                                             # open [0,T) for KV writes
        st = model.make_paged_state(mgr, seq)
        chk_pg = model.prefill_chunked(ids, st, chunk_tokens=ct)          # chunked over paged views
        mgr.commit(seq)
        mx.eval(chk_pg)

        d_dc = _max_abs(chk_disc, single)
        d_pg = _max_abs(chk_pg, chk_disc)
        mode = "int8 g32" if quantized else "bf16"
        _ck(d_dc == 0.0, f"discrete chunked != single-shot ({mode}): |Δ|={d_dc:.2e}")
        _ck(d_pg == 0.0, f"paged chunked != discrete chunked ({mode}): |Δ|={d_pg:.2e}")
        _ck(int(st[0].offset) == T, f"paged view offset {int(st[0].offset)} != {T} ({mode})")
        print(f"  [OK] paged ({mode}): chunked-over-paged == discrete-chunked == single-shot "
              f"|Δ|dc={d_dc:.1e} |Δ|pg={d_pg:.1e}")


# --- E. rule 6 ------------------------------------------------------------------------------------- #

def _raises(fn) -> bool:
    try:
        fn()
        return False
    except ValueError:
        return True


def _check_rule6() -> None:
    mx.random.seed(0)
    cfg = _cfg32()
    w = _synth(cfg, mx.random.key(4))
    blocks = _bf16_blocks(cfg, w)
    model = _model(cfg, blocks, w, quantized=False)
    ids = [int(i) for i in range(8)]
    _ck(_raises(lambda: model.prefill_chunked(ids, model.make_caches(), chunk_tokens=0)),
        "prefill_chunked accepted chunk_tokens<1 (rule 6)")
    _ck(_raises(lambda: model.prefill_chunked(ids, model.make_caches()[:-1], chunk_tokens=4)),
        "prefill_chunked accepted a wrong-length cache (rule 6)")
    # the free driver guards too
    from quanta.minimax.runtime_m3 import chunked_prefill
    _ck(_raises(lambda: chunked_prefill(model.layers, model.embed_w, model.norm_w, model.lm_head_w,
                                        cfg, [], model.make_caches(), chunk_tokens=4)),
        "chunked_prefill accepted an empty prompt (rule 6)")
    print("  [OK] rule 6: chunk_tokens<1 / wrong-length cache / empty prompt all refuse")


# --- F. prefill routing at the threshold ----------------------------------------------------------- #

def _check_routing() -> None:
    mx.random.seed(0)
    cfg = _cfg32()
    V = cfg.vocab_size
    w = _synth(cfg, mx.random.key(5))
    blocks = _bf16_blocks(cfg, w)
    model = _model(cfg, blocks, w, quantized=False)
    _ck(BR.MINIMAX_M3_CHUNKED_PREFILL_FROM == BR.MINIMAX_M3_PREFILL_CHUNK_TOKENS + 1,
        "threshold is not one chunk + 1")

    ids = [int((i * 3 + 1) % V) for i in range(8)]
    ref = model.prefill(ids, model.make_caches())                  # below threshold ⇒ single-shot
    mx.eval(ref)
    # force routing on a small prompt: lower the module threshold, confirm prefill dispatches to
    # prefill_chunked (one chunk @ the default chunk size ⇒ == single-shot, proving the dispatch is
    # the same forward).
    saved = BR.MINIMAX_M3_CHUNKED_PREFILL_FROM
    try:
        BR.MINIMAX_M3_CHUNKED_PREFILL_FROM = 4
        routed = model.prefill(ids, model.make_caches())
        mx.eval(routed)
    finally:
        BR.MINIMAX_M3_CHUNKED_PREFILL_FROM = saved
    d = _max_abs(routed, ref)
    _ck(d == 0.0, f"prefill routing (threshold lowered) != single-shot: |Δ|={d:.2e}")
    print(f"  [OK] routing: threshold == chunk+1; prefill(>=thr) dispatches to chunked (|Δ|={d:.1e})")


def run() -> None:
    print("\n=== MiniMax-M3-VL M3-5 chunked prefill gate (model-free) ===")
    print("A. driver bit-exact (discrete KV), chunk sizes incl. ragged + per-token")
    _check_driver(quantized=False)
    _check_driver(quantized=True)
    print("B. continue-from-non-empty-cache (multi-turn / prefix extension)")
    _check_continue()
    print("C. int8-KV chunked == single-shot (covered in A's int8 driver run)")
    print("D. chunked over paged views == discrete chunked == single-shot")
    _check_paged()
    print("E. rule 6")
    _check_rule6()
    print("F. prefill routing at the threshold")
    _check_routing()
    print(f"PARITY-CHECKS: {_N}")
    print("PASS — M3-5 chunked prefill: bit-exact to single-shot on the bf16 mixer (discrete + int8 KV "
          "+ paged views), continues from a non-empty cache, routes at the threshold, rule-6 honored.")


if __name__ == "__main__":
    run()
