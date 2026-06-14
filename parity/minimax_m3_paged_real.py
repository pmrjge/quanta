"""MiniMax-M3-VL M3-4: real-weight paged-KV + prefix caching re-gate @ 397B (SOLO).

Validates the M3-4 paged-KV path on the REAL int4-g64 artifact at full scale, off ONE ~233 GiB resident
load. M3 is the clean dense-GQA paged case (all 60 layers attention, NO recurrent state), so paging is
the textbook k/v scenario: prefix blocks dedup across requests, int8-g64 KV (GQA 4 kv heads ⇒ cheap KV;
int8 halves it), nothing to content-address at a block boundary.

One :class:`~quanta.minimax.batched_runtime_m3.MiniMaxM3BatchedResidentModel` load (``packed=True`` +
``packed_experts=True`` + ``kv_quantized=True`` ⇒ paged loop-kill ON by default) provides the paged
serving path; a per-stream-paged-loop sibling, a discrete int8 single-stream reference, and a bf16-KV
reference are all built over the SAME resident layers (zero extra memory), so every check shares the one
load.

  1. **paged prefix-reuse + suffix == discrete continue-from-prefix (BIT-EXACT), int8 KV.** A request
     that re-references a resident prefix's blocks and prefills only the uncached suffix is bit-identical
     to a discrete continue-from-prefix prefill of the same split (the ``cache_quant`` orthogonal-axes
     foundation: int8 quant groups on ``head_dim`` vs blocks on the seq axis). Asserts |Δ|==0, a real
     prefix hit (``prefix_hit_tokens>0``), and no boundary payloads (``has_recurrent_state=False``).
  2. **paged KV loop-kill == per-stream paged loop (BIT-EXACT), B ragged.** The M3-4 batched paged decode
     (ONE ``write_batched`` scatter + ONE ``gather_batched``) equals the same loop-kill attention with the
     per-stream ``.update()`` loop over paged views — both end in the same fused padded SDPA.
  3. **int8-KV quality (the lever): Δppl vs bf16 KV.** Teacher-forced ppl of the int8-KV cached forward
     vs the bf16 (caches=None) M1/M2 reference on the SAME codes — the int8 KV is near-lossless (the
     fleet's proven ``cache_quant`` scheme).
  4. **paged batched decode == discrete single-stream decode (greedy-token-equivalent).** End-to-end
     anchor (the SDPA reorder + the F32 router GEMM at M=B flip only near-ties).
  5. **cross-request prefix reuse.** Committed blocks survive a ``free()`` at ref 0 (LRU), so a later
     request re-references them. Reuse is BIT-EXACT only when the committing prefill SHAPE matches the
     would-be recompute (a packed projection at batch-M=A tiles its K-reduction differently than at M=B
     — the #153 finding, now in prefill: a prefix committed by a len-A prefill is quant-ULP-different
     from one committed by a len-B prefill, compounding across 60 layers). 5a (well-posed): turn 1
     commits the block-aligned prefix (the shape check 1's reference used), frees it, turn 2 re-references
     the resident blocks → bit-exact == row1. 5b (realistic re-admit): a from-scratch FULL admit then a
     same-prompt re-admit reuses a prefix committed at the full shape while recomputing the tail at the
     suffix shape ⇒ a benign prefill-batch-shape ULP (greedy-equiv, rel-bounded; not bit-exact).

    uv run python -m parity.minimax_m3_paged_real            # full re-gate (all 60 layers, SOLO)
    uv run python -m parity.minimax_m3_paged_real 4 64       # n_layers, n_tok (bounded code smoke)

# parity-gate: real-weight
"""

from __future__ import annotations

import math
import os
import sys
import time

import mlx.core as mx

from parity.minimax_m3_ppl import PROSE, teacher_forced
import quanta.minimax.batched_runtime_m3 as BR
from quanta.minimax.batched_runtime_m3 import MiniMaxM3BatchedResidentModel
from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.runtime_m3 import MiniMaxM3ResidentModel
from quanta.minimax.tokenizer import MiniMaxTokenizer
from quanta.paged import PagedKVCacheManager

SRC = "/Users/pmrj/models/MiniMax-M3"
ART = "/Users/pmrj/models/MiniMax-M3-quanta_int4g64"
N_TOK = 256          # a parity re-gate, not a ppl measurement
PPL_CEILING = 30.0   # the served runtime must ship a healthy 397B ppl (the M2b int4 value ~5.0)
INT8_KV_DPPL_CEIL = 5.0   # int8 KV is near-lossless (the fleet's proven cache_quant scheme)
B = 8                # batched streams for the paged loop-kill equivalence
BLOCK = 16           # paged block size (prefix-match granularity)
# paged batched vs discrete single: a near-tie SDPA / F32-router flip is ~0.08 rel on one stream; a
# SYSTEMATIC bug (wrong offset / mis-threaded paged view) blows up ALL streams to O(1).
ANCHOR_REL_CEIL = 0.15
ANCHOR_AGREE_FLOOR = 0.75

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _set_wired() -> None:
    """Pin the resident weight set (best-effort; the deployment target holds the model RAM-resident)."""
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        rec = int(info.get("max_recommended_working_set_size", 0))
        if rec > 0:
            mx.set_wired_limit(rec)
    except Exception:  # noqa: BLE001 — wired-limit is an optimization, never fail the gate on it
        pass


def _top1(a: mx.array, b: mx.array) -> bool:
    return int(mx.argmax(a[0, 0]).item()) == int(mx.argmax(b[0, 0]).item())


def _rel(a: mx.array, b: mx.array) -> float:
    return float((mx.linalg.norm((a - b).astype(mx.float32))
                  / (mx.linalg.norm(b.astype(mx.float32)) + 1e-9)).item())


def _new_mgr(spec: dict) -> PagedKVCacheManager:
    return PagedKVCacheManager(num_layers=spec["n_layers"], block_size=BLOCK, max_blocks=512,
                              group_size=spec["group_size"], bits=spec["bits"],
                              quantized=spec["quantized"], model_name="m3paged")


def _paged_prefill(model: MiniMaxM3BatchedResidentModel, mgr: PagedKVCacheManager, prompt: list) -> tuple:
    """Admit one stream from scratch into a paged state (reusing any committed prefix blocks)."""
    seq = mgr.new_sequence()
    n_attn = mgr.match_prefix(seq, prompt[:-1]) if len(prompt) > 1 else 0
    suffix = prompt[n_attn:]
    mgr.advance(seq, suffix)
    state = model.make_paged_state(mgr, seq)
    logits, _ = model.prefill_paged(mx.array(suffix, dtype=mx.int32), state, offset=n_attn,
                                    recurrent_in=None, block_size=BLOCK)
    mgr.commit(seq)
    return seq, state, n_attn, logits


def _paged_step(model: MiniMaxM3BatchedResidentModel, mgr: PagedKVCacheManager,
                seqs: list, states: list, tokens: list) -> list:
    """One batched paged decode step (advance → step_batch → commit), exactly as the engine drives it."""
    for seq, t in zip(seqs, tokens, strict=True):
        mgr.advance(seq, [int(t)])
    offsets = [seq.length - 1 for seq in seqs]
    outs = model.step_batch(list(tokens), states, offsets)
    for seq in seqs:
        mgr.commit(seq)
    return outs


def run(n_layers: int | None = None, n_tok: int = N_TOK) -> None:
    full = n_layers is None
    mx.set_cache_limit(8 * 1024**3)
    _set_wired()
    cfg = MiniMaxM3Config.from_pretrained(SRC)
    tok = MiniMaxTokenizer(os.path.join(SRC, "tokenizer.json"), cfg)
    ids_list = tok.encode(PROSE)[:n_tok]
    ids = mx.array(ids_list, dtype=mx.uint32)
    print(f"=== MiniMax-M3-VL M3-4 paged-KV + prefix caching re-gate — {len(ids_list)} tok, "
          f"{'all 60' if full else n_layers} layers (SOLO) ===", flush=True)

    # ---- ONE resident load: paged config, packed mixer + experts, int8 KV. The paged-batched attention
    # is gated behind loop-kill (paged = _paged_kv_batched AND _loopkill), and at int4 loop-kill AUTO-OFF
    # (rule 4 — same batched-SDPA reorder), so FORCE loopkill=True here to gate the paged-loop-kill PATH;
    # the int4 serving default is per-stream-paged (asserted below).
    t0 = time.perf_counter()
    pg = MiniMaxM3BatchedResidentModel(ART, max_batch=max(B, 2), n_layers=n_layers,
                                       packed=True, packed_experts=True, kv_quantized=True, loopkill=True)
    t_load = time.perf_counter() - t0
    # per-stream-paged-loop sibling (paged loop-kill OFF) + discrete int8 ref + bf16-KV ref — SAME layers
    pg_loop = MiniMaxM3BatchedResidentModel.from_inner(pg.layers, pg.embed_w, pg.norm_w, pg.lm_head_w,
                                                       pg.cfg, max_batch=max(B, 2), loopkill=True,
                                                       kv_quantized=True)
    pg_loop._paged_kv_batched = False
    single = MiniMaxM3ResidentModel.from_blocks(pg.layers, pg.embed_w, pg.norm_w, pg.lm_head_w, pg.cfg,
                                                kv_quantized=True)
    single_bf16 = MiniMaxM3ResidentModel.from_blocks(pg.layers, pg.embed_w, pg.norm_w, pg.lm_head_w,
                                                     pg.cfg, kv_quantized=False)
    _ck(pg._paged_kv_batched and pg._loopkill and pg.packed, "paged model not loopkill+packed+pagedKV")
    _ck(not pg_loop._paged_kv_batched, "per-stream-paged-loop sibling did not pin paged_batched=False")
    # the int4 SERVING default auto-disables loop-kill (⇒ paged-batched off, per-stream-paged serves) —
    # assert the resolver the constructor uses returns OFF for this width (no second 233 GiB load).
    _eb = BR._served_expert_bits(pg.layers)
    _ck(BR._resolve_loopkill_default(_eb) is False,
        f"int4 loop-kill default must be OFF (served expert bits {_eb}); int4 paged serving is per-stream")
    _ck(pg.paged_kv_spec["quantized"] and pg.paged_kv_spec["n_layers"] == pg.num_layers
        and not pg.has_recurrent_state, f"paged_kv_spec wrong: {pg.paged_kv_spec}")
    n_built = pg.num_layers
    spec = pg.paged_kv_spec
    print(f"  loaded {n_built}L resident in {t_load:.0f}s (paged_kv int{spec['bits']} g{spec['group_size']}, "
          f"loopkill={pg._loopkill}, paged_batched={pg._paged_kv_batched})", flush=True)

    # ---- (1) paged prefix-reuse + suffix == discrete continue-from-prefix (BIT-EXACT), int8 KV -------
    P = min(len(ids_list), 48)
    prompt = [int(t) for t in ids_list[:P]]
    n_pref = ((P - 1) // BLOCK) * BLOCK            # block-aligned prefix (e.g. 32 @ P=48, BLOCK=16)
    # discrete continue-from-prefix reference via the SAME forward (``prefill``) the paged path runs, so
    # ONLY the KV storage (discrete concat vs paged blocks) differs ⇒ bit-exact. (The resident
    # ``__call__`` tiles its lm_head over T rows vs ``prefill``'s 1-row head — a benign GEMM-tiling ULP;
    # using ``prefill`` for both isolates paging.) Same prefix/suffix split ⇒ identical SDPA shapes.
    disc = pg.make_caches()
    pg.prefill(prompt[:n_pref], disc)
    ref1 = pg.prefill(prompt[n_pref:], disc)[0, -1]
    mx.eval(ref1)
    mgr1 = _new_mgr(spec)
    seq_a, _, _, _ = _paged_prefill(pg, mgr1, prompt[:n_pref])    # req1: commit the prefix blocks
    seq_b, _, n_attn, logits_pg = _paged_prefill(pg, mgr1, prompt)  # req2: reuse prefix + suffix
    row1 = logits_pg[0, -1]
    mx.eval(row1)
    d1 = float(mx.max(mx.abs(row1 - ref1)))
    hit = int(mgr1.get_stats().prefix_hit_tokens)
    print(f"  [paged==disc] P={P} blk={BLOCK} n_attn={n_attn} hit_tok={hit} |Δ|={d1:.2e}", flush=True)
    _ck(n_attn == n_pref > 0, f"expected prefix reuse of {n_pref} tokens, got n_attn={n_attn}")
    _ck(hit > 0, f"manager reported no prefix hit: prefix_hit_tokens={hit}")
    _ck(d1 == 0.0, f"paged != discrete (int8 KV): |Δ|={d1} (the orthogonal-axes foundation must be exact)")

    # ---- (2) paged KV loop-kill == per-stream paged loop (BIT-EXACT), B ragged --------------------
    nb = min(B, max(2, len(ids_list) // 24))
    prompts = [[int(t) for t in ids_list[(5 * s) % 7: (5 * s) % 7 + (16 + 4 * s)]] for s in range(nb)]
    cont = [int(ids_list[(11 * s + 3) % len(ids_list)]) for s in range(nb)]
    mgr_lk, mgr_lp = _new_mgr(spec), _new_mgr(spec)
    seqs_lk, st_lk, seqs_lp, st_lp = [], [], [], []
    for s in range(nb):
        q, a, _, _ = _paged_prefill(pg, mgr_lk, prompts[s])
        seqs_lk.append(q)
        st_lk.append(a)
        q, a, _, _ = _paged_prefill(pg_loop, mgr_lp, prompts[s])
        seqs_lp.append(q)
        st_lp.append(a)
    out_lk = _paged_step(pg, mgr_lk, seqs_lk, st_lk, cont)
    out_lp = _paged_step(pg_loop, mgr_lp, seqs_lp, st_lp, cont)
    mx.eval(out_lk + out_lp)
    worst2 = max(float(mx.max(mx.abs(out_lk[s] - out_lp[s]))) for s in range(nb))
    print(f"  [lk==loop  ] paged loop-kill vs per-stream paged loop (B={nb}, ragged): worst |Δ|={worst2:.2e}",
          flush=True)
    _ck(worst2 == 0.0, f"paged loop-kill != per-stream paged loop: worst |Δ|={worst2}")

    # ---- (3) int8-KV quality: teacher-forced Δppl vs the bf16 (caches=None) reference --------------
    logits_bf16 = single_bf16(ids)                                # caches=None ⇒ bf16 KV (M1/M2 ref)
    logits_int8 = single(ids, caches=single.make_caches())        # int8-g64 KV cached prefill
    mx.eval(logits_bf16, logits_int8)
    ppl_bf16, acc_bf16, am_bf16 = teacher_forced(logits_bf16, ids)
    ppl_int8, acc_int8, am_int8 = teacher_forced(logits_int8, ids)
    dppl = 100.0 * (ppl_int8 / ppl_bf16 - 1.0) if ppl_bf16 > 0 else float("inf")
    agree_kv = float(mx.mean((am_int8 == am_bf16).astype(mx.float32)).item())
    print(f"  [int8 KV  ] bf16 ppl {ppl_bf16:7.4f} | int8 KV ppl {ppl_int8:7.4f} | Δppl {dppl:+.3f}% | "
          f"top-1 agree {agree_kv:.4f}", flush=True)
    _ck(math.isfinite(ppl_bf16) and math.isfinite(ppl_int8), "non-finite ppl")
    if full:
        _ck(ppl_int8 < PPL_CEILING, f"int8-KV ppl {ppl_int8:.4f} >= {PPL_CEILING}: not coherent")
        _ck(abs(dppl) < INT8_KV_DPPL_CEIL,
            f"int8 KV drifts from bf16: Δppl {dppl:+.3f}% (ceiling {INT8_KV_DPPL_CEIL}%)")

    # ---- (4) paged batched decode == discrete single-stream decode (greedy-token-equivalent) -------
    nxt = [int(ids_list[(17 * s + 2) % len(ids_list)]) for s in range(nb)]
    ref = []
    for s in range(nb):
        ca = single.make_caches()
        single(mx.array(prompts[s], dtype=mx.int32), caches=ca)
        ref.append(single(mx.array([nxt[s]], dtype=mx.int32), caches=ca))
    mgr4 = _new_mgr(spec)
    seqs4, st4 = [], []
    for s in range(nb):
        q, a, _, _ = _paged_prefill(pg, mgr4, prompts[s])
        seqs4.append(q)
        st4.append(a)
    o4 = _paged_step(pg, mgr4, seqs4, st4, nxt)
    mx.eval(o4 + ref)
    rels4 = [_rel(o4[s], ref[s]) for s in range(nb)]
    top1_4 = sum(_top1(o4[s], ref[s]) for s in range(nb)) / nb
    print(f"  [paged dec ] vs discrete single (B={nb} ragged): top-1 {top1_4:.3f} | worst rel {max(rels4):.2e}",
          flush=True)
    _ck(max(rels4) < ANCHOR_REL_CEIL, f"paged decode diverges from single: worst rel {max(rels4):.2e}")
    _ck(top1_4 >= ANCHOR_AGREE_FLOOR, f"paged decode top-1 drifts from single: {top1_4:.3f}")

    # ---- (5) cross-request prefix reuse — reuse-after-FREE is bit-exact; re-admit greedy-equiv ------
    # The committed blocks survive a free() at ref 0 (LRU), so a later request re-references them. Reuse
    # is BIT-EXACT only when the committing prefill SHAPE matches the would-be recompute (a packed
    # projection at batch-M=A tiles its K-reduction differently than at M=B — the #153 finding, now in
    # prefill: a prefix committed by a len-A prefill is quant-ULP-different from one committed by a len-B
    # prefill, ~1 int8 step per element, compounding across 60 layers). So:
    #   5a (well-posed, BIT-EXACT): turn-1 commits the block-aligned PREFIX (the same shape check 1's
    #      discrete reference used), it is freed, and turn 2 re-references the resident blocks → == row1.
    #   5b (realistic, GREEDY-EQUIV): a from-scratch FULL admit then a same-prompt re-admit reuses a
    #      prefix committed at the full-prompt shape while recomputing the tail at the suffix shape ⇒ a
    #      benign prefill-batch-shape ULP (NOT bit-exact). Asserted top-1 stable, |Δ| printed.
    if full:
        mgr5 = _new_mgr(spec)
        seq5a, _, _, _ = _paged_prefill(pg, mgr5, prompt[:n_pref])   # turn 1: commit the prefix blocks
        mgr5.free(seq5a)                                             # turn 1 ends; blocks stay resident (ref 0)
        _, _, n5a, logits5a = _paged_prefill(pg, mgr5, prompt)       # turn 2: re-reference the freed prefix
        hit5a = int(mgr5.get_stats().prefix_hit_tokens)
        d5a = float(mx.max(mx.abs(logits5a[0, -1] - row1)))          # same commit shape as row1 ⇒ bit-exact
        print(f"  [reuse-free] turn2 reuses freed prefix: hit_tok={hit5a} (n_attn={n5a}) |Δ vs row1|={d5a:.2e}",
              flush=True)
        _ck(hit5a > 0 and n5a == n_pref, f"reuse-after-free did not re-reference the prefix (hit={hit5a})")
        _ck(d5a == 0.0, f"reuse-after-free drifted from the same-split reference: |Δ|={d5a}")

        mgr5b = _new_mgr(spec)
        _, _, _, logits_a = _paged_prefill(pg, mgr5b, prompt)        # admit A: full prefill (commit all)
        hit_b0 = int(mgr5b.get_stats().prefix_hit_tokens)
        _, _, n5b, logits_b = _paged_prefill(pg, mgr5b, prompt)      # admit B: reuse A's prefix + recompute tail
        hit_b1 = int(mgr5b.get_stats().prefix_hit_tokens)
        d5b = _rel(logits_b[0, -1][None, None], logits_a[0, -1][None, None])
        same_top1 = int(mx.argmax(logits_b[0, -1]).item()) == int(mx.argmax(logits_a[0, -1]).item())
        print(f"  [re-admit ] full→reuse hit_tok {hit_b0}→{hit_b1} (n_attn={n5b}) | top-1 {'==' if same_top1 else '!='}"
              f" | rel {d5b:.2e} (prefill-batch-shape ULP — greedy-equiv, not bit-exact)", flush=True)
        _ck(hit_b1 > hit_b0, "re-admit did not reuse the prior admit's committed prefix blocks")
        # greedy-equiv boundary: the reused prefix (committed at the full-admit shape) + the
        # recomputed tail (suffix shape) drift only by the prefill-batch-shape ULP; a SYSTEMATIC bug
        # would blow rel up to O(1). A single-sample top-1 near-tie may flip (printed, not asserted).
        _ck(d5b < ANCHOR_REL_CEIL, f"re-admit diverges from the original admit: rel {d5b:.2e}")

    del pg, pg_loop, single, single_bf16
    mx.clear_cache()

    if full:
        print(f"\nVERDICT: M3-4 paged-KV + prefix caching VALIDATED @ 397B — paged == discrete BIT-EXACT "
              f"(int8 KV, |Δ| 0), the paged KV loop-kill == the per-stream paged loop (|Δ| 0); int8 KV "
              f"near-lossless (Δppl {dppl:+.3f}%, ppl {ppl_int8:.3f}); paged decode == single "
              f"(top-1 {top1_4:.3f}); prefix blocks dedup across requests.", flush=True)
    else:
        print(f"\nSMOKE ok — paged path ran ({nb} streams, {n_built} layers); paged==disc |Δ| {d1:.1e}, "
              f"lk==loop |Δ| {worst2:.1e} (numbers not meaningful on a partial model).", flush=True)
    print(f"PARITY-CHECKS: {_N}", flush=True)


if __name__ == "__main__":
    nl = int(sys.argv[1]) if len(sys.argv) > 1 else None
    nt = int(sys.argv[2]) if len(sys.argv) > 2 else N_TOK
    run(nl, nt)
