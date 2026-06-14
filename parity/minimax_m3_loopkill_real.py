"""MiniMax-M3-VL M3-3: real-weight GQA loop-kill re-gate @ 397B (SOLO).

Validates the M3-3 GQA loop-kill on the REAL int4-g64 artifact at full scale, off ONE ~233 GiB
resident load. The loop-kill replaces M3-2's per-stream decode-attention loop with ONE batched
attention across all ``B`` streams (batched chunked q/k/v/o projections + per-stream RoPE + the shared
fused padded SDPA).

**At int4 the loop-kill is NOT the serving default** — :func:`_resolve_loopkill_default` auto-disables
it for int4 because the coarse int4 MoE amplifies the fused padded-SDPA reduction-order ULP past
bit-exactness (~0.875 token-agree @ B=8 vs int6's BIT-EXACT — the user's rule-4 call: fall back to the
proven per-stream loop). So this gate **force-enables** loop-kill (``loopkill=True``) to validate the
PATH is still the bounded SDPA-reorder regime (not a systematic bug), and separately asserts the int4
auto-default is OFF. The per-stream loop is the int4 serving attention.

One :class:`~quanta.minimax.batched_runtime_m3.MiniMaxM3BatchedResidentModel` load (``packed=True`` +
``packed_experts=True`` + explicit ``loopkill=True``) provides the loop-kill path; a per-stream-loop
sibling and a single-stream reference are built over the SAME resident layers (zero extra memory) via
``from_inner(..., loopkill=False)`` and :meth:`MiniMaxM3ResidentModel.from_blocks`, so all checks share
the one load.

  1. **loop-kill == per-stream loop, multi-step decode @ B (the M3-3 equivalence).** ``B`` streams
     prefilled to DIFFERENT lengths (ragged offsets), then ``N`` teacher-forced decode steps fed the
     SAME token to BOTH paths each step (lockstep, so the comparison never desyncs). Both paths run the
     IDENTICAL batched MoE sub-block (the router GEMM at M=B is shared), so this isolates ONLY the
     attention: the chunked projections are bit-exact (§M0) and the per-stream RoPE is bit-identical, so
     the lone divergence is the fused padded-SDPA softmax reduction-order ULP — greedy-token-equivalent
     (a rare near-tie may flip one (stream,step); a systematic bug would blow up ALL of them). The gate
     asserts high per-(stream,step) top-1 agreement + a bounded worst logit rel.
  2. **loop-kill == single-stream decode, ragged @ B (end-to-end anchor).** One batched loop-kill step
     vs each stream's single-stream decode at the same offset — the same greedy-token-equivalent claim
     the M3-2 re-gate makes for the per-stream path, now for the loop-kill.
  3. **ppl sanity.** The resident prefill ships a healthy 397B teacher-forced ppl (~5.0, the M2b int4
     value) — the loop-kill is decode-only (prefill is unchanged), so this just confirms the load.
  4. **decode throughput lever (informational).** Aggregate decode tok/s of the loop-kill vs the
     per-stream loop at B=1 and B=``B`` — the loop-kill reads each mixer weight ⌈B/chunk⌉× instead of
     B× (on top of M3-2's batched-MoE expert-read amortization), so the win grows with B. Printed.

    uv run python -m parity.minimax_m3_loopkill_real          # full re-gate (all 60 layers, SOLO)
    uv run python -m parity.minimax_m3_loopkill_real 4 64     # n_layers, n_tok (bounded code smoke)

# parity-gate: real-weight
"""

from __future__ import annotations

import math
import os
import sys
import time

import mlx.core as mx

from parity.minimax_m3_ppl import PROSE, streamed_logits, teacher_forced
from quanta.minimax.artifact_m3 import MiniMaxM3Artifact
import quanta.minimax.batched_runtime_m3 as BR
from quanta.minimax.batched_runtime_m3 import MiniMaxM3BatchedResidentModel
from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.runtime_m3 import MiniMaxM3ResidentModel
from quanta.minimax.tokenizer import MiniMaxTokenizer

SRC = "/Users/pmrj/models/MiniMax-M3"
ART = "/Users/pmrj/models/MiniMax-M3-quanta_int4g64"
N_TOK = 256          # a parity re-gate, not a ppl measurement
PPL_CEILING = 30.0   # the served runtime must ship a healthy 397B ppl (the M2b int4 value ~5.0)
B = 8                # batched streams for the loop-kill equivalence + throughput probe
N_DECODE = 8         # teacher-forced decode steps for the multi-step equivalence
# loop-kill vs per-stream loop differ ONLY by the fused padded-SDPA reorder (projections bit-exact
# chunked, RoPE bit-identical, the batched MoE shared). At int6 this was BIT-EXACT; at int4 the coarse
# MoE amplifies the reorder to a worst row of ~0.19 rel / 0.875 token-agree @ B=8 (scattered near-tie
# flips, NOT the O(1)-all-rows signature of a systematic bug). These int4-loosened bounds confirm the
# force-enabled path is that benign reorder regime — it is NOT graduated as the int4 serving default
# (auto-OFF), so this is a "not broken", not an equivalence, gate. (int6 used 0.15 / 0.90.)
LK_REL_CEIL = 0.30
LK_AGREE_FLOOR = 0.80  # fraction of (stream,step) comparisons whose loop-kill top-1 matches the loop

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


def run(n_layers: int | None = None, n_tok: int = N_TOK) -> None:
    full = n_layers is None
    mx.set_cache_limit(8 * 1024**3)
    _set_wired()
    cfg = MiniMaxM3Config.from_pretrained(SRC)
    tok = MiniMaxTokenizer(os.path.join(SRC, "tokenizer.json"), cfg)
    ids_list = tok.encode(PROSE)[:n_tok]
    ids = mx.array(ids_list, dtype=mx.uint32)
    print(f"=== MiniMax-M3-VL M3-3 GQA loop-kill re-gate — {len(ids_list)} tok, "
          f"{'all 60' if full else n_layers} layers (SOLO) ===", flush=True)

    # ---- ONE resident load: FORCE loop-kill ON (int4 default is OFF), packed mixer + packed experts --
    t0 = time.perf_counter()
    lk = MiniMaxM3BatchedResidentModel(ART, max_batch=max(B, 2), n_layers=n_layers,
                                       packed=True, packed_experts=True, loopkill=True)  # forced ON to gate
    t_load = time.perf_counter() - t0
    # the int4 SERVING default auto-disables loop-kill (rule 4 — parity not bit-exact at int4): assert the
    # resolver the constructor uses returns OFF for this artifact's width (no second 233 GiB load needed).
    eb = BR._served_expert_bits(lk.layers)
    _ck(BR._resolve_loopkill_default(eb) is False,
        f"int4 loop-kill default must be OFF (served expert bits {eb}); the serving attention is per-stream")
    # per-stream-loop sibling + single-stream ref over the SAME resident layers (no copy)
    loop = MiniMaxM3BatchedResidentModel.from_inner(lk.layers, lk.embed_w, lk.norm_w, lk.lm_head_w,
                                                    lk.cfg, max_batch=max(B, 2), loopkill=False)
    single = MiniMaxM3ResidentModel.from_blocks(lk.layers, lk.embed_w, lk.norm_w, lk.lm_head_w, lk.cfg)
    _ck(lk._loopkill and lk.packed, "loop-kill model is not loopkill+packed (forced)")
    _ck(not loop._loopkill, "per-stream-loop sibling did not pin loopkill=False")
    n_built = lk.num_layers
    print(f"  loaded {n_built}L resident in {t_load:.0f}s (loopkill={lk._loopkill}, "
          f"packed={lk.packed}, packed_experts={lk.packed_experts})", flush=True)

    # ---- ragged streams: stream s consumes a different-length prompt slice ------------------------
    nb = min(B, max(2, len(ids_list) // 16))
    lens = [12 + 3 * s for s in range(nb)]
    prompts = [[int(t) for t in ids_list[(7 * s) % 5: (7 * s) % 5 + lens[s]]] for s in range(nb)]
    # a shared held-out continuation fed to BOTH paths each decode step (teacher-forced lockstep)
    cont = [int(ids_list[(13 * j + 5) % len(ids_list)]) for j in range(N_DECODE)]

    c_lk = lk.make_batch_caches(nb)
    c_lp = loop.make_batch_caches(nb)
    for s in range(nb):
        lk.prefill(prompts[s], c_lk[s])
        loop.prefill(prompts[s], c_lp[s])

    # ---- (1) loop-kill == per-stream loop, multi-step teacher-forced decode -----------------------
    cur = [int(ids_list[(11 * s + 3) % len(ids_list)]) for s in range(nb)]   # first decode token / stream
    worst_rel, n_match, n_cmp = 0.0, 0, 0
    for j in range(N_DECODE):
        offs = [len(prompts[s]) + j for s in range(nb)]
        o_lk = lk.step_batch(cur, c_lk, offs)
        o_lp = loop.step_batch(cur, c_lp, offs)
        mx.eval(o_lk + o_lp)
        for s in range(nb):
            worst_rel = max(worst_rel, _rel(o_lk[s], o_lp[s]))
            n_match += int(_top1(o_lk[s], o_lp[s]))
            n_cmp += 1
        cur = [cont[j]] * nb                                  # same next token to BOTH (stay in lockstep)
    agree = n_match / max(n_cmp, 1)
    print(f"  [loop-kill vs loop] B={nb}, {N_DECODE} decode steps: top-1 agree {agree:.4f} "
          f"({n_match}/{n_cmp}) | worst rel {worst_rel:.2e}", flush=True)
    _ck(worst_rel < LK_REL_CEIL,
        f"loop-kill diverges from per-stream loop: worst rel {worst_rel:.2e} >= {LK_REL_CEIL}")
    _ck(agree >= LK_AGREE_FLOOR,
        f"loop-kill top-1 drifts from per-stream loop: {agree:.4f} < {LK_AGREE_FLOOR}")

    # ---- (2) loop-kill == single-stream decode, ragged (end-to-end anchor) ------------------------
    nxt = [int(ids_list[(17 * s + 2) % len(ids_list)]) for s in range(nb)]
    ref = []
    for s in range(nb):
        ca = single.make_caches()
        single(mx.array(prompts[s], dtype=mx.int32), caches=ca)
        ref.append(single(mx.array([nxt[s]], dtype=mx.int32), caches=ca))
    c2 = lk.make_batch_caches(nb)
    for s in range(nb):
        lk.prefill(prompts[s], c2[s])
    o2 = lk.step_batch(nxt, c2, [len(prompts[s]) for s in range(nb)])
    mx.eval(o2 + ref)
    rels2 = [_rel(o2[s], ref[s]) for s in range(nb)]
    top1_2 = sum(_top1(o2[s], ref[s]) for s in range(nb)) / nb
    print(f"  [loop-kill vs single] B={nb} ragged step: top-1 match {top1_2:.3f} | "
          f"worst rel {max(rels2):.2e}", flush=True)
    _ck(max(rels2) < LK_REL_CEIL,
        f"loop-kill step diverges from single-stream: worst rel {max(rels2):.2e} >= {LK_REL_CEIL}")
    _ck(top1_2 >= 0.75,
        f"loop-kill top-1 drifts from single-stream: {top1_2:.3f} < 0.75")

    # ---- (3) ppl sanity: the resident prefill ships a healthy 397B ppl ----------------------------
    logits_res = single(ids)
    mx.eval(logits_res)
    ppl_res, acc_res, argmax_res = teacher_forced(logits_res, ids)
    print(f"  [ppl      ] resident prefill ppl {ppl_res:7.4f}  acc {acc_res:.4f}", flush=True)
    _ck(math.isfinite(ppl_res), "non-finite resident ppl")
    if full:
        # cross-check against the streamed bf16 reference (same codes) — confirms the load is the M2b model
        art = MiniMaxM3Artifact(ART)
        logits_ref = streamed_logits(art, art.cfg, ids, n_layers=n_layers)
        ppl_ref, _, argmax_ref = teacher_forced(logits_ref, ids)
        del art
        mx.clear_cache()
        dppl = 100.0 * (ppl_res / ppl_ref - 1.0) if ppl_ref > 0 else float("inf")
        agree_p = float(mx.mean((argmax_res == argmax_ref).astype(mx.float32)).item())
        print(f"  [ppl      ] streamed bf16 ref ppl {ppl_ref:7.4f} | Δppl {dppl:+.3f}% | "
              f"top-1 agree {agree_p:.4f}", flush=True)
        _ck(ppl_res < PPL_CEILING,
            f"resident ppl {ppl_res:.4f} >= {PPL_CEILING}: not coherent (expected ~5.0)")

    # ---- (4) decode throughput lever: loop-kill vs per-stream loop (informational) ----------------
    if full:
        def _decode_tps(model: MiniMaxM3BatchedResidentModel, bsz: int, steps: int = 6) -> float:
            caches = model.make_batch_caches(bsz)
            seed = [int(ids_list[i % len(ids_list)]) for i in range(bsz)]
            for s in range(bsz):
                model.prefill([seed[s]], caches[s])
            cur_t = list(seed)
            t = time.perf_counter()
            for _ in range(steps):
                lg = model.step_batch(cur_t, caches, [c[0].offset for c in caches])
                mx.eval(lg)
                cur_t = [int(mx.argmax(lg[s][0, 0]).item()) for s in range(bsz)]
            return bsz * steps / (time.perf_counter() - t)
        lk1, lkB = _decode_tps(lk, 1), _decode_tps(lk, nb)
        lp1, lpB = _decode_tps(loop, 1), _decode_tps(loop, nb)
        print(f"  [throughput] decode tok/s — loop-kill: B=1 {lk1:.1f} | B={nb} {lkB:.1f}  ||  "
              f"per-stream loop: B=1 {lp1:.1f} | B={nb} {lpB:.1f}", flush=True)
        print(f"  [throughput] loop-kill speedup over loop @ B={nb}: {lkB / max(lpB, 1e-9):.2f}x "
              f"| aggregate B=1→B={nb} {lkB / max(lk1, 1e-9):.2f}x", flush=True)

    del lk, loop, single
    mx.clear_cache()

    if full:
        print(f"\nVERDICT: M3-3 GQA loop-kill @ 397B int4 — force-enabled path is the bounded SDPA-reorder "
              f"regime vs the per-stream loop (top-1 {agree:.4f}, worst rel {worst_rel:.1e} — not a "
              f"systematic bug) but NOT bit-exact at int4, so it is AUTO-OFF as the int4 serving default "
              f"(rule 4 — the per-stream loop serves); resident ppl {ppl_res:.3f} (healthy int4).",
              flush=True)
    else:
        print(f"\nSMOKE ok — loop-kill ran ({nb} streams, {n_built} layers); top-1 {agree:.4f}, "
              f"rel {worst_rel:.1e} (numbers not meaningful on a partial model).", flush=True)
    print(f"PARITY-CHECKS: {_N}", flush=True)


if __name__ == "__main__":
    nl = int(sys.argv[1]) if len(sys.argv) > 1 else None
    nt = int(sys.argv[2]) if len(sys.argv) > 2 else N_TOK
    run(nl, nt)
