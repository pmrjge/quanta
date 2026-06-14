"""Model-free M3-3 gate: the MiniMax-M3 GQA loop-kill (``loopkill`` ON) == the per-stream attention
loop (``loopkill`` OFF) — tiny synthetic, no real weights.

M3-3 replaces the Design-A per-stream attention loop (M3-2) with ONE batched attention across all ``B``
streams. M3 is all-GQA (no GDN hybrid, no YaRN), so this is the whole mixer:

* batched q/k/v/o projections — each mixer weight read ``⌈B/chunk⌉×`` instead of ``B×`` (the
  mixer-read bandwidth win, on top of M3-2's batched-MoE expert-read amortization);
* a per-stream RoPE *kernel* loop — only the absolute offset differs (M3 has ONE ``inv_freq``), so
  the exact :func:`quanta.minimax.model_m3.rope_fast` kernel is looped per row, never a batched reimpl
  (the bf16-drift trap, ``feedback_batched_rope_bf16``);
* the shared fused padded SDPA across all ``B`` streams
  (:func:`quanta.modeling.batched_attention.batched_decode_attention_kv`, the #153 primitive
  InternLM2.5 / Nemotron / qwen35 already use).

**Greedy-token-equivalent:** the projections are bit-exact once packed + chunked (``<=`` the loop-kill
chunk keeps each ``mx.quantized_matmul`` in the M=1 gemv regime) and the per-stream RoPE is
bit-identical, so the ONLY divergence vs the per-stream loop is the fused padded-SDPA softmax
reduction-order ULP — the class the project accepts for batched/tiled paths (top-1 exact, ~ULP logits).

**§M0 — option-B foundational mechanism (run first).** Locks the matmul the packed loop-kill rests on:
``mx.quantized_matmul`` is batch-M BIT-EXACT only for ``M<=~10`` (a per-row gemv kernel) and switches
to a reordering tiled GEMM at ``M>=12``. Chunking the batched projections into ``<=``-chunk slices keeps
every matmul in the bit-exact regime, equalling the per-stream ``M=1`` loop bit-for-bit at any ``B``.
Re-proven here for M3's **int8** mixer (the qwen35 #153 finding was int4) — same threshold; if a future
MLX drops the gemv→GEMM threshold below the chunk, §M0 fails loudly (the signal to lower the constant).

Checks (all on tiny synthetic dims; runs in the model-free sweep):
  1. **§M0 chunk-validation** — chunked-``chunk`` int8 ``quantized_matmul`` bit-exact vs the per-stream
     ``M=1`` loop at ``B∈{1,4,8,32}``; full-batch reorders at ``M>=12``. The chunk == the runtime
     constant :data:`quanta.minimax.batched_runtime_m3.MINIMAX_M3_LOOPKILL_CHUNK`.
  2. **loop-kill == per-stream loop, RAGGED ``B=4``** — same packed blocks / same prefilled caches, so
     this isolates the attention dispatch (the MoE sub-block is identical): every stream's argmax
     matches, tight logit rel.
  3. **loop-kill == single-stream decode, RAGGED ``B=4``** — the end-to-end greedy-token-equivalent
     claim that ships.
  4. **B=1 loop-kill == single-stream** (degenerate: only the SDPA ``mask=zeros`` vs ``mask=None``).
  5. **rule 4/6: loop-kill ⇒ packed** — a bf16 (non-packed) batched runtime refuses ``loopkill=True``
     at construction AND on every ``step_batch`` (a runtime toggle cannot bypass it).

    uv run python -m parity.minimax_m3_loopkill_test
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from parity.minimax_m3_batched_test import (
    _build_blocks,
    _cfg,
    _single,
    _synth,
)
from quanta.minimax import batched_runtime_m3 as BR
from quanta.minimax import model_m3 as M

CHUNK = BR.MINIMAX_M3_LOOPKILL_CHUNK   # the runtime loop-kill chunk under test (cross-checked in §M0)

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _rel(a: mx.array, b: mx.array) -> float:
    an, bn = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    return float(np.abs(an - bn).max() / (np.abs(bn).max() + 1e-9))


def _argmax_agree(a: mx.array, b: mx.array) -> float:
    aa = mx.argmax(a[0].astype(mx.float32), axis=-1)
    bb = mx.argmax(b[0].astype(mx.float32), axis=-1)
    return float(mx.mean((aa == bb).astype(mx.float32)).item())


# --- §M0: packed-projection batch-M parity (the option-B foundational mechanism) ------------------- #
# Decisive shape ~ a real M3 mixer projection ([B,1,6144]@[8192,6144], int8-g64): mx.quantized_matmul is
# batch-M BIT-EXACT only for M<=~10 (a per-row gemv); at M>=12 it switches to a tiled GEMM that REORDERS
# the K-reduction (bf16). Chunking into <=CHUNK rows keeps every matmul bit-exact at any B — the mechanism
# the packed loop-kill stepper (MiniMaxM3Attention.decode_step_batched) builds on.
_M0_IN, _M0_OUT = 6144, 8192
_M0_GS, _M0_BITS = 64, 8                # M3's served mixer scheme (int8-g64) — int8, not qwen35's int4


def _maxdiff(a: mx.array, b: mx.array) -> float:
    return float(mx.abs((a - b).astype(mx.float32)).max())


def _chunked(linear, x: mx.array, chunk: int) -> mx.array:
    b = int(x.shape[0])
    if b <= chunk:
        return linear(x)
    return mx.concatenate([linear(x[i:i + chunk]) for i in range(0, b, chunk)], axis=0)


def _batchm_diff(linear, B: int, seed: int, *, chunk: int | None = None) -> float:
    """Worst per-row ``|Δ|`` between ONE forward over a ``[B,1,in]`` batch (full-batch if ``chunk`` is
    None, else chunked) and ``B`` separate ``M=1`` forwards. Zero ⇔ batch-M invariant."""
    mx.random.seed(seed)
    x = mx.random.normal((B, 1, _M0_IN)).astype(mx.bfloat16)
    yb = linear(x) if chunk is None else _chunked(linear, x, chunk)
    mx.eval(yb)
    return max(_maxdiff(yb[s:s + 1], linear(x[s:s + 1])) for s in range(B))


def _m0_chunk_validation() -> None:
    mx.random.seed(133)
    w = mx.random.normal((_M0_OUT, _M0_IN)).astype(mx.bfloat16)
    wq, sc, bi = mx.quantize(w, group_size=_M0_GS, bits=_M0_BITS)
    ql = nn.QuantizedLinear(_M0_IN, _M0_OUT, bias=False, group_size=_M0_GS, bits=_M0_BITS)
    ql.weight, ql.scales, ql.biases = wq, sc, bi

    bs = (1, 4, 8, 32)
    qf = {B: _batchm_diff(ql, B, 400 + B) for B in bs}                  # quantized, full batch
    qc = {B: _batchm_diff(ql, B, 400 + B, chunk=CHUNK) for B in bs}     # quantized, chunked <=CHUNK
    chunk_exact = all(qc[B] == 0.0 for B in bs)        # THE FIX: chunked quantized bit-exact at every B
    full_threshold = qf[CHUNK] == 0.0 and qf[32] > 0.0  # WHY: full-batch exact <=CHUNK, reorders @ B=32
    qcs = " ".join(f"B{B}={qc[B]:.1e}" for B in bs)
    qfs = " ".join(f"B{B}={qf[B]:.1e}" for B in bs)
    _ck(CHUNK <= 8, f"loop-kill chunk {CHUNK} > 8 — outside the validated gemv regime")
    _ck(chunk_exact, f"chunked-{CHUNK} int8 quantized_matmul NOT bit-exact across B: [{qcs}]")
    _ck(full_threshold,
        f"int8 gemv→GEMM threshold moved: full-batch [{qfs}] — expected exact@B={CHUNK}, reorder@B=32 "
        f"(if the threshold dropped below {CHUNK}, lower MINIMAX_M3_LOOPKILL_CHUNK)")
    print(f"  [§M0] chunked-{CHUNK} int8 quantized BIT-EXACT [{qcs}] (fix={chunk_exact}) | "
          f"full-batch [{qfs}] (reorders@>=12={full_threshold})")


def _lk(cfg, blocks, w, *, max_batch=8) -> BR.MiniMaxM3BatchedResidentModel:
    """The serving config: loop-kill ON (requires packed — the blocks are packed mixer + experts)."""
    return BR.MiniMaxM3BatchedResidentModel.from_inner(
        blocks, w["embed"], M.one_plus(w["final_norm"]), w["lm_head"], cfg, max_batch=max_batch,
        loopkill=True)


def _loop(cfg, blocks, w, *, max_batch=8) -> BR.MiniMaxM3BatchedResidentModel:
    """The M3-2 Design-A reference: per-stream attention loop (loopkill OFF), bit-exact vs single."""
    return BR.MiniMaxM3BatchedResidentModel.from_inner(
        blocks, w["embed"], M.one_plus(w["final_norm"]), w["lm_head"], cfg, max_batch=max_batch,
        loopkill=False)


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except ValueError:
        return True


def run() -> None:
    mx.random.seed(0)
    cfg = _cfg()
    V = cfg.vocab_size
    w = _synth(cfg, mx.random.key(1))

    # (1) §M0 chunk-validation (the substrate the packed loop-kill rests on) -----------------------
    _m0_chunk_validation()

    # the serving model (packed mixer + packed experts): loop-kill ON, the per-stream loop, and a
    # single-stream ref all over the SAME blocks ⇒ checks isolate the attention dispatch, not quant.
    blk_serve = _build_blocks(cfg, w, packed_mixer=True, packed_experts=True)
    single = _single(cfg, blk_serve, w)
    lk = _lk(cfg, blk_serve, w, max_batch=8)
    loop = _loop(cfg, blk_serve, w, max_batch=8)
    _ck(lk._loopkill and lk.packed, "loop-kill model did not enable loopkill on a packed runtime")
    _ck(not loop._loopkill, "per-stream reference did not pin loopkill=False")

    # ragged prompts (lengths 3,4,5,6) → ragged per-stream offsets, the real serving case ----------
    B = 4
    prompts = [[(s * 13 + i * 7 + 1) % V for i in range(3 + s)] for s in range(B)]
    nxt = [int((s * 5 + 2) % V) for s in range(B)]

    # single-stream reference: decode each stream independently
    ref = []
    for s in range(B):
        ca = single.make_caches()
        single(mx.array(prompts[s]), caches=ca)
        ref.append(single(mx.array([nxt[s]]), caches=ca))

    # loop-kill batched step (own caches)
    c_lk = lk.make_batch_caches(B)
    for s in range(B):
        lk.prefill(prompts[s], c_lk[s])
    out_lk = lk.step_batch(nxt, c_lk, [len(prompts[s]) for s in range(B)])

    # per-stream-loop batched step (own caches; the Design-A reference)
    c_lp = loop.make_batch_caches(B)
    for s in range(B):
        loop.prefill(prompts[s], c_lp[s])
    out_lp = loop.step_batch(nxt, c_lp, [len(prompts[s]) for s in range(B)])
    mx.eval(out_lk + out_lp + ref)

    # (2) loop-kill == per-stream loop (isolates the attention dispatch; MoE identical) ------------
    w2_rel, w2_agree = 0.0, 1.0
    for s in range(B):
        r, a = _rel(out_lk[s], out_lp[s]), _argmax_agree(out_lk[s], out_lp[s])
        w2_rel, w2_agree = max(w2_rel, r), min(w2_agree, a)
        _ck(a >= 0.90, f"loop-kill stream {s} top-1 drifts from per-stream loop: agree {a:.4f}")
        _ck(r < 2e-2, f"loop-kill stream {s} logits != per-stream loop: rel {r:.2e}")

    # (3) loop-kill == single-stream decode (end-to-end greedy-token-equivalent) -------------------
    w3_rel, w3_agree = 0.0, 1.0
    for s in range(B):
        r, a = _rel(out_lk[s], ref[s]), _argmax_agree(out_lk[s], ref[s])
        w3_rel, w3_agree = max(w3_rel, r), min(w3_agree, a)
        _ck(a >= 0.90, f"loop-kill stream {s} top-1 drifts from single-stream: agree {a:.4f}")
        _ck(r < 2e-2, f"loop-kill stream {s} logits != single-stream: rel {r:.2e}")

    # (4) B=1 loop-kill == single-stream (degenerate — only mask=zeros vs mask=None) ---------------
    c1 = lk.make_batch_caches(1)
    lk.prefill(prompts[0], c1[0])
    out1 = lk.step_batch([nxt[0]], c1, [len(prompts[0])])
    _ck(_rel(out1[0], ref[0]) < 2e-2 and _argmax_agree(out1[0], ref[0]) >= 0.90,
        "B=1 loop-kill step != single-stream decode")

    # (5) rule 4/6: loop-kill ⇒ packed — bf16 runtime refuses loopkill at construction AND at step --
    blk_bf16 = _build_blocks(cfg, w, packed_mixer=False, packed_experts=True)  # bf16 mixer (nn.Linear)
    _ck(_raises(lambda: BR.MiniMaxM3BatchedResidentModel.from_inner(
        blk_bf16, w["embed"], M.one_plus(w["final_norm"]), w["lm_head"], cfg,
        max_batch=8, loopkill=True)),
        "from_inner accepted loopkill=True on a non-packed (bf16) mixer (rule 4/6)")
    # a packed loop-kill model whose .packed is forced False must refuse at step (toggle can't bypass)
    lk_bad = _lk(cfg, blk_serve, w, max_batch=8)
    lk_bad.packed = False
    cbad = lk_bad.make_batch_caches(1)
    lk_bad.prefill(prompts[0], cbad[0])
    _ck(_raises(lambda: lk_bad.step_batch([nxt[0]], cbad, [len(prompts[0])])),
        "step_batch ran the loop-kill after .packed was cleared (rule 6)")

    print("\n=== MiniMax-M3-VL M3-3 GQA loop-kill gate (model-free) ===")
    print(f"(2) loop-kill == per-stream loop (B={B}, ragged): worst agree {w2_agree:.4f}, "
          f"worst rel {w2_rel:.2e}")
    print(f"(3) loop-kill == single-stream decode (B={B}, ragged): worst agree {w3_agree:.4f}, "
          f"worst rel {w3_rel:.2e}")
    print("(4) B=1 loop-kill == single-stream (top-1 exact)")
    print("(5) rule 4/6: loop-kill ⇒ packed (refused at construction AND at step on a bf16 mixer)")
    print(f"PARITY-CHECKS: {_N}")
    print("PASS — M3-3 GQA loop-kill: ONE batched attention across streams is greedy-token-equivalent "
          "to the per-stream loop; chunked projections bit-exact (§M0); loop-kill ⇒ packed enforced.")


if __name__ == "__main__":
    run()
