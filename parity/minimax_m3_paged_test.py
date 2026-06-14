"""Model-free M3-4 gate: MiniMax-M3-VL paged-KV + prefix caching (int8 KV) — tiny synthetic.

M3 is the clean dense-GQA paged case: all 60 layers are full attention with a standard k/v pair and
NO recurrent/derived state (``has_recurrent_state=False``), so paging is the textbook k/v scenario —
every layer's :class:`quanta.minimax.model_m3.KVCache` is paged through the shared
:class:`quanta.paged.PagedKVCacheManager`, prefix blocks dedup across requests, and there is nothing to
content-address at a block boundary. Proven here on a tiny synthetic M3 decoder (NO real weights):

A. **core — paged prefix-reuse + suffix == discrete continue-from-prefix (bit-exact).** A paged forward
   that re-references a resident prefix's blocks and prefills only the uncached suffix is **bit-identical**
   to a discrete continue-from-prefix prefill of the same split (the ``cache_quant`` orthogonal-axes
   foundation: int8 quant groups on ``head_dim`` vs blocks on the seq axis). Checked for BOTH KV modes
   (int8 g32 + bf16); asserts ``prefill_paged`` emits **no** boundary payloads and the manager reports a
   real prefix hit (``prefix_hit_tokens > 0``).

B. **paged KV loop-kill == per-stream paged loop (bit-exact).** A batched paged decode step with the
   M3-4 loop-kill (``paged_batched`` → ONE ``write_batched`` scatter + ONE ``gather_batched``) equals the
   same loop-kill attention with the per-stream ``.update()`` loop over paged views — both end in the same
   fused padded SDPA, so bit-identical (the #153 batched scatter/gather == per-stream foundation). A light
   anchor confirms the paged batched step is greedy-token-equivalent to a discrete single-stream decode.

C. **rule 6** — ``prefill_paged`` refuses a non-None ``recurrent_in`` (dense GQA has no recurrent state)
   and a desynced view offset.

``head_dim=32`` (the smallest the int8 KV path accepts — ``mx.quantize``'s min group_size is 32);
``max_position_embeddings`` >> the test lengths so partial RoPE stays in-range. Real-model teacher-forced
parity (paged ON == OFF + the int8-KV Δppl) is the SOLO ``parity/minimax_m3_paged_real.py``.

    uv run python -m parity.minimax_m3_paged_test
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlx.core as mx

from parity.minimax_m3_batched_test import _build_blocks, _synth
from quanta.minimax import batched_runtime_m3 as BR
from quanta.minimax import model_m3 as M
from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.runtime_m3 import MiniMaxM3ResidentModel
from quanta.paged import PagedKVCacheManager

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _cfg32() -> MiniMaxM3Config:
    """Tiny M3 config with ``head_dim=32`` (int8 KV needs group_size>=32) + ``rotary_dim=16``; 2 dense +
    3 MoE layers, structurally faithful (mirrors ``minimax_m3_batched_test._cfg`` at a paged-viable hd)."""
    tc = {
        "vocab_size": 128, "hidden_size": 64, "intermediate_size": 32,
        "dense_intermediate_size": 96, "shared_intermediate_size": 32,
        "num_hidden_layers": 5,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 32,
        "rotary_dim": 16, "partial_rotary_factor": 0.5, "rope_theta": 5e6,
        "use_qk_norm": True, "qk_norm_type": "per_head", "use_gemma_norm": True,
        "attention_output_gate": False,
        "num_local_experts": 6, "num_experts_per_tok": 2, "n_shared_experts": 1,
        "scoring_func": "sigmoid", "use_routing_bias": True, "routed_scaling_factor": 2.0,
        "norm_topk_prob": True,
        "moe_layer_freq": [0, 0, 1, 1, 1],
        "hidden_act": "swigluoai", "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
        "rms_norm_eps": 1e-6, "max_position_embeddings": 1048576, "tie_word_embeddings": False,
        "eos_token_id": 200020,
    }
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "config.json").write_text(json.dumps({"model_type": "minimax_m3_vl", "text_config": tc}))
        return MiniMaxM3Config.from_pretrained(d)


def _single_kv(cfg, blocks, w, *, quantized: bool, kv_gs: int) -> MiniMaxM3ResidentModel:
    """Single-stream resident reference with the chosen KV mode (the discrete oracle paged is gated vs)."""
    return MiniMaxM3ResidentModel.from_blocks(
        blocks, w["embed"], M.one_plus(w["final_norm"]), w["lm_head"], cfg,
        kv_quantized=quantized, kv_group_size=kv_gs)


def _batched_kv(cfg, blocks, w, *, quantized: bool, kv_gs: int, max_batch: int = 4
                ) -> BR.MiniMaxM3BatchedResidentModel:
    """The serving config (packed mixer + packed experts ⇒ loop-kill ON) at the chosen KV mode."""
    return BR.MiniMaxM3BatchedResidentModel.from_inner(
        blocks, w["embed"], M.one_plus(w["final_norm"]), w["lm_head"], cfg, max_batch=max_batch,
        loopkill=True, kv_quantized=quantized, kv_group_size=kv_gs)


def _serve_blocks(cfg, w) -> list:
    return _build_blocks(cfg, w, packed_mixer=True, packed_experts=True)


# --- A. core: paged prefix-reuse + suffix == discrete continue-from-prefix ------------------------- #

def _check_regime(quantized: bool) -> None:
    mx.random.seed(0)
    cfg = _cfg32()
    V = cfg.vocab_size
    w = _synth(cfg, mx.random.key(1))
    blocks = _serve_blocks(cfg, w)
    kv_gs = 32 if quantized else 64
    batched = _batched_kv(cfg, blocks, w, quantized=quantized, kv_gs=kv_gs, max_batch=2)

    P, block = 11, 4
    prompt = [int((i * 7 + 3) % V) for i in range(P)]
    n_pref = ((P - 1) // block) * block          # block-aligned prefix the paged path reuses (= 8)

    # discrete CONTINUE-FROM-PREFIX reference via the SAME forward (``prefill``) the paged path runs, so
    # ONLY the KV storage (discrete concat vs paged blocks) differs ⇒ bit-exact. (The resident
    # ``__call__`` tiles its lm_head over T rows vs ``prefill``'s 1-row head — a benign GEMM-tiling ULP
    # that would mask the paging comparison; using ``prefill`` for both isolates paging.) The prefix/
    # suffix split matches the paged path so the SDPA shapes match exactly.
    disc = batched.make_caches()
    batched.prefill(prompt[:n_pref], disc)
    ref = batched.prefill(prompt[n_pref:], disc)[0, -1]
    mx.eval(ref)

    spec = batched.paged_kv_spec
    _ck(spec["quantized"] == quantized and spec["n_layers"] == cfg.num_hidden_layers,
        f"paged_kv_spec mismatch: {spec}")
    mgr = PagedKVCacheManager(num_layers=spec["n_layers"], block_size=block, max_blocks=64,
                              group_size=spec["group_size"], bits=spec["bits"],
                              quantized=spec["quantized"], model_name="m3t")

    # req1: prefill the block-aligned prefix so its full blocks get content-addressed (committed).
    seq1 = mgr.new_sequence()
    mgr.advance(seq1, prompt[:n_pref])
    st1 = batched.make_paged_state(mgr, seq1)
    batched.prefill_paged(mx.array(prompt[:n_pref], dtype=mx.int32), st1,
                          offset=0, recurrent_in=None, block_size=block)
    mgr.commit(seq1)

    # req2: full prompt — reuse the resident prefix blocks + prefill ONLY the uncached suffix.
    seq2 = mgr.new_sequence()
    n_attn = mgr.match_prefix(seq2, prompt[:-1])
    suffix = prompt[n_attn:]
    mgr.advance(seq2, suffix)
    st2 = batched.make_paged_state(mgr, seq2)
    logits_pg, boundaries = batched.prefill_paged(mx.array(suffix, dtype=mx.int32), st2,
                                                  offset=n_attn, recurrent_in=None, block_size=block)
    mgr.commit(seq2)
    row = logits_pg[0, -1]
    mx.eval(row)

    d = float(mx.max(mx.abs(row - ref)))
    hit = int(mgr.get_stats().prefix_hit_tokens)
    mode = f"int{spec['bits']} g{kv_gs}" if quantized else "bf16"
    print(f"  [{'OK' if (boundaries == [] and n_attn == n_pref and d == 0.0 and hit > 0) else 'XX'}] "
          f"{mode:<9} P={P} blk={block} n_attn={n_attn} suffix={len(suffix)} "
          f"hit_tok={hit} boundaries={len(boundaries)} |Δ|={d:.2e}")
    _ck(boundaries == [], f"dense M3 must emit no boundary payloads, got {boundaries}")
    _ck(n_attn == n_pref > 0, f"expected prefix reuse of {n_pref} tokens, got n_attn={n_attn}")
    _ck(d == 0.0, f"paged != discrete ({mode}): |Δ|={d}")
    _ck(hit > 0, f"manager reported no prefix hit ({mode}): prefix_hit_tokens={hit}")


# --- B. paged KV loop-kill == per-stream paged loop (bit-exact) ------------------------------------ #

def _paged_prefill(model, mgr, prompt) -> tuple:
    """Admit one stream from scratch into a paged state; returns (seq, state) at offset len(prompt)."""
    seq = mgr.new_sequence()
    mgr.advance(seq, prompt)
    state = model.make_paged_state(mgr, seq)
    model.prefill_paged(mx.array(prompt, dtype=mx.int32), state, offset=0,
                        recurrent_in=None, block_size=mgr.block_size)
    mgr.commit(seq)
    return seq, state


def _paged_step(model, mgr, seqs, states, tokens) -> list:
    """One batched paged decode step over the streams (advance → step_batch → commit), as the engine does."""
    for seq, t in zip(seqs, tokens, strict=True):
        mgr.advance(seq, [int(t)])
    offsets = [seq.length - 1 for seq in seqs]
    outs = model.step_batch(list(tokens), states, offsets)
    for seq in seqs:
        mgr.commit(seq)
    return outs


def _new_mgr(spec, block) -> PagedKVCacheManager:
    return PagedKVCacheManager(num_layers=spec["n_layers"], block_size=block, max_blocks=128,
                              group_size=spec["group_size"], bits=spec["bits"],
                              quantized=spec["quantized"], model_name="m3b")


def _rel(a: mx.array, b: mx.array) -> float:
    return float((mx.linalg.norm((a - b).astype(mx.float32))
                  / (mx.linalg.norm(b.astype(mx.float32)) + 1e-9)).item())


def _top1(a: mx.array, b: mx.array) -> bool:
    return int(mx.argmax(a[0, 0]).item()) == int(mx.argmax(b[0, 0]).item())


def _check_paged_batched() -> None:
    mx.random.seed(0)
    cfg = _cfg32()
    V = cfg.vocab_size
    w = _synth(cfg, mx.random.key(2))
    blocks = _serve_blocks(cfg, w)
    block = 4
    B = 3

    # two serving models over the SAME blocks: lk (paged loop-kill ON) vs lp (paged loop-kill OFF, the
    # per-stream .update() loop over paged views). Both run the IDENTICAL loop-kill attention + batched
    # MoE; only the KV write/gather differs ⇒ bit-identical (both end in the same fused padded SDPA).
    lk = _batched_kv(cfg, blocks, w, quantized=True, kv_gs=32, max_batch=B)
    lp = _batched_kv(cfg, blocks, w, quantized=True, kv_gs=32, max_batch=B)
    lp._paged_kv_batched = False
    _ck(lk._paged_kv_batched and lk._loopkill, "lk did not enable the paged loop-kill")
    _ck((not lp._paged_kv_batched) and lp._loopkill, "lp did not pin the per-stream paged loop")
    spec = lk.paged_kv_spec

    prompts = [[int((s * 13 + i * 7 + 1) % V) for i in range(4 + s)] for s in range(B)]  # ragged 4,5,6
    cont = [int((s * 5 + 2) % V) for s in range(B)]

    mgr_lk, mgr_lp = _new_mgr(spec, block), _new_mgr(spec, block)
    seqs_lk, st_lk, seqs_lp, st_lp = [], [], [], []
    for s in range(B):
        q, a = _paged_prefill(lk, mgr_lk, prompts[s])
        seqs_lk.append(q)
        st_lk.append(a)
        q, a = _paged_prefill(lp, mgr_lp, prompts[s])
        seqs_lp.append(q)
        st_lp.append(a)

    out_lk = _paged_step(lk, mgr_lk, seqs_lk, st_lk, cont)
    out_lp = _paged_step(lp, mgr_lp, seqs_lp, st_lp, cont)
    mx.eval(out_lk + out_lp)
    worst = 0.0
    for s in range(B):
        d = float(mx.max(mx.abs(out_lk[s] - out_lp[s])))
        worst = max(worst, d)
        _ck(d == 0.0, f"paged loop-kill != per-stream paged loop on stream {s}: |Δ|={d}")

    # anchor: paged batched step is greedy-token-equivalent to a discrete single-stream decode --------
    single = _single_kv(cfg, blocks, w, quantized=True, kv_gs=32)
    ref = []
    for s in range(B):
        ca = single.make_caches()
        single(mx.array(prompts[s], dtype=mx.int32), caches=ca)
        ref.append(single(mx.array([cont[s]], dtype=mx.int32), caches=ca))
    mx.eval(ref)
    a_rel, a_top1 = 0.0, 1.0
    for s in range(B):
        a_rel = max(a_rel, _rel(out_lk[s], ref[s]))
        a_top1 = min(a_top1, 1.0 if _top1(out_lk[s], ref[s]) else 0.0)
    print(f"  [OK] paged loop-kill == per-stream paged loop (B={B}, ragged): worst |Δ|={worst:.2e} | "
          f"vs discrete single: top-1 {a_top1:.3f}, rel {a_rel:.2e}")
    _ck(a_rel < 2e-2, f"paged batched decode diverges from discrete single: rel {a_rel:.2e}")
    _ck(a_top1 >= 0.99, f"paged batched decode top-1 drifts from discrete single: {a_top1:.3f}")


# --- C. rule 6 ------------------------------------------------------------------------------------- #

def _raises(fn) -> bool:
    try:
        fn()
        return False
    except ValueError:
        return True


def _check_rule6() -> None:
    mx.random.seed(0)
    cfg = _cfg32()
    w = _synth(cfg, mx.random.key(3))
    blocks = _serve_blocks(cfg, w)
    batched = _batched_kv(cfg, blocks, w, quantized=True, kv_gs=32, max_batch=2)
    spec = batched.paged_kv_spec
    block = 4
    mgr = _new_mgr(spec, block)
    prompt = [int(i) for i in range(8)]
    seq = mgr.new_sequence()
    mgr.advance(seq, prompt)
    st = batched.make_paged_state(mgr, seq)

    # recurrent_in must be None (M3 is dense GQA — no recurrent state)
    _ck(_raises(lambda: batched.prefill_paged(mx.array(prompt, dtype=mx.int32), st, offset=0,
                                              recurrent_in=object(), block_size=block)),
        "prefill_paged accepted a non-None recurrent_in (rule 6)")
    # a desynced view offset (views at 0, claim offset 3) must refuse
    _ck(_raises(lambda: batched.prefill_paged(mx.array(prompt, dtype=mx.int32), st, offset=3,
                                              recurrent_in=None, block_size=block)),
        "prefill_paged accepted a desynced offset (rule 6)")
    print("  [OK] rule 6: prefill_paged refuses recurrent_in!=None and a desynced offset")


def run() -> None:
    print("\n=== MiniMax-M3-VL M3-4 paged-KV + prefix caching gate (model-free) ===")
    print("A. core: paged prefix-reuse + suffix == discrete continue-from-prefix (per KV mode)")
    _check_regime(quantized=True)
    _check_regime(quantized=False)
    print("B. paged KV loop-kill == per-stream paged loop (bit-exact) + greedy-token-equiv anchor")
    _check_paged_batched()
    print("C. rule 6")
    _check_rule6()
    print(f"PARITY-CHECKS: {_N}")
    print("PASS — M3-4 paged-KV: paged == discrete bit-exact (int8 + bf16), prefix blocks dedup, the "
          "paged KV loop-kill == the per-stream paged loop; dense M3 emits no boundary payloads.")


if __name__ == "__main__":
    run()
