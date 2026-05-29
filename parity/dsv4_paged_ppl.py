"""Teacher-forced ppl PAGED ON == OFF for DeepSeek-V4-Flash (#179) — loads int4-g64 (~180 GiB resident), ONE model.

#152 flipped ``PAGED_KV_DEFAULT`` True, so the prefix-sharing paged path is now the engine default for
the three keepers. The model-free ``parity/dsv4_paged_latent_test.py`` proves the single-stream codec +
snapshot/restore bit-exact across the dense/ratio-4/ratio-128 regimes; the real greedy gate
``parity/dsv4_paged_real_test.py`` proves a prefix-reuse decode (paged latent-KV blocks + the derived
compressed-KV / lightning-indexer / raw-hidden ring restored from a boundary snapshot) matches discrete.
THIS closes the project's e2e arbiter (CLAUDE.md methodology #4) on the real model: teacher-forced
perplexity computed with paging ON must equal paging OFF — i.e. the paged write+gather latent path,
exercised at EVERY position of a multi-block passage (not just K greedy steps), must not move the
headline number.

From-scratch (``offset=0``, ``recurrent_in=None``) so the whole passage flows through fresh paged latent
blocks AND a fresh derived (ckv/ikv/ring) state — there is NO boundary snapshot to restore here (that is
the reuse gate's job). The ONE variable is the latent KV storage: the discrete
:class:`~quanta.dsv4.decode.DSV4Cache` latent stream vs the paged single-stream
:class:`~quanta.paged.PagedLatentCacheView`. The derived streams are per-stream in BOTH paths and are
pooled from raw hidden identically (they are NOT paged), so they cannot diverge here. int8 latent quant
groups sit on ``head_dim`` while blocks cut the seq axis — orthogonal, so the paged gather is
bit-identical to the discrete cache fed the same tokens ⇒ the gate is BIT-EXACT (|Δlogits| = 0,
Δppl = 0), the B=1 equivalence class ([[feedback-batched-rope-bf16]]). A nonzero Δ here is a real paging
bug in the live forward, not bf16 noise (both paths are one-shot over the *same* q_len, so there is none
of the continue-from-prefix q_len-mismatch noise the #175 reuse gate documents).

The shared manager MUST be built with ``single_stream=True`` (``paged_kv_spec``) — DSV4's latent is one
vector per token (MQA latent), not a k/v pair. DSV4's inner ``__call__`` returns all-position logits
already, so ppl scoring (all positions) calls it directly through the CANONICAL paged state
(``rt.paged_kv_spec`` + ``rt.make_paged_state``) — the exact views the engine's ``_admit_paged`` builds.

DSV4 is the biggest + riskiest keeper — loads ~180 GiB resident (43 int4-packed expert layers via
gather_qmm at ~3.75 GiB/layer; the bf16 ``packed_experts=False`` debug path is ~389 GiB). Each passage
is capped at ``MAX_TOK`` tokens so the token-by-token prefill stays bounded. Run ALONE, never beside another large-resident job (the
OOM-reboot hazard, [[feedback-memory-safety]]).

    uv run --with regex python -m parity.dsv4_paged_ppl
"""

from __future__ import annotations

import mlx.core as mx

from parity.dsv4_ppl import PROSE
from quanta.dsv4.batched_runtime import DSV4BatchedResidentModel
from quanta.dsv4.decode import DSV4Cache
from quanta.dsv4.tokenizer import DeepSeekV4Tokenizer
from quanta.paged import PagedKVCacheManager

ART = "/Users/pmrj/models/DeepSeek-V4-Flash-quanta_int4g64"
BLOCK = 16
MAX_TOK = 128                          # cap per passage (8 blocks) — bounded prefill on the riskiest keeper

PASSAGES: dict[str, str] = {
    # repetition probe: a sound forward nails it near ppl 1 — the strongest "is the read path correct" check.
    "repeat": ("The quick brown fox jumps over the lazy dog. " * 8).strip(),
    # fluent prose: the recognizable dsv4_ppl/dsv4_int4_ppl passage (capped to MAX_TOK), a realistic ppl.
    "prose": PROSE,
}


def _ppl_top1(logits: mx.array, ids: mx.array) -> tuple[float, float]:
    """Teacher-forced perplexity + top-1 next-token accuracy. ``logits`` ``[1,T,vocab]``, ``ids`` ``[T]``."""
    lg = logits.astype(mx.float32)[0][:-1]               # [T-1, vocab] — predict positions 1..T-1
    tgt = ids[1:]                                        # [T-1]
    lse = mx.logsumexp(lg, axis=-1)
    tok = mx.take_along_axis(lg, tgt[:, None], axis=-1)[:, 0]
    ppl = float(mx.exp(mx.mean(lse - tok)).item())
    top1 = float(mx.mean((mx.argmax(lg, axis=-1) == tgt).astype(mx.float32)).item())
    return ppl, top1


def _score_off(rt: DSV4BatchedResidentModel, ids: mx.array) -> mx.array:
    """PAGED OFF: one-shot prefill into a fresh discrete ``DSV4Cache`` (discrete latent stream + fresh
    derived ckv/ikv/ring), all positions. A discrete cache (not ``rt.make_cache``, which since #18 M4
    leases an arena row when ``kv_arena`` defaults on) keeps this the plain non-arena scoring path."""
    return rt._inner(ids, caches=DSV4Cache(rt.num_layers), offset=0)


def _score_paged(rt: DSV4BatchedResidentModel, ids: mx.array,
                 ids_list: list[int]) -> tuple[mx.array, object]:
    """PAGED ON: one-shot prefill from scratch (offset=0, no prefix reuse) through the canonical paged
    state — every latent written to + read from the paged blocks — all positions. Same inner call as
    OFF; the ONLY difference is the latent KV storage (paged single-stream codec vs discrete)."""
    spec = rt.paged_kv_spec
    mgr = PagedKVCacheManager(num_layers=spec["n_layers"], block_size=BLOCK, max_blocks=4096,
                              group_size=spec["group_size"], bits=spec["bits"],
                              quantized=spec["quantized"], model_name="dsv4-ppl-paged",
                              single_stream=spec.get("single_stream", False))
    seq = mgr.new_sequence()
    mgr.advance(seq, ids_list)                           # open [0, T) for latent writes (no prefix reuse)
    state = rt.make_paged_state(mgr, seq)                # paged DSV4Cache (PagedLatentCacheView per layer)
    logits = rt._inner(ids, caches=state, offset=0)
    mgr.commit(seq)                                      # content-hash filled blocks
    return logits, mgr.get_stats()


def run() -> None:
    rt = DSV4BatchedResidentModel(ART, max_batch=1)
    tok = DeepSeekV4Tokenizer.from_pretrained(ART)
    spec_layers = rt.paged_kv_spec["n_layers"]           # paged (latent) layer count — the paging-engaged bar

    print(f"{'passage':8s} {'tok':>4s} {'ppl_off':>9s} {'ppl_on':>9s} {'Δppl':>9s} "
          f"{'t1_off':>7s} {'t1_on':>7s} {'|Δlogit|':>10s} {'blk':>4s}")
    ok = True
    for name, text in PASSAGES.items():
        ids_list = [int(t) for t in tok.encode(text, add_bos=True)][:MAX_TOK]
        ids = mx.array(ids_list)

        off = _score_off(rt, ids)
        on, st = _score_paged(rt, ids, ids_list)
        mx.eval(off, on)

        lmax = float(mx.max(mx.abs(off.astype(mx.float32) - on.astype(mx.float32))).item())
        ppl_off, t1_off = _ppl_top1(off, ids)
        ppl_on, t1_on = _ppl_top1(on, ids)

        exact = lmax == 0.0
        paged_engaged = st.allocated_blocks >= spec_layers and st.prefix_hit_tokens == 0
        row_ok = exact and (ppl_on == ppl_off) and (t1_on == t1_off) and paged_engaged
        ok = ok and row_ok

        flag = "OK" if row_ok else "FAIL"
        print(f"{name:8s} {len(ids_list):4d} {ppl_off:9.4f} {ppl_on:9.4f} {ppl_on - ppl_off:+9.2e} "
              f"{t1_off * 100:6.1f}% {t1_on * 100:6.1f}% {lmax:10.3e} {st.allocated_blocks:4d} [{flag}]")
        if not paged_engaged:
            print(f"    [WARN] paging not engaged: allocated_blocks={st.allocated_blocks} "
                  f"prefix_hit_tokens={st.prefix_hit_tokens}")

    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
