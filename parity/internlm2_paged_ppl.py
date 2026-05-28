"""Teacher-forced ppl PAGED ON == OFF for InternLM2.5-7B-Chat-1M (#179) — loads int8-g64, ONE model.

#152 flipped ``PAGED_KV_DEFAULT`` True, so the prefix-sharing paged path is now the engine default for
the three keepers. The model-free ``parity/internlm2_paged_test.py`` proves the session orchestration
bit-exact on a tiny model; the real greedy gate ``parity/internlm2_paged_real_test.py`` proves a
prefix-reuse decode matches discrete. THIS closes the project's e2e arbiter (CLAUDE.md methodology #4)
on the real model: teacher-forced perplexity computed with paging ON must equal paging OFF — i.e. the
paged write+gather KV path, exercised at EVERY position of a multi-block passage (not just K greedy
steps), must not move the headline number.

From-scratch (``offset=0``) so the whole passage flows through fresh paged blocks. Dense GQA + KV quant
on ``head_dim`` is orthogonal to the blocks cut on the seq axis, so the paged gather is bit-identical to
the discrete :class:`~quanta.internlm2.attention.KVCache` fed the same tokens ⇒ the gate is BIT-EXACT
(|Δlogits| = 0, Δppl = 0), the B=1 equivalence class ([[feedback-batched-rope-bf16]]). A nonzero Δ here
is a real paging bug in the live forward, not bf16 noise (both paths are one-shot over the *same* q_len,
so there is none of the continue-from-prefix/q_len-mismatch noise the #176 reuse gate documents).

Both runtime prefill entry points force ``last_only=True`` (memory-safe generation — only the sampler's
row), so ppl scoring (all positions) calls the inner ResidentModel with ``last_only=False`` through the
CANONICAL paged state (``rt.paged_kv_spec`` + ``rt.make_paged_state``) — the exact views the engine's
``_admit_paged`` builds.

Loads ~6 GB. Run ALONE — never beside another large-resident job (the OOM-reboot hazard).

    uv run --with tokenizers python -m parity.internlm2_paged_ppl
"""

from __future__ import annotations

import mlx.core as mx

from quanta.internlm2.batched_runtime import InternLM2BatchedResidentModel
from quanta.internlm2.tokenizer import InternLM2Tokenizer
from quanta.paged import PagedKVCacheManager

ART = "/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64"
BLOCK = 16

PASSAGES: dict[str, str] = {
    # repetition probe: a sound forward nails it near ppl 1 — the strongest "is the read path correct" check.
    "repeat": ("The quick brown fox jumps over the lazy dog. " * 8).strip(),
    # fluent prose: a realistic ppl (same passage as internlm2_packed_ppl so the number is recognizable).
    "prose": (
        "Good engineering begins with a clear understanding of the problem at hand. Before writing "
        "a single line of code, a careful developer studies how the existing system behaves, "
        "identifies the precise point at which it fails, and considers the smallest change that "
        "would restore correct behavior. This discipline saves time in the long run, because broad "
        "rewrites tend to introduce new defects faster than they remove old ones."
    ),
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


def _score_off(rt: InternLM2BatchedResidentModel, ids: mx.array) -> mx.array:
    """PAGED OFF: one-shot prefill into a fresh discrete InternLM2Cache, all positions."""
    return rt._inner(ids, cache=rt.new_cache(), last_only=False)


def _score_paged(rt: InternLM2BatchedResidentModel, ids: mx.array,
                 ids_list: list[int]) -> tuple[mx.array, object]:
    """PAGED ON: one-shot prefill from scratch (offset=0) through the canonical paged state — every
    position written to + read from the paged blocks — all positions."""
    spec = rt.paged_kv_spec
    mgr = PagedKVCacheManager(num_layers=spec["n_layers"], block_size=BLOCK, max_blocks=4096,
                              group_size=spec["group_size"], bits=spec["bits"],
                              quantized=spec["quantized"], model_name="ilm2-ppl-paged")
    seq = mgr.new_sequence()
    mgr.advance(seq, ids_list)                           # open [0, T) for KV writes (no prefix reuse)
    state = rt.make_paged_state(mgr, seq)                # one PagedKVCacheView per layer (engine-canonical)
    logits = rt._inner(ids, caches=state, offset=0, last_only=False)
    mgr.commit(seq)                                      # content-hash filled blocks
    return logits, mgr.get_stats()


def run() -> None:
    rt = InternLM2BatchedResidentModel(ART, max_batch=1)
    tok = InternLM2Tokenizer.from_pretrained(ART)

    print(f"{'passage':8s} {'tok':>4s} {'ppl_off':>9s} {'ppl_on':>9s} {'Δppl':>9s} "
          f"{'t1_off':>7s} {'t1_on':>7s} {'|Δlogit|':>10s} {'blk':>4s}")
    ok = True
    for name, text in PASSAGES.items():
        ids_list = [int(t) for t in tok.encode(text, add_bos=True)]
        ids = mx.array(ids_list)

        off = _score_off(rt, ids)
        on, st = _score_paged(rt, ids, ids_list)
        mx.eval(off, on)

        lmax = float(mx.max(mx.abs(off.astype(mx.float32) - on.astype(mx.float32))).item())
        ppl_off, t1_off = _ppl_top1(off, ids)
        ppl_on, t1_on = _ppl_top1(on, ids)

        exact = lmax == 0.0
        paged_engaged = st.allocated_blocks >= rt.num_layers and st.prefix_hit_tokens == 0
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
