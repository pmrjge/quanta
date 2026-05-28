"""Paged HYBRID engine equivalence (model-free, tiny stub runtime) — #152 step 3, the rule-4 bar.

Proves the unified batched session's PAGED path (``quanta.shim.omlx._BaseBatchedSession`` in paged
mode) is output-equivalent to the discrete path for a HYBRID model — i.e. a second request that shares
a prompt prefix reuses the resident attention-KV blocks AND restores the recurrent boundary state,
prefills only the suffix, and then decodes **token-for-token identically** to a from-scratch full
prefill + decode. This is the engine-level composition of two separately-gated facts:
``parity/paged_cache_test.py`` (paged KV == discrete) + ``parity/paged_recurrent_suffix_test.py``
(restore + suffix == full recurrence).

The stub is a faithful tiny hybrid (no real weights, ~0 GB — SAFE beside a live GPU job): per token a
recurrent accumulator ``rec = decay*rec + E[token]`` (history- and order-dependent, the essential
Mamba/GDN property) AND an attention layer whose value stream is cached in the REAL
:class:`quanta.nemotron.attention.KVCache` (discrete path) or :class:`quanta.paged.PagedKVCacheView`
(paged path); the next-token logits read BOTH (a position-ramp-weighted readout of the full cached
value stream + the recurrent state), so a session bug in EITHER the KV reuse or the recurrent restore
diverges the decode. The forward is cache-agnostic — the exact same code runs on a ``KVCache`` and on
a paged view — so the only thing under test is whether the session drives the paged cache + recurrent
cache correctly.

Checks: (1) paged decode tokens == discrete decode tokens (bit-exact argmax chain); (2) the prefix was
actually reused (manager prefix-hit tokens == the shared prefix length) and the recurrent boundary was
restored (recurrent snapshot hit); (3) a decode-crossing block boundary writes a fresh recurrent
snapshot; (4) ``prefix_cache_enabled`` / ``get_cache_stats`` report the live paged state.

    uv run python -m parity.paged_engine_equiv_test

deferred (one model at a time, GPU): real Nemotron teacher-forced ppl with paged ON == paged OFF.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.nemotron.attention import KVCache
from quanta.paged import PagedKVCacheManager, RecurrentPrefixCache
from quanta.shim.omlx import _BaseBatchedSession

D = 64
GROUP = 64
BLOCK = 4
VOCAB = 96
DECAY = 0.9


class _State:
    """Per-stream stub state: an attention value cache (``KVCache`` or ``PagedKVCacheView``, both expose
    ``offset`` + ``update(k,v)->(k_full,v_full)``) and a recurrent accumulator ``rec`` [D]."""

    def __init__(self, kv, rec: mx.array) -> None:
        self.kv = kv
        self.rec = rec

    @property
    def offset(self) -> int:
        return self.kv.offset


class _StubHybrid:
    """Tiny hybrid runtime exposing BOTH the unpaged contract (make_stream_state / prefill / step_batch)
    and the paged contract (paged_kv_spec / has_recurrent_state / make_paged_state / prefill_paged /
    get_recurrent_state). Fixed random weights shared across both paths so paged == discrete is a real
    equivalence, not a tautology."""

    has_recurrent_state = True
    num_layers = 1

    def __init__(self, seed: int = 0) -> None:
        mx.random.seed(seed)
        self.embed = mx.random.normal((VOCAB, D)).astype(mx.bfloat16)
        self.w_out = mx.random.normal((VOCAB, D)).astype(mx.bfloat16)

    # --- shared cache-agnostic forward ---------------------------------------
    def _readout(self, v_full: mx.array) -> mx.array:
        """Position-ramp-weighted reduction of the full cached value stream [1,1,S,D] -> [D]. Ramp
        weights make it order-/position-sensitive (a mis-placed KV row diverges it)."""
        vs = v_full[0, 0].astype(mx.float32)                       # [S, D]
        s = vs.shape[0]
        w = mx.arange(1, s + 1).astype(mx.float32)                 # [S]
        return (w[:, None] * vs).sum(0) / w.sum()                  # [D]

    def _logits(self, rec: mx.array, v_full: mx.array) -> mx.array:
        h = rec.astype(mx.float32) + self._readout(v_full)         # [D]
        return (h @ self.w_out.T.astype(mx.float32))[None, None]   # [1,1,VOCAB]

    def _recur(self, rec: mx.array, ids: list[int], base: int, block_size: int
               ) -> tuple[mx.array, list[tuple[int, mx.array]]]:
        """Advance the recurrence over ``ids`` (first at absolute position ``base``); capture the state
        at each absolute position landing on a block boundary."""
        snaps: list[tuple[int, mx.array]] = []
        for j, t in enumerate(ids):
            rec = DECAY * rec + self.embed[t].astype(mx.float32)
            pos = base + j + 1
            if block_size and pos % block_size == 0:
                snaps.append((pos, rec))
        return rec, snaps

    def _consume(self, state: _State, ids: list[int], base: int, block_size: int
                 ) -> tuple[mx.array, list[tuple[int, mx.array]]]:
        """Write ``ids``' value stream into ``state.kv`` (one-shot append at ``base``), advance the
        recurrence, and return (last-position logits [1,1,VOCAB], recurrent boundary snapshots)."""
        x = self.embed[mx.array(ids)][None, None]                  # [1,1,T,D]
        _k_full, v_full = state.kv.update(x, x)                    # value stream cached + gathered
        state.rec, snaps = self._recur(state.rec, ids, base, block_size)
        mx.eval(v_full, state.rec)
        return self._logits(state.rec, v_full), snaps

    @staticmethod
    def _ids(seq) -> list[int]:
        arr = seq if isinstance(seq, list) else mx.array(seq).reshape(-1).tolist()
        return [int(t) for t in arr]

    # --- unpaged contract (discrete reference) -------------------------------
    def make_stream_state(self) -> _State:
        return _State(KVCache(quantized=True, group_size=GROUP), mx.zeros((D,), dtype=mx.float32))

    def prefill(self, prompt_ids, state: _State) -> mx.array:
        logits, _ = self._consume(state, self._ids(prompt_ids), base=0, block_size=0)
        return logits

    def step_batch(self, stream_token_ids, states, offsets=None) -> list[mx.array]:
        del offsets  # each state's cache owns its offset (paged view or KVCache)
        out: list[mx.array] = []
        for tok, st in zip(stream_token_ids, states, strict=True):
            base = st.offset
            logits, _ = self._consume(st, self._ids(tok), base=base, block_size=0)
            out.append(logits)
        return out

    # --- paged contract -------------------------------------------------------
    paged_kv_spec = {"n_layers": 1, "group_size": GROUP, "bits": 8, "quantized": True}

    def make_paged_state(self, manager: PagedKVCacheManager, seq) -> _State:
        return _State(manager.view(seq, 0), mx.zeros((D,), dtype=mx.float32))

    def prefill_paged(self, suffix_ids, state: _State, *, offset: int, recurrent_in, block_size: int
                      ) -> tuple[mx.array, list[tuple[int, mx.array]]]:
        if recurrent_in is not None:
            state.rec = recurrent_in
        logits, snaps = self._consume(state, self._ids(suffix_ids), base=offset, block_size=block_size)
        return logits, snaps

    def get_recurrent_state(self, state: _State) -> mx.array:
        return state.rec


def _argmax(row: mx.array) -> int:
    return int(mx.argmax(row.reshape(-1)).item())


def _discrete_decode(rt: _StubHybrid, ids: list[int], n_steps: int) -> list[int]:
    """From-scratch reference: full prefill + greedy decode, driving the stub directly (no session)."""
    state = rt.make_stream_state()
    row = rt.prefill(ids, state)[0, -1]
    toks = [_argmax(row)]
    for _ in range(n_steps - 1):
        row = rt.step_batch([toks[-1]], [state])[0][0, -1]
        toks.append(_argmax(row))
    return toks


def _paged_decode(session: _BaseBatchedSession, slot: int, prompt_ids: list[int], n_steps: int) -> list[int]:
    row = session.admit(slot, prompt_ids)
    toks = [_argmax(row)]
    for _ in range(n_steps - 1):
        row = session.step_batch({slot: toks[-1]})[slot]
        toks.append(_argmax(row))
    return toks


def run() -> None:
    ok = True
    rt = _StubHybrid(seed=0)
    prefix = list(range(10, 18))          # 8 tokens = 2 full blocks (BLOCK=4) — the shared prefix
    suffix = [70, 71, 72]                 # 3-token private suffix
    full = prefix + suffix
    n_steps = 6

    # discrete reference: full prefill + decode, from scratch
    discrete_tokens = _discrete_decode(rt, full, n_steps)

    # paged: a shared manager + recurrent cache; seed the prefix with seq A (concurrent), then admit B
    # which must reuse A's prefix blocks + restore the recurrent boundary and prefill only the suffix.
    mgr = PagedKVCacheManager(num_layers=1, block_size=BLOCK, max_blocks=128,
                              group_size=GROUP, bits=8, quantized=True, model_name="equiv-test")
    rec = RecurrentPrefixCache(block_size=BLOCK, model_name="equiv-test", capacity=128)
    session = _BaseBatchedSession(runtime=rt, capacity=2, manager=mgr, rec_cache=rec)

    _seed = session.admit(0, prefix)      # seq A: stores prefix KV blocks + recurrent boundary snapshots
    paged_tokens = _paged_decode(session, 1, full, n_steps)   # seq B: reuse + suffix-only prefill

    st = mgr.get_stats()
    rst = rec.get_stats()
    tokens_eq = paged_tokens == discrete_tokens
    reused_prefix = st.prefix_hit_tokens == len(prefix)        # B reused exactly the 8-token prefix
    rec_restored = rst.snapshot_hits >= 1                      # B restored the recurrent boundary
    # decode crossed a block boundary (B prefilled to 11; pos 12 lands mid-decode) -> a fresh snapshot
    decode_snapshot = rst.snapshot_stores >= (len(prefix) // BLOCK) + 1
    stats_live = session.prefix_cache_enabled and session.get_cache_stats() is not None

    ok = tokens_eq and reused_prefix and rec_restored and decode_snapshot and stats_live
    print(f"  paged_tokens={paged_tokens}")
    print(f"  discrete_tokens={discrete_tokens}")
    print(f"  [{'OK' if tokens_eq else 'FAIL'}] paged==discrete decode (rule-4 bar)")
    print(f"  [{'OK' if reused_prefix else 'FAIL'}] prefix reused: hit_tokens={st.prefix_hit_tokens} "
          f"(exp {len(prefix)}) hit_blocks={st.prefix_hit_blocks}")
    print(f"  [{'OK' if rec_restored else 'FAIL'}] recurrent boundary restored: "
          f"snapshot_hits={rst.snapshot_hits}")
    print(f"  [{'OK' if decode_snapshot else 'FAIL'}] decode-boundary snapshot stored: "
          f"snapshot_stores={rst.snapshot_stores}")
    print(f"  [{'OK' if stats_live else 'FAIL'}] stats live: prefix_cache_enabled="
          f"{session.prefix_cache_enabled}")
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
