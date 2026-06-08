"""Model-free gate: Nemotron tree-spec over a PAGED state == over the discrete state (#158-160 M3).

Nemotron's decode state is a ``(caches, ssm, conv)`` TRIPLE (per-attention-layer KV caches + per-mamba
recurrence), not a single cache object — so the tree-spec-over-paged wiring differs from DSV4's:

  * :func:`quanta.nemotron.batched_runtime.replicate_state` gained a PAGED branch — when the KV slots
    are paged views it forks the WHOLE sequence B ways (``PagedKVCacheManager.replicate``) and re-points
    each layer's view onto its fork (``PagedKVCacheView.rebind``), instead of the discrete
    ``KVCache._copy``; the Mamba ``(ssm, conv)`` is carried by the same per-replica list-spine clone.
  * the spec loop drives the paged manager lifecycle (``advance`` before a forward writes, ``commit``
    after, ``free`` on a discarded replica) via the ``(manager, seq)`` recovered from the paged views
    (:func:`quanta.paged.manager_seq_of`) — there is no cache object to carry the hooks.
  * a ``make_state`` factory threaded through ``spec_generate{,_k,_tree}`` (default ``None`` = the
    discrete path, byte-identical — rule 4) lets the serving layer pass a paged triple.

This drives the REAL paged KV lifecycle (k/v ``update`` into block pools, COW fork on ``replicate``,
paged ``offset``/``truncate``, ``free``) through the FULL tree-spec loop with a stub main model. The
stub's greedy (``greedy(t) = t + STEP``) is **latent-independent**, so paged == discrete token streams
iff the spec loop drives the triple's paged lifecycle exactly as the discrete one.

Asserts (batched ∈ {False, True} × MTP ∈ {perfect, wrong}, W4D2 batched, width-1 chain, eos, fresh-guard).

    uv run python -m parity.nemotron_spec_paged_test
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from quanta.nemotron.attention import KVCache
from quanta.nemotron.spec import spec_generate_k, spec_generate_tree
from quanta.paged import PagedKVCacheManager

VOCAB = 64
HIDDEN = 8
NL = 4               # stub decoder layers (all attention → every KV slot is paged)
STEP = 3             # deterministic chain: greedy(t) = t + STEP
EOS = 40
MAXN = 16
N_KV = 2
HEAD_DIM = 128       # one int8-g128 group per token (real Nemotron KV geometry)
GROUP = 128
BLOCK = 4

EMBED = mx.eye(VOCAB)             # one-hot rows — embed[t] has argmax t (the MTP reads token_emb)
HEAD = mx.zeros((VOCAB, HIDDEN))


def _greedy_next(t: int) -> int:
    return t + STEP


def _row(tok: int) -> mx.array:
    return mx.where(mx.arange(VOCAB) == tok, 30.0, -30.0)


# ---------------------------------------------------------------------------
# Stub main model driving EITHER a discrete (caches, ssm, conv) triple or a paged one.
# ---------------------------------------------------------------------------
class _PagedAwareNemotronStub:
    """Stub Nemotron model exposing single-stream ``__call__`` + ``batch_step``, each writing (zero)
    k/v to every attention layer's cache via the real ``update(k, v)`` protocol (discrete ``KVCache`` or
    paged ``PagedKVCacheView`` — same signature). ``greedy(t) = t + STEP`` is latent-independent, so
    identical token streams across state kinds isolate the spec loop's paged-lifecycle handling. The
    paged manager ``advance``/``commit`` is driven by the SPEC LOOP bracket (not here), exactly as the
    real model forward only writes."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(num_hidden_layers=NL)
        self.num_layers = NL

    def make_caches(self, *, max_rollback: int = 8):
        # default (make_state=None) discrete path: a per-layer KVCache list + empty mamba state.
        caches = [KVCache(group_size=GROUP, max_rollback=max_rollback) for _ in range(NL)]
        return caches, [None] * NL, [None] * NL

    @staticmethod
    def _write(caches, n: int) -> None:
        """Append ``n`` tokens of (zero) k/v to every layer via ``update`` — as the real model forward
        does. For paged views the spec loop has already ``advance``d the positions (begin_forward)."""
        k = mx.zeros((1, N_KV, n, HEAD_DIM))
        v = mx.zeros((1, N_KV, n, HEAD_DIM))
        for c in caches:
            if c is not None:
                c.update(k, v)

    def __call__(self, token_ids, *, caches=None, ssm=None, conv=None, offset=0, capture_layers=None):
        del ssm, conv, offset
        ids = [int(x) for x in (token_ids.tolist() if isinstance(token_ids, mx.array) else token_ids)]
        if caches is not None:
            self._write(caches, len(ids))
        t = len(ids)
        logits = mx.stack([_row(_greedy_next(tok)) for tok in ids])[None]        # [1,T,vocab]
        if not capture_layers:
            return logits
        last = max(capture_layers)
        feat = mx.broadcast_to(mx.arange(t, dtype=mx.float32)[:, None], (t, HIDDEN))
        return logits, {last: feat}

    def batch_step(self, tokens, *, replicas, offset, capture_layer=None):
        b = len(tokens)
        if len(replicas) != b:
            raise ValueError(f"batch_step len mismatch: replicas={len(replicas)} tokens={b}")
        for s, (caches_s, _ssm_s, _conv_s) in enumerate(replicas):     # offset precondition (rule 6)
            for li, c in enumerate(caches_s):
                if c is not None and c.offset != offset:
                    raise ValueError(f"batch_step replicas[{s}].caches[{li}].offset={c.offset} != {offset}")
        for caches_s, _ssm_s, _conv_s in replicas:                     # advance each replica by one token
            self._write(caches_s, 1)
        rows = mx.stack([_row(_greedy_next(int(t))) for t in tokens])           # [B,vocab]
        logits = rows[:, None]                                                   # [B,1,vocab]
        if capture_layer is None:
            return logits, None
        feat = mx.broadcast_to(mx.array(0.0, dtype=mx.float32), (b, 1, HIDDEN))
        return logits, feat


# ---------------------------------------------------------------------------
# MTP stubs (Nemotron signature: mtp(prev_hidden, token_emb, head, *, return_hidden)).
# ---------------------------------------------------------------------------
def _dummy_hidden() -> mx.array:
    return mx.zeros((1, 1, HIDDEN), dtype=mx.float32)


def _logits_top_w(greedy_tok: int, other_toks: list[int]) -> mx.array:
    arr = mx.full((VOCAB,), -100.0)
    arr = mx.where(mx.arange(VOCAB) == greedy_tok, 100.0, arr)
    for i, tok in enumerate(other_toks):
        arr = mx.where(mx.arange(VOCAB) == tok, 90.0 - 5.0 * i, arr)
    return arr[None, None]


def _wrongs_for(parent_tok: int, width: int, greedy_tok: int) -> list[int]:
    wrongs: list[int] = []
    candidate = (parent_tok + 1) % VOCAB
    while len(wrongs) < width - 1:
        if candidate != greedy_tok and candidate not in wrongs:
            wrongs.append(candidate)
        candidate = (candidate + 1) % VOCAB
    return wrongs


class _PerfectLeftmostMTP:
    def __init__(self, width: int) -> None:
        self.width = width

    def __call__(self, prev_hidden, token_emb, head, *, return_hidden=False):
        parent = int(mx.argmax(token_emb[0, 0]).item())
        greedy = _greedy_next(parent) % VOCAB
        return _logits_top_w(greedy, _wrongs_for(parent, self.width, greedy)), _dummy_hidden()


class _WrongAllMTP:
    def __init__(self, width: int) -> None:
        self.width = width

    def __call__(self, prev_hidden, token_emb, head, *, return_hidden=False):
        parent = int(mx.argmax(token_emb[0, 0]).item())
        greedy = _greedy_next(parent) % VOCAB
        wrongs = _wrongs_for(parent, self.width + 1, greedy)[: self.width]
        return _logits_top_w(wrongs[0], wrongs[1:]), _dummy_hidden()


def _paged_make_state():
    """Fresh paged-triple factory backed by a private k/v manager — the serving wiring of
    :meth:`NemotronBatchedResidentModel.make_paged_state` (each attention layer -> a paged view)."""
    mgr = PagedKVCacheManager(num_layers=NL, block_size=BLOCK, max_blocks=2048, group_size=GROUP,
                              bits=8, quantized=True, model_name="nemo-spec-paged")

    def make_state(*, max_rollback: int = 1):
        del max_rollback                                  # paged truncate handles rollback (no ring)
        seq = mgr.new_sequence()
        caches = [mgr.view(seq, i) for i in range(NL)]
        return caches, [None] * NL, [None] * NL

    return make_state


# ============================================================================
# Tests
# ============================================================================
def _run_pair(mtp_factory, *, width, depth, batched, eos_id=None, prompt=(2, 5, 7)):
    disc, st_d = spec_generate_tree(_PagedAwareNemotronStub(), mtp_factory(), EMBED, HEAD, list(prompt),
                                    width=width, depth=depth, max_new=MAXN, eos_id=eos_id, batched=batched)
    paged, st_p = spec_generate_tree(_PagedAwareNemotronStub(), mtp_factory(), EMBED, HEAD, list(prompt),
                                     width=width, depth=depth, max_new=MAXN, eos_id=eos_id,
                                     batched=batched, make_state=_paged_make_state())
    return disc, st_d, paged, st_p


def test_paged_equals_discrete_w2d2() -> None:
    for batched in (False, True):
        for name, mtp in (("perfect", lambda: _PerfectLeftmostMTP(2)), ("wrong", lambda: _WrongAllMTP(2))):
            disc, st_d, paged, st_p = _run_pair(mtp, width=2, depth=2, batched=batched)
            assert disc == paged, f"[{name} batched={batched}] paged != discrete: {paged} vs {disc}"
            assert st_d["mean_accept"] == st_p["mean_accept"] and st_d["rounds"] == st_p["rounds"]
            print(f"[OK] W2D2 {name} batched={batched}: paged == discrete "
                  f"(n={len(paged)}, mean_accept={st_p['mean_accept']:.2f})")


def test_paged_equals_discrete_w4d2_batched() -> None:
    disc, _st_d, paged, st_p = _run_pair(lambda: _PerfectLeftmostMTP(4), width=4, depth=2, batched=True)
    assert disc == paged, f"W4D2 paged != discrete: {paged} vs {disc}"
    assert st_p["paths_per_round"] == 16
    print(f"[OK] W4D2 batched (B=16 forked seqs): paged == discrete, mean_accept={st_p['mean_accept']:.2f}")


def test_paged_equals_discrete_width1_chain() -> None:
    disc, _ = spec_generate_tree(_PagedAwareNemotronStub(), _PerfectLeftmostMTP(1), EMBED, HEAD, [2, 5, 7],
                                 width=1, depth=3, max_new=MAXN)
    paged, _ = spec_generate_tree(_PagedAwareNemotronStub(), _PerfectLeftmostMTP(1), EMBED, HEAD, [2, 5, 7],
                                  width=1, depth=3, max_new=MAXN, make_state=_paged_make_state())
    k_paged, _ = spec_generate_k(_PagedAwareNemotronStub(), _PerfectLeftmostMTP(1), EMBED, HEAD, [2, 5, 7],
                                 k=3, max_new=MAXN, make_state=_paged_make_state())
    assert disc == paged == k_paged, f"width1 chain: disc={disc} paged={paged} k={k_paged}"
    print(f"[OK] width=1 chain (k=3): paged == discrete == spec_generate_k(make_state) (n={len(paged)})")


def test_paged_equals_discrete_eos() -> None:
    disc, _, paged, _ = _run_pair(lambda: _PerfectLeftmostMTP(2), width=2, depth=2, batched=True, eos_id=EOS)
    assert disc == paged, f"eos paged != discrete: {paged} vs {disc}"
    assert paged and paged[-1] == EOS and EOS not in paged[:-1]
    print(f"[OK] eos under paged: bit-identical to discrete (final={paged[-1]} = EOS)")


def test_fresh_state_guard() -> None:
    """A make_state returning a PRE-FILLED triple fails loud (rule 6) — the loop owns the prefill."""
    mgr = PagedKVCacheManager(num_layers=NL, block_size=BLOCK, max_blocks=64, group_size=GROUP,
                              bits=8, quantized=True, model_name="nemo-prefilled")

    def bad_make_state(*, max_rollback: int = 1):
        del max_rollback
        seq = mgr.new_sequence()
        caches = [mgr.view(seq, i) for i in range(NL)]
        mgr.advance(seq, [0])                                    # pre-fill ONE token (illegal)
        for c in caches:
            c.update(mx.zeros((1, N_KV, 1, HEAD_DIM)), mx.zeros((1, N_KV, 1, HEAD_DIM)))
        mgr.commit(seq)
        return caches, [None] * NL, [None] * NL

    raised = False
    try:
        spec_generate_tree(_PagedAwareNemotronStub(), _PerfectLeftmostMTP(2), EMBED, HEAD, [2, 5, 7],
                           width=2, depth=2, max_new=MAXN, make_state=bad_make_state)
    except ValueError as e:
        raised = "fresh" in str(e).lower() and "offset" in str(e).lower()
    assert raised, "make_state returning a pre-filled (offset>0) state must fail loud"
    print("[OK] fresh-state guard: a pre-filled make_state triple fails loud (offset must be 0)")


def main() -> int:
    tests = [
        test_paged_equals_discrete_w2d2,
        test_paged_equals_discrete_w4d2_batched,
        test_paged_equals_discrete_width1_chain,
        test_paged_equals_discrete_eos,
        test_fresh_state_guard,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    if failures:
        return 1
    print("PASS — Nemotron tree-spec over a paged (caches, ssm, conv) triple is bit-identical to the "
          "discrete path (replicate_state fork + manager lifecycle wired; #158-160 M3 spec-loop half)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
