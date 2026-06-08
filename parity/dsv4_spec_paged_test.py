"""Model-free gate: DSV4 tree-spec running over a PAGED cache == over the discrete cache (#158-160 M1).

M0 (``parity/dsv4_paged_replicate_test.py``) proved the cache half — :class:`PagedDSV4Cache.replicate`
forks the sequence COW and satisfies the batched tree-spec ``replicate(B)`` contract. M1 wires the
spec LOOP to use it: :func:`quanta.dsv4.spec.spec_generate_tree` (and the chain variants) gained a
``make_state`` factory (default ``None`` = the discrete path, byte-identical — rule 4) the serving
layer passes to build a paged :class:`~quanta.dsv4.decode.PagedDSV4Cache` instead of the discrete
per-stream :class:`~quanta.dsv4.decode.DSV4Cache`.

This drives the REAL paged cache lifecycle (manager block alloc, COW fork on ``replicate``, paged
``offset``/``truncate``, commit-forward) through the FULL tree-spec loop with a stub main model — no
checkpoint, no GPU, a few KB of tensors. The stub's greedy (``greedy(t) = t + STEP``) is
**latent-independent**, so the token stream depends ONLY on whether the spec loop drives the cache
protocol correctly. Therefore:

    spec_generate_tree(make_state=<paged>)  ==  spec_generate_tree()  [discrete]   ⟺   the loop
    handles the paged ``replicate``/``truncate``/``offset`` exactly as the discrete cache.

Asserts (for ``batched`` ∈ {False, True} × MTP ∈ {perfect, wrong}):

  A. **paged == discrete** — bit-identical token list + ``mean_accept``/``rounds``/``max_accept``.
  B. **width=1 chain** — the ``width=1`` short-circuit (→ ``spec_generate_k``) forwards ``make_state``
     too: paged == discrete on a pure chain.
  C. **fresh-cache guard** — a ``make_state`` returning a pre-filled cache fails loud (rule 6): the
     spec loop owns the prefill, a non-zero offset would double-count positions.

    uv run python -m parity.dsv4_spec_paged_test
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from quanta.dsv4.decode import DSV4Cache, paged_cache
from quanta.dsv4.spec import spec_generate_k, spec_generate_tree
from quanta.paged import PagedKVCacheManager

VOCAB = 64
HC = 4               # DSV4 hyper-connection residual dim (matches dsv4_batched_tree_verify_test)
DIM = 8
NL = 3               # stub "decoder layers" — only cfg.num_hidden_layers matters to the spec loop
STEP = 3             # deterministic chain: greedy(t) = t + STEP
EOS = 40
MAXN = 16
HEAD_DIM = 128       # one int8-g128 group per token — the real DSV4 latent geometry
GROUP = 128
BLOCK = 4


def _greedy_next(t: int) -> int:
    return t + STEP


def _row(tok: int) -> mx.array:
    """A logit row over VOCAB with a clear argmax on ``tok``."""
    return mx.where(mx.arange(VOCAB) == tok, 30.0, -30.0)


# ---------------------------------------------------------------------------
# A stub main model that drives EITHER a discrete DSV4Cache or a real PagedDSV4Cache.
# ---------------------------------------------------------------------------
class _PagedAwareStub:
    """Stub main model exposing both single-stream ``__call__`` and ``batch_step`` (the surfaces the
    spec loop consumes), each writing (zero) latent to the cache via its real protocol — paged
    (``manager.advance`` → per-layer ``append_kv`` → ``manager.commit``) or discrete (``append_kv``
    only). ``greedy(t) = t + STEP`` is latent-independent, so identical token streams across cache
    types isolate the spec loop's cache-protocol handling."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(num_hidden_layers=NL)
        self.num_layers = NL

    def make_caches(self, *, max_rollback: int = 1) -> DSV4Cache:
        # the default (make_state=None) discrete path uses this; quantized matches the paged codec.
        return DSV4Cache(NL, quantized=True, group_size=GROUP, max_rollback=max_rollback)

    @staticmethod
    def _write(cache, n: int) -> None:
        """Append ``n`` tokens of (zero) latent to every layer — EXACTLY as the real model forward
        does (one ``[1, n, head_dim]`` ``append_kv`` per layer). The paged manager lifecycle
        (``advance`` before, ``commit`` after) is driven by the SPEC LOOP's begin/end_forward bracket
        around this forward — NOT here — so the stub mimics the real model faithfully (a paged cache
        whose ``append_kv`` hits an un-advanced position would crash, exactly as the real model did
        before the bracket landed)."""
        lat = mx.zeros((1, n, HEAD_DIM))
        for L in range(NL):
            cache.layers[L].append_kv(lat)

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        ids = [int(x) for x in (token_ids.tolist() if isinstance(token_ids, mx.array) else token_ids)]
        if caches is not None:
            self._write(caches, len(ids))
        t = len(ids)
        logits = mx.stack([_row(_greedy_next(tok)) for tok in ids])[None]        # [1,T,vocab]
        if not capture_layers:
            return logits
        last = max(capture_layers)
        feat = mx.broadcast_to(mx.arange(t, dtype=mx.float32)[:, None, None], (t, HC, DIM))
        return logits, {last: feat}

    def batch_step(self, tokens, *, caches, offset, capture_layer=None):
        b = len(tokens)
        if len(caches) != b:
            raise ValueError(f"batch_step len mismatch: caches={len(caches)} tokens={b}")
        for s, c in enumerate(caches):
            if c.offset != offset:
                raise ValueError(f"batch_step caches[{s}].offset={c.offset} != offset={offset}")
        for c in caches:
            self._write(c, 1)                                                    # advance each replica
        rows = mx.stack([_row(_greedy_next(int(t))) for t in tokens])           # [B,vocab]
        logits = rows[:, None]                                                   # [B,1,vocab]
        if capture_layer is None:
            return logits, None
        feat = mx.broadcast_to(mx.array(0.0, dtype=mx.float32), (b, 1, HC, DIM))
        return logits, feat


# ---------------------------------------------------------------------------
# MTP stubs (perfect-leftmost / wrong-all), copied compactly from the batched-verify gate.
# ---------------------------------------------------------------------------
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
    """TOP-1 child == main-greedy → the leftmost path accepts all ``depth`` drafts."""

    def __init__(self, width: int) -> None:
        self.width = width

    def __call__(self, prev_hidden, next_ids, embed, head, *, return_hidden=False):
        parent = int(next_ids[0, 0].item())
        greedy = _greedy_next(parent) % VOCAB
        logits = _logits_top_w(greedy, _wrongs_for(parent, self.width, greedy))
        return (logits, mx.zeros((1, 1, HC, DIM))) if return_hidden else logits


class _WrongAllMTP:
    """TOP-W are ALL non-greedy → every path rejects on the first draft (mean_accept == 1)."""

    def __init__(self, width: int) -> None:
        self.width = width

    def __call__(self, prev_hidden, next_ids, embed, head, *, return_hidden=False):
        parent = int(next_ids[0, 0].item())
        greedy = _greedy_next(parent) % VOCAB
        wrongs = _wrongs_for(parent, self.width + 1, greedy)[: self.width]
        logits = _logits_top_w(wrongs[0], wrongs[1:])
        return (logits, mx.zeros((1, 1, HC, DIM))) if return_hidden else logits


# ---------------------------------------------------------------------------
# Paged make_state factory
# ---------------------------------------------------------------------------
def _paged_make_state():
    """A fresh paged-cache factory backed by a private manager (single-stream DSV4 latent codec).
    The spec loop calls it once per run with ``max_rollback`` — we build a new sequence + paged
    cache exactly as :meth:`DSV4BatchedResidentModel.make_paged_state` would for a serving request."""
    mgr = PagedKVCacheManager(num_layers=NL, block_size=BLOCK, max_blocks=512, group_size=GROUP,
                              bits=8, quantized=True, single_stream=True, model_name="spec-paged-test")

    def make_state(*, max_rollback: int = 1):
        return paged_cache(mgr, mgr.new_sequence(), NL, quantized=True, group_size=GROUP,
                           max_rollback=max_rollback)

    return make_state


def _embed_head():
    return mx.zeros((VOCAB, HC * DIM)), mx.zeros((VOCAB, HC * DIM))


# ============================================================================
# Tests
# ============================================================================
def _run_pair(mtp_factory, *, width: int, depth: int, batched: bool, eos_id=None, prompt=(2, 5, 7)):
    embed, head = _embed_head()
    disc, st_d = spec_generate_tree(_PagedAwareStub(), mtp_factory(), embed, head, list(prompt),
                                    width=width, depth=depth, max_new=MAXN, eos_id=eos_id,
                                    batched=batched)
    paged, st_p = spec_generate_tree(_PagedAwareStub(), mtp_factory(), embed, head, list(prompt),
                                     width=width, depth=depth, max_new=MAXN, eos_id=eos_id,
                                     batched=batched, make_state=_paged_make_state())
    return disc, st_d, paged, st_p


def test_paged_equals_discrete_w2d2() -> None:
    """(W=2, D=2) — paged tree-spec bit-identical to discrete, for both batched flags + both MTPs."""
    for batched in (False, True):
        for name, mtp in (("perfect", lambda: _PerfectLeftmostMTP(2)), ("wrong", lambda: _WrongAllMTP(2))):
            disc, st_d, paged, st_p = _run_pair(mtp, width=2, depth=2, batched=batched)
            assert disc == paged, f"[{name} batched={batched}] paged != discrete: {paged} vs {disc}"
            assert st_d["mean_accept"] == st_p["mean_accept"], f"[{name} batched={batched}] mean_accept"
            assert st_d["rounds"] == st_p["rounds"] and st_d["max_accept"] == st_p["max_accept"]
            print(f"[OK] W2D2 {name} batched={batched}: paged == discrete "
                  f"(n={len(paged)}, mean_accept={st_p['mean_accept']:.2f})")


def test_paged_equals_discrete_w4d2_batched() -> None:
    """(W=4, D=2) batched — B=16 paged replicas (16 COW forks/round) bit-identical to discrete."""
    disc, st_d, paged, st_p = _run_pair(lambda: _PerfectLeftmostMTP(4), width=4, depth=2, batched=True)
    assert disc == paged, f"W4D2 paged != discrete: {paged} vs {disc}"
    assert st_p["paths_per_round"] == 16
    print(f"[OK] W4D2 batched (B=16): paged == discrete, mean_accept={st_p['mean_accept']:.2f}")


def test_paged_equals_discrete_width1_chain() -> None:
    """width=1 short-circuits to spec_generate_k and forwards make_state — paged chain == discrete."""
    embed, head = _embed_head()
    disc, _ = spec_generate_tree(_PagedAwareStub(), _PerfectLeftmostMTP(1), embed, head, [2, 5, 7],
                                 width=1, depth=2, max_new=MAXN)
    paged, _ = spec_generate_tree(_PagedAwareStub(), _PerfectLeftmostMTP(1), embed, head, [2, 5, 7],
                                  width=1, depth=2, max_new=MAXN, make_state=_paged_make_state())
    # And direct spec_generate_k with make_state matches too (the entry the short-circuit calls).
    k_paged, _ = spec_generate_k(_PagedAwareStub(), _PerfectLeftmostMTP(1), embed, head, [2, 5, 7],
                                 k=2, max_new=MAXN, make_state=_paged_make_state())
    assert disc == paged == k_paged, f"width1 chain: disc={disc} paged={paged} k={k_paged}"
    print(f"[OK] width=1 chain: paged == discrete == spec_generate_k(make_state) (n={len(paged)})")


def test_paged_equals_discrete_eos() -> None:
    """eos terminates the paged stream at the first emitted eos (inclusive) — same as discrete."""
    disc, _, paged, _ = _run_pair(lambda: _PerfectLeftmostMTP(2), width=2, depth=2, batched=True,
                                  eos_id=EOS)
    assert disc == paged, f"eos paged != discrete: {paged} vs {disc}"
    assert paged and paged[-1] == EOS and EOS not in paged[:-1]
    print(f"[OK] eos under paged: bit-identical to discrete (final={paged[-1]} = EOS)")


def test_fresh_cache_guard() -> None:
    """A make_state returning a PRE-FILLED cache fails loud (rule 6) — the loop owns the prefill."""
    embed, head = _embed_head()
    mgr = PagedKVCacheManager(num_layers=NL, block_size=BLOCK, max_blocks=64, group_size=GROUP,
                              bits=8, quantized=True, single_stream=True, model_name="prefilled-test")

    def bad_make_state(*, max_rollback: int = 1):
        cache = paged_cache(mgr, mgr.new_sequence(), NL, quantized=True, group_size=GROUP,
                            max_rollback=max_rollback)
        mgr.advance(cache._seq, [0])                     # pre-fill ONE token → offset 1 (illegal)
        for L in range(NL):
            cache.layers[L].append_kv(mx.zeros((1, 1, HEAD_DIM)))
        mgr.commit(cache._seq)
        return cache

    raised = False
    try:
        spec_generate_tree(_PagedAwareStub(), _PerfectLeftmostMTP(2), embed, head, [2, 5, 7],
                           width=2, depth=2, max_new=MAXN, make_state=bad_make_state)
    except ValueError as e:
        raised = "fresh" in str(e).lower() and "offset" in str(e).lower()
    assert raised, "make_state returning a pre-filled (offset>0) cache must fail loud"
    print("[OK] fresh-cache guard: a pre-filled make_state cache fails loud (offset must be 0)")


def main() -> int:
    tests = [
        test_paged_equals_discrete_w2d2,
        test_paged_equals_discrete_w4d2_batched,
        test_paged_equals_discrete_width1_chain,
        test_paged_equals_discrete_eos,
        test_fresh_cache_guard,
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
    print("PASS — DSV4 tree-spec over a paged cache is bit-identical to the discrete path "
          "(make_state factory wired; #158-160 M1 spec-loop half)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
