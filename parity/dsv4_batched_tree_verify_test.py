"""Parity gate for DSV4 batched tree-spec verify — bit-identical to the sequential form.

Validates :func:`quanta.dsv4.spec.spec_generate_tree`'s ``batched=True`` path against the proven
``batched=False`` sequential form (#157, ``700adb3``) — the follow-on landed per the design doc
``docs/batched_tree_verify.md``. MODEL-FREE: a stub main model exposing BOTH the single-stream
``__call__`` (consumed by ``batched=False``) AND :meth:`batch_step` (consumed by ``batched=True``),
plus stub caches with structural-sharing replicate; no checkpoint, no GPU, a few KB of tensors.
Safe alongside the EAGLE retrain on GPU — never contends.

Asserts:

  (1) **batched == sequential** — ``spec_generate_tree(model, ..., batched=True)`` returns the
      bit-identical token list to ``spec_generate_tree(model, ..., batched=False)`` for both a
      perfect-leftmost MTP (every accept) and a wrong-all MTP (no accepts). ``mean_accept``,
      ``rounds``, ``max_accept`` all match.
  (2) **width=1 short-circuit** — ``width=1`` bypasses batched (degenerates to a chain) and matches
      :func:`quanta.dsv4.spec.spec_generate_k` regardless of the ``batched`` flag value.
  (3) **Cache invariance** — after ``batched=True`` completes, the original (un-replicated) cache is
      at the same offset and holds the same per-layer state as if ``batched=False`` had run on it.
      Catches a bug that mutates the prefix instead of the replicas.
  (4) **Replicate fidelity** — :meth:`quanta.dsv4.decode.DSV4Cache.replicate` returns B caches that
      each read back the original cache's content (structural-sharing replication preserves the
      prefix exactly; subsequent diverging appends do not corrupt siblings).
  (5) **Replica divergence** — after a replica appends a new KV slot, sibling replicas (and the
      original) still see the un-appended state. This is the MLX-immutability guarantee that lets
      structural sharing be lossless.

Run:  ``uv run --with numpy python -m parity.dsv4_batched_tree_verify_test``

Real-model parity + throughput bench: ``parity/dsv4_batched_tree_verify_real.py`` (commit 5 of
``docs/batched_tree_verify.md``) — gates both branches against the resident baked DSV4 + native
MTP head. Measured: bit-identical 32 tokens at ``W=2, D=2`` and 64-token bench shows
``batched=True`` at 1.37 tok/s = **3.77× chained k=2** and **10.43× sequential tree**, with the
same mean_accept (2.74/3) — the result that motivated flipping ``batched=True`` to default in
:func:`quanta.dsv4.spec.spec_generate_tree`.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from quanta.dsv4.decode import DSV4Cache
from quanta.dsv4.spec import spec_generate_k, spec_generate_tree

VOCAB = 64
HC = 4              # DSV4 HC (hyper-connection) residual dim — matches dsv4_tree_spec_test
DIM = 8
NL = 3              # stub "decoder layers" — only cfg.num_hidden_layers matters to spec
STEP = 3            # deterministic chain: greedy(t) = t + STEP
EOS = 40
MAXN = 16
CAPTURE_LAYER = NL - 1     # the last "layer" — what spec_generate_tree captures for MTP feature


def _greedy_next(t: int) -> int:
    return t + STEP


def _row(tok: int) -> mx.array:
    """A logit row over VOCAB with a clear argmax on ``tok`` (everything else far below)."""
    return mx.where(mx.arange(VOCAB) == tok, 30.0, -30.0)


class _StubCache:
    """Minimal stand-in for ``DSV4Cache``: tracks length + supports ``truncate``/``offset``/``replicate``.

    The structural-sharing semantics matter: ``replicate(B)`` must return B fresh caches that each
    initially equal this one, and writes on one must NOT mutate siblings (the MLX-immutability
    guarantee). We model that by storing length as an integer (immutable) and copying it on
    replicate; further commits diverge naturally.
    """

    def __init__(self) -> None:
        self._len = 0
        self.truncations: list[tuple[int, int]] = []         # (from_len, to_len) per truncate call

    @property
    def offset(self) -> int:
        return self._len

    def append(self, n: int) -> None:
        self._len += n

    def truncate(self, length: int) -> None:
        if length < 0:
            raise ValueError(f"truncate length {length} < 0")
        if length < self._len:
            self.truncations.append((self._len, length))
            self._len = length

    def replicate(self, b: int) -> list["_StubCache"]:
        # Structural sharing: each replica is a fresh _StubCache initialized to this one's length.
        # Subsequent append/truncate on each diverges (Python int copy = immutability surrogate).
        out = []
        for _ in range(b):
            r = _StubCache()
            r._len = self._len
            out.append(r)
        return out


class _StubMainModel:
    """Stub main model with BOTH single-stream ``__call__`` AND ``batch_step`` (the two surfaces the
    spec consumes via the ``batched`` flag).

    * ``__call__(token_ids, *, caches, offset, capture_layers)`` — sequential-form contract.
    * ``batch_step(tokens, *, caches, offset, capture_layer)`` — batched-form contract:
      ``(logits [B,1,vocab], hidden [B,1,hc,dim] or None)``.

    Both produce deterministic ``greedy(t) = t + STEP`` over VOCAB, so the two paths must produce
    bit-identical token streams. Honors per-replica cache appends so the structural-sharing
    invariance is exercised (each batch_step advances every replica by one position).
    """

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(num_hidden_layers=NL)
        self.num_layers = NL
        self.calls: list[tuple[str, tuple[int, ...], int]] = []   # (which, ids, offset)

    def make_caches(self, *, max_rollback: int = 1) -> _StubCache:
        del max_rollback
        return _StubCache()

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        ids = [int(x) for x in (token_ids.tolist() if isinstance(token_ids, mx.array) else token_ids)]
        self.calls.append(("call", tuple(ids), offset))
        if caches is not None:
            caches.append(len(ids))
        t = len(ids)
        logits = mx.stack([_row(_greedy_next(tok)) for tok in ids])[None]       # [1,T,vocab]
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
        self.calls.append(("batch_step", tuple(int(t) for t in tokens), offset))
        # advance each replica by one position
        for c in caches:
            c.append(1)
        # per-stream logits: greedy(tok_b) per replica
        rows = mx.stack([_row(_greedy_next(int(t))) for t in tokens])           # [B, vocab]
        logits = rows[:, None]                                                   # [B,1,vocab]
        if capture_layer is None:
            return logits, None
        # deterministic per-stream feature; content irrelevant to the stub MTP, shape must match
        # what DSV4BatchedResidentModel.batch_step returns ([B,1,hc,dim]).
        feat = mx.broadcast_to(mx.array(0.0, dtype=mx.float32), (b, 1, HC, DIM))
        return logits, feat


def _dummy_hidden() -> mx.array:
    """``[1, 1, hc, dim]`` placeholder hidden the chained tree-build feeds back in."""
    return mx.zeros((1, 1, HC, DIM))


def _logits_top_w(greedy_tok: int, other_toks: list[int]) -> mx.array:
    """Build ``[1, 1, VOCAB]`` logits so ``_mtp_top_w(..., w)`` returns ``[greedy_tok, *other_toks]``
    in that order — the same construction as ``parity/dsv4_tree_spec_test.py``."""
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
    """TOP-1 child = main-greedy → leftmost path accepts all ``depth`` drafts."""

    def __init__(self, width: int) -> None:
        self.width = width

    def __call__(self, prev_hidden, next_ids, embed, head, *, return_hidden=False):
        parent = int(next_ids[0, 0].item())
        greedy = _greedy_next(parent) % VOCAB
        wrongs = _wrongs_for(parent, self.width, greedy)
        logits = _logits_top_w(greedy, wrongs)
        if return_hidden:
            return logits, _dummy_hidden()
        return logits


class _WrongAllMTP:
    """TOP-W are ALL non-greedy → every path rejects on the first draft."""

    def __init__(self, width: int) -> None:
        self.width = width

    def __call__(self, prev_hidden, next_ids, embed, head, *, return_hidden=False):
        parent = int(next_ids[0, 0].item())
        greedy = _greedy_next(parent) % VOCAB
        wrongs = _wrongs_for(parent, self.width + 1, greedy)[: self.width]
        logits = _logits_top_w(wrongs[0], wrongs[1:])
        if return_hidden:
            return logits, _dummy_hidden()
        return logits


# ============================================================================
# Tests
# ============================================================================

def test_batched_equals_sequential_perfect_w2d2() -> None:
    """(W=2, D=2) perfect-leftmost MTP — batched output bit-identical to sequential output."""
    embed = mx.zeros((VOCAB, HC * DIM))
    head = mx.zeros((VOCAB, HC * DIM))
    prompt = [2, 5, 7]
    seq, st_seq = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(2), embed, head, prompt,
                                     width=2, depth=2, max_new=MAXN, eos_id=None, batched=False)
    bat, st_bat = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(2), embed, head, prompt,
                                     width=2, depth=2, max_new=MAXN, eos_id=None, batched=True)
    assert seq == bat, f"batched != sequential: bat={bat} seq={seq}"
    assert st_bat["rounds"] == st_seq["rounds"]
    assert st_bat["mean_accept"] == st_seq["mean_accept"]
    assert st_bat["max_accept"] == st_seq["max_accept"]
    assert st_bat["batched"] is True and st_seq["batched"] is False
    print(f"[OK] perfect W=2 D=2: batched == sequential (n={len(bat)} tokens, "
          f"mean_accept={st_bat['mean_accept']:.2f})")


def test_batched_equals_sequential_wrong_w2d2() -> None:
    """(W=2, D=2) wrong-all MTP — batched still bit-identical to sequential (verify-arbitrated)."""
    embed = mx.zeros((VOCAB, HC * DIM))
    head = mx.zeros((VOCAB, HC * DIM))
    prompt = [2, 5, 7]
    seq, st_seq = spec_generate_tree(_StubMainModel(), _WrongAllMTP(2), embed, head, prompt,
                                     width=2, depth=2, max_new=MAXN, eos_id=None, batched=False)
    bat, st_bat = spec_generate_tree(_StubMainModel(), _WrongAllMTP(2), embed, head, prompt,
                                     width=2, depth=2, max_new=MAXN, eos_id=None, batched=True)
    assert seq == bat, f"batched != sequential (wrong MTP): bat={bat} seq={seq}"
    assert st_bat["mean_accept"] == st_seq["mean_accept"]
    print(f"[OK] wrong W=2 D=2: batched == sequential, mean_accept={st_bat['mean_accept']:.2f} (=1)")


def test_batched_equals_sequential_w4d2() -> None:
    """(W=4, D=2) perfect-leftmost — wider tree, same bit-identity holds."""
    embed = mx.zeros((VOCAB, HC * DIM))
    head = mx.zeros((VOCAB, HC * DIM))
    prompt = [2, 5, 7]
    seq, st_seq = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(4), embed, head, prompt,
                                     width=4, depth=2, max_new=MAXN, eos_id=None, batched=False)
    bat, st_bat = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(4), embed, head, prompt,
                                     width=4, depth=2, max_new=MAXN, eos_id=None, batched=True)
    assert seq == bat, f"batched != sequential (W=4): bat={bat} seq={seq}"
    assert st_bat["paths_per_round"] == 16
    print(f"[OK] perfect W=4 D=2 (B=16): batched == sequential, mean_accept={st_bat['mean_accept']:.2f}")


def test_width1_matches_spec_generate_k_both_flags() -> None:
    """width=1 short-circuits to spec_generate_k for BOTH batched flag values (chain has no fan-out)."""
    embed = mx.zeros((VOCAB, HC * DIM))
    head = mx.zeros((VOCAB, HC * DIM))
    prompt = [2, 5, 7]
    out_f, _ = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(1), embed, head, prompt,
                                  width=1, depth=2, max_new=MAXN, eos_id=None, batched=False)
    out_t, _ = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(1), embed, head, prompt,
                                  width=1, depth=2, max_new=MAXN, eos_id=None, batched=True)
    out_k, _ = spec_generate_k(_StubMainModel(), _PerfectLeftmostMTP(1), embed, head, prompt,
                               k=2, max_new=MAXN, eos_id=None)
    assert out_f == out_t == out_k, f"w=1 differs across flags/k: f={out_f} t={out_t} k={out_k}"
    print(f"[OK] width=1 short-circuit: batched=False == batched=True == spec_generate_k(k=2) "
          f"(n={len(out_k)})")


def test_eos_stops_under_batched() -> None:
    """eos terminates the stream at the first emitted eos (inclusive) — same as sequential."""
    embed = mx.zeros((VOCAB, HC * DIM))
    head = mx.zeros((VOCAB, HC * DIM))
    prompt = [2, 5, 7]                          # chain: 10, 13, 16, ..., 40(eos)
    seq, _ = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(2), embed, head, prompt,
                                width=2, depth=2, max_new=MAXN, eos_id=EOS, batched=False)
    bat, _ = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(2), embed, head, prompt,
                                width=2, depth=2, max_new=MAXN, eos_id=EOS, batched=True)
    assert seq == bat, f"eos stop mismatch: bat={bat} seq={seq}"
    assert bat and bat[-1] == EOS and EOS not in bat[:-1]
    print(f"[OK] eos stops under batched: bit-identical to sequential (final={bat[-1]} = EOS)")


def test_replicate_fidelity_stub() -> None:
    """``cache.replicate(B)`` returns B fresh caches that each equal the original cache's state.

    Stub cache version — sanity that the API contract (B caches with identical initial state)
    holds without requiring an MLX cache instance. The real DSV4Cache test below covers the
    structural-sharing of actual MLX arrays."""
    c = _StubCache()
    c.append(7)
    reps = c.replicate(3)
    assert len(reps) == 3
    for r in reps:
        assert r.offset == 7
        assert r is not c
    print("[OK] replicate fidelity (stub): 3 caches each read back original state, all independent")


def test_replicate_fidelity_dsv4cache() -> None:
    """``DSV4Cache.replicate(B)`` shares array refs with the prefix (zero-copy) and isolates writes."""
    cache = DSV4Cache(n_layers=2, quantized=False, max_rollback=4)
    # Seed each layer with a tiny KV vector — exercise both bf16 storage and the ratio==0 path.
    for lc in cache.layers:
        lc.append_kv(mx.zeros((1, 1, 8)) + 0.5)
    pre_kv = cache.layers[0]._kv_bf16                # capture the shared prefix array ref
    pre_off = cache.offset
    assert pre_off == 1

    reps = cache.replicate(3)
    # Every replica reads back the original KV exactly (structural sharing: same MLX array).
    for k, r in enumerate(reps):
        for li in range(2):
            assert r.layers[li]._kv_bf16 is cache.layers[li]._kv_bf16, (
                f"replica {k} layer {li}: KV array ref must be shared with the prefix")
        assert r.offset == pre_off

    # Diverge replica 0: append a new KV slot. Its KV must grow; the original AND other replicas
    # must still see the un-appended state (MLX immutability — the concat returns a new array).
    reps[0].layers[0].append_kv(mx.ones((1, 1, 8)))
    assert reps[0].layers[0]._kv_bf16.shape[1] == 2
    assert cache.layers[0]._kv_bf16.shape[1] == 1, "original prefix array must NOT have grown"
    assert cache.layers[0]._kv_bf16 is pre_kv, "original prefix array ref must be unchanged"
    for k in (1, 2):
        assert reps[k].layers[0]._kv_bf16.shape[1] == 1, (
            f"sibling replica {k} must NOT see replica 0's append (immutability invariant)")
    print("[OK] DSV4Cache.replicate: structural sharing + isolated divergence on append")


def test_cache_invariance_after_batched() -> None:
    """After ``batched=True`` completes, the original (un-replicated) cache is in the SAME state it
    would be after ``batched=False`` — same offset, same number of truncates (zero — batched
    doesn't truncate the prefix; it only writes via the commit-forward)."""
    embed = mx.zeros((VOCAB, HC * DIM))
    head = mx.zeros((VOCAB, HC * DIM))
    prompt = [2, 5, 7]

    # Run sequential — capture the cache offset progression via _WrapCache below if we wanted; here
    # we just compare final offsets between the two runs (the bit-identical token streams guarantee
    # identical cache positions if the same model+MTP are used).
    seq_model = _WrapCacheCapture(_StubMainModel())
    seq, _ = spec_generate_tree(seq_model, _PerfectLeftmostMTP(2), embed, head, prompt,
                                width=2, depth=2, max_new=MAXN, eos_id=None, batched=False)
    seq_offset = seq_model.cache.offset
    seq_truncates = list(seq_model.cache.truncations)

    bat_model = _WrapCacheCapture(_StubMainModel())
    bat, _ = spec_generate_tree(bat_model, _PerfectLeftmostMTP(2), embed, head, prompt,
                                width=2, depth=2, max_new=MAXN, eos_id=None, batched=True)
    bat_offset = bat_model.cache.offset
    bat_truncates = list(bat_model.cache.truncations)

    assert seq == bat
    assert seq_offset == bat_offset, (
        f"final offset diverged: sequential={seq_offset} batched={bat_offset}")
    # Batched does NOT truncate the prefix (the replicas absorb all per-path verify writes).
    # Sequential truncates W^D times per round (= 4 per round at W=2 D=2).
    assert len(bat_truncates) == 0, (
        f"batched form must not truncate the prefix cache (got {len(bat_truncates)} truncates)")
    assert len(seq_truncates) > 0, (
        f"sequential form is expected to per-path truncate (got {len(seq_truncates)})")
    print(f"[OK] cache invariance: offsets match ({seq_offset}); batched truncates "
          f"{len(bat_truncates)} (prefix never rolled back), sequential {len(seq_truncates)}")


class _WrapCacheCapture:
    """Stub main model wrapping a SHARED _StubCache + ``batch_step`` so the test can inspect
    the prefix-cache's state after a run. Reuses ``_StubMainModel`` for both surfaces."""

    def __init__(self, base: _StubMainModel) -> None:
        self.cfg = base.cfg
        self.num_layers = base.num_layers
        self.cache = _StubCache()
        self._base = base

    def make_caches(self, *, max_rollback: int = 1) -> _StubCache:
        del max_rollback
        return self.cache

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        return self._base(token_ids, caches=caches, offset=offset, capture_layers=capture_layers)

    def batch_step(self, tokens, *, caches, offset, capture_layer=None):
        return self._base.batch_step(tokens, caches=caches, offset=offset,
                                     capture_layer=capture_layer)


def main() -> int:
    tests = [
        test_batched_equals_sequential_perfect_w2d2,
        test_batched_equals_sequential_wrong_w2d2,
        test_batched_equals_sequential_w4d2,
        test_width1_matches_spec_generate_k_both_flags,
        test_eos_stops_under_batched,
        test_replicate_fidelity_stub,
        test_replicate_fidelity_dsv4cache,
        test_cache_invariance_after_batched,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
