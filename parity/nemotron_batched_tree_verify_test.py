"""Parity gate for Nemotron-H batched tree-spec verify — bit-identical to the sequential form.

Validates :func:`quanta.nemotron.spec.spec_generate_tree`'s ``batched=True`` path against the
proven ``batched=False`` sequential form (#157, ``ff202da``) — the follow-on landed per the design
doc ``docs/batched_tree_verify.md``. MODEL-FREE: a stub main model exposing BOTH the single-stream
``__call__`` (consumed by ``batched=False``) AND :meth:`batch_step` (consumed by ``batched=True``),
plus a stub cache with structural-sharing replicate; no checkpoint, no GPU, a few KB of tensors.
Safe alongside the EAGLE retrain on GPU — never contends.

Asserts:

  (1) **batched == sequential** — for both a perfect-leftmost MTP and a wrong-all MTP at
      ``(W=2,D=2)`` and ``(W=4,D=2)``; ``mean_accept`` / ``rounds`` / ``max_accept`` match.
  (2) **width=1 short-circuit** — bypasses batched (chain) and matches
      :func:`quanta.nemotron.spec.spec_generate_k` regardless of the ``batched`` flag value.
  (3) **Cache invariance** — after ``batched=True`` completes, the original KV cache offset matches
      the sequential run AND the prefix KV is never truncated under the batched path (replicas
      absorb per-path divergence; only the final commit-replay touches the original cache).
  (4) **Replicate fidelity** — :func:`quanta.nemotron.batched_runtime.replicate_state` returns B
      triples that read back the original state's content (KV per-layer ``_copy`` + ssm/conv list
      clones). Subsequent writes on one replica do not corrupt siblings.
  (5) **Replica divergence** — after one replica's KV update + ssm/conv mutation, the original and
      sibling replicas still see the un-mutated state (MLX-immutability + per-replica list-spine
      ownership is the contract that makes structural sharing lossless).
  (6) **eos** terminates at the first emitted eos (inclusive) under batched — matches sequential.

Run:  ``uv run --with numpy python -m parity.nemotron_batched_tree_verify_test``

Deferred (GPU/memory-available session) — docstring-only entry per ``parity/dsv4_int4_ppl.py``:

  * **Real-model parity** — load the resident baked Nemotron-H + native MTP, run
    ``spec_generate_tree(W=2, D=2, batched=True)`` vs ``=False`` on a real prompt, assert tokens
    match bit-for-bit (SDPA + sorted-MoE may reorder reductions; fall back to
    ``argmax_match >= 0.99``).
  * **Throughput bench** — measure tok/s for ``spec_generate_k(k=2)`` vs
    ``spec_generate_tree(W=2, D=2, batched=False)`` vs ``=True``; expected economics in
    ``docs/batched_tree_verify.md``'s table.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from quanta.nemotron.spec import spec_generate_k, spec_generate_tree

VOCAB = 64
HIDDEN = 8
NL = 4               # stub "decoder layers"
STEP = 3             # deterministic chain: greedy(t) = t + STEP
EOS = 40
MAXN = 16

EMBED = mx.eye(VOCAB)             # [VOCAB, VOCAB] one-hot rows — embed[t] has argmax t
HEAD = mx.zeros((VOCAB, HIDDEN))  # unused by stub MTPs; the real signature passes it


def _greedy_next(t: int) -> int:
    return t + STEP


def _row(tok: int) -> mx.array:
    return mx.where(mx.arange(VOCAB) == tok, 30.0, -30.0)


class _StubCache:
    """Stand-in for the Nemotron decode cache (which is normally a per-layer ``KVCache``).

    Tracks length, supports ``truncate`` / ``offset`` / ``_copy`` (the structural-sharing op
    that ``replicate_state`` uses). The stub model's logits depend only on input tokens, so the
    cache only needs the rollback surface + a per-replica _copy."""

    def __init__(self) -> None:
        self._len = 0
        self.truncations: list[tuple[int, int]] = []

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

    def _copy(self) -> "_StubCache":
        new = _StubCache()
        new._len = self._len
        return new


class _StubMainModel:
    """Stub Nemotron model with BOTH single-stream ``__call__`` AND ``batch_step``.

    * ``__call__(token_ids, *, caches, ssm, conv, offset, capture_layers)`` — sequential contract.
    * ``batch_step(tokens, *, replicas, offset, capture_layer)`` — batched contract returning
      ``(logits [B,1,vocab], hidden [B,1,hidden] or None)``.

    Both produce deterministic ``greedy(t) = t + STEP``, so the two paths must produce bit-identical
    token streams. The "cache" surface here matches the stub used in
    ``parity/nemotron_tree_spec_test.py`` (a single ``_StubCache`` with shared state across the
    stub's whole stack); the batched ``replicas`` argument is a list of triples whose KV entry IS
    a ``_StubCache``."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(num_hidden_layers=NL)
        self.num_layers = NL
        self.calls: list[tuple[str, tuple[int, ...], int]] = []
        self.cache: _StubCache | None = None

    def make_caches(self) -> _StubCache:
        self.cache = _StubCache()
        return self.cache

    def __call__(self, token_ids, *, caches=None, ssm=None, conv=None, offset=0,
                 capture_layers=None):
        del ssm, conv
        ids = [int(x) for x in (token_ids.tolist() if isinstance(token_ids, mx.array) else token_ids)]
        self.calls.append(("call", tuple(ids), offset))
        if caches is not None and hasattr(caches, "append"):
            caches.append(len(ids))
        t = len(ids)
        logits = mx.stack([_row(_greedy_next(tok)) for tok in ids])[None]       # [1,T,vocab]
        if not capture_layers:
            return logits
        last = max(capture_layers)
        feat = mx.broadcast_to(mx.arange(t, dtype=mx.float32)[:, None], (t, HIDDEN))
        return logits, {last: feat}

    def batch_step(self, tokens, *, replicas, offset, capture_layer=None):
        b = len(tokens)
        if len(replicas) != b:
            raise ValueError(f"batch_step len mismatch: replicas={len(replicas)} tokens={b}")
        # Each replica is (caches, ssm, conv); validate offset on the cache surface (the stub's
        # "cache" stands in for the per-layer KV list — here a single tracked length).
        for s, (caches_s, _ssm_s, _conv_s) in enumerate(replicas):
            if caches_s is not None and hasattr(caches_s, "offset") and caches_s.offset != offset:
                raise ValueError(f"batch_step replicas[{s}] offset={caches_s.offset} != {offset}")
        self.calls.append(("batch_step", tuple(int(t) for t in tokens), offset))
        # advance each replica's "cache" surface by one token
        for caches_s, _ssm_s, _conv_s in replicas:
            if caches_s is not None and hasattr(caches_s, "append"):
                caches_s.append(1)
        rows = mx.stack([_row(_greedy_next(int(t))) for t in tokens])           # [B, vocab]
        logits = rows[:, None]                                                  # [B,1,vocab]
        if capture_layer is None:
            return logits, None
        feat = mx.broadcast_to(mx.array(0.0, dtype=mx.float32), (b, 1, HIDDEN))
        return logits, feat


# `_capture_state` from quanta.nemotron.spec calls `model.make_caches()` and expects either a
# triple or a bare caches — for our stub we want a triple where ``caches`` is the _StubCache and
# the other slots are None. The wrapper below adapts the bare-cache return.
class _NemotronStateAdapter:
    """Wraps the stub model so ``make_caches`` returns a ``(caches, None, None)`` triple — the
    contract :func:`quanta.nemotron.spec._capture_state` expects. (The bare-cache fallback in
    ``_capture_state`` ALREADY does this for us, but making it explicit here matches what the
    batched form sees: ``replicate_state`` requires a triple.)"""

    def __init__(self, base: _StubMainModel) -> None:
        self.cfg = base.cfg
        self.num_layers = base.num_layers
        self._base = base

    def make_caches(self) -> tuple:
        # Return a SINGLE cache surface that the spec's _rollback path also accepts (via
        # cache.truncate); the spec's existing logic treats this as the cache argument threaded
        # through every _forward call. Tested against the same surface in nemotron_tree_spec_test.
        self._base.cache = _StubCache()
        # For batched, the spec expects replicate_state(state, b). To make that work with our
        # _StubCache, we'd need replicate_state to handle a non-list ``caches`` slot. The real
        # nemotron batched_runtime.replicate_state expects a list-of-layers, so we wrap the
        # stub's single _StubCache as a one-element list to satisfy the API.
        return [self._base.cache], None, None

    def __call__(self, token_ids, *, caches=None, ssm=None, conv=None, offset=0,
                 capture_layers=None):
        # caches comes in as the list-of-layers form here; pass through the single cache.
        single_cache = caches[0] if isinstance(caches, list) and caches else caches
        return self._base(token_ids, caches=single_cache, ssm=ssm, conv=conv, offset=offset,
                          capture_layers=capture_layers)

    def batch_step(self, tokens, *, replicas, offset, capture_layer=None):
        return self._base.batch_step(tokens, replicas=replicas, offset=offset,
                                     capture_layer=capture_layer)


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
    """Top-1 child = main-greedy → leftmost path accepts all ``depth`` drafts.
    Signature mirrors Nemotron MTP: ``mtp(prev_hidden, token_emb, head, *, return_hidden=False)``."""

    def __init__(self, width: int) -> None:
        self.width = width

    def __call__(self, prev_hidden, token_emb, head, *, return_hidden=False):
        parent = int(mx.argmax(token_emb[0, 0]).item())
        greedy = _greedy_next(parent) % VOCAB
        wrongs = _wrongs_for(parent, self.width, greedy)
        logits = _logits_top_w(greedy, wrongs)
        _ = return_hidden
        return logits, _dummy_hidden()


class _WrongAllMTP:
    def __init__(self, width: int) -> None:
        self.width = width

    def __call__(self, prev_hidden, token_emb, head, *, return_hidden=False):
        parent = int(mx.argmax(token_emb[0, 0]).item())
        greedy = _greedy_next(parent) % VOCAB
        wrongs = _wrongs_for(parent, self.width + 1, greedy)[: self.width]
        logits = _logits_top_w(wrongs[0], wrongs[1:])
        _ = return_hidden
        return logits, _dummy_hidden()


# ============================================================================
# Tests
# ============================================================================

def test_batched_equals_sequential_perfect_w2d2() -> None:
    seq, st_seq = spec_generate_tree(_NemotronStateAdapter(_StubMainModel()), _PerfectLeftmostMTP(2),
                                     EMBED, HEAD, [2, 5, 7], width=2, depth=2, max_new=MAXN,
                                     eos_id=None, batched=False)
    bat, st_bat = spec_generate_tree(_NemotronStateAdapter(_StubMainModel()), _PerfectLeftmostMTP(2),
                                     EMBED, HEAD, [2, 5, 7], width=2, depth=2, max_new=MAXN,
                                     eos_id=None, batched=True)
    assert seq == bat, f"batched != sequential: bat={bat} seq={seq}"
    assert st_bat["rounds"] == st_seq["rounds"]
    assert st_bat["mean_accept"] == st_seq["mean_accept"]
    assert st_bat["max_accept"] == st_seq["max_accept"]
    assert st_bat["batched"] is True and st_seq["batched"] is False
    print(f"[OK] perfect W=2 D=2: batched == sequential (n={len(bat)}, "
          f"mean_accept={st_bat['mean_accept']:.2f})")


def test_batched_equals_sequential_wrong_w2d2() -> None:
    seq, st_seq = spec_generate_tree(_NemotronStateAdapter(_StubMainModel()), _WrongAllMTP(2),
                                     EMBED, HEAD, [2, 5, 7], width=2, depth=2, max_new=MAXN,
                                     eos_id=None, batched=False)
    bat, st_bat = spec_generate_tree(_NemotronStateAdapter(_StubMainModel()), _WrongAllMTP(2),
                                     EMBED, HEAD, [2, 5, 7], width=2, depth=2, max_new=MAXN,
                                     eos_id=None, batched=True)
    assert seq == bat, f"batched != sequential (wrong): bat={bat} seq={seq}"
    assert st_bat["mean_accept"] == st_seq["mean_accept"]
    print(f"[OK] wrong W=2 D=2: batched == sequential, mean_accept={st_bat['mean_accept']:.2f} (=1)")


def test_batched_equals_sequential_w4d2() -> None:
    seq, _ = spec_generate_tree(_NemotronStateAdapter(_StubMainModel()), _PerfectLeftmostMTP(4),
                                EMBED, HEAD, [2, 5, 7], width=4, depth=2, max_new=MAXN,
                                eos_id=None, batched=False)
    bat, st = spec_generate_tree(_NemotronStateAdapter(_StubMainModel()), _PerfectLeftmostMTP(4),
                                 EMBED, HEAD, [2, 5, 7], width=4, depth=2, max_new=MAXN,
                                 eos_id=None, batched=True)
    assert seq == bat, f"batched != sequential (W=4): bat={bat} seq={seq}"
    assert st["paths_per_round"] == 16
    print(f"[OK] perfect W=4 D=2 (B=16): batched == sequential, mean_accept={st['mean_accept']:.2f}")


def test_width1_matches_spec_generate_k_both_flags() -> None:
    out_f, _ = spec_generate_tree(_NemotronStateAdapter(_StubMainModel()), _PerfectLeftmostMTP(1),
                                  EMBED, HEAD, [2, 5, 7], width=1, depth=2, max_new=MAXN,
                                  eos_id=None, batched=False)
    out_t, _ = spec_generate_tree(_NemotronStateAdapter(_StubMainModel()), _PerfectLeftmostMTP(1),
                                  EMBED, HEAD, [2, 5, 7], width=1, depth=2, max_new=MAXN,
                                  eos_id=None, batched=True)
    out_k, _ = spec_generate_k(_NemotronStateAdapter(_StubMainModel()), _PerfectLeftmostMTP(1),
                               EMBED, HEAD, [2, 5, 7], k=2, max_new=MAXN, eos_id=None)
    assert out_f == out_t == out_k, f"width=1 differs: f={out_f} t={out_t} k={out_k}"
    print(f"[OK] width=1 short-circuit: batched=False == True == spec_generate_k(k=2) (n={len(out_k)})")


def test_eos_stops_under_batched() -> None:
    seq, _ = spec_generate_tree(_NemotronStateAdapter(_StubMainModel()), _PerfectLeftmostMTP(2),
                                EMBED, HEAD, [2, 5, 7], width=2, depth=2, max_new=MAXN,
                                eos_id=EOS, batched=False)
    bat, _ = spec_generate_tree(_NemotronStateAdapter(_StubMainModel()), _PerfectLeftmostMTP(2),
                                EMBED, HEAD, [2, 5, 7], width=2, depth=2, max_new=MAXN,
                                eos_id=EOS, batched=True)
    assert seq == bat, f"eos stop mismatch: bat={bat} seq={seq}"
    assert bat and bat[-1] == EOS and EOS not in bat[:-1]
    print(f"[OK] eos stops under batched: bit-identical to sequential (final={bat[-1]} = EOS)")


def test_replicate_fidelity_stub() -> None:
    """``_StubCache._copy`` returns a fresh cache initialized to the same length; structurally
    independent from the original (a write on one does NOT affect the other)."""
    c = _StubCache()
    c.append(7)
    copies = [c._copy() for _ in range(3)]
    for r in copies:
        assert r.offset == 7
        assert r is not c
    copies[0].append(2)
    assert copies[0].offset == 9
    assert c.offset == 7, "original cache must NOT have been mutated"
    for k in (1, 2):
        assert copies[k].offset == 7, f"sibling {k} must NOT see copy 0's append"
    print("[OK] replicate fidelity (stub): 3 copies independent; sibling divergence isolated")


def test_replicate_state_fidelity_nemotron() -> None:
    """``replicate_state`` returns B triples sharing array refs with the prefix; per-replica
    appends/commits diverge naturally under MLX immutability + per-replica list ownership."""
    from quanta.nemotron.attention import KVCache
    from quanta.nemotron.batched_runtime import replicate_state

    # Build a fake state triple: 2 layers, layer 0 is attention (KV), layer 1 is mamba (ssm/conv).
    kv = KVCache(quantized=False)
    kv.update(mx.zeros((1, 2, 1, 4)) + 0.5, mx.zeros((1, 2, 1, 4)) + 0.25)
    pre_k = kv.k

    ssm0 = mx.zeros((1, 2, 4, 4)) + 0.7
    conv0 = mx.zeros((1, 3, 8)) + 0.3

    caches = [kv, None]
    ssm = [None, ssm0]
    conv = [None, conv0]
    state = (caches, ssm, conv)

    reps = replicate_state(state, 3)
    assert len(reps) == 3
    for k, (c_r, ssm_r, conv_r) in enumerate(reps):
        # KV array ref shared
        assert c_r[0].k is kv.k, f"replica {k}: KV array ref must be shared with prefix"
        # ssm/conv lists are independent (new list spine); array refs shared
        assert ssm_r is not ssm, f"replica {k}: ssm list spine must be independent"
        assert ssm_r[1] is ssm0, f"replica {k}: ssm[1] array ref must be shared with prefix"
        assert conv_r[1] is conv0

    # Diverge replica 0: KV update + ssm/conv reassign.
    reps[0][0][0].update(mx.ones((1, 2, 1, 4)), mx.ones((1, 2, 1, 4)))
    reps[0][1][1] = mx.zeros((1, 2, 4, 4)) + 1.0
    reps[0][2][1] = mx.zeros((1, 3, 8)) + 1.0

    # Replica 0 KV grew; original + siblings still hold 1-token KV
    assert reps[0][0][0].k.shape[2] == 2
    assert kv.k.shape[2] == 1, "original KV must NOT have grown"
    assert kv.k is pre_k
    for k in (1, 2):
        assert reps[k][0][0].k.shape[2] == 1, f"sibling {k} KV must NOT have grown"

    # Replica 0 ssm/conv reassigned; original + siblings hold the prefix value
    assert reps[0][1][1] is not ssm0
    assert ssm[1] is ssm0, "original ssm[1] must be unchanged"
    for k in (1, 2):
        assert reps[k][1][1] is ssm0, f"sibling {k} ssm[1] must still reference the prefix"
    print("[OK] replicate_state: structural sharing (KV + ssm + conv) + isolated divergence")


def test_cache_invariance_after_batched() -> None:
    """After ``batched=True`` completes, the original cache offset matches the sequential run."""
    seq_model = _NemotronStateAdapter(_StubMainModel())
    seq, _ = spec_generate_tree(seq_model, _PerfectLeftmostMTP(2), EMBED, HEAD, [2, 5, 7],
                                width=2, depth=2, max_new=MAXN, eos_id=None, batched=False)
    seq_offset = seq_model._base.cache.offset
    seq_truncates = list(seq_model._base.cache.truncations)

    bat_model = _NemotronStateAdapter(_StubMainModel())
    bat, _ = spec_generate_tree(bat_model, _PerfectLeftmostMTP(2), EMBED, HEAD, [2, 5, 7],
                                width=2, depth=2, max_new=MAXN, eos_id=None, batched=True)
    bat_offset = bat_model._base.cache.offset
    bat_truncates = list(bat_model._base.cache.truncations)

    assert seq == bat
    assert seq_offset == bat_offset, f"final offset diverged: seq={seq_offset} bat={bat_offset}"
    # Batched form does NOT truncate the prefix cache (the replicas absorb the per-path
    # divergence; only the final commit-replay touches the original cache).
    assert len(bat_truncates) == 0, (
        f"batched form must not truncate the prefix cache (got {len(bat_truncates)})")
    assert len(seq_truncates) > 0
    print(f"[OK] cache invariance: offsets match ({seq_offset}); batched truncates "
          f"{len(bat_truncates)} (prefix never rolled back), sequential {len(seq_truncates)}")


def main() -> int:
    tests = [
        test_batched_equals_sequential_perfect_w2d2,
        test_batched_equals_sequential_wrong_w2d2,
        test_batched_equals_sequential_w4d2,
        test_width1_matches_spec_generate_k_both_flags,
        test_eos_stops_under_batched,
        test_replicate_fidelity_stub,
        test_replicate_state_fidelity_nemotron,
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
