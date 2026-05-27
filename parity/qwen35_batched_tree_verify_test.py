"""Parity gate for Qwen3.5 batched tree-spec verify — bit-identical to the sequential form.

Validates :func:`quanta.qwen35.spec.spec_generate_tree`'s ``batched=True`` path against the proven
``batched=False`` sequential form (#157, ``99040c6``) — the follow-on landed per the design doc
``docs/batched_tree_verify.md``. MODEL-FREE: a stub main model exposing BOTH the single-stream
``__call__`` (consumed by ``batched=False``) AND :meth:`batch_step` (consumed by ``batched=True``),
plus stub caches with structural-sharing replicate; no checkpoint, no GPU, a few KB of tensors.
Safe alongside the EAGLE retrain on GPU — never contends.

Asserts:

  (1) **batched == sequential** — for both a perfect-leftmost MTP and a wrong-all MTP at
      ``(W=2,D=2)`` and ``(W=4,D=2)``; ``mean_accept`` / ``rounds`` / ``max_accept`` match.
  (2) **width=1 short-circuit** — bypasses batched (chain) and matches
      :func:`quanta.qwen35.spec.spec_generate_k` regardless of the ``batched`` flag value.
  (3) **Cache invariance** — after ``batched=True`` completes, the original cache offset matches the
      sequential run; the prefix is never truncated under the batched path (replicas absorb the
      per-path divergence).
  (4) **Replicate fidelity** — :meth:`quanta.qwen35.decode.Qwen35Cache.replicate` returns B caches
      that read back the original cache's content (structural-sharing replication preserves the
      prefix exactly for BOTH regime types — KV full-attn AND GDN recurrent snapshots — and
      subsequent appends/commits do not corrupt siblings).
  (5) **Replica divergence** — after one replica's KV ``update`` and another replica's GDN
      ``commit``, the original and sibling replicas still see the un-mutated state (MLX immutability
      isolating sibling caches is the contract that makes structural sharing lossless).
  (6) **eos** terminates at the first emitted eos (inclusive) under batched — matches sequential.

Run:  ``uv run --with numpy python -m parity.qwen35_batched_tree_verify_test``

Deferred (GPU/memory-available session) — docstring-only entry per ``parity/dsv4_int4_ppl.py``:

  * **Real-model parity** — load the resident baked Qwen3.5 + native MTP, run
    ``spec_generate_tree(W=2, D=2, batched=True)`` vs ``=False`` on a real prompt, assert tokens
    match bit-for-bit (SDPA + sorted-MoE may reorder; fall back to ``argmax_match >= 0.99``).
  * **Throughput bench** — measure tok/s for ``spec_generate_k(k=2)`` vs
    ``spec_generate_tree(W=2, D=2, batched=False)`` vs ``=True``; expected economics in
    ``docs/batched_tree_verify.md``'s table.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from quanta.qwen35.spec import spec_generate_k, spec_generate_tree

VOCAB = 64
HIDDEN = 8
NL = 3              # stub "decoder layers"
STEP = 3            # deterministic chain: greedy(t) = t + STEP
EOS = 40
MAXN = 16
CAPTURE_LAYER = NL - 1


def _greedy_next(t: int) -> int:
    return t + STEP


def _row(tok: int) -> mx.array:
    return mx.where(mx.arange(VOCAB) == tok, 30.0, -30.0)


class _StubCache:
    """Stand-in for ``Qwen35Cache``: tracks length, supports ``truncate``/``offset``/``replicate``.

    Models structural-sharing semantics: ``replicate(B)`` returns B fresh caches that initially
    equal the original; subsequent writes diverge naturally."""

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

    def replicate(self, b: int) -> list["_StubCache"]:
        out = []
        for _ in range(b):
            r = _StubCache()
            r._len = self._len
            out.append(r)
        return out


class _StubMainModel:
    """Stub with BOTH single-stream ``__call__`` AND ``batch_step`` — the two surfaces the spec
    consumes via the ``batched`` flag. Honors per-replica cache appends so structural sharing
    is exercised."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(num_hidden_layers=NL)
        self.num_layers = NL
        self.calls: list[tuple[str, tuple[int, ...], int]] = []

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
        feat = mx.broadcast_to(mx.arange(t, dtype=mx.float32)[:, None], (t, HIDDEN))
        return logits, {last: feat}

    def batch_step(self, tokens, *, caches, offset, capture_layer=None):
        b = len(tokens)
        if len(caches) != b:
            raise ValueError(f"batch_step len mismatch: caches={len(caches)} tokens={b}")
        for s, c in enumerate(caches):
            if c.offset != offset:
                raise ValueError(f"batch_step caches[{s}].offset={c.offset} != offset={offset}")
        self.calls.append(("batch_step", tuple(int(t) for t in tokens), offset))
        for c in caches:
            c.append(1)
        rows = mx.stack([_row(_greedy_next(int(t))) for t in tokens])           # [B, vocab]
        logits = rows[:, None]                                                  # [B,1,vocab]
        if capture_layer is None:
            return logits, None
        feat = mx.broadcast_to(mx.array(0.0, dtype=mx.float32), (b, 1, HIDDEN))
        return logits, feat


def _dummy_hidden() -> mx.array:
    """``[1, 1, hidden]`` placeholder hidden for the chained tree-build."""
    return mx.zeros((1, 1, HIDDEN))


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

    def __call__(self, prev_hidden, next_ids, embed, head, *, return_hidden=False):
        parent = int(next_ids[0, 0].item())
        greedy = _greedy_next(parent) % VOCAB
        wrongs = _wrongs_for(parent, self.width, greedy)
        logits = _logits_top_w(greedy, wrongs)
        if return_hidden:
            return logits, _dummy_hidden()
        return logits


class _WrongAllMTP:
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
    embed = mx.zeros((VOCAB, HIDDEN))
    head = mx.zeros((VOCAB, HIDDEN))
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
    embed = mx.zeros((VOCAB, HIDDEN))
    head = mx.zeros((VOCAB, HIDDEN))
    prompt = [2, 5, 7]
    seq, st_seq = spec_generate_tree(_StubMainModel(), _WrongAllMTP(2), embed, head, prompt,
                                     width=2, depth=2, max_new=MAXN, eos_id=None, batched=False)
    bat, st_bat = spec_generate_tree(_StubMainModel(), _WrongAllMTP(2), embed, head, prompt,
                                     width=2, depth=2, max_new=MAXN, eos_id=None, batched=True)
    assert seq == bat, f"batched != sequential (wrong MTP): bat={bat} seq={seq}"
    assert st_bat["mean_accept"] == st_seq["mean_accept"]
    print(f"[OK] wrong W=2 D=2: batched == sequential, mean_accept={st_bat['mean_accept']:.2f} (=1)")


def test_batched_equals_sequential_w4d2() -> None:
    embed = mx.zeros((VOCAB, HIDDEN))
    head = mx.zeros((VOCAB, HIDDEN))
    prompt = [2, 5, 7]
    seq, _ = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(4), embed, head, prompt,
                                width=4, depth=2, max_new=MAXN, eos_id=None, batched=False)
    bat, st = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(4), embed, head, prompt,
                                 width=4, depth=2, max_new=MAXN, eos_id=None, batched=True)
    assert seq == bat, f"batched != sequential (W=4): bat={bat} seq={seq}"
    assert st["paths_per_round"] == 16
    print(f"[OK] perfect W=4 D=2 (B=16): batched == sequential, mean_accept={st['mean_accept']:.2f}")


def test_width1_matches_spec_generate_k_both_flags() -> None:
    embed = mx.zeros((VOCAB, HIDDEN))
    head = mx.zeros((VOCAB, HIDDEN))
    prompt = [2, 5, 7]
    out_f, _ = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(1), embed, head, prompt,
                                  width=1, depth=2, max_new=MAXN, eos_id=None, batched=False)
    out_t, _ = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(1), embed, head, prompt,
                                  width=1, depth=2, max_new=MAXN, eos_id=None, batched=True)
    out_k, _ = spec_generate_k(_StubMainModel(), _PerfectLeftmostMTP(1), embed, head, prompt,
                               k=2, max_new=MAXN, eos_id=None)
    assert out_f == out_t == out_k, f"width=1 differs: f={out_f} t={out_t} k={out_k}"
    print(f"[OK] width=1 short-circuit: batched=False == True == spec_generate_k(k=2) (n={len(out_k)})")


def test_eos_stops_under_batched() -> None:
    embed = mx.zeros((VOCAB, HIDDEN))
    head = mx.zeros((VOCAB, HIDDEN))
    prompt = [2, 5, 7]
    seq, _ = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(2), embed, head, prompt,
                                width=2, depth=2, max_new=MAXN, eos_id=EOS, batched=False)
    bat, _ = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(2), embed, head, prompt,
                                width=2, depth=2, max_new=MAXN, eos_id=EOS, batched=True)
    assert seq == bat, f"eos stop mismatch: bat={bat} seq={seq}"
    assert bat and bat[-1] == EOS and EOS not in bat[:-1]
    print(f"[OK] eos stops under batched: bit-identical to sequential (final={bat[-1]} = EOS)")


def test_replicate_fidelity_stub() -> None:
    c = _StubCache()
    c.append(7)
    reps = c.replicate(3)
    assert len(reps) == 3
    for r in reps:
        assert r.offset == 7
        assert r is not c
    print("[OK] replicate fidelity (stub): 3 caches each read back original state, all independent")


def test_replicate_fidelity_qwen35cache_kv() -> None:
    """``Qwen35Cache.replicate(B)`` shares array refs for the KV cache + GDN snapshot ring across
    replicas; sibling appends/commits do not corrupt the original or siblings."""
    from quanta.qwen35.attention import KVCache
    from quanta.qwen35.decode import Qwen35Cache, _GDNLayerState

    # Hybrid cache: layer 0 linear (GDN), layer 1 full (KV).
    def is_linear(i: int) -> bool:
        return i == 0

    cache = Qwen35Cache(n_layers=2, layer_is_linear=is_linear, quantized=False, max_rollback=4)

    # Seed the KV layer with a tiny K/V update.
    kv_lc = cache.layers[1]
    assert isinstance(kv_lc, KVCache)
    k0, v0 = mx.zeros((1, 2, 1, 4)) + 0.5, mx.zeros((1, 2, 1, 4)) + 0.25
    kv_lc.update(k0, v0)
    pre_k = kv_lc.k

    # Seed the GDN layer with a tiny recurrent state.
    gdn_lc = cache.layers[0]
    assert isinstance(gdn_lc, _GDNLayerState)
    gdn_lc.commit(mx.zeros((1, 3, 8)) + 0.7, mx.zeros((1, 2, 4, 4)) + 0.3)

    reps = cache.replicate(3)
    # Every replica reads back the original KV exactly (same array ref → bit-identical content).
    for k, r in enumerate(reps):
        assert r.layers[1].k is cache.layers[1].k, (
            f"replica {k} KV layer: k array ref must be shared with the prefix")
        assert r.layers[1].v is cache.layers[1].v
        # GDN: same conv/recurrent + same snapshot tuple refs (list cloned, content shared).
        assert r.layers[0].conv_state is cache.layers[0].conv_state
        assert r.layers[0].recurrent_state is cache.layers[0].recurrent_state
        assert r.layers[0].offset == cache.layers[0].offset

    # Diverge replica 0 — KV update + GDN commit. The original prefix arrays/snapshots are unchanged.
    reps[0].layers[1].update(mx.ones((1, 2, 1, 4)), mx.ones((1, 2, 1, 4)))
    reps[0].layers[0].commit(mx.zeros((1, 3, 8)) + 1.0, mx.zeros((1, 2, 4, 4)) + 1.0)
    assert reps[0].layers[1].k.shape[2] == 2
    assert cache.layers[1].k.shape[2] == 1, "original KV must NOT have grown"
    assert cache.layers[1].k is pre_k, "original KV array ref must be unchanged"
    # Siblings: still single-position KV, single-commit GDN.
    for k in (1, 2):
        assert reps[k].layers[1].k.shape[2] == 1
        assert reps[k].layers[0].offset == 1, (
            f"sibling replica {k}: GDN offset must NOT advance from replica 0's commit")
    print("[OK] Qwen35Cache.replicate: structural sharing (KV + GDN) + isolated divergence")


def test_cache_invariance_after_batched() -> None:
    """After ``batched=True`` completes, the original (un-replicated) cache offset matches the
    sequential run AND the prefix is never truncated (replicas absorb per-path divergence)."""
    embed = mx.zeros((VOCAB, HIDDEN))
    head = mx.zeros((VOCAB, HIDDEN))
    prompt = [2, 5, 7]

    seq_model = _WrapCacheCapture(_StubMainModel())
    seq, _ = spec_generate_tree(seq_model, _PerfectLeftmostMTP(2), embed, head, prompt,
                                width=2, depth=2, max_new=MAXN, eos_id=None, batched=False)
    seq_truncates = list(seq_model.cache.truncations)
    seq_offset = seq_model.cache.offset

    bat_model = _WrapCacheCapture(_StubMainModel())
    bat, _ = spec_generate_tree(bat_model, _PerfectLeftmostMTP(2), embed, head, prompt,
                                width=2, depth=2, max_new=MAXN, eos_id=None, batched=True)
    bat_truncates = list(bat_model.cache.truncations)
    bat_offset = bat_model.cache.offset

    assert seq == bat
    assert seq_offset == bat_offset, f"final offset diverged: seq={seq_offset} bat={bat_offset}"
    assert len(bat_truncates) == 0, (
        f"batched form must not truncate the prefix cache (got {len(bat_truncates)})")
    assert len(seq_truncates) > 0
    print(f"[OK] cache invariance: offsets match ({seq_offset}); batched truncates "
          f"{len(bat_truncates)} (prefix never rolled back), sequential {len(seq_truncates)}")


class _WrapCacheCapture:
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
        test_replicate_fidelity_qwen35cache_kv,
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
