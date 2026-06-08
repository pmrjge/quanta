"""Lossless gate for DSV4 W-parallel chain-verify tree drafting (#157 follow-on).

Validates :func:`quanta.dsv4.spec.spec_generate_tree` — the interface-uniform form of tree drafting
that mirrors the qwen35 / nemotron implementations: each round drafts a ``(width, depth)`` MTP tree,
enumerates ``W ** D`` root-to-leaf paths, and chain-verifies each (cache snapshot/restore between
paths) so the main model always sees a contiguous chain. MODEL-FREE — a stub main model + stub MTP
+ stub cache; no checkpoint, no GPU, a few KB of tensors — safe alongside another large GPU-resident
job (rule-8: no large allocations).

Scope: this gate exercises the **sequential per-path** verify (``batched=False``) — the path the
single-stream stub main model (no ``.batch_step``) and ``test_truncate_pattern``'s per-path
snapshot/restore rollback assertion are written against. The BATCHED tree-verify path
(``batched=True``, the default since the ``replicate`` + ``batch_step`` fan-out landed) has its own
dedicated model-free gate in ``parity/dsv4_batched_tree_verify_test.py``; the two are complementary,
not redundant. Every ``spec_generate_tree`` call below pins ``batched=False`` explicitly so a future
default flip can't silently route these single-stream stubs through ``batch_verify`` again.

Asserts:
  (1) For a "perfect-leftmost" MTP (top-1 child = main-greedy at every node), the leftmost path
      always accepts ALL ``depth`` drafts → ``mean_accept == depth + 1`` and output is bit-identical
      to a plain greedy reference decode. Tested at ``(W=2, D=2)`` and ``(W=4, D=2)``.
  (2) For a "wrong-all" MTP (no candidate is main-greedy), every path rejects on the first draft
      → ``mean_accept == 1``, yet output is STILL bit-identical to greedy (verify-arbitrated
      losslessness — rule 4 / rule 6).
  (3) ``width=1`` degenerates to a chain of length ``depth`` AND produces identical output +
      ``mean_accept`` to :func:`quanta.dsv4.spec.spec_generate_k` with ``k=depth`` (the documented
      short-circuit). This proves the degenerate case is wired to the proven chained path.
  (4) eos stops generation at the first emitted eos (inclusive) — same as plain greedy — for both
      a perfect MTP and a wrong MTP.
  (5) The decode cache's ``truncate`` is driven exactly as the per-path verify expects: every round
      records ``W ** D`` per-path rollbacks (one per path verify, each dropping ``depth + 1`` tokens
      back to the round start), and the final cache offset reflects only committed tokens.

Run:  ``uv run --with numpy python -m parity.dsv4_tree_spec_test``
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from quanta.dsv4.spec import spec_generate_k, spec_generate_tree

VOCAB = 64
HC = 4              # DSV4 HC (hyper-connection) residual dim — matches dsv4_mtp_spec_test
DIM = 8
NL = 3              # stub "decoder layers" — only cfg.num_hidden_layers matters to spec
STEP = 3            # deterministic chain: greedy(t) = t + STEP
EOS = 40
MAXN = 16


def _greedy_next(t: int) -> int:
    return t + STEP


def _row(tok: int) -> mx.array:
    """A logit row over VOCAB with a clear argmax on ``tok`` (everything else far below)."""
    return mx.where(mx.arange(VOCAB) == tok, 30.0, -30.0)


class _StubCache:
    """Minimal stand-in for ``DSV4Cache``: tracks a length, supports ``truncate`` / ``offset``.

    Records each truncate so the test can confirm the per-path rollback pattern. The stub main
    model ignores cache *contents* (its logits depend only on the input tokens), so the cache need
    only honor the rollback surface the spec loop drives."""

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


class _StubMainModel:
    """Deterministic stub of ``DSV4ResidentModel``: greedy(t) = t + STEP, with MTP-feature capture.

    Honors the consumed contract ``(token_ids, *, caches, offset, capture_layers)`` →
    ``(logits [1,T,vocab], {last: hidden [T,hc,dim]})``. Advances the cache by ``len(token_ids)`` so
    ``offset`` stays consistent after rollbacks, returns a deterministic per-position capture so the
    spec loop's feature plumbing is exercised. Records each call so the test can introspect."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(num_hidden_layers=NL)
        self.num_layers = NL
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def make_caches(self, *, max_rollback: int = 1) -> _StubCache:
        # We accept the kw-arg the spec loop passes — the stub doesn't need to bound rollback depth
        # (no recurrent state to snapshot), but the signature must match for the spec to use it.
        del max_rollback
        return _StubCache()

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        ids = [int(x) for x in (token_ids.tolist() if isinstance(token_ids, mx.array) else token_ids)]
        self.calls.append((tuple(ids), offset))
        if caches is not None:
            caches.append(len(ids))
        t = len(ids)
        logits = mx.stack([_row(_greedy_next(tok)) for tok in ids])[None]   # [1,T,vocab]
        if not capture_layers:
            return logits
        last = max(capture_layers)
        # deterministic [T, hc, dim] feature; content is irrelevant to the stub MTP but shape must
        # match what DSV4ResidentModel returns (the HC residual stream after the last layer).
        feat = mx.broadcast_to(mx.arange(t, dtype=mx.float32)[:, None, None], (t, HC, DIM))
        return logits, {last: feat}


class _WrapCache:
    """Stub main model with a SHARED, test-provided cache so the test can inspect truncates."""

    def __init__(self, base: _StubMainModel, cache: _StubCache) -> None:
        self.cfg = base.cfg
        self.num_layers = base.num_layers
        self._cache = cache

    def make_caches(self, *, max_rollback: int = 1) -> _StubCache:
        del max_rollback
        return self._cache

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        ids = [int(x) for x in (token_ids.tolist() if isinstance(token_ids, mx.array) else token_ids)]
        if caches is not None:
            caches.append(len(ids))
        t = len(ids)
        logits = mx.stack([_row(_greedy_next(tok)) for tok in ids])[None]
        if not capture_layers:
            return logits
        last = max(capture_layers)
        feat = mx.broadcast_to(mx.arange(t, dtype=mx.float32)[:, None, None], (t, HC, DIM))
        return logits, {last: feat}


def _dummy_hidden() -> mx.array:
    """``[1, 1, hc, dim]`` placeholder hidden the chained tree-build feeds back in. Content is
    irrelevant to these stub drafters (which only look at ``next_ids``); the shape must match the
    real DSV4 ``mtp_forward(...)`` return so the spec loop's plumbing is exercised."""
    return mx.zeros((1, 1, HC, DIM))


def _logits_top_w(greedy_tok: int, other_toks: list[int]) -> mx.array:
    """Build ``[1, 1, VOCAB]`` logits so ``_mtp_top_w(..., w)`` returns ``[greedy_tok, *other_toks]``
    in that order. ``greedy_tok`` scores highest; ``other_toks`` get descending scores in order.

    The spec's ``_mtp_top_w`` uses ``mx.argsort(-logits)`` (descending), so any monotone-descending
    score assignment yields the documented ranking. We make the gap large enough that any small
    bf16 / fp32 rounding can't reorder."""
    arr = mx.full((VOCAB,), -100.0)
    arr = mx.where(mx.arange(VOCAB) == greedy_tok, 100.0, arr)
    for i, tok in enumerate(other_toks):
        # gap of 5 between consecutive rungs; well above any rounding noise
        arr = mx.where(mx.arange(VOCAB) == tok, 90.0 - 5.0 * i, arr)
    return arr[None, None]


def _wrongs_for(parent_tok: int, width: int, greedy_tok: int) -> list[int]:
    """Return ``width - 1`` distinct token ids that are not ``greedy_tok`` (the sibling fillers).

    Uses a stable, parent-dependent rotation so each parent's siblings are distinct from neighbouring
    parents' (helps tree-shape diversity during testing). All wrongs ≠ greedy_tok ≠ each other."""
    wrongs: list[int] = []
    candidate = (parent_tok + 1) % VOCAB
    while len(wrongs) < width - 1:
        if candidate != greedy_tok and candidate not in wrongs:
            wrongs.append(candidate)
        candidate = (candidate + 1) % VOCAB
    return wrongs


class _PerfectLeftmostMTP:
    """A drafter whose TOP-1 child is always the main-greedy → the leftmost root-to-leaf path
    always accepts all ``depth`` drafts. Siblings (indices ≥ 1 in the top-W) are non-greedy and
    will be rejected. ``mean_accept == depth + 1`` for any (W, D)."""

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
    """A drafter whose TOP-W are ALL non-greedy → every path rejects on the first draft.
    ``mean_accept == 1``; output STILL matches greedy (verify-arbitrated)."""

    def __init__(self, width: int) -> None:
        self.width = width

    def __call__(self, prev_hidden, next_ids, embed, head, *, return_hidden=False):
        parent = int(next_ids[0, 0].item())
        greedy = _greedy_next(parent) % VOCAB
        # Put the greedy LAST in our ranked top-(W+1); pick W distinct wrongs as the actual top-W.
        wrongs = _wrongs_for(parent, self.width + 1, greedy)[: self.width]
        # Top-W slot 0 gets wrongs[0] (highest score) — never equals greedy.
        logits = _logits_top_w(wrongs[0], wrongs[1:])
        if return_hidden:
            return logits, _dummy_hidden()
        return logits


def _greedy_reference(prompt: list[int], max_new: int, eos_id) -> list[int]:
    """Plain greedy decode on the SAME stub main model — the bit-identity target."""
    model = _StubMainModel()
    caches = model.make_caches()
    logits = model(mx.array(prompt), caches=caches, offset=0)
    cur = int(mx.argmax(logits[0, -1]).item())
    out = [cur]
    q = len(prompt) - 1
    while len(out) < max_new and (eos_id is None or cur != eos_id):
        logits = model(mx.array([cur]), caches=caches, offset=q + 1)
        q += 1
        cur = int(mx.argmax(logits[0, -1]).item())
        out.append(cur)
    out = out[:max_new]
    if eos_id is not None and eos_id in out:
        out = out[: out.index(eos_id) + 1]
    return out


def test_perfect_leftmost_w2d2() -> None:
    """``(W=2, D=2)`` perfect-leftmost MTP: spec output bit-identical to greedy, mean_accept = D+1."""
    embed = mx.zeros((VOCAB, HC * DIM))
    head = mx.zeros((VOCAB, HC * DIM))
    prompt = [2, 5, 7]
    greedy = _greedy_reference(prompt, MAXN, eos_id=None)
    spec, st = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(2), embed, head, prompt,
                                  width=2, depth=2, max_new=MAXN, eos_id=None, batched=False)
    assert spec == greedy, f"spec != greedy: spec={spec} greedy={greedy}"
    assert abs(st["mean_accept"] - 3.0) < 1e-9, (
        f"mean_accept expected 3 (depth+1), got {st['mean_accept']}"
    )
    assert st["width"] == 2 and st["depth"] == 2 and st["paths_per_round"] == 4
    print(f"[OK] perfect-leftmost W=2 D=2: spec==greedy, mean_accept={st['mean_accept']:.2f} "
          f"rounds={st['rounds']} paths/round={st['paths_per_round']}")


def test_perfect_leftmost_w4d2() -> None:
    """``(W=4, D=2)`` perfect-leftmost: same losslessness + mean_accept holds at a wider tree."""
    embed = mx.zeros((VOCAB, HC * DIM))
    head = mx.zeros((VOCAB, HC * DIM))
    prompt = [2, 5, 7]
    greedy = _greedy_reference(prompt, MAXN, eos_id=None)
    spec, st = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(4), embed, head, prompt,
                                  width=4, depth=2, max_new=MAXN, eos_id=None, batched=False)
    assert spec == greedy, f"spec != greedy: spec={spec} greedy={greedy}"
    assert abs(st["mean_accept"] - 3.0) < 1e-9, (
        f"mean_accept expected 3 (depth+1), got {st['mean_accept']}"
    )
    assert st["paths_per_round"] == 16
    print(f"[OK] perfect-leftmost W=4 D=2: spec==greedy, mean_accept={st['mean_accept']:.2f} "
          f"rounds={st['rounds']} paths/round={st['paths_per_round']}")


def test_wrong_all_w2d2() -> None:
    """``(W=2, D=2)`` wrong-all MTP: still bit-identical to greedy, mean_accept = 1."""
    embed = mx.zeros((VOCAB, HC * DIM))
    head = mx.zeros((VOCAB, HC * DIM))
    prompt = [2, 5, 7]
    greedy = _greedy_reference(prompt, MAXN, eos_id=None)
    spec, st = spec_generate_tree(_StubMainModel(), _WrongAllMTP(2), embed, head, prompt,
                                  width=2, depth=2, max_new=MAXN, eos_id=None, batched=False)
    assert spec == greedy, f"spec != greedy: spec={spec} greedy={greedy}"
    assert abs(st["mean_accept"] - 1.0) < 1e-9, (
        f"mean_accept expected 1 (all wrong), got {st['mean_accept']}"
    )
    print(f"[OK] wrong-all W=2 D=2: spec==greedy, mean_accept={st['mean_accept']:.2f} (≈1)")


def test_width1_matches_spec_generate_k() -> None:
    """``width=1`` degenerates to a chain — output + stats identical to spec_generate_k(k=depth)."""
    embed = mx.zeros((VOCAB, HC * DIM))
    head = mx.zeros((VOCAB, HC * DIM))
    prompt = [2, 5, 7]
    # Use a width-1 PerfectLeftmostMTP — its top-1 IS the greedy, so chain accepts all depth drafts.
    spec_tree, st_tree = spec_generate_tree(
        _StubMainModel(), _PerfectLeftmostMTP(1), embed, head, prompt,
        width=1, depth=2, max_new=MAXN, eos_id=None, batched=False,
    )
    spec_k, st_k = spec_generate_k(_StubMainModel(), _PerfectLeftmostMTP(1), embed, head, prompt,
                                   k=2, max_new=MAXN, eos_id=None)
    assert spec_tree == spec_k, f"width=1 spec != spec_generate_k(k=2): {spec_tree} vs {spec_k}"
    # spec_generate_k reports "k" in stats; spec_generate_tree reports "width" / "depth" — compare
    # the bit-identical pieces (tokens + rounds + mean_accept).
    assert st_tree["rounds"] == st_k["rounds"]
    assert abs(st_tree["mean_accept"] - st_k["mean_accept"]) < 1e-9
    print(f"[OK] width=1 == spec_generate_k(k=depth): "
          f"tokens match (n={len(spec_tree)}), mean_accept={st_tree['mean_accept']:.2f}")


def test_eos_stops() -> None:
    """eos terminates the stream at the first emitted eos (inclusive), matching plain greedy."""
    embed = mx.zeros((VOCAB, HC * DIM))
    head = mx.zeros((VOCAB, HC * DIM))
    prompt = [2, 5, 7]                                  # chain: 10, 13, 16, ..., 40(eos)
    greedy_e = _greedy_reference(prompt, MAXN, eos_id=EOS)
    # perfect-leftmost MTP + eos
    spec_p, _ = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(2), embed, head, prompt,
                                   width=2, depth=2, max_new=MAXN, eos_id=EOS, batched=False)
    assert spec_p == greedy_e, f"perfect MTP + eos: spec != greedy: {spec_p} vs {greedy_e}"
    assert spec_p and spec_p[-1] == EOS and EOS not in spec_p[:-1]
    # wrong-all MTP + eos
    spec_w, _ = spec_generate_tree(_StubMainModel(), _WrongAllMTP(2), embed, head, prompt,
                                   width=2, depth=2, max_new=MAXN, eos_id=EOS, batched=False)
    assert spec_w == greedy_e, f"wrong MTP + eos: spec != greedy: {spec_w} vs {greedy_e}"
    print(f"[OK] eos stops (perfect + wrong): all bit-identical to greedy, "
          f"final={spec_p[-1] if spec_p else None} (=EOS)")


def test_truncate_pattern() -> None:
    """The per-path rollback pattern: every round records exactly ``W ** D`` rollbacks (one per
    path verify, each rolling back the full ``depth + 1`` drafts to round start). The commit-forward
    re-feeds the accepted prefix, advancing the cache to the committed offset (visible as a non-
    truncate ``append`` between rounds)."""
    embed = mx.zeros((VOCAB, HC * DIM))
    head = mx.zeros((VOCAB, HC * DIM))
    prompt = [2, 5, 7]
    base = _StubMainModel()
    cache = _StubCache()
    spec, st = spec_generate_tree(_WrapCache(base, cache), _PerfectLeftmostMTP(2), embed, head,
                                  prompt, width=2, depth=2, max_new=MAXN, eos_id=None, batched=False)
    paths_per_round = st["paths_per_round"]
    rounds = st["rounds"]
    # Each round contributes exactly ``paths_per_round`` truncations (one per path verify).
    expected_min = rounds * paths_per_round
    assert len(cache.truncations) >= expected_min, (
        f"expected ≥{expected_min} truncates ({rounds} rounds × {paths_per_round} paths), "
        f"got {len(cache.truncations)}"
    )
    # Every truncate drops exactly ``depth + 1`` tokens (the per-path verify's full drafts+cur).
    bad = [(frm, to) for frm, to in cache.truncations if frm - to != 3]   # depth+1=3 here
    assert not bad, f"per-path rollback dropped ≠ depth+1 tokens: {bad}"
    # Final cache offset reflects only committed tokens: prompt + len(spec) - 1 (the last emitted
    # bonus is in ``out`` but NOT yet re-fed to the cache — it becomes next round's ``cur``).
    expected_offset = len(prompt) + len(spec) - 1
    assert cache.offset == expected_offset, (
        f"cache offset {cache.offset} != expected {expected_offset} "
        f"(prompt {len(prompt)} + spec {len(spec)} - 1 tail)"
    )
    print(f"[OK] per-path rollback pattern: rounds={rounds} paths={paths_per_round} "
          f"truncates={len(cache.truncations)} (each drops {2 + 1} = depth+1), "
          f"final offset={cache.offset}")


def main() -> int:
    tests = [
        test_perfect_leftmost_w2d2,
        test_perfect_leftmost_w4d2,
        test_wrong_all_w2d2,
        test_width1_matches_spec_generate_k,
        test_eos_stops,
        test_truncate_pattern,
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
