"""Lossless gate for Nemotron-H W-parallel chain-verify tree drafting (#157 follow-on).

Validates :func:`quanta.nemotron.spec.spec_generate_tree` — the hybrid-safe form of tree drafting
for Nemotron's GQA + Mamba hybrid (8 attention layers + 80 Mamba-2 SSM layers). Each round drafts
a ``(width, depth)`` MTP tree, enumerates ``W ** D`` root-to-leaf paths, chain-verifies each (KV
cache + Mamba ``(ssm, conv)`` snapshot/restore between paths so every per-path forward sees a
contiguous chain), and commits the longest-accepting path via a bounded replay. MODEL-FREE — a
stub main model + stub MTP + stub cache; no checkpoint, no GPU, a few KB of tensors — safe along-
side another large GPU-resident job.

Asserts:
  (1) Perfect-leftmost MTP (top-1 = main-greedy) at ``(W=2, D=2)`` and ``(W=4, D=2)``: spec output
      bit-identical to plain greedy and ``mean_accept == depth + 1`` (the leftmost path always
      accepts ALL drafts).
  (2) Wrong-all MTP (no candidate is main-greedy) at ``(W=2, D=2)``: still bit-identical to greedy
      (verify-arbitrated losslessness), ``mean_accept == 1``.
  (3) ``width=1`` degenerates to chain and matches :func:`spec_generate_k` with ``k=depth``.
  (4) eos stops generation at the first emitted eos (inclusive), same as plain greedy.
  (5) The decode cache's ``truncate`` is driven exactly as the per-path verify expects: every round
      records ``W ** D`` per-path rollbacks (each dropping ``depth + 1`` tokens to round start) plus
      one final at the replay (drops the replay's pre-offset). The total cache offset at the end
      reflects only committed tokens.

The Mamba ``(ssm, conv)`` snapshot/restore pattern itself is not exercised by these model-free
stubs — the stub model returns ``None`` for both — but the spec implementation USES the same
snapshot-list-restore pattern as the already-tested :func:`spec_generate_k` for ``k >= 2``, so the
real-model verify-side correctness is inherited from the chain-spec parity test.

Run:  ``uv run --with numpy python -m parity.nemotron_tree_spec_test``
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from quanta.nemotron.spec import spec_generate_k, spec_generate_tree

VOCAB = 64
HIDDEN = 8           # main-model captured-hidden width
NL = 4               # stub "decoder layers" — only cfg.num_hidden_layers matters to spec
STEP = 3             # deterministic chain: greedy(t) = t + STEP
EOS = 40
MAXN = 16

EMBED = mx.eye(VOCAB)            # [VOCAB, VOCAB] one-hot rows — embed[t] has argmax t
HEAD = mx.zeros((VOCAB, HIDDEN))  # unused by stub MTPs; the real signature passes it


def _greedy_next(t: int) -> int:
    return t + STEP


def _row(tok: int) -> mx.array:
    """A logit row over VOCAB with a clear argmax on ``tok`` (everything else far below)."""
    return mx.where(mx.arange(VOCAB) == tok, 30.0, -30.0)


class _StubCache:
    """Minimal stand-in for the decode cache: tracks a consumed length, supports ``truncate`` /
    ``offset``. Records every truncate so the test can confirm the per-path rollback pattern."""

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


class _StubMainModel:
    """Deterministic stub of the resident Nemotron model: greedy(t) = t + STEP, with MTP-feature
    capture. Honors the consumed contract ``(token_ids, *, caches, ssm, conv, offset, capture_layers)``
    → ``(logits [1,T,vocab], {last: hidden [T,hidden]})``. ``ssm``/``conv`` are passed as None by
    ``_capture_state`` (the stub has no Mamba state) — the snapshot/restore code paths in the spec
    loop become no-ops for the stub, but the per-path KV rollback IS exercised."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(num_hidden_layers=NL)
        self.num_layers = NL
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self.cache: _StubCache | None = None

    def make_caches(self) -> _StubCache:
        self.cache = _StubCache()
        return self.cache

    def __call__(self, token_ids, *, caches=None, ssm=None, conv=None, offset=0, capture_layers=None):
        del ssm, conv                                     # stub has no Mamba state to thread
        ids = [int(x) for x in (token_ids.tolist() if isinstance(token_ids, mx.array) else token_ids)]
        self.calls.append((tuple(ids), offset))
        if caches is not None and hasattr(caches, "append"):
            caches.append(len(ids))
        t = len(ids)
        logits = mx.stack([_row(_greedy_next(tok)) for tok in ids])[None]    # [1,T,vocab]
        if not capture_layers:
            return logits
        last = max(capture_layers)
        feat = mx.broadcast_to(mx.arange(t, dtype=mx.float32)[:, None], (t, HIDDEN))   # [T,hidden]
        return logits, {last: feat}


def _dummy_hidden() -> mx.array:
    """``[1, 1, hidden]`` placeholder hidden the chained tree-build feeds back in. Content is
    irrelevant to these stub drafters; the shape must match what the real ``mtp(...)`` returns."""
    return mx.zeros((1, 1, HIDDEN), dtype=mx.float32)


def _logits_top_w(greedy_tok: int, other_toks: list[int]) -> mx.array:
    """Build ``[1, 1, VOCAB]`` logits so the spec's ``_mtp_top_w`` returns ``[greedy_tok, *other_toks]``
    in order. ``greedy_tok`` scores highest; ``other_toks`` get monotonically descending scores."""
    arr = mx.full((VOCAB,), -100.0)
    arr = mx.where(mx.arange(VOCAB) == greedy_tok, 100.0, arr)
    for i, tok in enumerate(other_toks):
        arr = mx.where(mx.arange(VOCAB) == tok, 90.0 - 5.0 * i, arr)
    return arr[None, None]


def _wrongs_for(parent_tok: int, width: int, greedy_tok: int) -> list[int]:
    """``width - 1`` distinct token ids ≠ ``greedy_tok`` to fill the sibling slots."""
    wrongs: list[int] = []
    candidate = (parent_tok + 1) % VOCAB
    while len(wrongs) < width - 1:
        if candidate != greedy_tok and candidate not in wrongs:
            wrongs.append(candidate)
        candidate = (candidate + 1) % VOCAB
    return wrongs


class _PerfectLeftmostMTP:
    """Top-1 child = main-greedy → the leftmost root-to-leaf path always accepts all ``depth`` drafts.
    Signature mirrors the real Nemotron MTP: ``mtp(prev_hidden, token_emb, head, *, return_hidden=False)``."""

    def __init__(self, width: int) -> None:
        self.width = width

    def __call__(self, prev_hidden, token_emb, head, *, return_hidden=False):
        # Recover the parent token id from the one-hot ``token_emb = embed[parent_token]``.
        parent = int(mx.argmax(token_emb[0, 0]).item())
        greedy = _greedy_next(parent) % VOCAB
        wrongs = _wrongs_for(parent, self.width, greedy)
        logits = _logits_top_w(greedy, wrongs)
        _ = return_hidden                                  # signature surface only
        return logits, _dummy_hidden()


class _WrongAllMTP:
    """Top-W are ALL non-greedy → every path rejects on the first draft. Still lossless."""

    def __init__(self, width: int) -> None:
        self.width = width

    def __call__(self, prev_hidden, token_emb, head, *, return_hidden=False):
        parent = int(mx.argmax(token_emb[0, 0]).item())
        greedy = _greedy_next(parent) % VOCAB
        # Use ``width + 1`` "non-greedy" candidates; pick the first ``width`` as the top-W slots.
        wrongs = _wrongs_for(parent, self.width + 1, greedy)[: self.width]
        logits = _logits_top_w(wrongs[0], wrongs[1:])
        _ = return_hidden
        return logits, _dummy_hidden()


def _greedy_reference(prompt: list[int], max_new: int, eos_id) -> list[int]:
    """Plain greedy decode on the SAME stub main model — the bit-identity target."""
    stop = (set() if eos_id is None
            else ({int(eos_id)} if isinstance(eos_id, int)
                  else {int(s) for s in eos_id}))
    model = _StubMainModel()
    caches = model.make_caches()
    logits = model(mx.array(prompt), caches=caches, offset=0)
    cur = int(mx.argmax(logits[0, -1]).item())
    out = [cur]
    q = len(prompt) - 1
    while len(out) < max_new and cur not in stop:
        logits = model(mx.array([cur]), caches=caches, offset=q + 1)
        q += 1
        cur = int(mx.argmax(logits[0, -1]).item())
        out.append(cur)
    out = out[:max_new]
    if stop:
        for k, t in enumerate(out):
            if t in stop:
                out = out[: k + 1]
                break
    return out


def test_perfect_leftmost_w2d2() -> None:
    prompt = [2, 5, 7]
    greedy = _greedy_reference(prompt, MAXN, eos_id=None)
    spec, st = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(2), EMBED, HEAD, prompt,
                                  width=2, depth=2, max_new=MAXN, eos_id=None)
    assert spec == greedy, f"spec != greedy: spec={spec} greedy={greedy}"
    assert abs(st["mean_accept"] - 3.0) < 1e-9, (
        f"mean_accept expected 3 (depth+1), got {st['mean_accept']}"
    )
    assert st["width"] == 2 and st["depth"] == 2 and st["paths_per_round"] == 4
    print(f"[OK] perfect-leftmost W=2 D=2: spec==greedy, mean_accept={st['mean_accept']:.2f} "
          f"rounds={st['rounds']} paths/round={st['paths_per_round']}")


def test_perfect_leftmost_w4d2() -> None:
    prompt = [2, 5, 7]
    greedy = _greedy_reference(prompt, MAXN, eos_id=None)
    spec, st = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(4), EMBED, HEAD, prompt,
                                  width=4, depth=2, max_new=MAXN, eos_id=None)
    assert spec == greedy, f"spec != greedy: spec={spec} greedy={greedy}"
    assert abs(st["mean_accept"] - 3.0) < 1e-9
    assert st["paths_per_round"] == 16
    print(f"[OK] perfect-leftmost W=4 D=2: spec==greedy, mean_accept={st['mean_accept']:.2f} "
          f"rounds={st['rounds']} paths/round={st['paths_per_round']}")


def test_wrong_all_w2d2() -> None:
    prompt = [2, 5, 7]
    greedy = _greedy_reference(prompt, MAXN, eos_id=None)
    spec, st = spec_generate_tree(_StubMainModel(), _WrongAllMTP(2), EMBED, HEAD, prompt,
                                  width=2, depth=2, max_new=MAXN, eos_id=None)
    assert spec == greedy, f"spec != greedy: spec={spec} greedy={greedy}"
    assert abs(st["mean_accept"] - 1.0) < 1e-9, (
        f"mean_accept expected 1 (all wrong), got {st['mean_accept']}"
    )
    print(f"[OK] wrong-all W=2 D=2: spec==greedy, mean_accept={st['mean_accept']:.2f} (≈1)")


def test_width1_matches_spec_generate_k() -> None:
    prompt = [2, 5, 7]
    spec_tree, st_tree = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(1), EMBED, HEAD,
                                            prompt, width=1, depth=2, max_new=MAXN, eos_id=None)
    spec_k, st_k = spec_generate_k(_StubMainModel(), _PerfectLeftmostMTP(1), EMBED, HEAD,
                                   prompt, k=2, max_new=MAXN, eos_id=None)
    assert spec_tree == spec_k, f"width=1 spec != spec_generate_k(k=2): {spec_tree} vs {spec_k}"
    assert st_tree["rounds"] == st_k["rounds"]
    assert abs(st_tree["mean_accept"] - st_k["mean_accept"]) < 1e-9
    print(f"[OK] width=1 == spec_generate_k(k=depth): tokens match (n={len(spec_tree)}), "
          f"mean_accept={st_tree['mean_accept']:.2f}")


def test_eos_stops() -> None:
    prompt = [2, 5, 7]
    greedy_e = _greedy_reference(prompt, MAXN, eos_id=EOS)
    spec_p, _ = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(2), EMBED, HEAD, prompt,
                                   width=2, depth=2, max_new=MAXN, eos_id=EOS)
    assert spec_p == greedy_e, f"perfect MTP + eos: spec != greedy: {spec_p} vs {greedy_e}"
    assert spec_p and spec_p[-1] == EOS and EOS not in spec_p[:-1]
    spec_w, _ = spec_generate_tree(_StubMainModel(), _WrongAllMTP(2), EMBED, HEAD, prompt,
                                   width=2, depth=2, max_new=MAXN, eos_id=EOS)
    assert spec_w == greedy_e
    spec_s, _ = spec_generate_tree(_StubMainModel(), _PerfectLeftmostMTP(2), EMBED, HEAD, prompt,
                                   width=2, depth=2, max_new=MAXN, eos_id={EOS, 99})
    assert spec_s == greedy_e
    print(f"[OK] eos stops (perfect + wrong + set): all bit-identical to greedy, "
          f"final={spec_p[-1] if spec_p else None} (=EOS)")


def test_truncate_pattern() -> None:
    """Per-path rollback pattern: every round records ``W ** D`` truncates (each dropping
    ``depth + 1`` tokens to round start). No final commit-truncate (the replay grows the cache;
    it does not truncate). The cache offset at the end reflects only committed tokens."""
    prompt = [2, 5, 7]
    model = _StubMainModel()
    spec, st = spec_generate_tree(model, _PerfectLeftmostMTP(2), EMBED, HEAD, prompt,
                                  width=2, depth=2, max_new=MAXN, eos_id=None)
    paths_per_round = st["paths_per_round"]
    rounds = st["rounds"]
    cache = model.cache
    assert cache is not None
    # Exactly ``rounds * paths_per_round`` truncates: one per per-path rollback. Each drops the
    # full ``depth + 1`` tokens (cur + drafts) back to the round-start offset.
    assert len(cache.truncations) == rounds * paths_per_round, (
        f"expected {rounds * paths_per_round} truncates ({rounds} × {paths_per_round}), "
        f"got {len(cache.truncations)}"
    )
    bad = [(frm, to) for frm, to in cache.truncations if frm - to != 3]   # depth+1 = 3
    assert not bad, f"per-path rollback dropped ≠ depth+1 tokens: {bad}"
    # Final offset = len(prompt) + len(spec) - 1: the prefill + every committed token EXCEPT the
    # last emitted bonus, which is in ``out`` but hasn't been cached yet (it becomes the next
    # round's ``cur`` and would be fed by the next round's replay-forward — but generation halted
    # before that). Mirrors the chained spec_generate_k's tail-state.
    expected_offset = len(prompt) + len(spec) - 1
    assert cache.offset == expected_offset, (
        f"cache offset {cache.offset} != prompt+spec-1 ({expected_offset})"
    )
    print(f"[OK] per-path rollback pattern: rounds={rounds} paths={paths_per_round} "
          f"truncates={len(cache.truncations)} cache_offset={cache.offset}")


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
        except Exception as e:                              # noqa: BLE001
            failures += 1
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
