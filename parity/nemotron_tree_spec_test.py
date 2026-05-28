"""Lossless gate for Nemotron-H W-parallel chain-verify tree drafting (#157 follow-on).

Validates :func:`quanta.nemotron.spec.spec_generate_tree` — the hybrid-safe form of tree drafting
for Nemotron's GQA + Mamba hybrid (8 attention layers + 80 Mamba-2 SSM layers). Each round drafts
a ``(width, depth)`` MTP tree, enumerates ``W ** D`` root-to-leaf paths, chain-verifies each (KV
cache + Mamba ``(ssm, conv)`` snapshot/restore between paths so every per-path forward sees a
contiguous chain), and commits the longest-accepting path via a bounded replay. MODEL-FREE — a
stub main model + stub MTP + stub cache; no checkpoint, no GPU, a few KB of tensors — safe along-
side another large GPU-resident job. The stub main model exposes BOTH the single-stream ``__call__``
and the ``batch_step`` shared-offset verify surface, so the output-equivalence + stats subtests
(1)–(4) run the DEFAULT ``batched=True`` verify (bit-identical to the sequential form, cross-checked
in ``parity/nemotron_batched_tree_verify_test.py``); the per-path rollback subtest (5) pins
``batched=False`` to assert the sequential cache-truncate pattern directly.

Asserts:
  (1) Perfect-leftmost MTP (top-1 = main-greedy) at ``(W=2, D=2)`` and ``(W=4, D=2)``: spec output
      bit-identical to plain greedy and ``mean_accept == depth + 1`` (the leftmost path always
      accepts ALL drafts).
  (2) Wrong-all MTP (no candidate is main-greedy) at ``(W=2, D=2)``: still bit-identical to greedy
      (verify-arbitrated losslessness), ``mean_accept == 1``.
  (3) ``width=1`` degenerates to chain and matches :func:`spec_generate_k` with ``k=depth``.
  (4) eos stops generation at the first emitted eos (inclusive), same as plain greedy.
  (5) Under ``batched=False`` the decode cache's ``truncate`` is driven exactly as the per-path
      verify expects: every round records ``W ** D`` per-path rollbacks (each dropping ``depth + 1``
      tokens to round start); the commit-replay grows the cache rather than truncating, so there is
      no final truncate, and the total cache offset at the end reflects only committed tokens. (The
      default ``batched=True`` path replicates the prefix instead of rolling it back — zero truncates
      on the original cache — gated separately in ``parity/nemotron_batched_tree_verify_test.py``.)

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

    def _copy(self) -> "_StubCache":
        """Structural-sharing copy for :func:`quanta.nemotron.batched_runtime.replicate_state` — a
        fresh cache at the same consumed length, independent of the original (a write on one does not
        touch the other). Mirrors the ``_StubCache._copy`` in ``nemotron_batched_tree_verify_test``."""
        new = _StubCache()
        new._len = self._len
        return new


class _StubMainModel:
    """Deterministic stub of the resident Nemotron model: greedy(t) = t + STEP, with MTP-feature
    capture. Exposes BOTH spec contracts so :func:`spec_generate_tree` works on either flag:

    * single-stream ``__call__(token_ids, *, caches, ssm, conv, offset, capture_layers)`` →
      ``(logits [1,T,vocab], {last: hidden [T,hidden]})`` — drives the prefill, the commit-replay,
      and the ``batched=False`` per-path verify;
    * ``batch_step(tokens, *, replicas, offset, capture_layer)`` → ``(logits [B,1,vocab],
      hidden [B,1,hidden] or None)`` — drives the default ``batched=True`` verify, mirroring
      :meth:`quanta.nemotron.batched_runtime.NemotronBatchedResidentModel.batch_step` (shared offset
      across replicas).

    ``make_caches`` returns the Nemotron triple ``([cache], None, None)`` — the per-layer-list ``caches``
    form :func:`quanta.nemotron.batched_runtime.replicate_state` requires — with ``ssm``/``conv`` left
    ``None`` (the stub has no Mamba state), so the spec loop's snapshot/restore paths are no-ops while
    the KV rollback (``batched=False``) and the batched replicate (``batched=True``) ARE exercised.
    Folds the ``_StubMainModel`` + ``_NemotronStateAdapter`` pair from
    ``parity/nemotron_batched_tree_verify_test.py`` into one class (no adapter needed here)."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(num_hidden_layers=NL)
        self.num_layers = NL
        self.calls: list[tuple[str, tuple[int, ...], int]] = []
        self.cache: _StubCache | None = None

    def make_caches(self) -> tuple[list, None, None]:
        self.cache = _StubCache()
        return [self.cache], None, None

    def __call__(self, token_ids, *, caches=None, ssm=None, conv=None, offset=0, capture_layers=None):
        del ssm, conv                                     # stub has no Mamba state to thread
        single = caches[0] if isinstance(caches, list) and caches else caches   # unwrap per-layer list
        ids = [int(x) for x in (token_ids.tolist() if isinstance(token_ids, mx.array) else token_ids)]
        self.calls.append(("call", tuple(ids), offset))
        if single is not None and hasattr(single, "append"):
            single.append(len(ids))
        t = len(ids)
        logits = mx.stack([_row(_greedy_next(tok)) for tok in ids])[None]    # [1,T,vocab]
        if not capture_layers:
            return logits
        last = max(capture_layers)
        feat = mx.broadcast_to(mx.arange(t, dtype=mx.float32)[:, None], (t, HIDDEN))   # [T,hidden]
        return logits, {last: feat}

    def batch_step(self, tokens, *, replicas, offset, capture_layer=None):
        """Shared-offset batched verify step → ``(logits [B,1,vocab], hidden [B,1,hidden] or None)``.

        Mirrors :meth:`NemotronBatchedResidentModel.batch_step`: every replica's per-layer KV must sit
        at the shared ``offset`` (rule 6 / no silent drift), each advances one token, and the logits are
        the same deterministic ``greedy(t) = t + STEP`` as ``__call__`` — so the batched verify is
        bit-identical to the single-stream path. ``capture_layer`` is always set by ``batch_verify``,
        so the captured hidden is never ``None`` on the spec hot path."""
        b = len(tokens)
        if len(replicas) != b:
            raise ValueError(f"batch_step len mismatch: replicas={len(replicas)} tokens={b}")
        for s, (caches_s, _ssm_s, _conv_s) in enumerate(replicas):
            for c in (caches_s or []):                    # per-layer KV list (None on non-attn layers)
                if c is not None and c.offset != offset:
                    raise ValueError(f"batch_step replicas[{s}] offset={c.offset} != {offset}")
        self.calls.append(("batch_step", tuple(int(t) for t in tokens), offset))
        for caches_s, _ssm_s, _conv_s in replicas:
            for c in (caches_s or []):
                if c is not None and hasattr(c, "append"):
                    c.append(1)
        rows = mx.stack([_row(_greedy_next(int(t))) for t in tokens])           # [B, vocab]
        logits = rows[:, None]                                                  # [B,1,vocab]
        if capture_layer is None:
            return logits, None
        feat = mx.broadcast_to(mx.array(0.0, dtype=mx.float32), (b, 1, HIDDEN))  # [B,1,hidden]
        return logits, feat


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
    caches, _ssm, _conv = model.make_caches()       # ([cache], None, None) — pass the per-layer list
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
    """Per-path rollback pattern (``batched=False``): every round records ``W ** D`` truncates (each
    dropping ``depth + 1`` tokens to round start). No final commit-truncate (the replay grows the
    cache; it does not truncate). The cache offset at the end reflects only committed tokens. Pinned
    to the sequential path on purpose — the default ``batched=True`` verify replicates the prefix
    instead of rolling the original cache back (zero truncates on the original; that invariant is
    gated in ``parity/nemotron_batched_tree_verify_test.py::test_cache_invariance_after_batched``)."""
    prompt = [2, 5, 7]
    model = _StubMainModel()
    spec, st = spec_generate_tree(model, _PerfectLeftmostMTP(2), EMBED, HEAD, prompt,
                                  width=2, depth=2, max_new=MAXN, eos_id=None, batched=False)
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
