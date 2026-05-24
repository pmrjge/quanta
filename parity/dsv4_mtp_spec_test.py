"""Lossless gate for DSV4 native-MTP spec-decode (#78): spec_generate reproduces greedy decode.

MODEL-FREE — builds a STUB main model and a STUB MTP head over a tiny vocab (like the EAGLE / oMLX
fake runtimes), with a stub decode cache supporting ``truncate`` / ``offset``. No checkpoint, no GPU,
a few KB of tensors — safe to run while a large model is resident. The verify step makes losslessness
hold for ANY MTP quality (the head only changes *speed*), so this validates the draft → verify →
accept-or-bonus → rollback LOGIC, not the real weights.

The stub main model is a deterministic next-token chain ``next = t + STEP`` with a spike on that
token, so greedy decode is well-defined; per-position over a verify window ``[cur, draft]`` it returns
``greedy(cur)`` then ``greedy(draft)`` (exactly what the rollback logic consumes). Asserts:
  (1) ``spec_generate`` output is BIT-IDENTICAL to a plain greedy reference decode on the same stub
      main model (losslessness — the core invariant), for a perfect MTP AND a wrong MTP;
  (2) a correct-drafting MTP raises ``mean_accept`` (>1, →2 here), and an always-wrong MTP still
      reproduces greedy with ``mean_accept`` ≈ 1;
  (3) eos stops generation (the chain hits eos; spec terminates there, matching greedy).

    uv run --with numpy python -m parity.dsv4_mtp_spec_test

    # deferred (needs the resident DSV4 model — do NOT run while another large job is resident):
    #   real MTP accept-rate / decode-speedup benchmark for #78 against DSV4ResidentModel +
    #   the baked MTP head, asserting spec == greedy on real prose and reporting mean_accept.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from quanta.dsv4.spec import spec_generate

VOCAB = 64
HC = 4
DIM = 8
NL = 3          # stub "decoder layers" — only cfg.num_hidden_layers matters to spec_generate
STEP = 3        # deterministic chain: greedy(t) = t + STEP
EOS = 40
MAXN = 16


def _greedy_next(t: int) -> int:
    """The stub main model's greedy next token after token ``t`` (a fixed deterministic chain)."""
    return t + STEP


def _row(tok: int) -> mx.array:
    """A logit row over VOCAB with a clear argmax on ``tok`` (everything else far below)."""
    return mx.where(mx.arange(VOCAB) == tok, 30.0, -30.0)


class _StubCache:
    """Minimal stand-in for ``DSV4Cache``: tracks a length, supports ``truncate`` / ``offset``.

    The stub main model ignores cache *contents* (its logits depend only on the input tokens), so the
    cache need only honor the rollback surface the spec loop drives. ``append`` advances the length;
    ``truncate`` rolls it back (and must be exact — the losslessness proof depends on it)."""

    def __init__(self) -> None:
        self._len = 0
        self.truncations: list[int] = []

    @property
    def offset(self) -> int:
        return self._len

    def append(self, n: int) -> None:
        self._len += n

    def truncate(self, length: int) -> None:
        if length < 0:
            raise ValueError(f"truncate length {length} < 0")
        if length < self._len:
            self._len = length
            self.truncations.append(length)


class _StubMainModel:
    """Deterministic stub of ``DSV4ResidentModel``: greedy(t) = t + STEP, with the MTP-feature capture.

    ``__call__`` matches the consumed contract — ``(token_ids, *, caches, offset, capture_layers)`` ->
    ``(logits [1,T,vocab], {last: hidden [T,hc,dim]})``. It advances the stub cache by the input length
    (so ``offset`` stays consistent after rollbacks) and returns a deterministic per-position capture so
    the spec loop's feature plumbing is exercised. Records each call for the test to inspect."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(num_hidden_layers=NL)
        self.num_layers = NL
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def make_caches(self) -> _StubCache:
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
        # deterministic [T,hc,dim] feature; content is irrelevant to the stub MTP but shape must match
        feat = mx.broadcast_to(mx.arange(t, dtype=mx.float32)[:, None, None], (t, HC, DIM))
        return logits, {last: feat}


class _PerfectMTP:
    """A drafter that always predicts the main model's greedy token from ``next_ids`` (= ``cur``) →
    every draft is accepted → mean_accept rises to 2. Mirrors ``mtp(prev_hidden, next_ids, embed,
    head)``; ignores ``prev_hidden`` content (only the *next* token determines the draft here)."""

    def __call__(self, prev_hidden, next_ids, embed, head):
        cur = int(next_ids[0, 0].item())
        return _row(_greedy_next(cur))[None, None]            # [1,1,vocab]


class _WrongMTP:
    """A drafter that always proposes a token the main model would NOT pick → every draft is rejected
    → mean_accept ≈ 1, yet the output is still bit-identical to greedy (the verify guarantees it)."""

    def __call__(self, prev_hidden, next_ids, embed, head):
        cur = int(next_ids[0, 0].item())
        wrong = (_greedy_next(cur) + 1) % VOCAB               # != greedy(cur)
        return _row(wrong)[None, None]


def _greedy_reference(model: _StubMainModel, prompt, max_new: int, eos_id: int | None) -> list[int]:
    """Plain greedy decode on the SAME stub main model — one token per forward, argmax each step,
    terminate at the first eos (inclusive). The bit-identity target for spec_generate."""
    caches = model.make_caches()
    logits = model(mx.array(prompt), caches=caches, offset=0)
    cur = int(mx.argmax(logits[0, -1]).item())
    out = [cur]
    q = len(prompt) - 1
    while len(out) < max_new and not (eos_id is not None and cur == eos_id):
        logits = model(mx.array([cur]), caches=caches, offset=q + 1)
        q += 1
        cur = int(mx.argmax(logits[0, -1]).item())
        out.append(cur)
    out = out[:max_new]
    if eos_id is not None and eos_id in out:
        out = out[: out.index(eos_id) + 1]
    return out


def run() -> None:
    ok = True
    embed = mx.zeros((VOCAB, DIM))   # unused by the stub MTP, but the real signature passes them
    head = mx.zeros((VOCAB, DIM))
    prompt = [2, 5, 7]               # last token 7 → chain 10,13,16,19,22,25,28,31,34,37,40(eos)

    # reference greedy decode (no eos) for the bit-identity checks
    greedy = _greedy_reference(_StubMainModel(), prompt, MAXN, eos_id=None)

    # (1)+(2a) perfect MTP: bit-identical to greedy AND mean_accept rises to 2
    m = _StubMainModel()
    spec_p, st_p = spec_generate(m, _PerfectMTP(), embed, head, prompt, max_new=MAXN, eos_id=None)
    good = spec_p == greedy and st_p["mean_accept"] > 1.0
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] perfect MTP: spec==greedy={spec_p == greedy} "
          f"mean_accept={st_p['mean_accept']:.2f} rounds={st_p['rounds']}")
    print(f"             greedy[:10]={greedy[:10]}")
    print(f"             spec  [:10]={spec_p[:10]}")
    # a perfect drafter must verify ~half as often as greedy emits tokens
    n_main = len(m.calls)
    good = n_main < len(spec_p)
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] perfect MTP fewer main forwards than tokens: "
          f"forwards={n_main} tokens={len(spec_p)}")

    # (1)+(2b) wrong MTP: still bit-identical to greedy, mean_accept ≈ 1
    spec_w, st_w = spec_generate(_StubMainModel(), _WrongMTP(), embed, head, prompt,
                                 max_new=MAXN, eos_id=None)
    good = spec_w == greedy and abs(st_w["mean_accept"] - 1.0) < 1e-9
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] wrong MTP: spec==greedy={spec_w == greedy} "
          f"mean_accept={st_w['mean_accept']:.2f} (≈1)")

    # (3) eos stops generation — greedy and spec both terminate at the first eos (inclusive)
    greedy_e = _greedy_reference(_StubMainModel(), prompt, MAXN, eos_id=EOS)
    spec_e, st_e = spec_generate(_StubMainModel(), _PerfectMTP(), embed, head, prompt,
                                 max_new=MAXN, eos_id=EOS)
    good = (spec_e == greedy_e and len(spec_e) > 0 and spec_e[-1] == EOS
            and EOS not in spec_e[:-1])
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] eos stops: spec==greedy={spec_e == greedy_e} "
          f"ends_with_eos={spec_e[-1] == EOS if spec_e else False} len={len(spec_e)}")
    print(f"             spec_eos={spec_e}")

    # the wrong MTP with eos must ALSO match greedy's eos stop (losslessness under rejection + eos)
    spec_we, _ = spec_generate(_StubMainModel(), _WrongMTP(), embed, head, prompt,
                               max_new=MAXN, eos_id=EOS)
    good = spec_we == greedy_e
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] wrong MTP + eos: spec==greedy={good}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
