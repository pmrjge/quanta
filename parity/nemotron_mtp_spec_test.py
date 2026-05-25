"""Lossless gate for Nemotron-H native-MTP spec-decode (#40): spec_generate reproduces greedy decode.

MODEL-FREE — builds a STUB main model + a STUB MTP head over a tiny vocab (like the DSV4 / GLM / EAGLE
fake-runtime spec tests), with a stub decode cache supporting ``truncate`` / ``offset``. No checkpoint,
no GPU, a few KB of tensors — safe to run while a large model is resident. Because the verify step
makes losslessness hold for ANY MTP quality (the head only changes *speed*), this validates the
draft → verify → accept-or-bonus → rollback LOGIC, not the real weights.

The stub main model is a deterministic next-token chain ``g(t) = t + STEP`` with a clear argmax on that
token, so greedy decode is well-defined; over a verify window ``[cur, draft]`` it returns ``g(cur)``
then ``g(draft)`` per position (exactly what the rollback logic consumes), and a per-position hidden
capture so the feature plumbing is exercised. The stub embedding table is one-hot, so the stub MTP can
recover ``cur`` from the ``token_emb`` vector it is handed (the real MTP signature passes the embedding
*vector*, not the id). Asserts:
  (1) ``spec_generate`` output is BIT-IDENTICAL to a plain greedy reference decode on the same stub
      main model (losslessness — the core invariant), for a perfect MTP AND a wrong MTP;
  (2) a correct-drafting MTP makes accept length maximal (every draft accepted ⇒ ``mean_accept`` → 2)
      and drops ``rounds`` (fewer main forwards than tokens), while an always-wrong MTP still
      reproduces greedy with ``mean_accept`` ≈ 1;
  (3) on mismatch the spec loop rolls the cache back correctly (the cache offset tracks exactly the
      accepted positions, never the rejected draft) and still matches greedy;
  (4) eos stops generation (inclusive), matching greedy's eos stop, for both perfect and wrong MTP and
      for eos given as an int and as a set.

    uv run --with numpy python -m parity.nemotron_mtp_spec_test

    # deferred (needs the resident Nemotron model — do NOT run while another large job is resident):
    #   real MTP accept-rate / decode-speedup benchmark for #40 against NemotronResidentModel + the
    #   baked MTP head (NemotronMTP filled from the mtp.layers.0/1 tensors), asserting spec == greedy
    #   on real prose and reporting mean_accept, e.g.:
    #     model = NemotronResidentModel("~/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64")
    #     mtp   = load_nemotron_mtp(model.art, model.cfg)   # fills NemotronMTP from mtp.* tensors
    #     embed, head = model.embed_w, model.lm_head_w
    #     toks, st = spec_generate(model, mtp, embed, head, prompt_ids, max_new=256,
    #                              eos_id=model.cfg.eos_token_id)
    #     assert toks == greedy(model, prompt_ids, 256)     # losslessness on real weights
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from quanta.nemotron.spec import spec_generate

VOCAB = 64
DIM = VOCAB      # one-hot embedding: embed[t] has argmax t, so the stub MTP recovers the id
HIDDEN = 8       # main-model captured-hidden width (irrelevant to the stub MTP; only the id matters)
NL = 4           # stub "decoder layers" — only cfg.num_hidden_layers matters to spec_generate
STEP = 3         # deterministic chain: greedy(t) = t + STEP
EOS = 40
MAXN = 16

EMBED = mx.eye(VOCAB)            # [VOCAB, DIM] one-hot rows
HEAD = mx.zeros((VOCAB, HIDDEN))  # unused by the stub MTP (it emits fixed logits), but the real sig needs it


def _greedy_next(t: int) -> int:
    """The stub main model's greedy next token after token ``t`` (a fixed deterministic chain)."""
    return t + STEP


def _row(tok: int) -> mx.array:
    """A logit row over VOCAB with a clear argmax on ``tok`` (everything else far below)."""
    return mx.where(mx.arange(VOCAB) == tok, 30.0, -30.0)


class _StubCache:
    """Minimal stand-in for the decode cache: tracks a consumed length, supports ``truncate`` /
    ``offset``. The stub main model ignores cache *contents* (its logits depend only on the input
    tokens), so the cache need only honor the rollback surface the spec loop drives. ``append``
    advances the length; ``truncate`` rolls it back (and must be exact — losslessness depends on it)."""

    def __init__(self) -> None:
        self._len = 0
        self.truncations: list[int] = []
        self.max_len = 0

    @property
    def offset(self) -> int:
        return self._len

    def append(self, n: int) -> None:
        self._len += n
        self.max_len = max(self.max_len, self._len)

    def truncate(self, length: int) -> None:
        if length < 0:
            raise ValueError(f"truncate length {length} < 0")
        if length < self._len:
            self._len = length
            self.truncations.append(length)


class _StubMainModel:
    """Deterministic stub of the resident Nemotron model: greedy(t) = t + STEP, with the MTP-feature
    capture. ``__call__`` matches the consumed contract — ``(token_ids, *, caches, ssm, conv, offset,
    capture_layers)`` -> ``(logits [1,T,vocab], {last: hidden [T,hidden]})`` — advancing the stub cache
    by the input length so ``offset`` stays consistent after rollbacks, and returning a deterministic
    per-position hidden so the feature plumbing is exercised. Records every call for the test."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(num_hidden_layers=NL)
        self.num_layers = NL
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self.cache: _StubCache | None = None

    def make_caches(self) -> _StubCache:
        self.cache = _StubCache()
        return self.cache

    def __call__(self, token_ids, *, caches=None, ssm=None, conv=None, offset=0, capture_layers=None):
        ids = [int(x) for x in (token_ids.tolist() if isinstance(token_ids, mx.array) else token_ids)]
        self.calls.append((tuple(ids), offset))
        if caches is not None and hasattr(caches, "append"):
            caches.append(len(ids))
        t = len(ids)
        logits = mx.stack([_row(_greedy_next(tok)) for tok in ids])[None]   # [1,T,vocab]
        if not capture_layers:
            return logits
        last = max(capture_layers)
        # deterministic [T,hidden] feature; content is irrelevant to the stub MTP but shape must match
        feat = mx.broadcast_to(mx.arange(t, dtype=mx.float32)[:, None], (t, HIDDEN))
        return logits, {last: feat}


class _PerfectMTP:
    """A drafter that always predicts the main model's greedy token from the handed ``token_emb`` (the
    one-hot embedding of ``cur``) → every draft is accepted → mean_accept rises to 2. Mirrors the real
    surface ``mtp(prev_hidden, token_emb, head) -> (logits, new_hidden)``; ignores ``prev_hidden``
    content (only the *next* token determines the draft here). Returns a dummy ``new_hidden``."""

    def __call__(self, prev_hidden, token_emb, head):
        cur = int(mx.argmax(token_emb[0, 0]).item())          # recover the id from the one-hot embedding
        return _row(_greedy_next(cur))[None, None], prev_hidden  # [1,1,vocab], dummy new_hidden


class _WrongMTP:
    """A drafter that always proposes a token the main model would NOT pick → every draft is rejected
    → mean_accept ≈ 1, yet the output is still bit-identical to greedy (the verify guarantees it)."""

    def __call__(self, prev_hidden, token_emb, head):
        cur = int(mx.argmax(token_emb[0, 0]).item())
        wrong = (_greedy_next(cur) + 1) % VOCAB               # != greedy(cur)
        return _row(wrong)[None, None], prev_hidden


def _greedy_reference(model: _StubMainModel, prompt, max_new: int, eos_id=None) -> list[int]:
    """Plain greedy decode on the SAME stub main model — one token per forward, argmax each step,
    terminate at the first eos (inclusive). The bit-identity target for spec_generate."""
    stop = set() if eos_id is None else ({int(eos_id)} if isinstance(eos_id, int) else {int(s) for s in eos_id})
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
        for k, tok in enumerate(out):
            if tok in stop:
                return out[: k + 1]
    return out


def run() -> None:
    ok = True
    prompt = [2, 5, 7]               # last token 7 → chain 10,13,16,19,22,25,28,31,34,37,40(eos)

    # reference greedy decode (no eos) for the bit-identity checks
    greedy = _greedy_reference(_StubMainModel(), prompt, MAXN, eos_id=None)

    # (1)+(2a) perfect MTP: bit-identical to greedy AND every draft accepted (mean_accept → 2)
    m = _StubMainModel()
    spec_p, st_p = spec_generate(m, _PerfectMTP(), EMBED, HEAD, prompt, max_new=MAXN, eos_id=None)
    good = spec_p == greedy and st_p["mean_accept"] == 2.0 and st_p["max_accept"] == 2
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] perfect MTP: spec==greedy={spec_p == greedy} "
          f"mean_accept={st_p['mean_accept']:.2f} rounds={st_p['rounds']}")
    print(f"             greedy[:10]={greedy[:10]}")
    print(f"             spec  [:10]={spec_p[:10]}")
    # a perfect drafter must verify with fewer main forwards than tokens emitted (the speedup)
    n_main = len(m.calls)
    good = n_main < len(spec_p)
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] perfect MTP fewer main forwards than tokens: "
          f"forwards={n_main} tokens={len(spec_p)}")

    # (1)+(2b) wrong MTP: still bit-identical to greedy, no draft accepted (mean_accept ≈ 1)
    mw = _StubMainModel()
    spec_w, st_w = spec_generate(mw, _WrongMTP(), EMBED, HEAD, prompt, max_new=MAXN, eos_id=None)
    good = spec_w == greedy and abs(st_w["mean_accept"] - 1.0) < 1e-9
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] wrong MTP: spec==greedy={spec_w == greedy} "
          f"mean_accept={st_w['mean_accept']:.2f} (≈1)")

    # (3) rollback: the wrong MTP forces a rejected draft every round → the cache must be truncated
    #     back to exactly the accepted length each round (never leaving the rejected draft resident),
    #     and the final consumed length must equal prompt + emitted tokens (no drift).
    trunc_ok = len(mw.cache.truncations) == st_w["rounds"]            # one rollback per round
    # after a rejected draft (j=0) each round keeps (q+1)+1 consumed *input* positions; the consumed
    # offset is (last verified input position)+1 = len(prompt)+len(out)-1 (the final emitted token is
    # predicted, not yet fed back as an input). The rejected draft is never left in the cache.
    expect_off = len(prompt) + len(spec_w) - 1
    final_len_ok = mw.cache.offset == expect_off
    # the cache never grew past the verify window beyond the accepted length (no rejected draft left)
    max_ok = mw.cache.max_len == expect_off + 1                       # the single in-flight draft, then rolled back
    good = trunc_ok and final_len_ok and max_ok
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] rollback on mismatch: truncations={len(mw.cache.truncations)} "
          f"(rounds={st_w['rounds']}) final_offset={mw.cache.offset} expected={expect_off} "
          f"max_len={mw.cache.max_len}")

    # the perfect MTP accepts every draft → it should never need to roll back (truncate is a no-op
    # because the kept length already equals the consumed length).
    good = len(m.cache.truncations) == 0
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] perfect MTP never rolls back: truncations={len(m.cache.truncations)}")

    # (4) eos stops generation — greedy and spec both terminate at the first eos (inclusive)
    greedy_e = _greedy_reference(_StubMainModel(), prompt, MAXN, eos_id=EOS)
    spec_e, st_e = spec_generate(_StubMainModel(), _PerfectMTP(), EMBED, HEAD, prompt,
                                 max_new=MAXN, eos_id=EOS)
    good = (spec_e == greedy_e and len(spec_e) > 0 and spec_e[-1] == EOS and EOS not in spec_e[:-1])
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] eos (int) stops: spec==greedy={spec_e == greedy_e} "
          f"ends_with_eos={spec_e[-1] == EOS if spec_e else False} len={len(spec_e)}")
    print(f"             spec_eos={spec_e}")

    # wrong MTP + eos must ALSO match greedy's eos stop (losslessness under rejection + eos)
    spec_we, _ = spec_generate(_StubMainModel(), _WrongMTP(), EMBED, HEAD, prompt,
                               max_new=MAXN, eos_id=EOS)
    good = spec_we == greedy_e
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] wrong MTP + eos (int): spec==greedy={good}")

    # eos as a SET must behave identically (the stop surface accepts a collection)
    spec_es, _ = spec_generate(_StubMainModel(), _PerfectMTP(), EMBED, HEAD, prompt,
                               max_new=MAXN, eos_id={EOS})
    good = spec_es == greedy_e
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] eos (set) stops: spec==greedy={good}")

    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
