"""Gate: Qwen3.5 sampling math + the generate loop bound — model-free, tiny tensors, ~0 GB.

Answers two questions without loading the model (safe to run while a 398 GB capture is GPU-resident):

  (a) **Sampling math** (:func:`quanta.qwen35.generate.sample_logits`): greedy == argmax;
      top-k / top-p / min-p restrict the support to exactly the right tokens; and a ``seed`` makes
      temperature sampling reproducible (same seed → identical draws, different seed differs). This is
      the same math :func:`quanta.dsv4.generate.sample_logits` / the oMLX shim use, so the engine and
      the standalone generator agree token-for-token.
  (b) **Generate-loop bound** (:func:`quanta.qwen35.generate.generate`): driven by a STUB model (a tiny
      object exposing ``.cfg`` / ``.num_layers`` and the single-token ``__call__(token_ids, caches=,
      offset=)`` contract, returning fixed logits) — ``generate`` seeds the cache by stepping the prompt
      (asserted via recorded offsets) then decodes; an eos-less run is bounded by ``max_new_tokens`` (a
      loop can never run unbounded), a stub whose argmax is ``eos_id`` stops early, an ``eos_id`` SET is
      honored, and an empty prompt fails loud. A stub cache is passed in, so the real decode module is
      not imported and nothing model-sized is allocated.

    uv run --with numpy python -m parity.qwen35_generate_test
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from quanta.qwen35.generate import generate, sample_logits

V = 64  # tiny vocab


class _FixedModel:
    """Returns the SAME logits every call; ``argmax == arg``. ``__call__`` is the single-token decode
    contract (``caches``/``offset``) — it ignores the (stub) cache but records each call's offset and
    token, so the test can assert ``generate`` seeds the cache by stepping the prompt then decodes.
    ``.cfg``/``.num_layers`` satisfy ``generate``'s cache construction (unused when a cache is passed)."""

    def __init__(self, arg: int, n_layers: int = 2) -> None:
        self.cfg = SimpleNamespace(num_hidden_layers=n_layers)
        self.num_layers = n_layers
        self._arg = arg
        self.calls: list[tuple[int, int]] = []  # (offset, token) per __call__

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        flat = mx.array(token_ids).reshape(-1)
        t = int(flat.shape[0])
        self.calls.append((int(offset), int(flat[0].item())))
        row = (mx.arange(V) == self._arg).astype(mx.float32) * 60.0 - 30.0  # argmax == arg
        return mx.broadcast_to(row, (1, t, V))


class _StubCache:
    """A non-``None`` cache placeholder so ``generate`` skips building a real ``Qwen35Cache`` (keeps the
    gate model-free); the stub model never touches it."""


def _support(logits: mx.array, **kw) -> set[int]:
    """The set of token ids sample_logits can ever draw (non -inf after filtering), via a many-sample
    draw. Greedy is excluded; here temperature>0 so the filtered logits define the support."""
    keys = mx.random.split(mx.random.key(0), 4000)
    return {int(sample_logits(logits, temperature=1.0, key=k, **kw).item()) for k in keys}


def run() -> None:
    ok = True

    # (a) greedy == argmax (temperature == 0), for several arglocs
    good = all(int(sample_logits(
        ((mx.arange(V) == a).astype(mx.float32) * 5.0), temperature=0.0).item()) == a
        for a in (0, 7, V - 1))
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] greedy == argmax")

    # build distinct-valued logits so thresholds are unambiguous: logit[i] = i
    ramp = mx.arange(V).astype(mx.float32)

    # (a) top_k keeps exactly the k largest (ids V-k .. V-1)
    sup = _support(ramp, top_k=5)
    good = sup == set(range(V - 5, V))
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] top_k=5 support={sorted(sup)} (expect {list(range(V - 5, V))})")

    # (a) min_p prunes all but the peak on a peaked distribution
    peak = mx.where(mx.arange(V) == V - 1, 50.0, 0.0)
    sup = _support(peak, min_p=0.5)
    good = sup == {V - 1}
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] min_p=0.5 on a peak support={sorted(sup)} (expect [{V - 1}])")

    # (a) top_p nucleus on a peak -> just the peak
    sup = _support(peak, top_p=0.5)
    good = sup == {V - 1}
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] top_p=0.5 on a peak support={sorted(sup)} (expect [{V - 1}])")

    # (a) top_p keeps the crossing token. Skewed probs on ids {2,1,0}; with top_p=0.7 the 2nd token is
    #     the crossing token (before-mass ~0.665 < 0.7 -> KEPT); at 0.5 only the peak is kept.
    skew = mx.where(mx.arange(V) == 2, 1.0, mx.where(mx.arange(V) == 1, 0.0,
                    mx.where(mx.arange(V) == 0, -1.0, -50.0)))
    sup7 = _support(skew, top_p=0.7)
    sup5 = _support(skew, top_p=0.5)
    good = sup7 == {1, 2} and sup5 == {2}
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] top_p keeps crossing token: p0.7={sorted(sup7)} (expect [1,2]) "
          f"p0.5={sorted(sup5)} (expect [2])")

    # (a) seed reproducibility: same seed identical draws, different seed differs (broad support)
    flat = mx.where(mx.arange(V) < 16, 0.0, -50.0)
    def _draws(seed: int) -> list[int]:
        key = mx.random.key(seed)
        out = []
        for _ in range(20):
            key, sub = mx.random.split(key)
            out.append(int(sample_logits(flat, temperature=1.0, key=sub).item()))
        return out
    d1, d1b, d2 = _draws(123), _draws(123), _draws(456)
    good = d1 == d1b and d1 != d2 and all(t < 16 for t in d1)
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] seed: same-seed identical={d1 == d1b} diff-seed differs={d1 != d2}")

    # (b) generate-loop bound: eos-less greedy run stops at exactly max_new_tokens (never unbounded)
    NON_EOS, EOS = 7, 11
    prompt = [1, 2, 3]
    stub = _FixedModel(arg=NON_EOS)
    out = generate(stub, prompt, max_new_tokens=8, temperature=0.0, eos_id=EOS, cache=_StubCache())
    seed_calls = stub.calls[:len(prompt)]
    decode_calls = stub.calls[len(prompt):]
    seeded_ok = seed_calls == [(0, 1), (1, 2), (2, 3)]
    offsets_ok = [o for o, _ in decode_calls] == list(range(3, 3 + 8))
    good = len(out) == 8 and all(t == NON_EOS for t in out) and seeded_ok and offsets_ok
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] max_new_tokens bounds loop: n={len(out)} "
          f"all_nonEOS={all(t == NON_EOS for t in out)} seeded={seeded_ok} decode_offsets={offsets_ok}")

    # (b) stub whose argmax IS eos_id -> stops early (first sampled token is eos -> empty output)
    stub_eos = _FixedModel(arg=EOS)
    out = generate(stub_eos, prompt, max_new_tokens=8, temperature=0.0, eos_id=EOS, cache=_StubCache())
    good = out == [] and stub_eos.calls == [(0, 1), (1, 2), (2, 3)]
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] argmax==eos stops early: out={out} "
          f"prompt_seeded={stub_eos.calls == [(0, 1), (1, 2), (2, 3)]}")

    # (b) eos_id as a SET of stop ids is honored too
    out = generate(_FixedModel(arg=EOS), prompt, max_new_tokens=8, temperature=0.0,
                   eos_id={5, EOS}, cache=_StubCache())
    good = out == []
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] eos_id set honored: out={out} (expect [])")

    # (b) empty prompt fails loud (rule 6 — never silently prefill nothing)
    try:
        generate(_FixedModel(arg=NON_EOS), [], max_new_tokens=4, cache=_StubCache())
        good = False
    except ValueError:
        good = True
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] empty prompt -> ValueError (no silent empty prefill)")

    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    run()
