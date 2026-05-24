"""Gate: DSV4 sampling math + the generate loop bound — model-free, tiny tensors, ~0 GB.

Answers two questions without loading the model (safe to run while a big model is GPU-resident):

  (a) **Sampling math** (:func:`quanta.dsv4.generate.sample_logits`): greedy == argmax;
      top-k / top-p / min-p restrict the support to exactly the right tokens; and a ``seed`` makes
      temperature sampling reproducible (same seed → identical draws, different seed differs). Mirrors
      ``parity/omlx_sampling_test.py`` — the sampler is the same math the oMLX shim uses, so the engine
      and the standalone generator agree token-for-token.
  (b) **Generate-loop bound** (:func:`quanta.dsv4.generate.generate`): driven by a STUB model (a tiny
      object exposing ``.cfg`` / ``.num_layers`` and the single-token ``__call__(token_ids, caches=,
      offset=)`` contract, returning fixed logits) — ``generate`` seeds the cache by stepping the
      prompt (asserted via recorded offsets) then decodes; an eos-less run is bounded by
      ``max_new_tokens`` (a loop can never run unbounded), a stub whose argmax is ``eos_id`` stops
      early, and an empty prompt fails loud. A stub cache is passed in, so the real decode module is
      not imported and nothing model-sized is allocated.

    uv run --with numpy python -m parity.dsv4_generate_test

Deferred (#77, run later ON the M3 Ultra — loads the ~real artifact, NOT model-free):
    # from quanta.dsv4.runtime import DSV4ResidentModel
    # from quanta.dsv4.generate import generate
    # m = DSV4ResidentModel("/Users/pmrj/models/DeepSeek-V4-Flash-quanta_<type>")
    # print(generate(m, [m.cfg.bos_token_id, 1, 2, 3], max_new_tokens=32,
    #                temperature=0.7, top_p=0.9, eos_id=set(m.cfg.eos_token_ids), seed=0))
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from quanta.dsv4.generate import generate, sample_logits

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
    """A non-``None`` cache placeholder so ``generate`` skips building a real ``DSV4Cache`` (keeps the
    gate model-free); the stub model never touches it."""


def _support(logits: mx.array, **kw) -> set[int]:
    """The set of token ids sample_logits can ever draw (non -inf after filtering), via a many-sample
    draw. Greedy is excluded; here temperature>0 so the filtered logits define the support."""
    keys = mx.random.split(mx.random.key(0), 4000)
    draws = {int(sample_logits(logits, temperature=1.0, key=k, **kw).item()) for k in keys}
    return draws


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

    # (a) min_p: with logit ramp the max prob is token V-1; keep tokens with p >= min_p*p_max.
    #     Use a peaked distribution: one big logit + rest tiny, so min_p prunes all but the peak.
    peak = mx.where(mx.arange(V) == V - 1, 50.0, 0.0)
    sup = _support(peak, min_p=0.5)
    good = sup == {V - 1}
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] min_p=0.5 on a peak support={sorted(sup)} (expect [{V - 1}])")

    # (a) top_p nucleus: peaked dist (token V-1 ~ all mass) -> nucleus is just the peak
    sup = _support(peak, top_p=0.5)
    good = sup == {V - 1}
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] top_p=0.5 on a peak support={sorted(sup)} (expect [{V - 1}])")

    # (a) top_p keeps the crossing token (the one that crosses the mass threshold, not just those
    #     strictly under it). Skewed probs p=[~0.665, ~0.245, ~0.090] on ids {2,1,0}; with top_p=0.7
    #     the strictly-before mass at the 2nd token is ~0.665 < 0.7 so it's KEPT (the crossing token),
    #     and at the 3rd is ~0.910 so it's dropped -> support {1,2}. Same dist at top_p=0.5 keeps only
    #     the peak (before-mass at the 2nd is ~0.665, not < 0.5).
    skew = mx.where(mx.arange(V) == 2, 1.0, mx.where(mx.arange(V) == 1, 0.0,
                    mx.where(mx.arange(V) == 0, -1.0, -50.0)))
    sup7 = _support(skew, top_p=0.7)
    sup5 = _support(skew, top_p=0.5)
    good = sup7 == {1, 2} and sup5 == {2}
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] top_p keeps crossing token: p0.7={sorted(sup7)} (expect [1,2]) "
          f"p0.5={sorted(sup5)} (expect [2])")

    # (c) seed reproducibility: same seed identical draws, different seed differs (broad support)
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
    # prompt seeded at offsets 0,1,2 (token ids 1,2,3), then 8 decode steps at offsets 3..10
    seed_calls = stub.calls[:len(prompt)]
    decode_calls = stub.calls[len(prompt):]
    seeded_ok = seed_calls == [(0, 1), (1, 2), (2, 3)]
    offsets_ok = [o for o, _ in decode_calls] == list(range(3, 3 + 8))
    good = len(out) == 8 and all(t == NON_EOS for t in out) and seeded_ok and offsets_ok
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] max_new_tokens bounds loop: n={len(out)} all_nonEOS={all(t == NON_EOS for t in out)} "
          f"seeded={seeded_ok} decode_offsets={offsets_ok}")

    # (b) stub whose argmax IS eos_id -> stops early (first sampled token is eos -> empty output)
    stub_eos = _FixedModel(arg=EOS)
    out = generate(stub_eos, prompt, max_new_tokens=8, temperature=0.0, eos_id=EOS, cache=_StubCache())
    # cache still seeded over the prompt; no decode step appended (stopped before the first append)
    good = out == [] and stub_eos.calls == [(0, 1), (1, 2), (2, 3)]
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] argmax==eos stops early: out={out} prompt_seeded={stub_eos.calls == [(0, 1), (1, 2), (2, 3)]}")

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


if __name__ == "__main__":
    run()
