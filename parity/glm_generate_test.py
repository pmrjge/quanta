"""Gate: GLM-5.1 sampling math + the generate loop bound — model-free, tiny tensors, ~0 GB.

Same contract as ``parity/dsv4_generate_test.py`` (the sampler is shared math): (a) greedy == argmax;
top-k / top-p / min-p restrict the support to exactly the right tokens; a ``seed`` makes temperature
sampling reproducible. (b) :func:`quanta.glm.generate.generate`, driven by a STUB model (``.num_layers``
+ the single-token ``__call__`` contract), seeds the cache by stepping the prompt (asserted via recorded
offsets) then decodes; an eos-less run is bounded by ``max_new_tokens``, a stub whose argmax is ``eos_id``
stops early, an eos *set* is honored, and an empty prompt fails loud. A stub cache is passed so the real
decode module is not imported and nothing model-sized is allocated.

    uv run --with numpy python -m parity.glm_generate_test
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from quanta.glm.generate import generate, sample_logits

V = 64


class _FixedModel:
    """Returns the SAME logits every call; ``argmax == arg``. Records each call's (offset, token) so the
    test can assert ``generate`` seeds the cache by stepping the prompt then decodes."""

    def __init__(self, arg: int, n_layers: int = 2) -> None:
        self.cfg = SimpleNamespace(num_hidden_layers=n_layers)
        self.num_layers = n_layers
        self._arg = arg
        self.calls: list[tuple[int, int]] = []

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        flat = mx.array(token_ids).reshape(-1)
        t = int(flat.shape[0])
        self.calls.append((int(offset), int(flat[0].item())))
        row = (mx.arange(V) == self._arg).astype(mx.float32) * 60.0 - 30.0
        return mx.broadcast_to(row, (1, t, V))


class _StubCache:
    """Non-``None`` cache placeholder so ``generate`` skips building a real GLMCache (keeps it model-free)."""


def _support(logits: mx.array, **kw) -> set[int]:
    keys = mx.random.split(mx.random.key(0), 4000)
    return {int(sample_logits(logits, temperature=1.0, key=k, **kw).item()) for k in keys}


def run() -> None:
    ok = True

    good = all(int(sample_logits(((mx.arange(V) == a).astype(mx.float32) * 5.0), temperature=0.0).item()) == a
               for a in (0, 7, V - 1))
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] greedy == argmax")

    ramp = mx.arange(V).astype(mx.float32)
    sup = _support(ramp, top_k=5)
    good = sup == set(range(V - 5, V))
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] top_k=5 support={sorted(sup)}")

    peak = mx.where(mx.arange(V) == V - 1, 50.0, 0.0)
    good = _support(peak, min_p=0.5) == {V - 1} and _support(peak, top_p=0.5) == {V - 1}
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] min_p/top_p on a peak -> {{{V - 1}}}")

    skew = mx.where(mx.arange(V) == 2, 1.0, mx.where(mx.arange(V) == 1, 0.0,
                    mx.where(mx.arange(V) == 0, -1.0, -50.0)))
    good = _support(skew, top_p=0.7) == {1, 2} and _support(skew, top_p=0.5) == {2}
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] top_p keeps the crossing token")

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
    print(f"  [{'OK' if good else 'FAIL'}] seed: same identical={d1 == d1b} diff differs={d1 != d2}")

    NON_EOS, EOS = 7, 11
    prompt = [1, 2, 3]
    stub = _FixedModel(arg=NON_EOS)
    out = generate(stub, prompt, max_new_tokens=8, temperature=0.0, eos_id=EOS, cache=_StubCache())
    seeded_ok = stub.calls[:3] == [(0, 1), (1, 2), (2, 3)]
    offsets_ok = [o for o, _ in stub.calls[3:]] == list(range(3, 11))
    good = len(out) == 8 and all(t == NON_EOS for t in out) and seeded_ok and offsets_ok
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] max_new_tokens bounds loop: n={len(out)} seeded={seeded_ok} offsets={offsets_ok}")

    stub_eos = _FixedModel(arg=EOS)
    out = generate(stub_eos, prompt, max_new_tokens=8, temperature=0.0, eos_id=EOS, cache=_StubCache())
    good = out == [] and stub_eos.calls == [(0, 1), (1, 2), (2, 3)]
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] argmax==eos stops early: out={out}")

    out = generate(_FixedModel(arg=EOS), prompt, max_new_tokens=8, temperature=0.0,
                   eos_id={5, EOS}, cache=_StubCache())
    good = out == []
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] eos_id set honored: out={out}")

    try:
        generate(_FixedModel(arg=NON_EOS), [], max_new_tokens=4, cache=_StubCache())
        good = False
    except ValueError:
        good = True
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] empty prompt -> ValueError")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
