"""Model-free gate: ``quanta.generate.sample_logits`` min-p truncation (rule 4: min_p=0 is a no-op).

Adds the min-p sampler lever (drop tokens whose probability is below ``min_p * max_prob``) so the
Nemotron continuous-batching sampler (:func:`quanta.nemotron.batched_generate.batched_generate`) no
longer rejects ``min_p > 0``. Proves:

  A. **min_p == 0 is byte-identical** to the top-k/top-p-only path (every existing caller unchanged).
  B. **a positive min_p deterministically prunes the tail** — with logits whose runner-up sits below
     the threshold, sampling collapses to the surviving head token regardless of the rng key, while
     the no-min_p baseline still surfaces the runner-up (so the pruning is meaningful, not vacuous).

    uv run python -m parity.sample_min_p_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.generate import sample_logits


def test_min_p_zero_is_noop() -> None:
    x = mx.array([3.0, 2.5, 2.0, 1.0, 0.0, -1.0])
    for kk in range(8):
        key = mx.random.key(kk)
        a = sample_logits(x, temperature=0.8, top_k=4, top_p=0.95, key=key)
        b = sample_logits(x, temperature=0.8, top_k=4, top_p=0.95, min_p=0.0, key=key)
        assert int(a.item()) == int(b.item()), \
            f"min_p=0 changed the sample at key {kk}: {a.item()} != {b.item()}"
    print("A min_p=0.0 is byte-identical to the top-k/top-p-only sampler (rule-4 no-op)  ok")


def test_min_p_prunes_tail() -> None:
    # softmax([10,9,0,0]) ~ [0.731, 0.269, 3e-5, 3e-5]. min_p=0.5 -> threshold 0.5*0.731=0.366, so the
    # runner-up (0.269) is pruned and only the head (0.731) survives -> categorical must return 0.
    x = mx.array([10.0, 9.0, 0.0, 0.0])
    for kk in range(16):
        key = mx.random.key(kk)
        tok = int(sample_logits(x, temperature=1.0, min_p=0.5, key=key).item())
        assert tok == 0, f"min_p=0.5 should collapse to the head token, got {tok} at key {kk}"
    # sanity: WITHOUT min_p the runner-up does appear across keys, so the collapse above is non-trivial.
    seen = {int(sample_logits(x, temperature=1.0, key=mx.random.key(kk)).item()) for kk in range(64)}
    assert 1 in seen, "without min_p the runner-up token should appear -> min_p pruning is meaningful"
    print(f"B min_p=0.5 collapses to the head token; baseline samples {sorted(seen)} (runner-up present)  ok")


def run() -> None:
    test_min_p_zero_is_noop()
    test_min_p_prunes_tail()
    print("PASS — sample_logits min-p truncation (min_p=0 no-op; positive min_p prunes the tail)")


if __name__ == "__main__":
    run()
