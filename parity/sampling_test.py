"""Unit checks for the vectorized next-token sampler (no model, instant, CPU).

Verifies greedy == argmax, seed determinism, and that top-k / top-p truncation actually
bound the sampled token to the intended candidate set — all on batched logits so the
vectorized (no-loop) multi-row path is exercised too.

    uv run python -m parity.sampling_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.generate import sample_logits

V = 2000


def run() -> None:
    mx.random.seed(0)
    logits = mx.random.normal((V,))
    amax = int(mx.argmax(logits).item())
    batch = mx.broadcast_to(logits[None], (64, V))  # 64 independent draws in one call

    greedy_ok = int(sample_logits(logits, temperature=0.0).item()) == amax

    k = mx.random.key(7)
    det_ok = bool(mx.all(sample_logits(batch, temperature=1.0, key=k)
                         == sample_logits(batch, temperature=1.0, key=k)).item())

    t1 = sample_logits(batch, temperature=1.0, top_k=1, key=mx.random.key(1))
    topk1_ok = bool(mx.all(t1 == amax).item())  # only the max survives top_k=1

    t5 = sample_logits(batch, temperature=1.0, top_k=5, key=mx.random.key(2))
    thresh5 = mx.sort(logits)[-5]
    topk5_ok = bool(mx.all(logits[t5] >= thresh5).item())  # every draw is in the top-5

    peaked = logits + 0.0
    peaked[amax] = 50.0  # dominate so the nucleus is just the top token
    tp = sample_logits(mx.broadcast_to(peaked[None], (64, V)), temperature=1.0, top_p=0.5,
                       key=mx.random.key(3))
    topp_ok = bool(mx.all(tp == amax).item())

    print("\n=== sampler unit checks ===")
    print(f"greedy == argmax     : {greedy_ok}")
    print(f"seed determinism     : {det_ok}")
    print(f"top_k=1 -> argmax    : {topk1_ok}")
    print(f"top_k=5 in top-5 set : {topk5_ok}")
    print(f"tiny top_p -> argmax : {topp_ok}")
    assert all([greedy_ok, det_ok, topk1_ok, topk5_ok, topp_ok]), "sampler check failed"
    print("all sampler checks passed")


if __name__ == "__main__":
    run()
