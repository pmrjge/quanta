"""Model-free parity gate for :func:`quanta.modeling.batched_attention.batched_decode_attention`
(Approach 1 — vectorize the per-stream decode-attention loop with one padded+masked SDPA).

Tiny random tensors ONLY (no model load) — safe to run alongside a GPU job. The arbiter: row ``b``
of the single batched ``mx.fast.scaled_dot_product_attention`` over ``B`` ragged-length streams must
equal the **per-stream** ``mx.fast.scaled_dot_product_attention(q_b, repeat(k_b), repeat(v_b), scale,
mask=None)`` that the current ``step_batch`` loop runs (the InternLM2 / Nemotron / DSV4 decode path),
to within SDPA reduction-tiling ULPs — NOT a logic change. Covers: ragged lengths incl. a length-1
stream, GQA (``n_rep>1``) and MHA (``n_rep==1``), ``B==1`` degenerate, equal-length (tight) case,
the ``[1,H,1,D]`` batch-axis input form, and the loud memory guard.

    uv run --with numpy python -m parity.batched_attention_test
"""

from __future__ import annotations

import math

import mlx.core as mx

from quanta.modeling.batched_attention import (
    _guard_padded_kv,
    batched_decode_attention,
    decode_pad_mask,
)

TOL = 1e-4  # fp32; padded-tail masked to ~zero ⇒ residual is SDPA tiling reorder on valid cols only


def _ref_per_stream(q, k, v, *, scale: float, n_rep: int) -> mx.array:
    """The per-stream path the batched call must match: one SDPA over exactly this stream's L_b keys.

    q: ``[H,1,D]``  k,v: ``[n_kv,L_b,D]`` → ``[1,H,1,D]`` (GQA repeat then ``mask=None`` last-token).
    """
    qb = q[None]                                            # [1,H,1,D]
    kb = mx.repeat(k[None], n_rep, axis=1)                  # [1,H,L_b,D]
    vb = mx.repeat(v[None], n_rep, axis=1)
    return mx.fast.scaled_dot_product_attention(qb, kb, vb, scale=scale, mask=None)


def _rand(key, shape):
    key, sub = mx.random.split(key)
    return key, mx.random.normal(shape, key=sub)


def _make_streams(lengths, *, nh, nkv, d, seed=0):
    """Build ragged per-stream (q [H,1,D], k/v [n_kv,L_b,D]) fp32 tensors."""
    key = mx.random.key(seed)
    qs, ks, vs = [], [], []
    for lb in lengths:
        key, q = _rand(key, (nh, 1, d))
        key, k = _rand(key, (nkv, lb, d))
        key, v = _rand(key, (nkv, lb, d))
        qs.append(q)
        ks.append(k)
        vs.append(v)
    return qs, ks, vs


def _max_abs(a, b) -> float:
    return float(mx.max(mx.abs(a - b)).item())


def _check(name, lengths, *, nh, nkv, d, batch_axis=False) -> float:
    n_rep = nh // nkv
    scale = 1.0 / math.sqrt(d)
    qs, ks, vs = _make_streams(lengths, nh=nh, nkv=nkv, d=d)
    q_in = [q[None] for q in qs] if batch_axis else qs     # exercise the [1,H,1,D] input form too
    out = batched_decode_attention(q_in, ks, vs, scale=scale, n_rep=n_rep)  # [B,H,1,D]
    mx.eval(out)
    worst = 0.0
    for b, (q, k, v) in enumerate(zip(qs, ks, vs)):
        ref = _ref_per_stream(q, k, v, scale=scale, n_rep=n_rep)            # [1,H,1,D]
        worst = max(worst, _max_abs(out[b:b + 1], ref))
    status = "OK" if worst < TOL else "FAIL"
    print(f"  [{status}] {name:<34} B={len(lengths):>2} nh={nh} nkv={nkv} "
          f"L={lengths} |Δ|={worst:.2e}")
    assert worst < TOL, f"{name}: |Δ|={worst:.2e} >= {TOL:.0e}"
    return worst


def test_mask_shape_and_values() -> None:
    """The additive mask is ``[B,1,1,L_max]``, 0 on valid columns and large-negative on padding."""
    m = decode_pad_mask([2, 4, 1], 4, mx.float32)
    assert tuple(m.shape) == (3, 1, 1, 4), m.shape
    m0 = m[:, 0, 0, :]                                       # [B, L_max]
    # row 0: valid cols {0,1}; row 1: all 4 valid; row 2: only col 0
    assert float(m0[0, 0].item()) == 0.0 and float(m0[0, 2].item()) <= -1e8
    assert float(m0[1, 3].item()) == 0.0
    assert float(m0[2, 0].item()) == 0.0 and float(m0[2, 1].item()) <= -1e8
    print("  [OK] decode_pad_mask shape+values [B,1,1,L_max], 0 valid / -1e9 padded")


def test_memory_guard() -> None:
    """The pre-alloc guard fails loud above the byte cap instead of allocating (memory-safety)."""
    raised = False
    try:
        _guard_padded_kv(b=32, n_kv=8, l_max=10_000_000, d=128, itemsize=2)  # ~6 TiB → refuse
    except ValueError as e:
        raised = "memory-safety" in str(e)
    assert raised, "expected a loud ValueError from _guard_padded_kv"
    _guard_padded_kv(b=32, n_kv=8, l_max=2048, d=128, itemsize=2)            # ~1.6 GiB → fine
    print("  [OK] memory guard: raises above 64 GiB cap, passes a normal B=32 L=2048 case")


def main() -> None:
    print("A. batched padded+masked SDPA == per-stream loop (ragged decode):")
    _check("GQA ragged (rep=4)", [3, 7, 5, 1], nh=8, nkv=2, d=8)
    _check("GQA ragged incl len-1", [1, 6, 2], nh=4, nkv=2, d=8)
    _check("MHA ragged (rep=1)", [4, 9, 5], nh=4, nkv=4, d=16)
    _check("equal lengths (tight)", [6, 6, 6, 6], nh=8, nkv=2, d=8)
    _check("B=1 degenerate", [5], nh=4, nkv=2, d=8)
    _check("[1,H,1,D] input form", [3, 5], nh=4, nkv=2, d=8, batch_axis=True)
    print("B. mask + memory-safety:")
    test_mask_shape_and_values()
    test_memory_guard()
    print("\nPASS — batched decode attention is per-stream-equivalent (within SDPA tiling ULPs)")


if __name__ == "__main__":
    main()
