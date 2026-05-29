"""Gate: the #18 batched KV arena round-trips bit-identically to the per-stream ``_LayerCache`` —
MODEL-FREE (random tensors, no runtime load, no GPU weights).

The batched DSV4 decode steppers grow ``B`` ragged per-stream ``_LayerCache`` latent streams with a
per-stream Python loop (quantize the new token + ``mx.concatenate``) and re-pad them every step via
``_pad_stack``. #18 replaces that with a persistent ``max_batch``-sized arena: ONE scatter write
(``arena[rows, cols, :] = codes``) + ONE gather read (``mx.take`` + a batched dequant). This gate proves
the arena store is a faithful drop-in BEFORE any stepper is wired to it (M0):

  1. **round-trip (int8 g128, head_dim=128)** — seed ``B`` ragged-length rows via the per-row prefill
     write (``_KVArena.append_row``) then advance them with the batched hot-path scatter
     (``append_batched``). Each row read back (``read_row``) is BIT-IDENTICAL to the matching
     ``_LayerCache.append_kv`` / ``.kv`` stream, and the batched read (``read_batched``) matches
     ``_pad_stack([lc.kv ...])`` on every valid (unpadded) position. Rows are non-contiguous
     (``[4,1,3,0]``) so the test would catch any row-index confusion.
  2. **bf16 path (head_dim=16, quantization disabled)** — same round-trip, bit-exact (no codec).
  3. **_ArenaLayerView** — the ``_LayerCache``-duck prefill uses: it seeds the arena row through
     ``append_kv`` / ``.kv`` / ``kv_length`` / ``truncate_kv`` exactly like a ``_LayerCache``.
  4. **free-list + growth** — ``_KVArenaSet.alloc`` hands out distinct rows and fails loud when
     exhausted; ``free`` resets the row in every layer and lets it be re-leased; double-free fails loud;
     ``L_cap`` grows by doubling while preserving content (bit-identical to a single-shot append).

    uv run python -m parity.dsv4_kv_arena_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.dsv4.decode import (
    _ArenaLayerView,
    _KVArena,
    _KVArenaSet,
    _LayerCache,
    _pad_stack,
)


def _rand(shape: tuple[int, ...]) -> mx.array:
    """A bf16 normal draw from the current (seeded) RNG — caller seeds once for determinism."""
    return mx.random.normal(shape).astype(mx.bfloat16)


def _eq(a: mx.array, b: mx.array) -> bool:
    """Bit-exact array equality (shape + every element)."""
    return tuple(a.shape) == tuple(b.shape) and bool(mx.all(a == b).item())


def _run_roundtrip(*, quantized: bool, head_dim: int, group_size: int, label: str) -> bool:
    """Seed B ragged rows via per-row prefill writes, advance them via the batched scatter, and assert
    each row + the batched read are bit-identical to the per-stream ``_LayerCache`` path."""
    mx.random.seed(0)
    b = 4
    rows = [4, 1, 3, 0]                 # arbitrary non-contiguous arena rows (R=6)
    r_total = 6
    pre_len = [5, 2, 8, 1]              # ragged prefill lengths
    steps = 3                          # decode tokens appended to every row
    d = head_dim

    pre = [_rand((1, pre_len[s], d)) for s in range(b)]      # per-stream prefill chunk
    dec = [_rand((1, steps, d)) for s in range(b)]           # per-stream decode tokens

    # reference: B independent _LayerCache streams (prefill chunk, then token-by-token decode).
    lcs = [_LayerCache(quantized=quantized, group_size=group_size) for _ in range(b)]
    for s in range(b):
        lcs[s].append_kv(pre[s])
    for t in range(steps):
        for s in range(b):
            lcs[s].append_kv(dec[s][:, t:t + 1])

    # arena: per-row prefill writes, then the batched hot-path scatter once per decode step.
    arena = _KVArena(r_total, group_size=group_size, quantized=quantized)
    for s in range(b):
        arena.append_row(rows[s], pre[s])
    for t in range(steps):
        kv_step = mx.concatenate([dec[s][:, t:t + 1] for s in range(b)], axis=0)   # [B,1,D]
        arena.append_batched(rows, kv_step)

    ok = True
    # (a) per-row read == _LayerCache.kv, bit-exact; lengths agree.
    for s in range(b):
        ok = ok and arena.length(rows[s]) == lcs[s].kv_length() == pre_len[s] + steps
        ok = ok and _eq(arena.read_row(rows[s]), lcs[s].kv)

    # (b) batched read: shape [B, L_max, D]; every valid position == per-stream == _pad_stack.
    batched = arena.read_batched(rows)
    pad = _pad_stack([lc.kv for lc in lcs])                  # [B, L_max, D] zero-tail reference
    l_max = max(pre_len) + steps
    ok = ok and batched.shape[1] == l_max == pad.shape[1]
    for s in range(b):
        n = pre_len[s] + steps
        ok = ok and _eq(batched[s:s + 1, :n], lcs[s].kv)
        ok = ok and _eq(batched[s:s + 1, :n], pad[s:s + 1, :n])

    print(f"  [{'OK' if ok else 'FAIL'}] {label}: per-row + batched read bit-identical to _LayerCache "
          f"(B={b} rows={rows} lens={[n + steps for n in pre_len]})")
    return ok


def _run_view() -> bool:
    """``_ArenaLayerView`` (the prefill duck) drives the arena row through the ``_LayerCache`` latent
    surface — append / read / length / truncate all match a plain ``_LayerCache``."""
    mx.random.seed(1)
    d, gs = 128, 128
    n1, n2 = 7, 3
    chunk, tok = _rand((1, n1, d)), _rand((1, n2, d))

    arena = _KVArena(2, group_size=gs, quantized=True)
    view = _ArenaLayerView(arena, 1, quantized=True, group_size=gs)
    lc = _LayerCache(quantized=True, group_size=gs)
    for piece in (chunk, tok):
        view.append_kv(piece)
        lc.append_kv(piece)

    ok = view.kv_length() == lc.kv_length() == n1 + n2 and _eq(view.kv, lc.kv)
    view.truncate_kv(n1)
    lc.truncate_kv(n1)
    ok = ok and view.kv_length() == lc.kv_length() == n1 and _eq(view.kv, lc.kv)

    print(f"  [{'OK' if ok else 'FAIL'}] _ArenaLayerView: append/read/length/truncate match _LayerCache")
    return ok


def _run_freelist_growth() -> bool:
    """``_KVArenaSet`` free-list (distinct rows, exhaustion + double-free fail loud, reset on free) and
    ``_KVArena`` doubling growth (content preserved bit-identically across a re-alloc)."""
    ok = True
    d, gs = 128, 128
    mx.random.seed(2)

    aset = _KVArenaSet(n_layers=3, rows=2, group_size=gs, quantized=True)
    r0, r1 = aset.alloc(), aset.alloc()
    ok = ok and r0 != r1 and {r0, r1} == {0, 1}

    exhausted = False
    try:
        aset.alloc()
    except RuntimeError:
        exhausted = True
    ok = ok and exhausted

    aset[0].append_row(r0, _rand((1, 4, d)))               # write row r0 in layer 0
    ok = ok and aset[0].length(r0) == 4
    aset.free(r0)                                          # free resets the row in every layer
    ok = ok and aset[0].length(r0) == 0
    ok = ok and aset.alloc() == r0                         # re-leased (LIFO)

    double_free = False
    try:
        aset.free(r1)
        aset.free(r1)
    except RuntimeError:
        double_free = True
    ok = ok and double_free

    # doubling growth: append past the initial cap; content stays bit-identical to a single _LayerCache.
    ar = _KVArena(1, group_size=gs, quantized=True)
    a, c = _rand((1, 3, d)), _rand((1, 10, d))
    ar.append_row(0, a)
    cap1 = ar.l_cap
    ar.append_row(0, c)
    ok = ok and ar.l_cap > cap1 and ar.l_cap >= 13
    lc = _LayerCache(quantized=True, group_size=gs)
    lc.append_kv(a)
    lc.append_kv(c)
    ok = ok and ar.length(0) == 13 and _eq(ar.read_row(0), lc.kv)

    print(f"  [{'OK' if ok else 'FAIL'}] free-list (alloc/exhaust/free-reset/re-lease/double-free) + "
          f"doubling growth (content preserved)")
    return ok


def run() -> None:
    ok = True
    print("\n=== #18 batched KV arena: round-trip vs per-stream _LayerCache (model-free) ===")
    ok &= _run_roundtrip(quantized=True, head_dim=128, group_size=128, label="int8 g128 head_dim=128")
    ok &= _run_roundtrip(quantized=False, head_dim=16, group_size=128, label="bf16 head_dim=16    ")
    ok &= _run_view()
    ok &= _run_freelist_growth()
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    run()
