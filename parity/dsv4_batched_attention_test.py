"""Parity: DSV4 batched single-token decode attention == the per-stream Design-A loop.

Model-free gate for the per-stream-OFFSET batched steppers in :mod:`quanta.dsv4.decode`
(``decode_step_dense_batched`` / ``decode_step_compressed_batched``) — the siblings that collapse the
Design-A per-stream attention loop (one ``decode_step`` per stream, every layer) into ONE projection
+ ONE windowed-sink SDPA across ``B`` ragged-offset streams. The compressor pooling state machine,
the Lightning-indexer top-k and the cache appends stay per-stream (data-dependent / IO, rule-3); only
the offset-independent matmuls + the SDPA are fused over the batch.

Proves, on tiny random params (no artifact), against the single-stream steppers the runtime uses:

  A. **dense (ratio-0)** — batched == per-stream loop, **B=1 bit-exact** and **B≥2 tight** (pad+mask
     SDPA ULPs), across **ragged offsets** (streams seeded to different lengths) + the grown caches.
     Also drives the **#18 KV arena** stepper (ONE scatter write + ONE gather read replacing the
     per-stream ``append_kv`` loop + ``_pad_stack``): arena == per-stream loop (same bar) AND
     arena == the ``_LayerCache`` batched path bit-exactly (identical codes, padding, SDPA).
  B. **compressed (ratio-4 indexer / ratio-128)** — same, exercising the window-closing pool, the
     indexer top-k selection and the ragged compressed-KV stream. (Arena wiring lands in #18 M3.)

    uv run --with numpy python -m parity.dsv4_batched_attention_test
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from parity.dsv4_batched_test import _attn_params, _cfg, _r
from quanta.dsv4.attention import rope_cos_sin
from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.decode import (
    DSV4Cache,
    _CompArena,
    _KVArena,
    _LayerCache,
    decode_step_compressed,
    decode_step_compressed_batched,
    decode_step_dense,
    decode_step_dense_batched,
)


def _maxdiff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _rope_full(cfg: DeepSeekV4Config, layer_id: int, length: int) -> tuple[mx.array, mx.array]:
    """Full per-layer RoPE tables for ``[0, length)`` — mirrors ``DSV4ResidentModel._rope``."""
    orig, theta = cfg.attn_rope(layer_id)
    return rope_cos_sin(cfg.rope_head_dim, length, orig, theta,
                        cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)


def _seed(cfg: DeepSeekV4Config, p: dict, layer_id: int, seqs: list[list[mx.array]],
          step) -> list[DSV4Cache]:
    """One fresh ``DSV4Cache`` per stream, grown to ``len(seqs[b])`` tokens by stepping the
    single-stream ``step`` so each stream lands at its own (ragged) offset."""
    caches: list[DSV4Cache] = []
    for ins in seqs:
        c = DSV4Cache(cfg.num_hidden_layers)
        for t, x in enumerate(ins):
            cos, sin = _rope_full(cfg, layer_id, t + 1)
            step(x, p, cfg, layer_id, c, cos, sin, t)
        caches.append(c)
    return caches


def _arena_from_layer_caches(lcs: list[_LayerCache], rows: list[int], r_total: int) -> _KVArena:
    """Code-copy ``B`` per-stream latent-KV streams into a fresh :class:`_KVArena` (bit-exact starting
    state), so the M1 dense check isolates the arena READ/WRITE WIRING from the codec — the codec
    round-trip (raw input → quantize → arena → dequantize) is already M0's ``dsv4_kv_arena_test`` gate.
    Stream ``s``'s stored trio (int8 codes) or bf16 stream is written into arena row ``rows[s]`` at
    ``[0:len]``; the unfilled tail stays zero (inert under the SDPA window/pad mask)."""
    f = lcs[0]
    arena = _KVArena(r_total, group_size=f.group_size, quantized=f.quantized)
    nmax = max(lc.kv_length() for lc in lcs)
    if f.quantized:
        arena._q = mx.zeros((r_total, nmax, f._kv_q.shape[-1]), dtype=f._kv_q.dtype)
        arena._s = mx.zeros((r_total, nmax, f._kv_s.shape[-1]), dtype=f._kv_s.dtype)
        arena._b = mx.zeros((r_total, nmax, f._kv_b.shape[-1]), dtype=f._kv_b.dtype)
        arena.l_cap = nmax
        for lc, row in zip(lcs, rows, strict=True):
            n = lc.kv_length()
            arena._q[row, :n] = lc._kv_q[0]
            arena._s[row, :n] = lc._kv_s[0]
            arena._b[row, :n] = lc._kv_b[0]
            arena.lengths[row] = n
    else:
        arena._bf16 = mx.zeros((r_total, nmax, f._kv_bf16.shape[-1]), dtype=f._kv_bf16.dtype)
        arena.l_cap = nmax
        for lc, row in zip(lcs, rows, strict=True):
            n = lc.kv_length()
            arena._bf16[row, :n] = lc._kv_bf16[0]
            arena.lengths[row] = n
    return arena


def _comp_arena_from_layer_caches(lcs: list[_LayerCache], rows: list[int], r_total: int,
                                  ratio: int, overlap: bool, has_idx: bool) -> tuple[_KVArena, _CompArena]:
    """Code-copy ``B`` per-stream COMPRESSED ``_LayerCache``s into a fresh latent :class:`_KVArena` +
    :class:`_CompArena` (#18 M3) — a bit-exact starting state that isolates the arena READ/WRITE WIRING
    from the codec/pool arithmetic (codec gated in M0's ``dsv4_kv_arena_test``, pool in
    ``dsv4_decode_attn_test``). Reuses :func:`_arena_from_layer_caches` for the latent stream, and
    additionally seeds the compressed extras: ckv (same codec), ikv (bf16) and the fixed-width
    ``[R, cap, dim]`` ring — each per-stream ring RIGHT-aligned (newest at ``[:, -1]``, zero-padded at
    the FRONT, the :func:`quanta.dsv4.decode._push_ring_batched` layout). Stream ``s`` → arena row
    ``rows[s]`` (non-contiguous, to catch row confusion)."""
    f = lcs[0]
    latent = _arena_from_layer_caches(lcs, rows, r_total)
    comp = _CompArena(r_total, ratio=ratio, overlap=overlap, max_rollback=f.max_rollback,
                      group_size=f.group_size, quantized=f.quantized, has_indexer=has_idx)
    ncmax = max(lc.n_comp() for lc in lcs)
    if ncmax > 0:
        fc = next(lc for lc in lcs if lc.n_comp() > 0)           # a stream with a populated ckv (shape src)
        if f.quantized:                                          # ckv int8 codes (real DSV4 head_dim=128)
            comp.ckv._q = mx.zeros((r_total, ncmax, fc._ckv_q.shape[-1]), dtype=fc._ckv_q.dtype)
            comp.ckv._s = mx.zeros((r_total, ncmax, fc._ckv_s.shape[-1]), dtype=fc._ckv_s.dtype)
            comp.ckv._b = mx.zeros((r_total, ncmax, fc._ckv_b.shape[-1]), dtype=fc._ckv_b.dtype)
            comp.ckv.l_cap = ncmax
            for lc, row in zip(lcs, rows, strict=True):
                n = lc.n_comp()
                if n:
                    comp.ckv._q[row, :n] = lc._ckv_q[0]
                    comp.ckv._s[row, :n] = lc._ckv_s[0]
                    comp.ckv._b[row, :n] = lc._ckv_b[0]
                comp.ckv.lengths[row] = n
        else:                                                    # ckv bf16 (tiny-head_dim test path)
            comp.ckv._bf16 = mx.zeros((r_total, ncmax, fc._ckv_bf16.shape[-1]), dtype=fc._ckv_bf16.dtype)
            comp.ckv.l_cap = ncmax
            for lc, row in zip(lcs, rows, strict=True):
                n = lc.n_comp()
                if n:
                    comp.ckv._bf16[row, :n] = lc._ckv_bf16[0]
                comp.ckv.lengths[row] = n
        if has_idx:                                              # ikv stream (always bf16)
            fik = next(lc.ikv for lc in lcs if lc.ikv is not None)
            comp.ikv._bf16 = mx.zeros((r_total, ncmax, fik.shape[-1]), dtype=fik.dtype)
            comp.ikv.l_cap = ncmax
            for lc, row in zip(lcs, rows, strict=True):
                n = lc.n_comp()
                if n and lc.ikv is not None:
                    comp.ikv._bf16[row, :n] = lc.ikv[0]
                comp.ikv.lengths[row] = n
    rdim = next((lc.ring.shape[-1] for lc in lcs if lc.ring is not None), None)
    if rdim is not None:                                         # raw-hidden ring, right-aligned, zero front
        cap, rdtype = comp.ring_cap, next(lc.ring.dtype for lc in lcs if lc.ring is not None)
        comp.ring = mx.zeros((r_total, cap, rdim), dtype=rdtype)
        for lc, row in zip(lcs, rows, strict=True):
            if lc.ring is not None:
                w = min(lc.ring.shape[1], cap)
                comp.ring[row, -w:] = lc.ring[0, -w:]
    return latent, comp


def _check(cfg: DeepSeekV4Config, layer_id: int, step, step_b, seed_lens: list[int],
           rng, *, arena_path: bool = False, comp_arena: bool = False) -> tuple[bool, str]:
    """Seed ``B`` streams to ragged offsets, then compare ONE more decode step: per-stream ``step``
    loop (reference) vs the batched ``step_b`` (one call over the stacked streams). Independently
    seeded cache sets so the in-place appends don't cross-contaminate.

    ``arena_path`` (dense / #18 M1): ALSO run ``step_b`` on a :class:`_KVArena` (one scatter write +
    one gather read replacing the per-stream ``append_kv`` loop + ``_pad_stack``) and compare to BOTH
    the per-stream reference (the M1 bar: B=1 bit-exact, ragged B≥2 ``<5e-4``, ``length`` matches) AND
    the per-stream ``_LayerCache`` batched path (must be BIT-EXACT — identical codes, padding, SDPA)."""
    dim = cfg.hidden_size
    b = len(seed_lens)
    seqs = [[_r(rng, 1, 1, dim) for _ in range(n)] for n in seed_lens]
    ref_caches = _seed(cfg, P[layer_id], layer_id, seqs, step)
    bat_caches = _seed(cfg, P[layer_id], layer_id, seqs, step)
    offsets = list(seed_lens)                                  # each stream's next abs position
    new_x = [_r(rng, 1, 1, dim) for _ in range(b)]
    x_b = mx.concatenate(new_x, axis=0)                        # [B,1,dim]

    # reference: per-stream single-stream step (each at its own offset, its own full RoPE table)
    ref = []
    for s in range(b):
        cos, sin = _rope_full(cfg, layer_id, offsets[s] + 1)
        ref.append(step(new_x[s], P[layer_id], cfg, layer_id, ref_caches[s], cos, sin, offsets[s]))
    mx.eval(ref)

    # batched (per-stream _LayerCache loop): ONE call with the full table to max offset + per-stream gather
    cos, sin = _rope_full(cfg, layer_id, max(offsets) + 1)
    lcs = [bat_caches[s][layer_id] for s in range(b)]
    bat = step_b(x_b, P[layer_id], cfg, layer_id, lcs, cos, sin, offsets)
    mx.eval(bat)

    diffs = [_maxdiff(bat[s:s + 1], ref[s]) for s in range(b)]
    # cache lengths must match per stream (each grew by exactly one latent token)
    len_ok = all(bat_caches[s][layer_id].kv_length() == ref_caches[s][layer_id].kv_length()
                 for s in range(b))
    good = max(diffs) < 5e-4 and len_ok
    msg = (f"max|Δ|={max(diffs):.2e} per-stream=[{', '.join(f'{d:.2e}' for d in diffs)}] "
           f"len_ok={len_ok}")

    if arena_path:
        r_total = b + 2
        rows = list(reversed(range(b)))                        # non-contiguous → catches row confusion
        arn_caches = _seed(cfg, P[layer_id], layer_id, seqs, step)   # third identical set
        arena = _arena_from_layer_caches([arn_caches[s][layer_id] for s in range(b)], rows, r_total)
        arn = step_b(x_b, P[layer_id], cfg, layer_id, None, cos, sin, offsets, arena=arena, rows=rows)
        mx.eval(arn)
        arn_ref = [_maxdiff(arn[s:s + 1], ref[s]) for s in range(b)]   # arena vs per-stream loop (M1 bar)
        arn_bat = _maxdiff(arn, bat)                          # arena vs _LayerCache batched (bit-exact)
        arn_len_ok = all(arena.length(rows[s]) == ref_caches[s][layer_id].kv_length()
                         for s in range(b))
        good = good and max(arn_ref) < 5e-4 and arn_bat == 0.0 and arn_len_ok
        msg += (f" | arena |Δ|ref={max(arn_ref):.2e} |Δ|bat={arn_bat:.2e} len_ok={arn_len_ok}")

    if comp_arena:                                             # #18 M3: compressed stepper on the arena
        ratio, overlap = cfg.compress_ratio(layer_id), cfg.overlap(layer_id)
        has_idx = cfg.has_indexer(layer_id)
        r_total = b + 2
        rows = list(reversed(range(b)))                        # non-contiguous → catches row confusion
        arn_caches = _seed(cfg, P[layer_id], layer_id, seqs, step)   # third identical set
        latent, comp = _comp_arena_from_layer_caches(
            [arn_caches[s][layer_id] for s in range(b)], rows, r_total, ratio, overlap, has_idx)
        arn = step_b(x_b, P[layer_id], cfg, layer_id, None, cos, sin, offsets,
                     arena=latent, rows=rows, comp=comp)        # ONE scatter/gather + masked pool
        mx.eval(arn)
        arn_ref = [_maxdiff(arn[s:s + 1], ref[s]) for s in range(b)]   # arena vs per-stream loop (M1 bar)
        arn_bat = _maxdiff(arn, bat)                          # arena vs _LayerCache batched (bit-exact)
        arn_len_ok = all(latent.length(rows[s]) == ref_caches[s][layer_id].kv_length()
                         for s in range(b))
        arn_nc_ok = all(comp.n_comp(rows[s]) == ref_caches[s][layer_id].n_comp() for s in range(b))
        n_closed = sum(1 for o in offsets if (o + 1) % ratio == 0)    # rows that pool a new window now
        good = good and max(arn_ref) < 5e-4 and arn_bat == 0.0 and arn_len_ok and arn_nc_ok
        msg += (f" | arena |Δ|ref={max(arn_ref):.2e} |Δ|bat={arn_bat:.2e} "
                f"len_ok={arn_len_ok} nc_ok={arn_nc_ok} closed={n_closed}/{b}")

    return good, msg


# Shared tiny params per layer (built once; layer 0 dense, layer 1 ratio-4 indexer, layer 2 ratio-3).
cfg_g = _cfg()
rng_g = np.random.default_rng(7)
P = [_attn_params(rng_g, cfg_g, i) for i in range(cfg_g.num_hidden_layers)]


def run() -> None:
    ok = True

    # B=1 must be bit-exact (no padding); ragged B>=2 tight (pad+mask SDPA ULPs).
    # arena_path=True also drives the #18 KV-arena stepper (scatter write + gather read) vs the loop.
    print("A. dense (ratio-0, layer 0): batched [+arena #18] == per-stream loop")
    for tag, lens in (("B=1", [4]), ("ragged B=4", [5, 2, 8, 1]), ("ragged B=3", [7, 7, 3])):
        good, msg = _check(cfg_g, 0, decode_step_dense, decode_step_dense_batched, lens,
                           np.random.default_rng(100 + len(lens)), arena_path=True)
        ok = ok and good
        print(f"  [{'OK' if good else 'XX'}] {tag:>10}: {msg}")

    # comp_arena=True also drives the #18 M3 compressed arena stepper (ONE latent scatter, ONE ring
    # roll, ONE compute-all pool masked-scattered into ckv/ikv) vs the per-stream loop. The "close"
    # cases seed offsets so SOME streams cross a window boundary THIS step — exercising the masked
    # scatter-append + n_comp bump (and, on ratio-4, BOTH prev-valid c>=1 and the c==0 window-0 pad).
    print("B. compressed (ratio-4 indexer = layer 1, ratio-3 = layer 2): batched [+arena #18 M3] == "
          "per-stream loop")
    comp_lens = {
        # ratio-4 (overlap) closes at offset in {3,7,11}: c==0 at offset 3 (prev-invalid pad)
        1: (("B=1 no-close", [6]), ("ragged no-close B=4", [9, 4, 12, 2]), ("ragged no-close B=2", [10, 5]),
            ("B=1 close", [7]), ("ragged close B=4", [7, 4, 11, 3]), ("all-close B=2", [3, 7])),
        # ratio-3 (no overlap) closes at offset in {2,5,8,11}
        2: (("B=1 no-close", [7]), ("ragged no-close B=4", [10, 4, 13, 7]), ("ragged no-close B=2", [10, 7]),
            ("B=1 close", [5]), ("ragged close B=4", [8, 5, 2, 9]), ("all-close B=2", [2, 5])),
    }
    for layer_id, name in ((1, "ratio-4 idx"), (2, "ratio-3")):
        for idx, (tag, lens) in enumerate(comp_lens[layer_id]):
            good, msg = _check(cfg_g, layer_id, decode_step_compressed, decode_step_compressed_batched,
                               lens, np.random.default_rng(200 + layer_id * 100 + idx), comp_arena=True)
            ok = ok and good
            print(f"  [{'OK' if good else 'XX'}] {name:>11} {tag:>18}: {msg}")

    print("PASS — DSV4 batched decode attention is token-identical to the per-stream loop"
          if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
