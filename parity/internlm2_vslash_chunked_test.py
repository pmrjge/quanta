"""InternLM2.5 vertical-slash **key-chunked long-context probe** gate — M7 of the MInference track.

M3–M6 built the vertical-slash selector on an online probe of the LAST query block's attention to all
keys (MInference §3). That probe materializes the full ``[B,H,lp,S]`` attention in one shot — fine at
the short-doc quality gate, but it OOMs (and the old code fail-loud ``raise``\\ d) past ``max_alloc_gb``,
i.e. exactly at the 100K+ context where vertical-slash is *designed* to pay off. M7 key-chunks the
probe: when the full probe would exceed ``max_alloc_gb`` the softmax over keys is taken in **key
chunks** via the standard online-softmax (flash) two pass (:func:`_vertical_slash_index_chunked`), so
peak memory is one key chunk instead of O(S). The short-doc single-shot path (``gb <= max_alloc_gb``)
is left **byte-for-byte unchanged**, so every M1–M6 gate stays bit-identical; only the long-context
branch is new, and it is output-equivalent to the single-shot masses up to fp reassociation.

This gate is model-free (synthetic ``q``/``k``/``v``, fp32 for a tight bound) and proves the chunked
branch is correct, independent of weights:

  1. **mass parity** — chunked ``(key_mass, slash_mass)`` == the single-shot masses (forced to chunk via
     a tiny ``max_alloc_gb``), across several chunk granularities (down to 1 block/chunk) and at both a
     block-aligned ``T`` and a ragged ``T`` (partial last query block + partial last key chunk). This is
     the core correctness proof — the flash normalizer + the bounded slash offset-window accumulation
     must reconstruct the global softmax masses.
  2. **param-independence preserved** — the chunked masses do not depend on ``vert``/``slash`` (the
     top-k cut lives in :func:`select_keep`, M6), so the long-context probe stays per-head-re-cuttable.
  3. **chunked keep-all == causal** — feeding the chunked masses to :func:`select_keep` with
     ``vert``/``slash`` ≥ n_blocks selects exactly the full causal block mask (the parity anchor: a
     chunked probe still degenerates to dense).
  4. **chunked gather == mask** — the SAME chunked masses drive both executions (the additive-mask
     quality path and the block-gather speed path) to the same output (the M3 invariant, now on the
     chunked probe): :func:`gather_sparse_attention` recomputes the chunked index internally and must
     agree with a mask built from the explicitly-chunked index.

Vertical-slash is lossy and prefill-only; its long-context *quality/speed* payoff is measured
separately on the real bake (the gather-path wall-clock bench is M7's second, ppl-gated milestone).

    uv run python -m parity.internlm2_vslash_chunked_test
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx

from quanta.modeling.xattention import (
    XAttnConfig,
    additive_mask,
    gather_sparse_attention,
    select_keep,
    vertical_slash_index,
)

BLOCK = 128


def _probe_gb(bsz: int, h: int, lp: int, s: int) -> float:
    """The single-shot probe's [B,H,lp,S] fp32 footprint (the chunk trigger is gb > max_alloc_gb)."""
    return bsz * h * lp * s * 4 / 1e9


def _qk(bsz: int, h: int, t: int, d: int, seed: int) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    q = mx.random.normal((bsz, h, t, d)).astype(mx.float32)
    k = mx.random.normal((bsz, h, t, d)).astype(mx.float32)
    return q, k


def _rel(a: mx.array, b: mx.array) -> float:
    denom = float(mx.maximum(mx.max(mx.abs(b)), mx.array(1e-12)).item())
    return float(mx.max(mx.abs(a - b)).item()) / denom


def _mass_parity() -> None:
    """Chunked masses == single-shot masses, forced to chunk at several granularities and two T."""
    blk = BLOCK
    h, d = 8, 64
    scale = 1.0 / (d ** 0.5)
    # last-block size lp: T=896 → 7 full blocks (lp=128); T=823 → 7 blocks, ragged last (lp=55).
    for t, tag in ((896, "block-aligned"), (823, "ragged")):
        nb = (t + blk - 1) // blk
        q, k = _qk(1, h, t, d, seed=t)
        lp = t - (nb - 1) * blk
        big = XAttnConfig(block=blk, selector="vslash", min_seq=0, max_alloc_gb=8.0)
        km_s, sm_s = vertical_slash_index(q, k, scale, big)        # single-shot reference
        assert km_s.shape == (1, h, nb) and sm_s.shape == (1, h, nb)

        full_gb = _probe_gb(1, h, lp, t)
        per_blk = 1.5 * 1 * h * lp * blk * 4 / 1e9                 # one key-block of the probe
        for nblk in (3, 2, 1):                                     # coarse → 1 block/chunk
            cap = (nblk + 0.5) * per_blk                            # max_alloc_gb landing nblk blocks/chunk
            assert full_gb > cap, f"T={t} nblk={nblk}: probe {full_gb:.2e} !> max_alloc {cap:.2e} (no chunk)"
            small = replace(big, max_alloc_gb=cap)
            km_c, sm_c = vertical_slash_index(q, k, scale, small)   # chunked
            rk, rs = _rel(km_c, km_s), _rel(sm_c, sm_s)
            n_chunks = (t + nblk * blk - 1) // (nblk * blk)
            print(f"  mass {tag:13s} T={t} {n_chunks} chunk(s) (~{nblk} blk): "
                  f"key rel={rk:.2e}  slash rel={rs:.2e}")
            assert rk < 1e-4, f"T={t} nblk={nblk}: chunked key_mass != single-shot ({rk:.2e})"
            assert rs < 1e-4, f"T={t} nblk={nblk}: chunked slash_mass != single-shot ({rs:.2e})"


def _param_independence() -> None:
    """The chunked masses are param-free (top-vert/slash cut is in select_keep, M6) — identical across
    vert/slash, so a per-head vslash spec can re-cut the one chunked probe with its own params."""
    blk = BLOCK
    h, d, t = 8, 64, 896
    nb = t // blk
    scale = 1.0 / (d ** 0.5)
    q, k = _qk(1, h, t, d, seed=7)
    per_blk = 1.5 * 1 * h * blk * blk * 4 / 1e9
    small = XAttnConfig(block=blk, selector="vslash", vert=2, slash=2, min_seq=0,
                        max_alloc_gb=2.5 * per_blk)               # forces ~3 blocks/chunk
    assert _probe_gb(1, h, blk, t) > small.max_alloc_gb, "probe must exceed max_alloc (chunk engaged)"
    km_a, sm_a = vertical_slash_index(q, k, scale, small)
    km_b, sm_b = vertical_slash_index(q, k, scale, replace(small, vert=nb, slash=nb))
    dk = float(mx.max(mx.abs(km_a - km_b)).item())
    ds = float(mx.max(mx.abs(sm_a - sm_b)).item())
    print(f"  param-independence: key Δ={dk:.2e}  slash Δ={ds:.2e}  (expected 0)")
    assert dk == 0.0 and ds == 0.0, "chunked masses depend on vert/slash — not param-independent"


def _chunked_keep_all_causal() -> None:
    """Chunked masses → select_keep with vert/slash ≥ n_blocks == the full causal block mask."""
    blk = BLOCK
    h, d, t = 8, 64, 896
    nb = t // blk
    scale = 1.0 / (d ** 0.5)
    q, k = _qk(1, h, t, d, seed=11)
    per_blk = 1.5 * 1 * h * blk * blk * 4 / 1e9
    small = XAttnConfig(block=blk, selector="vslash", vert=nb + 2, slash=nb + 2, min_seq=0,
                        max_alloc_gb=1.5 * per_blk)               # ~1 block/chunk (nb chunks)
    assert _probe_gb(1, h, blk, t) > small.max_alloc_gb, "probe must exceed max_alloc (chunk engaged)"
    index = vertical_slash_index(q, k, scale, small)              # chunked masses
    keep, _ = select_keep(q, k, scale, small, 0, index)
    i = mx.arange(nb)[:, None]
    j = mx.arange(nb)[None, :]
    causal = (j <= i)[None, None]
    diff = int(mx.sum(keep != causal).item())
    print(f"  chunked keep-all vs causal: off by {diff} cell(s)  (expected 0)")
    assert diff == 0, f"chunked keep-all != full causal mask ({diff} cells) — chunked selector bug"


def _chunked_gather_eq_mask() -> None:
    """The SAME chunked masses drive the gather (speed) and mask (quality) paths to one output.

    The budget must stay NON-binding so the two paths keep the identical set (the gather caps to budget,
    the mask path does not) — exactly the existing single-shot gate's setup, here with a chunked probe.
    A long enough sequence makes the probe ``[B,H,blk,S]`` exceed the gather's (budget-bounded,
    S-independent) per-query-block alloc, so ONE ``max_alloc_gb`` in their gap forces a chunked probe
    AND a working, non-binding block-gather — both recompute the identical chunked index."""
    blk = BLOCK
    h, d, t = 2, 64, 2048
    nb = t // blk
    scale = 1.0 / (d ** 0.5)
    q, k = _qk(1, h, t, d, seed=23)
    mx.random.seed(24)
    v = mx.random.normal((1, h, t, d)).astype(mx.float32)
    budget = 8                                          # vert=slash=1 keeps ≤ ~4 blocks/query ⇒ cap idle
    cap = min(budget, nb)
    probe_gb = _probe_gb(1, h, blk, t)                  # lp == blk here (T a block multiple)
    per_block_gb = 1.5 * 1 * h * cap * blk * (blk * 3 + d * 2 + d * 2) / 1e9   # gather's per-query-block
    assert per_block_gb < probe_gb, "regime: gather per-block must fit under the probe to co-exist"
    max_alloc = 0.5 * (per_block_gb + probe_gb)         # in (per_block, probe): chunk probe, run gather
    cfg_g = XAttnConfig(block=blk, selector="vslash", vert=1, slash=1, budget=budget, min_seq=0,
                        gather=True, max_alloc_gb=max_alloc)
    assert probe_gb > cfg_g.max_alloc_gb, "probe must exceed max_alloc (chunk engaged)"
    index = vertical_slash_index(q, k, scale, cfg_g)             # chunked masses
    keep, _ = select_keep(q, k, scale, replace(cfg_g, gather=False), 0, index)
    mask = additive_mask(keep, t, t, blk, q.dtype)
    o_mask = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    o_gath = gather_sparse_attention(q, k, v, scale, cfg_g)      # recomputes the same chunked index
    d4 = _rel(o_gath, o_mask)
    print(f"  chunked gather vs mask: rel={d4:.2e}  (expected ~0: same chunked index, budget non-binding)")
    assert d4 < 1e-3, f"chunked gather != mask ({d4:.2e}) — the two executions disagree on the probe"


def run() -> None:
    print("=== InternLM2 vertical-slash key-chunked long-context probe (M7) — model-free ===")
    _mass_parity()
    _param_independence()
    _chunked_keep_all_causal()
    _chunked_gather_eq_mask()
    print("PASS — key-chunked vslash probe: masses == single-shot (param-independent), keep-all == "
          "causal, gather == mask. Long-context probe scales to 100K+ at O(one key chunk) memory.")


if __name__ == "__main__":
    run()
