"""InternLM2.5 per-head-GROUPED gather "fold" gate — M9-speed of the MInference sparse-prefill track.

M4–M6 route each query head to its own selector (kind, params) — the MInference per-head assignment.
That folds **quality** (each head gets its cheapest-sufficient pattern). It does NOT fold **speed**: the
block-gather sizes its work by ONE global ``max_kept`` = the densest head's kept-block count (it builds a
rectangular ``[B,H,m,max_kept,blk]`` gather), so a mix of a cheap static pattern (ashape, ~3% kept at
long context) with a dense one (xattn nucleus, ~63% kept — see the M8 wall-clock bench) makes EVERY head
pay the dense head's budget. Combining the approaches does not, by itself, fold the speed.

``XAttnConfig.grouped_gather`` is the fold: partition heads by their DISTINCT spec and gather each group
at its OWN ``max_kept`` (a bounded loop over distinct specs, rule 3), so the cheap-pattern heads finally
run cheap. It is an **output-equivalent optimization** (rule 4): head ``i`` attends exactly the same kept
blocks either way — the naive path's extra ``-inf`` gather slots contribute nothing to the softmax — so
it is kept behind the ``grouped_gather`` flag (default False ⇒ the naive single-``max_kept`` path) until
this gate proves the equivalence; the wall-clock win is measured separately by the prefill bench.

Model-free (synthetic ``q``/``k``/``v``, fp32 for a tight bound). Proves, on a per-head assignment that
mixes all three selector kinds at different params:

  1. **grouped == naive (head_specs)** — the folded gather is bit-equivalent to the naive per-head gather.
  2. **grouped == mask (head_specs)** — composing the M6 invariant: the fold also matches the additive-mask
     quality path (so it is correct end-to-end, not just equal to the naive gather).
  3. **grouped == naive (head_selectors)** — the M4 kind-only per-head config folds too.
  4. **budget-bound** — with a binding kept-block budget, grouped still == naive (the per-group and global
     caps select the same per-head blocks).
  5. **fold premise** — the per-group ``max_kept`` of the cheap group is far below the global ``max_kept``
     (which the dense group sets), so the fold has real work to save (informational, the bench times it).

    uv run python -m parity.internlm2_grouped_gather_test
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx

from quanta.modeling.xattention import (
    HeadSpec,
    XAttnConfig,
    _uses_vslash,
    additive_mask,
    gather_sparse_attention,
    select_keep,
    vertical_slash_index,
)

BLOCK = 128
T = 1024                      # 8 blocks — enough that ashape (cheap) and xattn (dense) keep different #s
H, D = 8, 64


def _qkv(seed: int) -> tuple[mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    q = mx.random.normal((1, H, T, D)).astype(mx.float32)
    k = mx.random.normal((1, H, T, D)).astype(mx.float32)
    v = mx.random.normal((1, H, T, D)).astype(mx.float32)
    return q, k, v


def _rel(a: mx.array, b: mx.array) -> float:
    return float((mx.max(mx.abs(a - b)) / (mx.max(mx.abs(b)) + 1e-12)).item())


# A per-head assignment mixing all three kinds at different params (the fold's stress case):
# 4 cheap static ashape heads, 2 vslash heads, 2 dense xattn heads.
MIXED_SPECS = (
    HeadSpec("ashape", local=2), HeadSpec("ashape", local=2),
    HeadSpec("ashape", local=4), HeadSpec("ashape", local=2),
    HeadSpec("vslash", vert=2, slash=2), HeadSpec("vslash", vert=3, slash=3),
    HeadSpec("xattn", threshold=0.9), HeadSpec("xattn", threshold=0.95),
)
MIXED_KINDS = ("ashape", "ashape", "ashape", "ashape", "vslash", "vslash", "xattn", "xattn")


def _mask_path(q: mx.array, k: mx.array, v: mx.array, scale: float, cfg: XAttnConfig) -> mx.array:
    """The per-head additive-mask quality path (gather=False) for the same selection."""
    index = vertical_slash_index(q, k, scale, cfg) if _uses_vslash(cfg) else None
    keep, _ = select_keep(q, k, scale, replace(cfg, gather=False), 0, index)
    mask = additive_mask(keep, T, T, BLOCK, q.dtype)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def _kept_per_head(q: mx.array, k: mx.array, scale: float, cfg: XAttnConfig) -> mx.array:
    """Mean kept key-blocks per head ``[H]`` for a per-head config (the gather's per-head work)."""
    index = vertical_slash_index(q, k, scale, cfg) if _uses_vslash(cfg) else None
    keep, _ = select_keep(q, k, scale, replace(cfg, gather=False), 0, index)
    return mx.mean(mx.sum(keep.astype(mx.float32), axis=-1), axis=(0, 2))      # [H]


def run() -> None:
    print("=== InternLM2 per-head-grouped gather fold (M9-speed) — model-free ===")
    q, k, v = _qkv(0)
    scale = 1.0 / (D ** 0.5)

    # 1. grouped == naive (head_specs)
    cfg_naive = XAttnConfig(block=BLOCK, stride=16, head_specs=MIXED_SPECS, min_seq=0, gather=True)
    cfg_fold = replace(cfg_naive, grouped_gather=True)
    naive = gather_sparse_attention(q, k, v, scale, cfg_naive)
    fold = gather_sparse_attention(q, k, v, scale, cfg_fold)
    d1 = _rel(fold, naive)
    print(f"  grouped vs naive (head_specs)   rel={d1:.2e}  (expect ~0: output-equivalent fold)")
    assert d1 < 1e-5, f"grouped per-head gather != naive ({d1:.2e}) — the fold is not output-equivalent"

    # 2. grouped == mask (compose the M6 gather==mask invariant — correct end-to-end, not just == naive)
    o_mask = _mask_path(q, k, v, scale, cfg_naive)
    d2 = _rel(fold, o_mask)
    print(f"  grouped vs mask   (head_specs)  rel={d2:.2e}  (expect <1e-3: same selection, gather==mask)")
    assert d2 < 1e-3, f"grouped per-head gather != mask path ({d2:.2e}) — selection disagrees"

    # 3. grouped == naive (head_selectors, the M4 kind-only per-head config)
    cn = XAttnConfig(block=BLOCK, stride=16, head_selectors=MIXED_KINDS, threshold=0.9, local=2,
                     vert=2, slash=2, min_seq=0, gather=True)
    d3 = _rel(gather_sparse_attention(q, k, v, scale, replace(cn, grouped_gather=True)),
              gather_sparse_attention(q, k, v, scale, cn))
    print(f"  grouped vs naive (head_selectors) rel={d3:.2e}  (expect ~0)")
    assert d3 < 1e-5, f"grouped head_selectors gather != naive ({d3:.2e})"

    # 4. budget-bound: a binding kept-block cap still folds equivalently (per-group cap == global cap
    #    select the same per-head blocks by rank)
    cb_naive = replace(cfg_naive, budget=2)
    d4 = _rel(gather_sparse_attention(q, k, v, scale, replace(cb_naive, grouped_gather=True)),
              gather_sparse_attention(q, k, v, scale, cb_naive))
    print(f"  grouped vs naive (budget=2 bind)  rel={d4:.2e}  (expect ~0)")
    assert d4 < 1e-5, f"grouped != naive under a binding budget ({d4:.2e})"

    # 5. fold premise: the cheap group's per-group max_kept is far below the global max_kept (set by the
    #    dense xattn group) — the naive gather makes every head pay the global; the fold saves exactly that.
    kept = _kept_per_head(q, k, scale, cfg_naive)                  # [H] mean kept blocks/head
    ashape_max = float(mx.max(kept[:4]).item())                    # cheap ashape group
    xattn_max = float(mx.max(kept[6:]).item())                     # dense xattn group
    global_max = float(mx.max(kept).item())
    print(f"  fold premise: per-group max_kept ashape={ashape_max:.1f}  xattn={xattn_max:.1f}  "
          f"global={global_max:.1f}  (naive pays global for all; fold pays per-group)")
    assert ashape_max < xattn_max, "expected the static ashape heads to keep fewer blocks than xattn"
    assert global_max == xattn_max, "global max_kept should be set by the dense (xattn) group"

    print("PASS — per-head-grouped gather fold: output-equivalent to the naive per-head gather (and the "
          "mask path), for head_specs & head_selectors, with/without a binding budget; the cheap groups "
          "keep far fewer blocks than the dense group (the speed the bench measures).")


if __name__ == "__main__":
    run()
