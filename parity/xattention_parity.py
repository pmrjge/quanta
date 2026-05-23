"""XAttention sanity: keep-all == dense, causal-safe, and actually sparse.

1. threshold=1.0 keeps every causal block ⇒ sparse output == dense causal attention
   (numeric parity — proves the block/mask machinery is correct).
2. The additive mask never allows a future key (b>a) at any threshold.
3. A lower threshold drops blocks (non-trivial sparsity) and changes the output —
   so the lever does something (its quality cost is what the long-doc ppl gate measures).

    uv run python -m parity.xattention_parity
"""

from __future__ import annotations

import mlx.core as mx

from quanta.modeling.xattention import (
    NEG_INF,
    XAttnConfig,
    block_scores,
    gather_sparse_attention,
    select_blocks,
    sparse_prefill_mask,
)

BSZ, H, T, D, VHD = 1, 4, 500, 192, 128  # T not a multiple of block → exercises padding
SCALE = D ** -0.5


def _attn(q, k, v, add_mask):
    sc = (q @ mx.swapaxes(k, -1, -2)) * SCALE + add_mask
    w = mx.softmax(sc.astype(mx.float32), axis=-1).astype(q.dtype)
    return w @ v


def run() -> None:
    mx.random.seed(0)
    q = mx.random.normal((BSZ, H, T, D))
    k = mx.random.normal((BSZ, H, T, D))
    v = mx.random.normal((BSZ, H, T, VHD))

    a = mx.arange(T)[:, None]
    b = mx.arange(T)[None, :]
    causal = mx.where(b <= a, mx.array(0.0), mx.array(NEG_INF))[None, None]
    dense = _attn(q, k, v, causal)

    cfg_full = XAttnConfig(block=128, stride=16, threshold=1.0)
    m_full = sparse_prefill_mask(q, k, SCALE, cfg_full)
    sparse_full = _attn(q, k, v, m_full)
    max_abs = mx.max(mx.abs(dense - sparse_full)).item()

    # future-leak check: wherever a future key (b>a) is allowed (mask > -inf) → fail
    future_allowed = mx.sum((m_full > NEG_INF) & (b > a)[None, None]).item()

    cfg_sp = XAttnConfig(block=128, stride=16, threshold=0.5)
    keep = select_blocks(block_scores(q, k, SCALE, 128, 16), 0.5)
    tq = keep.shape[-1]
    n_valid = (tq * (tq + 1)) // 2  # causal block pairs
    kept = int(mx.sum(keep).item()) // (BSZ * H)
    m_sp = sparse_prefill_mask(q, k, SCALE, cfg_sp)
    sparse_sp = _attn(q, k, v, m_sp)
    drift_sp = mx.max(mx.abs(dense - sparse_sp)).item()

    # gather (speed) path must be output-equivalent to the mask path / dense
    g_full = gather_sparse_attention(q, k, v, SCALE, XAttnConfig(threshold=1.0, gather=True))
    g_vs_dense = mx.max(mx.abs(dense - g_full)).item()
    g_sp = gather_sparse_attention(q, k, v, SCALE, cfg_sp)  # same τ=0.5 selection as mask path
    g_vs_mask = mx.max(mx.abs(sparse_sp - g_sp)).item()

    # memory guard must fail loud (raise before allocating), never OOM
    guard_fires = False
    try:
        gather_sparse_attention(q, k, v, SCALE, XAttnConfig(threshold=1.0, gather=True, max_alloc_gb=1e-9))
    except MemoryError:
        guard_fires = True
    # budget cap path runs and stays bounded
    g_budget = gather_sparse_attention(q, k, v, SCALE, XAttnConfig(threshold=1.0, gather=True, budget=2))
    budget_ok = g_budget.shape == (BSZ, H, T, VHD)

    # chunked execution (tiny max_alloc_gb → many 1-block chunks) must match single-chunk
    g_single = gather_sparse_attention(q, k, v, SCALE, XAttnConfig(threshold=0.8, gather=True))
    g_chunked = gather_sparse_attention(q, k, v, SCALE, XAttnConfig(threshold=0.8, gather=True, max_alloc_gb=0.004))
    chunk_vs_single = mx.max(mx.abs(g_single - g_chunked)).item()

    print("\n=== XAttention sanity ===")
    print(f"keep-all vs dense    : max_abs {max_abs:.3e}   (expect ~0)")
    print(f"future keys allowed  : {future_allowed}        (expect 0)")
    print(f"thr=0.5 blocks kept  : {kept}/{n_valid} causal blocks  (expect < all)")
    print(f"thr=0.5 vs dense     : max_abs {drift_sp:.3e}   (expect > 0, sparsity bites)")
    print(f"gather keep-all/dense: max_abs {g_vs_dense:.3e}   (expect ~0)")
    print(f"gather vs mask (0.5) : max_abs {g_vs_mask:.3e}   (expect ~0, same selection)")
    print(f"alloc guard fires    : {guard_fires}   (expect True — fails loud, no OOM)")
    print(f"budget=2 cap runs    : {budget_ok}   (bounded; shape {tuple(g_budget.shape)})")
    print(f"chunked vs 1-chunk   : max_abs {chunk_vs_single:.3e}   (expect ~0, chunking exact)")


if __name__ == "__main__":
    run()
