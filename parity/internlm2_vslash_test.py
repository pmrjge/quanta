"""InternLM2.5 vertical-slash selector integration gate — M3 of the MInference sparse-prefill track.

M1 measured XAttention's antidiagonal-nucleus selector; M2 added MInference's positional A-shape.
M3 adds MInference's **vertical-slash** selector (MInference §3): an online probe of the LAST query
block's attention to all keys builds ONE global pattern — top-``vert`` **vertical** key-blocks
(columns every query attends) ∪ top-``slash`` **slash** block-offset bands (diagonals at a fixed
query-minus-key offset) — which is then applied to every query block through the **same** validated
block-gather / additive-mask execution (:mod:`quanta.modeling.xattention`). Only the block-*selection*
differs (``XAttnConfig.selector``); the downstream gather/mask machinery is byte-identical.

Unlike xattn/ashape (which select locally per query block), vertical-slash's pattern is *global*:
the caller computes :func:`vertical_slash_index` once over the whole sequence and threads it into
every chunk of the gather path — so the gather and mask paths select identically. This gate proves
the *integration* is correct on the real attention module (covers the GQA ``mx.repeat``,
``attn_scale``, post-RoPE q/k), independent of weights:

  1. :func:`vertical_slash_index` + :func:`select_keep` produce a strictly **causal** keep-mask
     (no future block ever selected) that always contains the diagonal + sink, and degenerate to the
     full causal mask when ``vert``/``slash`` cover every block,
  2. vertical-slash with ``vert``/``slash`` ≥ n_blocks keeps every causal block ⇒ additive-mask
     path == dense (the parity anchor),
  3. ditto the block-gather path (the long-context speed path) == dense,
  4. vertical-slash **gather path == mask path** at a tight ``vert=slash=1`` pattern (the SAME global
     index drives both executions) — the M3 correctness invariant (the real-model ppl twin checks the
     same thing on the bake),
  5. a tight ``vert``/``slash`` drops blocks ⇒ the output changes (the lossy lever engages),
  6. the budget cap engages for vertical-slash (force-keeps sink+diagonal, bounded shape),
  7. ``min_seq > T`` gates sparsity off entirely,
  8. perturbing the final token cannot change an earlier query block's output (block 0's selectable
     set is force-pinned to {block 0} by causality, so it is probe-independent — no future leakage).

Model-free (tiny random ``InternLM2Attention``, fp32 for a tight numeric bound). Vertical-slash is a
data-dependent but prefill-only selector; its *quality* cost vs dense and vs the M1/M2 baselines is
measured separately on the real bake (CLAUDE.md: sparse attention is ppl-gated, not parity-gated).

    uv run python -m parity.internlm2_vslash_test
"""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_map

from quanta.internlm2.attention import InternLM2Attention
from quanta.internlm2.config import InternLM2Config
from quanta.modeling.xattention import XAttnConfig, select_keep, vertical_slash_index

T = 500       # not a multiple of block=128 → exercises padding; spans 4 blocks (ceil(500/128))
BLOCK = 128
NB = (T + BLOCK - 1) // BLOCK   # 4


def _cfg() -> InternLM2Config:
    return InternLM2Config(
        vocab_size=64, hidden_size=128, num_hidden_layers=1, intermediate_size=256,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, attention_bias=False,
        rope_theta=1.0e4, rope_scaling_type="dynamic", rope_scaling_factor=2.5,
        max_position_embeddings=4096, hidden_act="silu", norm_eps=1e-5, tie_word_embeddings=False,
        eos_token_id=2, eos_token_ids=(2,), pad_token_id=2, bos_token_id=1, add_bos_token=True,
    )


def _attn(cfg: InternLM2Config) -> InternLM2Attention:
    a = InternLM2Attention(cfg)
    a.update(tree_map(lambda p: p.astype(mx.float32), a.parameters()))  # fp32 → tight parity bound
    mx.eval(a.parameters())
    return a


def _prefill(a: InternLM2Attention, x: mx.array, sparse: XAttnConfig | None, *, use_fast: bool):
    a.sparse = sparse
    out = a(x, cache=None, use_fast=use_fast)   # cache=None ⇒ offset 0, kv_len == t ⇒ prefill
    mx.eval(out)
    return out


def _vs(vert: int, slash: int, *, gather: bool, min_seq: int = 0, budget: int | None = 64) -> XAttnConfig:
    return XAttnConfig(block=BLOCK, stride=16, selector="vslash", vert=vert, slash=slash,
                       min_seq=min_seq, gather=gather, budget=budget)


def _check_index() -> None:
    """The vertical-slash selection must be strictly causal, contain the diagonal+sink, and be exactly
    the full causal mask when ``vert``/``slash`` cover every block — a structural check on random q/k."""
    mx.random.seed(1)
    cfg = _cfg()
    h, d = 4, cfg.head_dim
    q = mx.random.normal((1, h, T, d)).astype(mx.float32)
    k = mx.random.normal((1, h, T, d)).astype(mx.float32)
    scale = cfg.attn_scale

    vert_keep, slash_keep, key_mass = vertical_slash_index(q, k, scale, _vs(2, 2, gather=False))
    assert vert_keep.shape == (1, h, NB) and vert_keep.dtype == mx.bool_, "vert_keep shape/dtype"
    assert slash_keep.shape == (1, h, NB) and slash_keep.dtype == mx.bool_, "slash_keep shape/dtype"
    assert key_mass.shape == (1, h, NB), "key_mass shape"
    assert bool(mx.all(mx.isfinite(key_mass)).item()), "key_mass must be finite"

    i = mx.arange(NB)[:, None]
    j = mx.arange(NB)[None, :]
    causal = j <= i
    forced = ((j == i) | (j == 0)) & causal

    keep, rank = select_keep(q, k, scale, _vs(1, 1, gather=False))
    assert keep.shape == (1, h, NB, NB) and rank.shape == (1, h, NB, NB), "keep/rank shape"
    future = int(mx.sum(keep & (j > i)[None, None]).item())
    assert future == 0, f"vertical-slash selected {future} non-causal (future) blocks"
    missing = int(mx.sum(forced[None, None] & ~keep).item())
    assert missing == 0, f"vertical-slash dropped {missing} forced diagonal/sink blocks"

    keepall, _ = select_keep(q, k, scale, _vs(NB, NB, gather=False))
    diff = int(mx.sum(keepall != causal[None, None]).item())
    assert diff == 0, f"vert=slash=n_blocks must equal the full causal mask, off by {diff} cells"
    print("vertical_slash_index causal / diag+sink-forced / keep-all==causal   OK")


def run() -> None:
    mx.random.seed(0)
    cfg = _cfg()
    a = _attn(cfg)
    x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.float32)

    _check_index()

    dense_fast = _prefill(a, x, None, use_fast=True)
    dense_naive = _prefill(a, x, None, use_fast=False)
    d0 = float(mx.max(mx.abs(dense_fast - dense_naive)))
    print(f"dense fast vs naive                 max|Δ| = {d0:.2e}")
    assert d0 < 5e-3, f"dense fast/naive disagree ({d0}) — a pre-existing issue, not sparse"

    # 2. keep-all (vert,slash ≥ n_blocks) additive-mask path == dense  (vertical-slash parity anchor)
    keepall = NB + 2
    d1 = float(mx.max(mx.abs(
        _prefill(a, x, _vs(keepall, keepall, gather=False), use_fast=True) - dense_fast)))
    print(f"vslash keep-all mask   vs dense     max|Δ| = {d1:.2e}")
    assert d1 < 1e-4, f"vertical-slash keep-all mask path != dense ({d1}) — selector integration bug"

    # 3. keep-all block-gather path == dense (the long-context speed path)
    d2 = float(mx.max(mx.abs(
        _prefill(a, x, _vs(keepall, keepall, gather=True), use_fast=True) - dense_fast)))
    print(f"vslash keep-all gather vs dense     max|Δ| = {d2:.2e}")
    assert d2 < 1e-3, f"vertical-slash keep-all gather path != dense ({d2}) — selector integration bug"

    # 4. gather path == mask path at a tight pattern (the SAME global index drives both) — the M3
    #    invariant. vert=slash=1 ⇒ sink + diagonal + 1 vertical block + 1 slash band per query block.
    tight_mask = _prefill(a, x, _vs(1, 1, gather=False), use_fast=True)
    tight_gath = _prefill(a, x, _vs(1, 1, gather=True), use_fast=True)
    d3 = float(mx.max(mx.abs(tight_mask - tight_gath)))
    print(f"vslash v=s=1 gather vs mask         max|Δ| = {d3:.2e}  (expected ~0: same global index)")
    assert d3 < 1e-3, f"vertical-slash gather != mask ({d3}) — the two executions disagree on selection"

    # 5. a tight pattern drops blocks ⇒ output changes (the lossy lever engages)
    d4 = float(mx.max(mx.abs(tight_mask - dense_fast)))
    print(f"vslash v=s=1 mask   vs dense        max|Δ| = {d4:.2e}  (expected > 0: sparsity active)")
    assert d4 > 1e-2, f"vertical-slash vert=slash=1 did not change the output ({d4}) — not engaging"

    # 6. budget cap engages: keep-all + budget=2 force-keeps sink+diagonal only, so it runs (bounded
    #    shape) and differs from the uncapped keep-all (== dense).
    capped = _prefill(a, x, _vs(keepall, keepall, gather=True, budget=2), use_fast=True)
    d5 = float(mx.max(mx.abs(capped - dense_fast)))
    print(f"vslash keep-all budget=2 vs dense   max|Δ| = {d5:.2e}  (expected > 0: cap drops blocks)")
    assert capped.shape == dense_fast.shape, "budget-capped vertical-slash changed output shape"
    assert d5 > 1e-2, f"budget=2 did not cap vertical-slash ({d5}) — the kept-block budget is not binding"

    # 7. min_seq gate: T below min_seq ⇒ dense even with a sparse config set
    d6 = float(mx.max(mx.abs(_prefill(a, x, _vs(1, 1, gather=True, min_seq=T + 1), use_fast=True)
                             - dense_fast)))
    print(f"vslash v=s=1 min_seq>T vs dense     max|Δ| = {d6:.2e}  (expected 0: gated off)")
    assert d6 < 1e-6, f"min_seq gate failed ({d6}) — vertical-slash engaged below min_seq"

    # 8. causal safety: perturb the FINAL token; block 0 (rows [0,BLOCK)) can only attend to key
    #    block 0 (causality), so its selectable set is {block 0} regardless of the probe — its output
    #    must be bit-unchanged even though the global pattern is probe-derived. No future leakage.
    x2 = mx.array(x)
    x2[:, -1, :] = mx.random.normal((cfg.hidden_size,)).astype(mx.float32)
    pert = _prefill(a, x2, _vs(1, 1, gather=True), use_fast=True)
    d7 = float(mx.max(mx.abs(tight_gath[:, :BLOCK] - pert[:, :BLOCK])))
    print(f"causal: perturb last tok, Δ blk0    max|Δ| = {d7:.2e}  (expected 0)")
    assert d7 < 1e-6, f"perturbing the final token changed block 0 ({d7}) — future leakage"

    print("PASS — InternLM2 vertical-slash selector: causal global pattern, dense-equivalent at "
          "keep-all (mask & gather), gather==mask, sparsity-active + budget-capped below it, "
          "min_seq-gated, causal-safe.")


if __name__ == "__main__":
    run()
