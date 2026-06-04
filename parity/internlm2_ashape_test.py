"""InternLM2.5 A-shape selector integration gate — M2 of the MInference sparse-prefill track.

M1 measured XAttention's antidiagonal-nucleus selector. M2 adds MInference's **A-shape** selector
(attention sink block 0 + a ``local``-block causal window — the StreamingLLM pattern) as an
*alternative selector* feeding the **same** validated block-gather / additive-mask execution
(:mod:`quanta.modeling.xattention`). A-shape picks the kept blocks *positionally* (no scoring); the
downstream gather/mask machinery is byte-identical to the XAttention path.

This gate proves the *integration* is correct on the real attention module (covers the GQA
``mx.repeat``, ``attn_scale``, post-RoPE q/k), independent of weights:

  1. ``ashape_keep`` selects exactly ``{0} ∪ {i-local+1..i}`` (closed-form, q_offset-shifted),
  2. A-shape with ``local ≥ n_blocks`` keeps every causal block ⇒ additive-mask path == dense,
  3. ditto the block-gather path (the long-context speed path) == dense — A-shape's parity anchor,
  4. A-shape **gather path == mask path** at a tight window (same selection, both executions agree)
     — the M2 correctness invariant (the real-model ppl twin checks the same thing on the bake),
  5. a tight ``local`` drops blocks ⇒ the output changes (the static lever actually engages),
  6. the budget cap engages for A-shape (force-keeps sink+diagonal, bounded shape),
  7. ``min_seq > T`` gates sparsity off entirely,
  8. perturbing the final token cannot change an earlier query block's output (no future leakage).

Model-free (tiny random ``InternLM2Attention``, fp32 for a tight numeric bound). A-shape is a static
selector — known lossier than the dynamic nucleus; its *quality* cost vs dense and vs the M1
XAttention baseline is measured separately on the real bake (CLAUDE.md: sparse attention is
ppl-gated, never numeric-parity-gated).

    uv run python -m parity.internlm2_ashape_test
"""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_map

from quanta.internlm2.attention import InternLM2Attention
from quanta.internlm2.config import InternLM2Config
from quanta.modeling.xattention import XAttnConfig, ashape_keep

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


def _ash(local: int, *, gather: bool, min_seq: int = 0, budget: int | None = 64) -> XAttnConfig:
    return XAttnConfig(block=BLOCK, stride=16, selector="ashape", local=local,
                       min_seq=min_seq, gather=gather, budget=budget)


def _check_selector() -> None:
    """``ashape_keep`` must equal the closed-form sink+window pattern, incl. a q_offset shift."""
    for q_offset in (0, 4):
        for local in (1, 2, 3):
            tq, tk = 4, 8
            keep = ashape_keep(tq, tk, q_offset, local)
            i = mx.arange(q_offset, q_offset + tq)[:, None]
            j = mx.arange(tk)[None, :]
            expect = ((j == 0) | ((j > i - local) & (j <= i))) & (j <= i)
            bad = int(mx.sum(keep != expect).item())
            assert bad == 0, f"ashape_keep mismatch (q_offset={q_offset}, local={local}): {bad} cells"
    print("ashape_keep closed-form (sink+window, q_offset-shifted)  OK")


def run() -> None:
    mx.random.seed(0)
    cfg = _cfg()
    a = _attn(cfg)
    x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.float32)

    _check_selector()

    dense_fast = _prefill(a, x, None, use_fast=True)
    dense_naive = _prefill(a, x, None, use_fast=False)
    d0 = float(mx.max(mx.abs(dense_fast - dense_naive)))
    print(f"dense fast vs naive                 max|Δ| = {d0:.2e}")
    assert d0 < 5e-3, f"dense fast/naive disagree ({d0}) — a pre-existing issue, not sparse"

    # 2. keep-all (local ≥ n_blocks) additive-mask path == dense  (A-shape parity anchor)
    keepall = NB + 4
    d1 = float(mx.max(mx.abs(_prefill(a, x, _ash(keepall, gather=False), use_fast=True) - dense_fast)))
    print(f"ashape keep-all mask   vs dense     max|Δ| = {d1:.2e}")
    assert d1 < 1e-4, f"A-shape keep-all mask path != dense ({d1}) — selector integration bug"

    # 3. keep-all block-gather path == dense (the long-context speed path)
    d2 = float(mx.max(mx.abs(_prefill(a, x, _ash(keepall, gather=True), use_fast=True) - dense_fast)))
    print(f"ashape keep-all gather vs dense     max|Δ| = {d2:.2e}")
    assert d2 < 1e-3, f"A-shape keep-all gather path != dense ({d2}) — selector integration bug"

    # 4. gather path == mask path at a tight window (same A-shape selection, two executions) — the
    #    M2 invariant. local=1 ⇒ each query block attends only sink (block 0) + its diagonal block.
    tight_mask = _prefill(a, x, _ash(1, gather=False), use_fast=True)
    tight_gath = _prefill(a, x, _ash(1, gather=True), use_fast=True)
    d3 = float(mx.max(mx.abs(tight_mask - tight_gath)))
    print(f"ashape L=1 gather vs mask           max|Δ| = {d3:.2e}  (expected ~0: same blocks)")
    assert d3 < 1e-3, f"A-shape gather != mask ({d3}) — the two executions disagree on selection"

    # 5. a tight window drops blocks ⇒ output changes (the static lever engages)
    d4 = float(mx.max(mx.abs(tight_mask - dense_fast)))
    print(f"ashape L=1 mask   vs dense          max|Δ| = {d4:.2e}  (expected > 0: sparsity active)")
    assert d4 > 1e-2, f"A-shape local=1 did not change the output ({d4}) — sparsity not engaging"

    # 6. budget cap engages for A-shape: keep-all + budget=2 force-keeps sink+diagonal only, so it
    #    runs (bounded shape) and differs from the uncapped keep-all (== dense).
    capped = _prefill(a, x, _ash(keepall, gather=True, budget=2), use_fast=True)
    d5 = float(mx.max(mx.abs(capped - dense_fast)))
    print(f"ashape keep-all budget=2 vs dense   max|Δ| = {d5:.2e}  (expected > 0: cap drops blocks)")
    assert capped.shape == dense_fast.shape, "budget-capped A-shape changed output shape"
    assert d5 > 1e-2, f"budget=2 did not cap A-shape ({d5}) — the kept-block budget is not binding"

    # 7. min_seq gate: T below min_seq ⇒ dense even with a sparse config set
    d6 = float(mx.max(mx.abs(_prefill(a, x, _ash(1, gather=True, min_seq=T + 1), use_fast=True)
                             - dense_fast)))
    print(f"ashape L=1 min_seq>T vs dense       max|Δ| = {d6:.2e}  (expected 0: gated off)")
    assert d6 < 1e-6, f"min_seq gate failed ({d6}) — A-shape engaged below min_seq"

    # 8. causal safety: perturb the FINAL token; block 0 (rows [0,BLOCK)) is strictly causally
    #    upstream of it, so its output must be bit-unchanged — no future key leaks through A-shape.
    x2 = mx.array(x)
    x2[:, -1, :] = mx.random.normal((cfg.hidden_size,)).astype(mx.float32)
    pert = _prefill(a, x2, _ash(1, gather=True), use_fast=True)
    d7 = float(mx.max(mx.abs(tight_gath[:, :BLOCK] - pert[:, :BLOCK])))
    print(f"causal: perturb last tok, Δ blk0    max|Δ| = {d7:.2e}  (expected 0)")
    assert d7 < 1e-6, f"perturbing the final token changed block 0 ({d7}) — future leakage"

    print("PASS — InternLM2 A-shape selector: dense-equivalent at keep-all (mask & gather), "
          "gather==mask, sparsity-active + budget-capped below it, min_seq-gated, causal-safe.")


if __name__ == "__main__":
    run()
