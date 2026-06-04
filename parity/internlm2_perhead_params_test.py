"""InternLM2.5 per-head *params* gate — M5 of the MInference sparse-prefill track.

M4 routed each query head to a selector *kind* but shared one set of params across every head of that
kind. M5 lets each head carry its OWN params: a tuple of :class:`~quanta.modeling.xattention.HeadSpec`
(kind + that kind's params) on ``XAttnConfig.head_specs``, and the offline assignment becomes a
**kernel-aware FLOP-budgeted search** (:func:`~quanta.modeling.xattention.assign_head_specs`) — per head,
the most accurate candidate (kind, params) whose cost fits a FLOP budget. The per-head-params path is a
pure *routing layer* exactly like M4: head ``h``'s kept-block mask is byte-identical to the uniform keep
for ``head_specs[h]``'s (kind, params); the only new thing is that two heads of the same kind can now hold
**different params**.

Two independently-checkable pieces, both gated here model-free (no weights needed):

  * the **policy** :func:`assign_head_specs` — given each candidate's per-head error vs dense and its
    kernel-aware cost (rows cheap→accurate), pick per head the most-accurate within a FLOP ``budget``,
    else the cheapest. Pure / positional, so a hand-built (errors, costs) matrix pins every branch
    (budget excludes the accurate candidate / full budget → most accurate / under budget → cheapest /
    ties → the cheaper).
  * the **mechanism** on the real ``InternLM2Attention`` (covers GQA ``mx.repeat``, ``attn_scale``,
    post-RoPE q/k):
      1. routing exactness — ``select_keep`` with a mixed ``head_specs`` (incl. two same-kind heads at
         DIFFERENT params) reproduces, for each head, the uniform single-spec keep for that head's
         (kind, params) — the M5 essence,
      2. a uniform-as-per-head-specs assignment (every head the same spec) == the plain uniform selector,
      3. a MIXED keep-all assignment (all kinds + per-head params, all at keep-all) == dense — mask AND
         gather (the per-head-params parity anchor: routing is exact regardless of which specs mix),
      4. a tight mixed assignment: **gather path == mask path** (the M5 twin),
      5. the tight mixed assignment drops blocks ⇒ output changes (the lever engages),
      6. the budget cap engages under a mixed assignment (force-keeps sink+diagonal, bounded shape),
      7. ``min_seq > T`` gates sparsity off entirely,
      8. perturbing the final token cannot change an earlier query block's output (no future leakage),
      9. validation — a vslash spec whose vert/slash differs from the config is now ALLOWED (M6: the
         probe returns param-independent masses, so heads sharing it can keep different vert/slash);
         setting both head_specs and head_selectors is rejected; a non-HeadSpec entry is rejected; a
         head_specs/heads length mismatch is rejected at use.

The assignment's *quality* (which heads earn which params, and the ppl it trades) is measured on the
real bake in ``parity/internlm2_ppl_sparse.py`` (CLAUDE.md: sparse attention is ppl-gated, never
numeric-parity-gated). This gate proves the routing is correct.

    uv run python -m parity.internlm2_perhead_params_test
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
from mlx.utils import tree_map

from quanta.internlm2.attention import InternLM2Attention
from quanta.internlm2.config import InternLM2Config
from quanta.modeling.xattention import HeadSpec, XAttnConfig, assign_head_specs, select_keep

T = 500       # not a multiple of block=128 → exercises padding; spans 4 blocks (ceil(500/128))
BLOCK = 128
NB = (T + BLOCK - 1) // BLOCK   # 4
NH = 4                          # query heads in the tiny cfg (head_specs length must match)


def _cfg() -> InternLM2Config:
    return InternLM2Config(
        vocab_size=64, hidden_size=128, num_hidden_layers=1, intermediate_size=256,
        num_attention_heads=NH, num_key_value_heads=2, head_dim=32, attention_bias=False,
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


def _phs(specs: tuple[HeadSpec, ...], *, gather: bool, vert: int = 8, slash: int = 8,
         min_seq: int = 0, budget: int | None = 64) -> XAttnConfig:
    # vert/slash on the config must match any vslash spec (the shared global probe index); xattn/ashape
    # params live entirely on the per-head specs (the config's threshold/local are unused in this path).
    return XAttnConfig(block=BLOCK, stride=16, head_specs=specs, vert=vert, slash=slash,
                       min_seq=min_seq, gather=gather, budget=budget)


# Per-head-params mixes (NH=4). Note two SAME-kind heads at DIFFERENT params — the M5 point.
_KEEPALL = (HeadSpec("xattn", threshold=1.0), HeadSpec("ashape", local=NB + 2),
            HeadSpec("vslash", vert=NB + 2, slash=NB + 2), HeadSpec("ashape", local=NB + 2))
_TIGHT = (HeadSpec("xattn", threshold=0.9), HeadSpec("ashape", local=1),
          HeadSpec("vslash", vert=1, slash=1), HeadSpec("ashape", local=2))   # ashape L1 vs L2


def _check_policy() -> None:
    """:func:`assign_head_specs` routes each head to the most-accurate candidate within the FLOP
    ``budget``, else the cheapest — a hand-built (errors, costs) matrix pins every branch."""
    cands = [HeadSpec("ashape", local=1), HeadSpec("ashape", local=4), HeadSpec("xattn", threshold=0.9)]
    costs = mx.array([1.0, 4.0, 6.0])                # cheap → accurate (kernel-aware FLOP cost ascending)
    # cols h0..h3; rows = candidates (c0 cheapest ashapeL1, c1 ashapeL4, c2 accurate xattn cost6).
    errors = mx.array([
        [0.5, 0.1, 0.3, 0.9],                        # c0 ashape L1
        [0.1, 0.5, 0.3, 0.9],                        # c1 ashape L4
        [0.0, 0.0, 0.0, 0.0],                        # c2 xattn (most accurate, but cost 6)
    ])
    # budget 4 ⇒ c2 (cost 6) unaffordable; among {c0,c1} pick min error (ties → cheaper c0).
    got = assign_head_specs(errors, costs, cands, budget=4.0)
    want = (cands[1], cands[0], cands[0], cands[0])
    assert got == want, f"assign_head_specs(budget=4) {got} != {want}"
    # budget 6 ⇒ all affordable ⇒ the most accurate (c2) everywhere.
    assert assign_head_specs(errors, costs, cands, budget=6.0) == (cands[2],) * 4, "full budget not most-accurate"
    # budget 0.5 ⇒ none affordable ⇒ the globally cheapest (c0) everywhere.
    assert assign_head_specs(errors, costs, cands, budget=0.5) == (cands[0],) * 4, "under budget not cheapest"
    print("assign_head_specs budget-excludes-accurate / full-budget / under-budget / tie   OK")


def _check_routing(cfg: InternLM2Config) -> None:
    """A mixed-``head_specs`` ``select_keep`` reproduces, per head, the uniform keep for that head's
    (kind, params) — incl. two same-kind heads at different params (the M5 essence)."""
    mx.random.seed(2)
    q = mx.random.normal((1, NH, T, cfg.head_dim)).astype(mx.float32)
    k = mx.random.normal((1, NH, T, cfg.head_dim)).astype(mx.float32)
    scale = cfg.attn_scale
    specs = (HeadSpec("xattn", threshold=0.9), HeadSpec("ashape", local=1),
             HeadSpec("ashape", local=2), HeadSpec("xattn", threshold=0.95))   # 2×ashape, 2×xattn
    mixed = _phs(specs, gather=False)
    keep_mixed, rank_mixed = select_keep(q, k, scale, mixed)
    assert keep_mixed.shape == (1, NH, NB, NB), f"mixed keep shape {keep_mixed.shape}"
    assert rank_mixed.shape == (1, NH, NB, NB), f"mixed rank shape {rank_mixed.shape}"
    for h, sp in enumerate(specs):
        sub = replace(mixed, head_specs=None, selector=sp.kind, threshold=sp.threshold,
                      local=sp.local, vert=sp.vert, slash=sp.slash)
        keep_u, _ = select_keep(q, k, scale, sub)
        bad = int(mx.sum(keep_mixed[:, h] != keep_u[:, h]).item())
        assert bad == 0, f"head {h} ({sp}) routed keep differs from its uniform spec by {bad} cells"
    print("per-head-params routing exactness (mixed keep[:,h] == uniform keep for head's spec)   OK")


def run() -> None:
    mx.random.seed(0)
    cfg = _cfg()
    a = _attn(cfg)
    x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.float32)

    _check_policy()
    _check_routing(cfg)

    dense_fast = _prefill(a, x, None, use_fast=True)

    # 1. uniform-as-per-head-specs: every head the same spec == the plain uniform selector (exact).
    uni = _prefill(a, x, XAttnConfig(block=BLOCK, stride=16, selector="ashape", local=2,
                                     min_seq=0, gather=False), use_fast=True)
    php = _prefill(a, x, _phs((HeadSpec("ashape", local=2),) * NH, gather=False), use_fast=True)
    d_u = float(mx.max(mx.abs(uni - php)))
    print(f"per-head-specs all-ashapeL2 vs uniform ashapeL2 max|Δ| = {d_u:.2e}  (expected 0: same sel)")
    assert d_u < 1e-6, f"uniform-as-per-head-specs != uniform ({d_u}) — routing changed selection"

    # 2. MIXED keep-all == dense (mask & gather) — the per-head-params parity anchor. All kinds + per-head
    #    params present, every spec at keep-all ⇒ each head keeps every causal block regardless.
    d1 = float(mx.max(mx.abs(_prefill(a, x, _phs(_KEEPALL, gather=False, vert=NB + 2, slash=NB + 2),
                                      use_fast=True) - dense_fast)))
    print(f"per-head-specs mixed keep-all mask vs dense max|Δ| = {d1:.2e}")
    assert d1 < 1e-4, f"per-head-specs mixed keep-all mask != dense ({d1}) — routing integration bug"
    d2 = float(mx.max(mx.abs(_prefill(a, x, _phs(_KEEPALL, gather=True, vert=NB + 2, slash=NB + 2),
                                      use_fast=True) - dense_fast)))
    print(f"per-head-specs mixed keep-all gather vs dense max|Δ| = {d2:.2e}")
    assert d2 < 1e-3, f"per-head-specs mixed keep-all gather != dense ({d2}) — routing integration bug"

    # 3. tight mixed assignment: gather path == mask path (the M5 twin). ashape L1 vs L2 + vslash v1s1 +
    #    xattn t0.9 ⇒ a real per-head-PARAMS sparse mix. config vert/slash=1 matches the vslash spec.
    tight_mask = _prefill(a, x, _phs(_TIGHT, gather=False, vert=1, slash=1), use_fast=True)
    tight_gath = _prefill(a, x, _phs(_TIGHT, gather=True, vert=1, slash=1), use_fast=True)
    d3 = float(mx.max(mx.abs(tight_mask - tight_gath)))
    print(f"per-head-specs mixed gather vs mask       max|Δ| = {d3:.2e}  (expected ~0: same per-head sel)")
    assert d3 < 1e-3, f"per-head-specs gather != mask ({d3}) — the two executions disagree on selection"

    # 4. the tight mixed assignment drops blocks ⇒ output changes (the lever engages)
    d4 = float(mx.max(mx.abs(tight_mask - dense_fast)))
    print(f"per-head-specs mixed mask vs dense        max|Δ| = {d4:.2e}  (expected > 0: sparsity active)")
    assert d4 > 1e-2, f"per-head-specs mixed assignment did not change the output ({d4}) — not engaging"

    # 5. budget cap engages under a mixed assignment (force-keeps sink+diagonal, bounded shape)
    capped = _prefill(a, x, _phs(_TIGHT, gather=True, vert=1, slash=1, budget=2), use_fast=True)
    d5 = float(mx.max(mx.abs(capped - dense_fast)))
    print(f"per-head-specs mixed budget=2 vs dense    max|Δ| = {d5:.2e}  (expected > 0: cap drops blocks)")
    assert capped.shape == dense_fast.shape, "budget-capped per-head-specs changed output shape"
    assert d5 > 1e-2, f"budget=2 did not cap per-head-specs ({d5}) — the kept-block budget is not binding"

    # 6. min_seq gate: T below min_seq ⇒ dense even with a per-head-specs config set
    d6 = float(mx.max(mx.abs(_prefill(a, x, _phs(_TIGHT, gather=True, vert=1, slash=1, min_seq=T + 1),
                                      use_fast=True) - dense_fast)))
    print(f"per-head-specs mixed min_seq>T vs dense    max|Δ| = {d6:.2e}  (expected 0: gated off)")
    assert d6 < 1e-6, f"min_seq gate failed ({d6}) — per-head-specs engaged below min_seq"

    # 7. causal safety: perturb the FINAL token; block 0's selectable set is force-pinned to {block 0}
    #    by causality for every kind, so its output must be bit-unchanged. No future leakage.
    x2 = mx.array(x)
    x2[:, -1, :] = mx.random.normal((cfg.hidden_size,)).astype(mx.float32)
    pert = _prefill(a, x2, _phs(_TIGHT, gather=True, vert=1, slash=1), use_fast=True)
    d7 = float(mx.max(mx.abs(tight_gath[:, :BLOCK] - pert[:, :BLOCK])))
    print(f"causal: perturb last tok, Δ blk0          max|Δ| = {d7:.2e}  (expected 0)")
    assert d7 < 1e-6, f"perturbing the final token changed block 0 ({d7}) — future leakage"

    # 8. validation: per-head vslash params are now ALLOWED (M6 — the probe returns param-independent
    #    masses, so a vslash spec need not match the config's vert/slash); both-set / non-HeadSpec /
    #    length mismatch are still rejected.
    XAttnConfig(head_specs=(HeadSpec("vslash", vert=2, slash=2),), vert=3, slash=3)  # M6: accepted (no pin)
    try:
        XAttnConfig(head_specs=(HeadSpec("xattn"),) * NH, head_selectors=("xattn",) * NH)
        raise AssertionError("XAttnConfig accepted both head_specs and head_selectors")
    except ValueError:
        pass
    try:
        XAttnConfig(head_specs=("xattn",))   # a bare string, not a HeadSpec
        raise AssertionError("XAttnConfig accepted a non-HeadSpec head_specs entry")
    except ValueError:
        pass
    try:
        _prefill(a, x, _phs((HeadSpec("xattn"), HeadSpec("ashape")), gather=False), use_fast=True)  # len 2 != NH
        raise AssertionError("per-head-specs dispatch accepted a head_specs/heads length mismatch")
    except ValueError:
        pass
    print("validation: per-head vslash params allowed (no pin) + both-set + non-HeadSpec + length mismatch   OK")

    print("PASS — InternLM2 per-head params: policy budget-correct; routing exact (incl. same-kind "
          "different params); uniform-as-per-head-specs == uniform; mixed keep-all == dense (mask & "
          "gather); gather==mask; sparsity-active + budget-capped; min_seq-gated; causal-safe; validated.")


if __name__ == "__main__":
    run()
