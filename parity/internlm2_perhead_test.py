"""InternLM2.5 per-head pattern-assignment gate — M4 of the MInference sparse-prefill track.

M1/M2/M3 added three *uniform* block selectors (XAttention antidiagonal nucleus, A-shape, vertical-
slash) onto one shared block-gather / additive-mask execution. M4 makes the selector **per head**: an
offline assignment routes each query head to the cheapest selector kind that still recalls its
attention, and :func:`~quanta.modeling.xattention.select_keep` dispatches per head over the validated
selectors. The per-head path is a pure *routing layer* — it adds no new selection math: head ``h``'s
kept-block mask is byte-identical to the uniform mask for ``head_selectors[h]``.

Two independently-checkable pieces, both gated here model-free (no weights needed):

  * the **policy** :func:`assign_head_selectors` — given each candidate's per-head error vs dense
    (rows ordered cheap→accurate), pick the cheapest within ``tol`` per head, else the accurate
    fallback. Pure / positional, so a hand-built error matrix pins every branch.
  * the **mechanism** on the real ``InternLM2Attention`` (covers GQA ``mx.repeat``, ``attn_scale``,
    post-RoPE q/k):
      1. routing exactness — ``select_keep`` with a mixed ``head_selectors`` reproduces, for each
         head, the uniform single-selector keep for that head's kind (incl. the vslash global index),
      2. a uniform-as-per-head assignment (every head one kind) == the plain uniform selector,
      3. a MIXED keep-all assignment (all kinds, params at keep-all) == dense — mask AND gather (the
         per-head parity anchor: routing is exact regardless of which kinds are mixed),
      4. a tight mixed assignment: **gather path == mask path** (the M4 twin — the per-head selection
         drives both executions identically),
      5. the tight mixed assignment drops blocks ⇒ output changes (the lever engages),
      6. the budget cap engages under a mixed assignment (force-keeps sink+diagonal, bounded shape),
      7. ``min_seq > T`` gates sparsity off entirely,
      8. perturbing the final token cannot change an earlier query block's output (no future leakage),
      9. validation — an unknown kind is rejected at config build; a head_selectors/heads length
         mismatch is rejected at use.

The per-head assignment's *quality* (which heads earn which selector, and the ppl it trades) is
measured on the real bake in ``parity/internlm2_ppl_sparse.py`` (CLAUDE.md: sparse attention is
ppl-gated, never numeric-parity-gated). This gate proves the routing is correct.

    uv run python -m parity.internlm2_perhead_test
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
from mlx.utils import tree_map

from quanta.internlm2.attention import InternLM2Attention
from quanta.internlm2.config import InternLM2Config
from quanta.modeling.xattention import XAttnConfig, assign_head_selectors, select_keep

T = 500       # not a multiple of block=128 → exercises padding; spans 4 blocks (ceil(500/128))
BLOCK = 128
NB = (T + BLOCK - 1) // BLOCK   # 4
NH = 4                          # query heads in the tiny cfg (head_selectors length must match)


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


def _ph(kinds: tuple[str, ...], *, gather: bool, threshold: float = 0.9, local: int = 1,
        vert: int = 1, slash: int = 1, min_seq: int = 0, budget: int | None = 64) -> XAttnConfig:
    return XAttnConfig(block=BLOCK, stride=16, head_selectors=kinds, threshold=threshold,
                       local=local, vert=vert, slash=slash, min_seq=min_seq, gather=gather,
                       budget=budget)


def _check_policy() -> None:
    """:func:`assign_head_selectors` routes each head to the cheapest candidate within ``tol``,
    else the accurate fallback — a hand-built error matrix pins every branch."""
    kinds = ["ashape", "vslash", "xattn"]            # cheap → accurate (kernel cost ascending)
    # cols: h0 ashape-ok, h1 vslash-ok, h2 xattn-ok, h3 none-ok (→fallback), h4 ashape at the tol edge,
    #       h5 all-ok (→cheapest=ashape).
    errors = mx.array([
        [0.01, 0.50, 0.50, 0.90, 0.02, 0.00],        # ashape
        [0.40, 0.01, 0.40, 0.80, 0.40, 0.00],        # vslash
        [0.30, 0.30, 0.00, 0.70, 0.30, 0.00],        # xattn
    ])
    got = assign_head_selectors(errors, kinds, tol=0.02)
    want = ("ashape", "vslash", "xattn", "xattn", "ashape", "ashape")
    assert got == want, f"assign_head_selectors {got} != {want}"
    # tol huge ⇒ every head takes the cheapest; tol below all errors ⇒ every head the fallback.
    assert assign_head_selectors(errors, kinds, tol=9.0) == ("ashape",) * 6, "tol=∞ not all-cheapest"
    assert assign_head_selectors(errors, kinds, tol=-1.0) == ("xattn",) * 6, "tol<0 not all-fallback"
    print("assign_head_selectors cheapest-within-tol / fallback / boundary   OK")


def _check_routing(cfg: InternLM2Config) -> None:
    """A mixed-``head_selectors`` ``select_keep`` reproduces, per head, the uniform keep for that
    head's kind — proving per-head dispatch is pure routing (no new selection math)."""
    mx.random.seed(2)
    q = mx.random.normal((1, NH, T, cfg.head_dim)).astype(mx.float32)
    k = mx.random.normal((1, NH, T, cfg.head_dim)).astype(mx.float32)
    scale = cfg.attn_scale
    kinds = ("xattn", "ashape", "vslash", "ashape")
    mixed = _ph(kinds, gather=False, threshold=0.9, local=2, vert=2, slash=2)
    keep_mixed, rank_mixed = select_keep(q, k, scale, mixed)
    assert keep_mixed.shape == (1, NH, NB, NB), f"mixed keep shape {keep_mixed.shape}"
    assert rank_mixed.shape == (1, NH, NB, NB), f"mixed rank shape {rank_mixed.shape}"
    for kind in set(kinds):
        sub = replace(mixed, head_selectors=None, selector=kind)
        keep_k, _ = select_keep(q, k, scale, sub)           # vslash index recomputed from same q,k
        for h, a in enumerate(kinds):
            if a != kind:
                continue
            bad = int(mx.sum(keep_mixed[:, h] != keep_k[:, h]).item())
            assert bad == 0, f"head {h} ({kind}) routed keep differs from uniform by {bad} cells"
    print("per-head routing exactness (mixed keep[:,h] == uniform keep for head's kind)   OK")


def run() -> None:
    mx.random.seed(0)
    cfg = _cfg()
    a = _attn(cfg)
    x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.float32)

    _check_policy()
    _check_routing(cfg)

    dense_fast = _prefill(a, x, None, use_fast=True)

    # 1. uniform-as-per-head: every head the same kind == the plain uniform selector (exact).
    uni = _prefill(a, x, XAttnConfig(block=BLOCK, stride=16, selector="xattn", threshold=0.9,
                                     min_seq=0, gather=False), use_fast=True)
    php = _prefill(a, x, _ph(("xattn",) * NH, gather=False, threshold=0.9), use_fast=True)
    d_u = float(mx.max(mx.abs(uni - php)))
    print(f"per-head all-xattn vs uniform xattn max|Δ| = {d_u:.2e}  (expected 0: same selection)")
    assert d_u < 1e-6, f"uniform-as-per-head != uniform xattn ({d_u}) — routing changed selection"

    # 2. MIXED keep-all == dense (mask & gather) — the per-head parity anchor. All three kinds present,
    #    every kind's params at keep-all ⇒ each head keeps every causal block regardless of its kind.
    mixed = ("xattn", "ashape", "vslash", "ashape")
    ka = dict(threshold=1.0, local=NB + 2, vert=NB + 2, slash=NB + 2)
    d1 = float(mx.max(mx.abs(_prefill(a, x, _ph(mixed, gather=False, **ka), use_fast=True) - dense_fast)))
    print(f"per-head mixed keep-all mask vs dense max|Δ| = {d1:.2e}")
    assert d1 < 1e-4, f"per-head mixed keep-all mask != dense ({d1}) — routing integration bug"
    d2 = float(mx.max(mx.abs(_prefill(a, x, _ph(mixed, gather=True, **ka), use_fast=True) - dense_fast)))
    print(f"per-head mixed keep-all gather vs dense max|Δ| = {d2:.2e}")
    assert d2 < 1e-3, f"per-head mixed keep-all gather != dense ({d2}) — routing integration bug"

    # 3. tight mixed assignment: gather path == mask path (the M4 twin — per-head selection drives
    #    both executions identically). threshold=0.9 / local=vert=slash=1 ⇒ a real sparse mix.
    tight_mask = _prefill(a, x, _ph(mixed, gather=False), use_fast=True)
    tight_gath = _prefill(a, x, _ph(mixed, gather=True), use_fast=True)
    d3 = float(mx.max(mx.abs(tight_mask - tight_gath)))
    print(f"per-head mixed gather vs mask       max|Δ| = {d3:.2e}  (expected ~0: same per-head sel)")
    assert d3 < 1e-3, f"per-head gather != mask ({d3}) — the two executions disagree on selection"

    # 4. the tight mixed assignment drops blocks ⇒ output changes (the lever engages)
    d4 = float(mx.max(mx.abs(tight_mask - dense_fast)))
    print(f"per-head mixed mask vs dense        max|Δ| = {d4:.2e}  (expected > 0: sparsity active)")
    assert d4 > 1e-2, f"per-head mixed assignment did not change the output ({d4}) — not engaging"

    # 5. budget cap engages under a mixed assignment (force-keeps sink+diagonal, bounded shape)
    capped = _prefill(a, x, _ph(mixed, gather=True, budget=2), use_fast=True)
    d5 = float(mx.max(mx.abs(capped - dense_fast)))
    print(f"per-head mixed budget=2 vs dense    max|Δ| = {d5:.2e}  (expected > 0: cap drops blocks)")
    assert capped.shape == dense_fast.shape, "budget-capped per-head changed output shape"
    assert d5 > 1e-2, f"budget=2 did not cap per-head ({d5}) — the kept-block budget is not binding"

    # 6. min_seq gate: T below min_seq ⇒ dense even with a per-head config set
    d6 = float(mx.max(mx.abs(_prefill(a, x, _ph(mixed, gather=True, min_seq=T + 1), use_fast=True)
                             - dense_fast)))
    print(f"per-head mixed min_seq>T vs dense    max|Δ| = {d6:.2e}  (expected 0: gated off)")
    assert d6 < 1e-6, f"min_seq gate failed ({d6}) — per-head engaged below min_seq"

    # 7. causal safety: perturb the FINAL token; block 0's selectable set is force-pinned to {block 0}
    #    by causality for every kind, so its output must be bit-unchanged. No future leakage.
    x2 = mx.array(x)
    x2[:, -1, :] = mx.random.normal((cfg.hidden_size,)).astype(mx.float32)
    pert = _prefill(a, x2, _ph(mixed, gather=True), use_fast=True)
    d7 = float(mx.max(mx.abs(tight_gath[:, :BLOCK] - pert[:, :BLOCK])))
    print(f"causal: perturb last tok, Δ blk0    max|Δ| = {d7:.2e}  (expected 0)")
    assert d7 < 1e-6, f"perturbing the final token changed block 0 ({d7}) — future leakage"

    # 8. validation: unknown kind rejected at config build; length mismatch rejected at use.
    try:
        XAttnConfig(head_selectors=("xattn", "nope"))
        raise AssertionError("XAttnConfig accepted an unknown head_selectors kind")
    except ValueError:
        pass
    try:
        _prefill(a, x, _ph(("xattn", "ashape"), gather=False), use_fast=True)   # len 2 != NH heads
        raise AssertionError("per-head dispatch accepted a head_selectors/heads length mismatch")
    except ValueError:
        pass
    print("validation: unknown kind + length mismatch rejected   OK")

    print("PASS — InternLM2 per-head pattern assignment: policy correct; routing exact; mixed "
          "keep-all == dense (mask & gather); gather==mask; sparsity-active + budget-capped; "
          "min_seq-gated; causal-safe; validated.")


if __name__ == "__main__":
    run()
