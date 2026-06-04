"""InternLM2.5 per-head vertical-slash *params* gate — M6 of the MInference sparse-prefill track.

M5 let each head carry its own (kind, params) via ``head_specs`` — but ``vslash`` was special-cased:
all vslash heads had to share the config's ``vert``/``slash`` (a fail-loud pin), because the online probe
baked the top-``vert``/``slash`` cut into the threaded global index. M6 removes that pin. The probe
(:func:`vertical_slash_index`) now returns **param-independent** masses ``(key_mass, slash_mass)`` and the
top-k cut is applied per spec in :func:`select_keep`, so two heads can read the ONE global probe yet keep
**different** vert/slash. This is the M5 deferral, bought back — and the load-bearing step toward the
long-context (key-chunked) probe of M7, since the masses are exactly what a chunked probe accumulates.

Everything stays a pure *routing layer*: head ``h``'s kept-block mask is byte-identical to the uniform
vslash keep for ``head_specs[h]``'s own vert/slash. Gated here model-free (no weights) on the real
``InternLM2Attention`` (covers GQA ``mx.repeat``, ``attn_scale``, post-RoPE q/k):

  1. **routing exactness** — a mixed ``head_specs`` with two vslash heads at DIFFERENT vert/slash
     (+ an ashape + an xattn head) reproduces, for each head, the uniform single-spec keep for that
     head's spec — including the two vslash heads, each cut to its own params from the shared masses,
  2. the two different-param vslash heads actually keep **different** block sets (the params bite — they
     did not silently collapse to one pattern),
  3. the config's ``vert``/``slash`` are **irrelevant** to per-head vslash specs (a config whose
     vert/slash match NEITHER spec routes identically) — the masses are param-independent,
  4. a MIXED keep-all assignment (two vslash params + ashape + xattn, all at keep-all) == dense — mask
     AND gather (the per-head-vslash-params parity anchor),
  5. a tight mixed assignment: **gather path == mask path** (the M6 twin — both re-cut the one global
     probe identically per head),
  6. the tight mixed assignment drops blocks ⇒ output changes (the lever engages),
  7. ``min_seq > T`` gates sparsity off entirely,
  8. perturbing the final token cannot change an earlier query block's output (block 0's selectable set
     is force-pinned to {block 0} by causality for every kind — no future leakage).

The assignment's *quality* (which heads earn which vslash params, and the ppl it trades) is measured on
the real bake in ``parity/internlm2_ppl_sparse.py`` (CLAUDE.md: sparse attention is ppl-gated, never
numeric-parity-gated). This gate proves the per-head vslash-params routing is correct.

    uv run python -m parity.internlm2_vslash_perhead_test
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
from mlx.utils import tree_map

from quanta.internlm2.attention import InternLM2Attention
from quanta.internlm2.config import InternLM2Config
from quanta.modeling.xattention import HeadSpec, XAttnConfig, select_keep

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


def _phs(specs: tuple[HeadSpec, ...], *, gather: bool, vert: int = 7, slash: int = 7,
         min_seq: int = 0, budget: int | None = 64) -> XAttnConfig:
    # config vert/slash are DELIBERATELY 7 — matching NEITHER vslash spec below — to prove they no
    # longer constrain a per-head vslash spec (the probe masses are param-independent; M6).
    return XAttnConfig(block=BLOCK, stride=16, head_specs=specs, vert=vert, slash=slash,
                       min_seq=min_seq, gather=gather, budget=budget)


# Two vslash heads at DIFFERENT params (the M6 point), plus an ashape + an xattn head.
_KEEPALL = (HeadSpec("vslash", vert=NB + 2, slash=NB + 2), HeadSpec("vslash", vert=NB + 1, slash=NB + 1),
            HeadSpec("ashape", local=NB + 2), HeadSpec("xattn", threshold=1.0))
_TIGHT = (HeadSpec("vslash", vert=1, slash=1), HeadSpec("vslash", vert=2, slash=2),
          HeadSpec("ashape", local=1), HeadSpec("xattn", threshold=0.9))   # vslash v1s1 vs v2s2


def _check_routing(cfg: InternLM2Config) -> None:
    """A mixed ``head_specs`` with two different-param vslash heads reproduces, per head, the uniform keep
    for that head's spec — and the two vslash heads keep DIFFERENT block sets (the params bite)."""
    mx.random.seed(3)
    q = mx.random.normal((1, NH, T, cfg.head_dim)).astype(mx.float32)
    k = mx.random.normal((1, NH, T, cfg.head_dim)).astype(mx.float32)
    scale = cfg.attn_scale
    mixed = _phs(_TIGHT, gather=False)
    keep_mixed, _ = select_keep(q, k, scale, mixed)
    assert keep_mixed.shape == (1, NH, NB, NB), f"mixed keep shape {keep_mixed.shape}"
    for h, sp in enumerate(_TIGHT):
        sub = replace(mixed, head_specs=None, selector=sp.kind, threshold=sp.threshold,
                      local=sp.local, vert=sp.vert, slash=sp.slash)
        keep_u, _ = select_keep(q, k, scale, sub)
        bad = int(mx.sum(keep_mixed[:, h] != keep_u[:, h]).item())
        assert bad == 0, f"head {h} ({sp}) routed keep differs from its uniform spec by {bad} cells"
    # the two vslash heads (v1s1 vs v2s2) must NOT keep the same set — else the per-head params were
    # silently ignored (collapsed onto one pattern).
    vdiff = int(mx.sum(keep_mixed[:, 0] != keep_mixed[:, 1]).item())
    assert vdiff > 0, "the two vslash heads at v1s1 / v2s2 kept identical blocks — per-head params ignored"
    print(f"per-head vslash-params routing exactness (mixed keep[:,h] == uniform; v1s1≠v2s2 by {vdiff})   OK")


def _check_param_independence(cfg: InternLM2Config) -> None:
    """The config's vert/slash must not affect a per-head vslash spec: the same head_specs routed under
    two different config vert/slash give byte-identical keeps (the masses are param-independent)."""
    mx.random.seed(4)
    q = mx.random.normal((1, NH, T, cfg.head_dim)).astype(mx.float32)
    k = mx.random.normal((1, NH, T, cfg.head_dim)).astype(mx.float32)
    scale = cfg.attn_scale
    keep_a, _ = select_keep(q, k, scale, _phs(_TIGHT, gather=False, vert=3, slash=3))
    keep_b, _ = select_keep(q, k, scale, _phs(_TIGHT, gather=False, vert=NB, slash=NB))
    bad = int(mx.sum(keep_a != keep_b).item())
    assert bad == 0, f"per-head vslash routing changed with the config's vert/slash ({bad} cells) — not param-independent"
    print("per-head vslash-params config-irrelevance (keep unchanged across config vert/slash)   OK")


def run() -> None:
    mx.random.seed(0)
    cfg = _cfg()
    a = _attn(cfg)
    x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.float32)

    _check_routing(cfg)
    _check_param_independence(cfg)

    dense_fast = _prefill(a, x, None, use_fast=True)

    # 1. uniform-vslash-as-per-head: every head the same vslash spec == the plain uniform vslash selector.
    uni = _prefill(a, x, XAttnConfig(block=BLOCK, stride=16, selector="vslash", vert=2, slash=2,
                                     min_seq=0, gather=False), use_fast=True)
    php = _prefill(a, x, _phs((HeadSpec("vslash", vert=2, slash=2),) * NH, gather=False), use_fast=True)
    d_u = float(mx.max(mx.abs(uni - php)))
    print(f"per-head all-vslash-v2s2 vs uniform v2s2  max|Δ| = {d_u:.2e}  (expected 0: same selection)")
    assert d_u < 1e-6, f"uniform-as-per-head-vslash != uniform ({d_u}) — routing changed selection"

    # 2. MIXED keep-all == dense (mask & gather) — the per-head-vslash-params parity anchor. Two vslash
    #    heads at DIFFERENT (keep-all) params + ashape + xattn, every spec at keep-all ⇒ all blocks kept.
    d1 = float(mx.max(mx.abs(_prefill(a, x, _phs(_KEEPALL, gather=False), use_fast=True) - dense_fast)))
    print(f"per-head vslash-params keep-all mask vs dense max|Δ| = {d1:.2e}")
    assert d1 < 1e-4, f"per-head vslash-params keep-all mask != dense ({d1}) — routing integration bug"
    d2 = float(mx.max(mx.abs(_prefill(a, x, _phs(_KEEPALL, gather=True), use_fast=True) - dense_fast)))
    print(f"per-head vslash-params keep-all gather vs dense max|Δ| = {d2:.2e}")
    assert d2 < 1e-3, f"per-head vslash-params keep-all gather != dense ({d2}) — routing integration bug"

    # 3. tight mixed (vslash v1s1 + v2s2 + ashape L1 + xattn t0.9): gather path == mask path (the M6 twin).
    tight_mask = _prefill(a, x, _phs(_TIGHT, gather=False), use_fast=True)
    tight_gath = _prefill(a, x, _phs(_TIGHT, gather=True), use_fast=True)
    d3 = float(mx.max(mx.abs(tight_mask - tight_gath)))
    print(f"per-head vslash-params gather vs mask     max|Δ| = {d3:.2e}  (expected ~0: same per-head cut)")
    assert d3 < 1e-3, f"per-head vslash-params gather != mask ({d3}) — the two executions disagree on selection"

    # 4. the tight mixed assignment drops blocks ⇒ output changes (the lever engages)
    d4 = float(mx.max(mx.abs(tight_mask - dense_fast)))
    print(f"per-head vslash-params mask vs dense      max|Δ| = {d4:.2e}  (expected > 0: sparsity active)")
    assert d4 > 1e-2, f"per-head vslash-params tight assignment did not change the output ({d4}) — not engaging"

    # 5. min_seq gate: T below min_seq ⇒ dense even with a per-head-vslash-params config set
    d5 = float(mx.max(mx.abs(_prefill(a, x, _phs(_TIGHT, gather=True, min_seq=T + 1), use_fast=True)
                             - dense_fast)))
    print(f"per-head vslash-params min_seq>T vs dense  max|Δ| = {d5:.2e}  (expected 0: gated off)")
    assert d5 < 1e-6, f"min_seq gate failed ({d5}) — per-head vslash-params engaged below min_seq"

    # 6. causal safety: perturb the FINAL token; block 0's selectable set is force-pinned to {block 0}
    #    by causality for every kind (incl. both vslash heads), so its output must be bit-unchanged.
    x2 = mx.array(x)
    x2[:, -1, :] = mx.random.normal((cfg.hidden_size,)).astype(mx.float32)
    pert = _prefill(a, x2, _phs(_TIGHT, gather=True), use_fast=True)
    d6 = float(mx.max(mx.abs(tight_gath[:, :BLOCK] - pert[:, :BLOCK])))
    print(f"causal: perturb last tok, Δ blk0          max|Δ| = {d6:.2e}  (expected 0)")
    assert d6 < 1e-6, f"perturbing the final token changed block 0 ({d6}) — future leakage"

    print("PASS — InternLM2 per-head vslash params: routing exact (two vslash heads at different "
          "vert/slash, each from the one shared probe); config-irrelevant; keep-all == dense (mask & "
          "gather); gather==mask; sparsity-active; min_seq-gated; causal-safe.")


if __name__ == "__main__":
    run()
