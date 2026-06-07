"""InternLM2.5 gather-path **wall-clock prefill bench** — M8 of the MInference sparse-prefill track.

The whole MInference lever is *prefill speed*: InternLM2.5-7B is the only serving keeper still paying
full **O(T²)** dense prefill. M1–M6 measured the lossy selectors' **quality** (Δppl) and proved the
``gather=True`` speed path selects the same blocks as the ``gather=False`` mask path — but, as the M1
harness notes, *"with fast SDPA + an additive block mask MLX still computes the full QKᵀ, so the mask
path measures quality only; the gather path is the actual FLOP/memory win"*. That win was an
**expectation, never timed**. M8 times it: real wall-clock of the block-gather prefill vs dense flash
SDPA, swept over context length, where vertical-slash's O(1)-blocks-per-query pattern should turn the
O(T²) attention into O(T).

Three things, on ONE resident decoder layer of the int8-g64 7B bake (dequantized to bf16; rule-8):

  1. **parity anchor (correctness before timing)** — the timed gather path must be *output-equivalent*
     to dense at keep-all (``ashape local = n_blocks`` / ``vslash vert = slash = n_blocks`` keep every
     causal block), so the speed numbers compare correct executions, not a broken kernel.
  2. **M7 chunked-probe on real weights** — at long context the vertical-slash probe key-chunks (M7);
     confirm on the real layer's post-RoPE GQA q/k that the chunked masses == the single-shot masses
     (the M7 model-free property, now on real weights) and that chunking costs ~nothing in wall-clock.
  3. **the headline — dense vs gather wall-clock across T** — time the full attention ``__call__``
     (projections + RoPE + attention; the projections are identical both sides, so the per-layer speedup
     is the honest end-to-end attention cost and grows as the O(T²) term comes to dominate). Selectors:
     ``ashape`` (sink + local window, the cheapest static), ``vslash`` (the long-context global pattern),
     ``xattn`` (antidiagonal nucleus, adaptive). Report ms, speedup vs dense, and the kept-block fraction.

This is a **speed characterization** (the speedup is the measured result, recorded not pass/fail — like
the Nemotron U4 measure-first benches); the only hard gates are the parity anchor + the chunked-probe
equivalence. Quality (long-context Δppl) is the separate M9 milestone (full-model teacher-forcing).

Heavy at large T (one attention layer at up to 64K context). Run ALONE on a free GPU.

    uv run --with sentencepiece python -m parity.internlm2_prefill_bench
    uv run --with sentencepiece python -m parity.internlm2_prefill_bench 32768   # cap the T sweep
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from quanta.internlm2.artifact import InternLM2Artifact
from quanta.internlm2.model import _DecoderLayer, _load_decoder_layer
from quanta.internlm2.tokenizer import InternLM2Tokenizer
from quanta.modeling.xattention import HeadSpec, XAttnConfig, vertical_slash_index

ARTIFACT = "/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64"
REPO = Path("/Users/pmrj/Environment/agentic_ai/finally_quanta")
BLOCK = 128
SWEEP = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
ITERS = 3                       # timed iters per (T, config) after one warmup
WIRED_GIB = 64                  # pin the ~9 GiB dequant layer-set; attention working grows in the rest
MEM_CEIL_GIB = 440.0            # skip any T whose projected peak would approach the working-set ceiling


def _corpus(tok: InternLM2Tokenizer, n: int) -> mx.array:
    """A long real-text token array (repo source + prose) in InternLM2's own SentencePiece, length n."""
    files = (sorted(REPO.glob("src/quanta/**/*.py")) + sorted(REPO.glob("parity/*.py"))
             + sorted(REPO.glob("*.md")) + sorted(REPO.glob("tests/**/*.py")))
    text = "\n\n".join(p.read_text() for p in files if p.exists())
    ids = tok.encode(text, add_bos=True)
    if len(ids) < n:                                   # tile if the corpus is shorter than the max T
        ids = (ids * ((n // len(ids)) + 1))[:n]
    return mx.array(ids[:n])


def _gib(nbytes: int) -> float:
    return nbytes / 1024 ** 3


def _bench_call(attn, xn: mx.array, iters: int = ITERS) -> float:
    """Mean wall-clock (ms) of attn(xn) prefill — one warmup, then ``iters`` timed, eval-synced."""
    o = attn(xn, use_fast=True)
    mx.eval(o)
    t0 = time.perf_counter()
    for _ in range(iters):
        o = attn(xn, use_fast=True)
        mx.eval(o)
    return (time.perf_counter() - t0) / iters * 1e3


def _kept_frac(layer: _DecoderLayer, xn: mx.array, cfg: XAttnConfig) -> float:
    """Mean kept key-blocks / mean causal key-blocks for ``cfg`` on this input — the sparsity the gather
    path realizes (the speedup's structural cause). Calibration-only (no forward)."""
    counts = layer.attention._attn_keep_counts(xn, cfg)          # [nh] mean kept blocks/query
    nb = (xn.shape[1] + BLOCK - 1) // BLOCK
    causal = (nb + 1) / 2.0                                       # avg causal blocks/query
    return float(mx.mean(counts).item()) / causal


def _gathers() -> list[tuple[str, XAttnConfig]]:
    """The gather=True selectors to time (uniform; per-head assignment is M9's deployment number)."""
    return [
        ("ashape L8", XAttnConfig(block=BLOCK, selector="ashape", local=8, min_seq=0, gather=True)),
        ("vslash v8s8", XAttnConfig(block=BLOCK, selector="vslash", vert=8, slash=8, min_seq=0,
                                    gather=True)),
        ("xattn t0.9", XAttnConfig(block=BLOCK, stride=16, threshold=0.9, min_seq=0, gather=True)),
    ]


def _perhead_mix(nh: int, n_dense: int = 4) -> tuple[HeadSpec, ...]:
    """A deployment-plausible per-head assignment: most heads on the cheap static ashape window, a few
    (``n_dense``) on the dense xattn nucleus (the heads that need its quality). This is the mix that
    bottlenecks the NAIVE gather — its one global ``max_kept`` = the dense xattn head's, so every cheap
    head pays the dense budget — and that the grouped fold rescues (each spec-group its own ``max_kept``)."""
    cheap, dense = HeadSpec("ashape", local=8), HeadSpec("xattn", threshold=0.9)
    return tuple(dense if i >= nh - n_dense else cheap for i in range(nh))


def _parity_anchor(layer: _DecoderLayer, xn: mx.array) -> bool:
    """keep-all gather == dense (output-equivalent): the timed gather path is correct before we time it."""
    t = xn.shape[1]
    nblk = (t + BLOCK - 1) // BLOCK
    layer.attention.sparse = None
    dense = layer.attention(xn, use_fast=True)
    oks = []
    for name, cfg in (("ashape", XAttnConfig(block=BLOCK, selector="ashape", local=nblk, min_seq=0,
                                             gather=True)),
                      ("vslash", XAttnConfig(block=BLOCK, selector="vslash", vert=nblk, slash=nblk,
                                             min_seq=0, gather=True))):
        layer.attention.sparse = cfg
        out = layer.attention(xn, use_fast=True)
        rel = float((mx.max(mx.abs(out - dense)) / (mx.max(mx.abs(dense)) + 1e-9)).item())
        print(f"  parity anchor  {name} keep-all gather vs dense  rel={rel:.2e}  (expect <1e-2: bf16 floor)")
        oks.append(rel < 1e-2)   # bf16 real-weight floor (gather's per-chunk SDPA vs dense flash SDPA);
        #                          the fp32-tight 1e-3 keep-all==dense is the model-free M3/M6 gate
    layer.attention.sparse = None
    return all(oks)


def _chunked_probe_check(layer: _DecoderLayer, xn: mx.array) -> bool:
    """M7 on real weights: the key-chunked vslash probe == the single-shot probe on the real layer's
    post-RoPE GQA q/k, and chunking costs ~nothing in wall-clock."""
    t = xn.shape[1]
    q, kr, _, _ = layer.attention._attn_qkv(xn, cache=None, use_fast=True)
    scale = layer.attention.scale
    h, lp = q.shape[1], min(BLOCK, t - ((t - 1) // BLOCK) * BLOCK)
    probe_gb = q.shape[0] * h * lp * t * 4 / 1e9
    big = XAttnConfig(block=BLOCK, selector="vslash", min_seq=0, max_alloc_gb=8.0)        # single-shot
    small = XAttnConfig(block=BLOCK, selector="vslash", min_seq=0, max_alloc_gb=probe_gb / 8.0)  # chunked
    km_s, sm_s = vertical_slash_index(q, kr, scale, big)
    km_c, sm_c = vertical_slash_index(q, kr, scale, small)
    mx.eval(km_s, sm_s, km_c, sm_c)
    rk = float((mx.max(mx.abs(km_c - km_s)) / (mx.max(mx.abs(km_s)) + 1e-12)).item())
    rs = float((mx.max(mx.abs(sm_c - sm_s)) / (mx.max(mx.abs(sm_s)) + 1e-12)).item())
    t_single = _time_probe(q, kr, scale, big)
    t_chunk = _time_probe(q, kr, scale, small)
    per_blk = 1.5 * q.shape[0] * h * lp * BLOCK * 4              # same chunking math as the helper
    sc_chunk = max(1, int(small.max_alloc_gb * 1e9 // per_blk)) * BLOCK
    nchunks = (t + sc_chunk - 1) // sc_chunk
    print(f"  chunked probe @T={t}: probe {probe_gb:.2f} GiB → {nchunks} chunk(s) @ "
          f"max_alloc={small.max_alloc_gb:.3f} GiB; masses key rel={rk:.2e} slash rel={rs:.2e}; "
          f"t_single={t_single:.1f}ms t_chunk={t_chunk:.1f}ms")
    return rk < 1e-3 and rs < 1e-3 and nchunks > 1


def _time_probe(q: mx.array, kr: mx.array, scale: float, cfg: XAttnConfig, iters: int = 3) -> float:
    o = vertical_slash_index(q, kr, scale, cfg)
    mx.eval(o)
    t0 = time.perf_counter()
    for _ in range(iters):
        o = vertical_slash_index(q, kr, scale, cfg)
        mx.eval(o)
    return (time.perf_counter() - t0) / iters * 1e3


def run(t_max: int = SWEEP[-1]) -> None:
    mx.set_wired_limit(int(WIRED_GIB * 1024 ** 3))
    art = InternLM2Artifact(ARTIFACT)
    cfg = art.cfg
    tok = InternLM2Tokenizer.from_pretrained(ARTIFACT)
    sweep = [t for t in SWEEP if t <= t_max]

    print("=== InternLM2.5 gather-path wall-clock prefill bench (M8) — SOLO ===")
    print(f"  artifact={ARTIFACT}")
    print(f"  heads nh={cfg.num_attention_heads} nkv={cfg.num_key_value_heads} hd={cfg.head_dim} "
          f"hidden={cfg.hidden_size}  block={BLOCK}  iters={ITERS}  sweep≤{t_max}")

    ids = _corpus(tok, max(sweep))
    layer = _DecoderLayer(cfg)
    _load_decoder_layer(layer, art, 0, mx.bfloat16)              # ONE decoder layer resident (rule-8)
    emb = art.embed()
    h_full = emb[ids][None]                                      # [1, T_max, H] bf16
    del emb
    art.release()
    mx.clear_cache()
    xn_full = layer.attention_norm(h_full)                      # the layer-0 attention input
    mx.eval(xn_full)

    # 1. parity anchor (a moderate T so dense fits cheaply): keep-all gather == dense.
    print("\n  correctness before timing:")
    anchor_ok = _parity_anchor(layer, xn_full[:, :4096])
    # 2. M7 chunked probe on real weights, at the largest swept T.
    probe_ok = _chunked_probe_check(layer, xn_full[:, : max(sweep)])

    # 3. the headline sweep: dense vs each gather selector, wall-clock across T.
    gathers = _gathers()
    names = [n for n, _ in gathers]
    print("\n  prefill wall-clock — dense (causal flash SDPA) vs gather selectors (one attention layer):")
    head = f"  {'T':>7}  {'dense ms':>9}  " + "  ".join(f"{n:>20}" for n in names)
    print(head)
    print(f"  {'':>7}  {'':>9}  " + "  ".join(f"{'ms / x / kept':>20}" for _ in names))
    for t in sweep:
        proj_gib = _gib(1 * cfg.num_attention_heads * t * cfg.head_dim * 2 * 4)  # q+kr+vr+out bf16 ≈
        if proj_gib > MEM_CEIL_GIB:
            print(f"  {t:>7}  (skipped — projected {proj_gib:.0f} GiB > ceil {MEM_CEIL_GIB:.0f})")
            break
        xn = xn_full[:, :t]
        mx.clear_cache()
        layer.attention.sparse = None
        td = _bench_call(layer.attention, xn)
        cells = []
        for _, gcfg in gathers:
            layer.attention.sparse = gcfg
            tg = _bench_call(layer.attention, xn)
            frac = _kept_frac(layer, xn, gcfg)
            cells.append(f"{tg:7.1f} /{td / tg:4.1f}x /{frac * 100:4.0f}%")
        layer.attention.sparse = None
        print(f"  {t:>7}  {td:>9.1f}  " + "  ".join(f"{c:>20}" for c in cells))

    # 4. the FOLD: a mixed per-head assignment (most cheap ashape + a few dense xattn). Naive gather sizes
    #    every head by one global max_kept = the dense head's (so it ≈ uniform xattn, the slow one); the
    #    grouped fold gathers each spec-group at its own max_kept, so the cheap heads run cheap. Output is
    #    bit-equivalent (model-free gate internlm2_grouped_gather_test) — this measures the speed it folds.
    nh = cfg.num_attention_heads
    mix = _perhead_mix(nh)
    naive_cfg = XAttnConfig(block=BLOCK, head_specs=mix, min_seq=0, gather=True, grouped_gather=False)
    fold_cfg = replace(naive_cfg, grouped_gather=True)
    print(f"\n  per-head FOLD — {nh - 4} cheap ashape + 4 dense xattn heads: naive (one global max_kept) "
          f"vs grouped fold (per-group max_kept); fold is bit-equivalent (gate), this times it:")
    print(f"  {'T':>7}  {'dense ms':>9}  {'naive ms / x':>16}  {'fold ms / x':>16}  {'fold/naive':>10}")
    fold_ok = True
    for t in sweep:
        proj_gib = _gib(1 * nh * t * cfg.head_dim * 2 * 4)
        if proj_gib > MEM_CEIL_GIB:
            break
        xn = xn_full[:, :t]
        mx.clear_cache()
        layer.attention.sparse = None
        td = _bench_call(layer.attention, xn)
        layer.attention.sparse = naive_cfg
        tn = _bench_call(layer.attention, xn)
        layer.attention.sparse = fold_cfg
        tf = _bench_call(layer.attention, xn)
        layer.attention.sparse = None
        print(f"  {t:>7}  {td:>9.1f}  {tn:>8.1f} /{td / tn:4.1f}x  {tf:>8.1f} /{td / tf:4.1f}x  "
              f"{tn / tf:>9.2f}x")
        if t == sweep[-1]:
            fold_ok = tf < tn * 0.95          # at the longest ctx the fold must materially beat naive
    print(f"  → at T={sweep[-1]} the fold runs the cheap heads cheap instead of paying the dense head's "
          f"block budget on all {nh} heads (naive is bottlenecked ≈ uniform xattn).")

    ok = anchor_ok and probe_ok and fold_ok
    print(f"\n{'PASS' if ok else 'FAIL'} — parity anchor (keep-all gather == dense): {anchor_ok}; "
          f"M7 chunked probe == single-shot on real weights: {probe_ok}; per-head fold beats naive at "
          f"max T: {fold_ok}. The speed tables are the measured result (gather wins once O(T²) dominates "
          f"the identical projection cost; the fold un-bottlenecks a mixed per-head assignment).")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else SWEEP[-1])
