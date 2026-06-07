"""InternLM2.5 LONG-CONTEXT per-head sparse-prefill ppl gate — M10 of the MInference track.

The **quality** companion to M8 (gather wall-clock speed) and M9 (the per-head-grouped "fold"). M1–M6
measured the per-head (kind, params) assignment's quality on a SHORT 7-block doc, where vertical-slash —
a long-RANGE pattern — earned only 4% of heads (its payoff was off-screen). M10 measures the SAME M6
machinery at a LONG context (128 blocks), where the three deferred long-context claims finally land
**end-to-end through the full 32-layer teacher-forced model**, on real int8-g64 weights:

  1. **the M7 key-chunked vslash probe, e2e in the full model.** A small ``max_alloc_gb`` forces the
     gather's vertical-slash probe to take its softmax in KEY CHUNKS (M7) — not in a standalone probe
     test, but inside every layer of the real forward. Gated by ``vslash v6s6 single`` (single-shot
     probe) vs ``vslash v6s6 chunk`` (chunked probe): SAME selection (same budget, non-binding), only
     ``max_alloc_gb`` differs, so the full-model ppl delta is *pure probe chunking* (M7's masses are
     output-equivalent up to fp reassociation — here confirmed at the ppl level over 32 layers).
  2. **the M9 grouped-fold gather == the additive-mask quality path, at long context.** ``perhd-p``
     (mask, ``gather=False``) is the quality reference; ``perhd-p fold`` (``grouped_gather=True``) is the
     real long-context speed path (each distinct-spec head-group gathered at its own ``max_kept``, M9,
     over the chunked probe, M7, with per-head vslash params, M6). Their ppl must agree (the twin).
  3. **vertical-slash's long-range payoff in the realized per-head mix.** The offline FLOP-budgeted
     search (:func:`assign_head_specs`) runs per layer over a bounded menu of CHEAP patterns (ashape +
     vslash at two param-points each). At 128 blocks xattn's antidiagonal nucleus keeps ~40–63% of
     blocks (measured here as the uniform ``xattn t0.9`` baseline) ⇒ its FLOP cost ≫ the budget ⇒ it is
     **priced out** of the per-head menu (the MInference thesis: reserve the dense kernel, spend the
     cheap bounded patterns). Expect vslash's share to rise above the 4% it earned at 7 blocks — its
     long-range vertical/slash structure has 128 blocks to recall. The realized mix is the measured
     result, not a pass/fail bar.

Sparse attention is **lossy ⇒ ppl-gated, never numeric-parity-gated** (CLAUDE.md rule 4). The hard
gates here are correctness invariants (mixed keep-all == dense; the two twins) + sanity; the per-head
Δppl and the realized selector mix are the measured quality result.

Memory note (rule 8): streams **one** ``_DecoderLayer`` resident at a time out of the int8-g64 artifact
(dequantized to bf16), pushing every variant's hidden state through each layer before releasing it, so
the whole sweep costs ~one layer of weights, not N copies of the 7B model. At T=16384 the additive-mask
path materializes a transient ``[1,32,16384,16384]`` mask (~26 GiB) for the two mask variants (anchor +
perhd-p); each is evaluated and freed before the next (peak one mask). Calibration candidates run on the
sparse GATHER path (cheap, no full mask). The candidate menu omits xattn (priced out at long context;
its cost is shown via the uniform baseline) so the realized assignment stays bounded-cost, which lets
the gather budget be non-binding (the twin's precondition) AND small enough to chunk the probe.

Heavy (loads the 7B bake dequantized; 26 GiB transient masks at 16K ctx). Run ALONE on a free GPU —
never beside another large-resident job (the OOM-reboot hazard):

    uv run --with sentencepiece python -m parity.internlm2_ppl_sparse_long            # 32 layers, 16384 tok
    uv run --with sentencepiece python -m parity.internlm2_ppl_sparse_long 12 8192    # n_layers, max_tokens
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace

import mlx.core as mx

from quanta.internlm2.artifact import InternLM2Artifact
from quanta.internlm2.model import _DecoderLayer, _load_decoder_layer
from quanta.internlm2.tokenizer import InternLM2Tokenizer
from quanta.modeling.xattention import HeadSpec, XAttnConfig, assign_head_specs
from parity.internlm2_ppl_sparse import _KERNEL_OVH, _metrics, _spec_label
from parity.internlm2_prefill_bench import _corpus

ARTIFACT = "/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64"
BLOCK = 128
T_LONG = 16384                  # 128 blocks — the long context (cf. M1–M6's 7 blocks)

MASK_ALLOC_GB = 40.0            # clears sparse_prefill_mask's [B,H,T,T] guard (~26 GiB at T=16384)
SINGLE_ALLOC_GB = 8.0          # gather with a single-shot vslash probe (probe ≪ this ⇒ no chunking)
GAT_ALLOC_GB = 0.20            # gather with a CHUNKED vslash probe (probe 0.27 GiB > this ⇒ M7 chunks)
GAT_BUDGET = 16               # gather kept-block cap — NON-BINDING for the bounded menu (max kept ≤14)
PH2_BUDGET = 16.0             # per-head FLOP-budgeted search budget (blocks); the cheap menu all fits

# Per-head search menu: CHEAP bounded patterns only (ashape window + vslash bands), two param-points per
# kind so the search can assign per-head PARAMS (incl. two vslash widths — the M6 lever). All keep ≤14
# blocks/query, so GAT_BUDGET=16 never binds (gather == mask). xattn is omitted: at 128 blocks its
# nucleus keeps ~40–63% (the uniform `xattn t0.9` baseline below), cost ≫ PH2_BUDGET ⇒ priced out.
PH2_CAND = [
    HeadSpec("ashape", local=4),
    HeadSpec("ashape", local=8),
    HeadSpec("vslash", vert=3, slash=3),
    HeadSpec("vslash", vert=6, slash=6),     # 2nd vslash param-point — per-head vslash params (M6)
]
VSL_VERT, VSL_SLASH = 6, 6     # the uniform vslash twin (single-shot vs chunked probe); max kept 14 ≤ budget


def _cand_cfg(sp: HeadSpec) -> XAttnConfig:
    """A candidate's uniform config for the offline calibration — the SPARSE gather path (cheap, no full
    [B,H,T,T] mask), ``budget=None`` so the kept-block count is honest and the output equals the mask
    path's (gather == mask) for the per-head error measurement."""
    return XAttnConfig(block=BLOCK, stride=16, selector=sp.kind, threshold=sp.threshold, local=sp.local,
                       vert=sp.vert, slash=sp.slash, min_seq=0, gather=True, budget=None,
                       max_alloc_gb=SINGLE_ALLOC_GB)


def _probe_chunks(nh: int, t: int) -> tuple[float, int]:
    """The uniform vslash chunk variant's single-shot probe footprint (GiB) + the # key chunks it takes
    under GAT_ALLOC_GB — the M7 chunk math (mirrors :func:`vertical_slash_index`). Asserted > 1 chunk."""
    nblk = (t + BLOCK - 1) // BLOCK
    lp = t - (nblk - 1) * BLOCK                          # last (real) query block size
    probe_gb = 1 * nh * lp * t * 4 / 1e9
    per_blk = 1.5 * 1 * nh * lp * BLOCK * 4
    sc_chunk = max(1, int(GAT_ALLOC_GB * 1e9 // per_blk)) * BLOCK
    return probe_gb, (t + sc_chunk - 1) // sc_chunk


def run(n_layers: int | None = None, max_tokens: int = T_LONG) -> None:
    art = InternLM2Artifact(ARTIFACT)
    cfg = art.cfg
    tok = InternLM2Tokenizer.from_pretrained(ARTIFACT)
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    nh = cfg.num_attention_heads

    arr = _corpus(tok, max_tokens)                       # long real-text tokens (repo ≫ max_tokens, no tiling)
    t = int(arr.shape[0])
    nblk = (t + BLOCK - 1) // BLOCK

    # --- static variants (fixed cfg). dense=None; the two mask variants need MASK_ALLOC_GB for the
    #     [B,H,T,T] mask; the two uniform vslash variants are the M7 chunked-probe twin (single-shot vs
    #     chunked, same NON-binding budget so only the probe path differs); xattn is the priced-out
    #     baseline (gather, ~lossless vs its mask twin, no 26 GiB mask). -----------------------------
    anchor_specs = tuple(
        [HeadSpec("xattn", threshold=1.0), HeadSpec("ashape", local=nblk),
         HeadSpec("vslash", vert=nblk, slash=nblk)][h % 3] for h in range(nh))
    statics: list[tuple[str, XAttnConfig | None]] = [
        ("dense", None),
        ("perhd-p anchor", XAttnConfig(block=BLOCK, head_specs=anchor_specs, min_seq=0, gather=False,
                                       max_alloc_gb=MASK_ALLOC_GB)),
        ("vslash v6s6 single", XAttnConfig(block=BLOCK, selector="vslash", vert=VSL_VERT, slash=VSL_SLASH,
                                           min_seq=0, gather=True, budget=GAT_BUDGET,
                                           max_alloc_gb=SINGLE_ALLOC_GB)),
        ("vslash v6s6 chunk", XAttnConfig(block=BLOCK, selector="vslash", vert=VSL_VERT, slash=VSL_SLASH,
                                          min_seq=0, gather=True, budget=GAT_BUDGET,
                                          max_alloc_gb=GAT_ALLOC_GB)),
        ("xattn t0.9 gat", XAttnConfig(block=BLOCK, stride=16, threshold=0.9, min_seq=0, gather=True,
                                       budget=None, max_alloc_gb=SINGLE_ALLOC_GB)),
    ]
    ph2_labels = ["perhd-p", "perhd-p fold"]
    labels = [lab for lab, _ in statics] + ph2_labels
    dense_i = 0                                          # hs[dense_i] is the dense stream (calibration input)
    dist2: dict[str, int] = {_spec_label(sp): 0 for sp in PH2_CAND}

    emb = art.embed()
    h0 = emb[arr][None]                                  # [1, T, H] bf16
    del emb
    art.release()
    mx.clear_cache()
    hs = [h0 for _ in labels]                            # all streams share the embedding, then diverge

    xattn_keep_frac = 0.0
    for i in range(n):
        layer = _DecoderLayer(cfg)
        _load_decoder_layer(layer, art, i, mx.bfloat16)

        # Offline per-head search on the DENSE stream's input (hs[dense_i] is still this layer's input).
        xn = layer.attention_norm(hs[dense_i])
        layer.attention.sparse = None
        dense_heads = layer.attention._attn_heads(xn, use_fast=True)              # [1, nh, T, hd]
        den = mx.sqrt(mx.sum(dense_heads.astype(mx.float32) ** 2, axis=(0, 2, 3))) + 1e-6   # [nh]
        errs, costs = [], []
        for sp in PH2_CAND:
            uc = _cand_cfg(sp)
            layer.attention.sparse = uc
            ch = layer.attention._attn_heads(xn, use_fast=True)
            errs.append(mx.sqrt(mx.sum((ch - dense_heads).astype(mx.float32) ** 2, axis=(0, 2, 3))) / den)
            costs.append(float(mx.mean(layer.attention._attn_keep_counts(xn, uc))) + _KERNEL_OVH[sp.kind])
        order = mx.argsort(mx.array(costs)).tolist()                              # cheap → accurate
        cand_o = [PH2_CAND[o] for o in order]
        errs_o = mx.stack([errs[o] for o in order], axis=0)                       # [C, nh] sorted
        costs_o = mx.array([costs[o] for o in order])
        assign2 = assign_head_specs(errs_o, costs_o, cand_o, PH2_BUDGET)          # tuple[HeadSpec]*nh
        for sp2 in assign2:
            dist2[_spec_label(sp2)] += 1
        ph2_mask_cfg = XAttnConfig(block=BLOCK, head_specs=assign2, min_seq=0, gather=False,
                                   max_alloc_gb=MASK_ALLOC_GB)
        ph2_fold_cfg = replace(ph2_mask_cfg, gather=True, grouped_gather=True, budget=GAT_BUDGET,
                               max_alloc_gb=GAT_ALLOC_GB)

        # measure the uniform xattn kept-fraction once (layer 0) — the "priced-out at long ctx" evidence.
        if i == 0:
            kc = layer.attention._attn_keep_counts(
                xn, XAttnConfig(block=BLOCK, stride=16, threshold=0.9, min_seq=0, gather=False))
            xattn_keep_frac = float(mx.mean(kc).item()) / ((nblk + 1) / 2.0)      # kept / avg causal blocks

        cfgs = [sp for _, sp in statics] + [ph2_mask_cfg, ph2_fold_cfg]
        for vi, sp in enumerate(cfgs):
            layer.attention.sparse = sp                  # M0 hook; None ⇒ dense, byte-unchanged
            hs[vi] = layer(hs[vi], use_fast=True)        # offset 0, t == kv_len ⇒ from-scratch prefill
            mx.eval(hs[vi])                              # serialize ⇒ free each 26 GiB mask before the next
        del layer
        art.release()
        mx.clear_cache()

    # final norm + LM head in fp32 (matches internlm2_ppl_sparse' ppl-precision convention)
    norm_w = art.final_norm().astype(mx.float32)
    lm_head = art.lm_head().astype(mx.float32)

    print(f"\n=== InternLM2.5 LONG-CONTEXT per-head sparse-prefill ppl (M10; int8-g64 bake, layers={n}, "
          f"tokens={t}, blocks={nblk}) ===")
    print(f"{'variant':<20} {'ppl':>10} {'top-1':>8} {'Δppl%':>8}")
    ppls: dict[str, float] = {}
    base_ppl: float | None = None
    for label, h in zip(labels, hs):
        hf = mx.fast.rms_norm(h.astype(mx.float32), norm_w, cfg.norm_eps)
        logits = (hf @ lm_head.T)[0]
        mx.eval(logits)
        ppl, acc = _metrics(logits, arr)
        ppls[label] = ppl
        if base_ppl is None:
            base_ppl, dpct = ppl, "—"
        else:
            dpct = f"{100 * (ppl - base_ppl) / base_ppl:+.2f}"
        print(f"{label:<20} {ppl:>10.4f} {acc:>8.3f} {dpct:>8}")

    # --- gates --------------------------------------------------------------------------------------
    dense = ppls["dense"]
    healthy = dense < 20.0

    # 1. mixed per-head-specs keep-all == dense — the per-head ROUTING parity anchor (exact regardless of
    #    which kinds/params mix). Mask vs the dense "causal" path: bf16-identical (both flash SDPA).
    anchor_ok = abs(ppls["perhd-p anchor"] - dense) / dense < 0.01

    # 2. M7 chunked vslash probe == single-shot probe, e2e in the full 32-layer model. Same selection
    #    (same non-binding budget), only the probe path differs ⇒ the delta is pure M7 chunking. Long-ctx
    #    bf16 reassociation floor (looser than the fp-tight model-free internlm2_vslash_chunked_test).
    vsl_single, vsl_chunk = ppls["vslash v6s6 single"], ppls["vslash v6s6 chunk"]
    chunk_twin = abs(vsl_chunk - vsl_single) / vsl_single < 0.02
    probe_gb, nchunks = _probe_chunks(nh, t)
    chunk_fired = nchunks > 1 and probe_gb > GAT_ALLOC_GB

    # 3. M9 grouped-fold gather == the additive-mask quality path, for the derived per-head assignment
    #    (the fold + per-head vslash params + the chunked probe, all e2e). Same bf16 long-ctx floor.
    ph2_mask, ph2_fold = ppls["perhd-p"], ppls["perhd-p fold"]
    fold_twin = abs(ph2_fold - ph2_mask) / ph2_mask < 0.02
    d_ph2 = 100 * (ph2_mask - dense) / dense
    ph2_sane = ph2_mask < 2.0 * dense                    # derived assignment not blown up (bug catch)

    n_assign = sum(dist2.values())
    mix = "  ".join(f"{lab} {dist2[lab]} ({100 * dist2[lab] / n_assign:.0f}%)"
                    for lab in dist2 if dist2[lab])
    vslash_share = sum(dist2[lab] for lab in dist2 if lab.startswith("vslash")) / max(1, n_assign)
    d_vsl = 100 * (vsl_single - dense) / dense
    d_xattn = 100 * (ppls["xattn t0.9 gat"] - dense) / dense

    print(f"\n  dense ppl<20: {healthy} ({dense:.3f})")
    print(f"  [1] perhd-p mixed keep-all == dense (<1%): {anchor_ok} "
          f"(Δ={abs(ppls['perhd-p anchor'] - dense):.2e})")
    print(f"  [2] M7 chunked probe e2e: vslash v6s6 single {vsl_single:.4f} vs chunk {vsl_chunk:.4f}  "
          f"twin(<2%): {chunk_twin} (Δ={abs(vsl_chunk - vsl_single):.2e}); "
          f"probe {probe_gb:.2f} GiB → {nchunks} chunk(s) @ {GAT_ALLOC_GB} GiB: {chunk_fired}")
    print(f"  [3] M9 fold == mask: perhd-p {ph2_mask:.4f} vs fold {ph2_fold:.4f}  twin(<2%): {fold_twin} "
          f"(Δ={abs(ph2_fold - ph2_mask):.2e})")
    print(f"  perhd-p Δppl = {d_ph2:+.2f}%  sane(<2x): {ph2_sane}  (budget={PH2_BUDGET} blocks; "
          f"assign Σlayers×heads: {mix})")
    print(f"  realized vslash share: {100 * vslash_share:.0f}%  (M6 saw 4% at 7 blocks — long-range "
          f"payoff lands at {nblk} blocks)")
    print(f"  cf. uniform ingredients — vslash v6s6 {d_vsl:+.2f}%   xattn t0.9 {d_xattn:+.2f}% "
          f"(keeps ~{100 * xattn_keep_frac:.0f}% of causal blocks ⇒ cost ≫ budget ⇒ priced out of the menu)")

    ok = (healthy and anchor_ok and chunk_twin and chunk_fired and fold_twin and ph2_sane)
    print(f"\n{'PASS' if ok else 'FAIL'} — long-context per-head sparse prefill: keep-all anchored (== "
          f"dense), M7 chunked probe == single-shot e2e, M9 fold == mask; per-head Δppl + realized mix "
          f"(vslash {100 * vslash_share:.0f}%) recorded. The quality companion to M8/M9's speed.")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    args = sys.argv[1:]
    nl = int(args[0]) if len(args) > 0 else None
    mt = int(args[1]) if len(args) > 1 else T_LONG
    _t0 = time.perf_counter()
    run(nl, mt)
    print(f"\n[wall {time.perf_counter() - _t0:.1f}s]")
