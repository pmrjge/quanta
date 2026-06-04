"""InternLM2.5 long-doc sparse-prefill ppl gate — M1 (XAttention) + M2 (A-shape) + M3 (vertical-slash)
+ M4 (per-head kind) + M5 (per-head params) + M6 (per-head vslash params).

Measures the **quality cost** of block-sparse prefill on the real int8-g64 InternLM2.5-7B bake,
per selector, against the dense teacher (ppl / top-1 / Δppl%):

* **M1 — XAttention** (antidiagonal nucleus): a ``threshold`` sweep; the additive-mask path is the
  quality number, the block-gather path its speed-path twin.
* **M2 — A-shape** (MInference: attention sink block 0 + a ``local``-block window, static/positional):
  the keep-all variant (``local = n_blocks``) is the parity anchor (== dense), L=4/L=2 the tightening
  windows whose cost is measured, plus an L=4 gather twin. A-shape feeds the **same** execution as
  XAttention — only the block-*selection* differs (``XAttnConfig.selector``).
* **M3 — vertical-slash** (MInference §3): a GLOBAL pattern from an online probe of the last query
  block — top-``vert`` vertical key-blocks ∪ top-``slash`` slash block-offset bands — applied to
  every query block through the same execution. keep-all (``vert = slash = n_blocks``) is the parity
  anchor (== dense); v3s3/v2s2 the tightening patterns whose cost is measured, plus a v3s3 gather twin.
* **M4 — per-head assignment**: an offline search routes each query head to the **cheapest** selector
  kind whose per-head attention-output error vs dense (measured on each layer's dense input — the
  calibration forward) is within a tolerance; ``XAttnConfig.head_selectors`` then dispatches per head
  through the same execution (each head's keep == the uniform keep for its kind). A MIXED keep-all
  assignment is the parity anchor (== dense); the derived assignment's ppl + its realized selector mix
  are recorded, with a gather twin. The win is cheaper kernels at bounded quality, not beating the
  single best uniform — so the derived Δppl is a measured result, not a pass/fail bar.
* **M5 — per-head params**: generalizes M4 from per-head *kind* to per-head *params* — each head carries
  its own ``HeadSpec`` (kind + params), and the offline policy becomes a **kernel-aware FLOP-budgeted
  search** (:func:`assign_head_specs`): per head, the most accurate candidate (kind, params) whose cost
  (measured mean kept blocks via ``_attn_keep_counts`` + a per-kind selection-kernel constant) fits a
  FLOP budget. The candidate grid spans multiple param-points per kind, so two heads of the same kind can
  land on different params. Same anchors as M4 (a MIXED keep-all == dense; a gather twin); the derived
  Δppl + realized (kind, params) mix are the measured result.
* **M6 — per-head vslash params**: removes M5's vslash-pin. The vertical-slash probe
  (:func:`vertical_slash_index`) now returns **param-independent** masses and the top-``vert``/``slash``
  cut is applied per spec in :func:`select_keep`, so two heads can read the ONE global probe yet keep
  different vert/slash. The M5 search grid here gains a 2nd vslash param-point (v2s2 **and** v3s3): a
  derived ``head_specs`` may now mix them per head, with no config-level vert/slash. Same anchors/twin.

Sparse attention is **lossy ⇒ ppl-gated, never numeric-parity-gated** (CLAUDE.md rule 4). This is
the quality baseline every later MInference selector (M2+) is judged against. Two things it proves
on the real weights:

* **the lossy lever's quality cost** — how much teacher-forced ppl XAttention trades for the
  prefill speedup at each threshold (≤ ~1–2% Δppl at threshold 0.9 ⇒ "free");
* **the M1 correctness invariant** — the ``gather=True`` speed path must match its ``gather=False``
  mask twin in ppl (they select the *same* blocks; only the execution differs). The mask path
  measures quality only — with fast SDPA + an additive block mask MLX still computes the full QKᵀ,
  so there is no prefill speedup there; the **gather path is the actual FLOP/memory win**.

Streams **one** decoder layer resident at a time (rule-8) out of the int8-g64 artifact
(dequantized to bf16 via :class:`~quanta.internlm2.artifact.InternLM2Artifact`), pushing *every*
variant's hidden state through each layer before releasing it — so the whole sweep costs ~one
layer of resident weights, not N copies of the 7B model. ``layer.attention.sparse`` is the M0 hook;
the packed ``mx.quantized_matmul`` runtime has no such hook, so the quality measurement runs through
the dequantized module path (the int8 weight quant is orthogonal to and ~lossless against the sparse
approximation it is measuring).

Heavy (loads the 7B bake, dequantized). Run ALONE on a free GPU — never beside another large-resident
job (the OOM-reboot hazard):

    uv run --with sentencepiece python -m parity.internlm2_ppl_sparse            # 32 layers, 1024 tok
    uv run --with sentencepiece python -m parity.internlm2_ppl_sparse 12 768     # n_layers, max_tokens
"""

from __future__ import annotations

import sys
from dataclasses import replace

import mlx.core as mx

from quanta.internlm2.artifact import InternLM2Artifact
from quanta.internlm2.model import _DecoderLayer, _load_decoder_layer
from quanta.internlm2.tokenizer import InternLM2Tokenizer
from quanta.modeling.xattention import (
    HeadSpec,
    XAttnConfig,
    assign_head_selectors,
    assign_head_specs,
)

ARTIFACT = "/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64"
THRESHOLDS = (0.95, 0.9, 0.8)

# Original expository prose (self-authored, no copyright). Long enough to span several block=128
# tiles so block-sparsity actually has off-diagonal blocks to drop — the short clean paragraph the
# bf16/int8 ppl gates use (~110 tokens, one block) cannot exercise the lever at all.
LONG_TEXT = (
    "For most of human history the measurement of time and the measurement of place were the same "
    "problem, and both were solved by watching the sky. A sailor far from land could find his "
    "latitude with relative ease, because the height of the noon sun or of the pole star above the "
    "horizon changes steadily as one travels north or south. Longitude was the harder riddle. To "
    "know how far east or west a ship had sailed, a navigator needed to compare the local time, read "
    "from the sun, against the time at a fixed reference meridian back home. The difference between "
    "the two clocks, converted at fifteen degrees to the hour, gave the distance travelled around "
    "the globe. The trouble was that no clock of the age could keep the reference time through "
    "months of damp, pitching, temperature-swinging life at sea.\n\n"
    "The cost of this ignorance was paid in wrecks. Fleets ran aground on coasts their captains "
    "believed were still leagues away, and cargoes and crews were lost to errors that a reliable "
    "timepiece would have erased. Governments offered enormous prizes to anyone who could solve the "
    "longitude problem, and the greatest scientific minds of the day assumed the answer would come "
    "from astronomy: from ever more precise tables of the moon's position against the stars, read "
    "with instruments of brass and glass. The moon does move across the sky like the hand of a vast "
    "celestial clock, and for a time the lunar-distance method was the only practical scheme that "
    "worked at all.\n\n"
    "The eventual winner came from the opposite direction. A self-taught carpenter and clockmaker "
    "spent decades building marine timekeepers of astonishing ingenuity, replacing the pendulum, "
    "which is useless on a rolling deck, with balanced springs that beat steadily in any motion. He "
    "compensated for the way metal expands in heat and contracts in cold by pairing brass and steel "
    "so that their opposite tendencies cancelled. His finest instrument, no larger than a heavy "
    "pocket watch, lost only a few seconds across a voyage to the Caribbean and back, a precision "
    "that the astronomers had insisted was impossible from mere machinery. The clock, not the "
    "telescope, had won.\n\n"
    "The same impulse that built a clock to tame the ocean would later build machines to tame "
    "calculation. A century after the longitude prize, an inventor imagined an engine of gears and "
    "cams that could evaluate mathematical tables without the errors that crept in whenever weary "
    "human computers did the arithmetic by hand. His designs were never fully completed in his "
    "lifetime, defeated by the cost and the limits of precision engineering, but the idea did not "
    "die. It contained, in mechanical form, the essential parts of every computer that followed: a "
    "store to hold numbers, a mill to operate on them, and a way to control the sequence of "
    "operations from instructions fed in from outside.\n\n"
    "When electricity replaced brass, those parts shrank and quickened beyond all expectation. The "
    "switch, whether a relay, a vacuum tube, or a transistor etched into silicon, let a machine make "
    "a simple decision millions and then billions of times a second. What had taken a room full of "
    "clerks a week could be done before the eye could blink. Yet the logic of the work did not "
    "change. A modern processor still holds numbers in a store, still operates on them in a unit "
    "built for arithmetic, and still follows a sequence of instructions, exactly as the unfinished "
    "engine of gears had promised it would.\n\n"
    "What did change was the discipline required to use such speed without drowning in mistakes. A "
    "machine that obeys instructions perfectly will also obey a flawed instruction perfectly, and so "
    "the burden shifted from the reliability of the hardware to the correctness of the program. Good "
    "engineering, in this newer craft as in the old one, begins with understanding the problem "
    "precisely, finding the smallest change that restores correct behaviour, and testing that change "
    "first against the narrow case that motivated it and then against everything the surrounding work "
    "depends upon. The clockmaker filing his springs by candlelight and the programmer reading a "
    "failing test are, in the end, practising the same patient art."
)


def _metrics(logits: mx.array, arr: mx.array) -> tuple[float, float]:
    """(ppl, top-1 next-token agreement) for ``logits`` ``[T, vocab]`` against targets ``arr`` ``[T]``."""
    lg = logits[:-1].astype(mx.float32)
    targets = arr[1:]
    ce = mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, targets[:, None], axis=-1)[:, 0]
    ppl = mx.exp(ce.mean()).item()
    acc = (mx.argmax(lg, axis=-1) == targets).astype(mx.float32).mean().item()
    return ppl, acc


# --- M5 per-head-params search helpers ------------------------------------------------------------
# Per-kind selection-KERNEL cost (block-equivalent), added to a candidate's measured mean kept-block
# count to make the FLOP budget kernel-aware: ashape is purely positional (no probe / no scoring),
# vslash pays one global probe, xattn pays per-block antidiagonal scoring. Coarse — the attention FLOP
# (∝ kept blocks) dominates; these only tilt ties toward the cheaper kernel at equal kept-block count.
_KERNEL_OVH = {"ashape": 0.0, "vslash": 0.25, "xattn": 0.5}


def _uni(sp: HeadSpec) -> XAttnConfig:
    """Uniform (single-selector) XAttnConfig realizing one HeadSpec — used to measure that candidate's
    per-head error and kept-block cost on the calibration forward."""
    return XAttnConfig(block=128, stride=16, selector=sp.kind, threshold=sp.threshold,
                       local=sp.local, vert=sp.vert, slash=sp.slash, min_seq=0, gather=False)


def _spec_label(sp: HeadSpec) -> str:
    """Short readable (kind, params) tag for the realized per-head-params mix readout."""
    if sp.kind == "ashape":
        return f"ashape:L{sp.local}"
    if sp.kind == "vslash":
        return f"vslash:v{sp.vert}s{sp.slash}"
    return f"xattn:t{sp.threshold:g}"


def run(n_layers: int | None = None, max_tokens: int = 1024) -> None:
    art = InternLM2Artifact(ARTIFACT)
    cfg = art.cfg
    tok = InternLM2Tokenizer.from_pretrained(ARTIFACT)
    n = cfg.num_hidden_layers if n_layers is None else n_layers

    ids = tok.encode(LONG_TEXT, add_bos=True)[:max_tokens]
    arr = mx.array(ids)

    # dense reference + the threshold sweep. mask path (gather=False) is the quality number; the
    # gather=True twin at t=0.9 is the speed path and must match its mask twin in ppl. min_seq=0
    # forces sparsity on for the whole passage (the substrate default min_seq=256 would gate short
    # tails dense). budget=64 (default) never binds below 8192 tokens, so it does not touch quality.
    variants: list[tuple[str, XAttnConfig | None]] = [("dense", None)]
    for t in THRESHOLDS:
        variants.append((f"xattn t={t:.2f}", XAttnConfig(block=128, stride=16, threshold=t,
                                                          min_seq=0, gather=False)))
    variants.append(("xattn t=0.90 gat", XAttnConfig(block=128, stride=16, threshold=0.9,
                                                     min_seq=0, gather=True)))

    # M2: A-shape selector (MInference) — sink (block 0) + a ``local``-block causal window, fed
    # through the SAME execution (only XAttnConfig.selector differs). keep-all (local = n_blocks)
    # keeps every causal block ⇒ == dense (the real-model parity anchor); L=4 (512 tok) / L=2 (256
    # tok) are tightening static windows whose ppl cost is measured; the L=4 gather twin checks the
    # gather speed-path == the mask quality-path on the bake (the M2 invariant).
    nblk = (len(ids) + 127) // 128
    variants.append(("ashape keepall", XAttnConfig(block=128, stride=16, selector="ashape",
                                                   local=nblk, min_seq=0, gather=False)))
    for lw in (4, 2):
        variants.append((f"ashape L={lw}", XAttnConfig(block=128, stride=16, selector="ashape",
                                                       local=lw, min_seq=0, gather=False)))
    variants.append(("ashape L=4 gat", XAttnConfig(block=128, stride=16, selector="ashape",
                                                   local=4, min_seq=0, gather=True)))

    # M3: vertical-slash selector (MInference §3) — a global pattern from the last query block's
    # probe: top-`vert` vertical key-blocks ∪ top-`slash` slash block-offset bands, applied to every
    # query block through the SAME execution (only XAttnConfig.selector differs). keep-all
    # (vert=slash=n_blocks) keeps every causal block ⇒ == dense (the real-model parity anchor); v3s3 /
    # v2s2 are tightening patterns whose ppl cost is measured; the v3s3 gather twin checks the gather
    # speed-path == the mask quality-path on the bake (the M3 invariant — both use the one global index).
    variants.append(("vslash keepall", XAttnConfig(block=128, stride=16, selector="vslash",
                                                   vert=nblk, slash=nblk, min_seq=0, gather=False)))
    for vv in (3, 2):
        variants.append((f"vslash v{vv}s{vv}", XAttnConfig(block=128, stride=16, selector="vslash",
                                                          vert=vv, slash=vv, min_seq=0, gather=False)))
    variants.append(("vslash v3s3 gat", XAttnConfig(block=128, stride=16, selector="vslash",
                                                    vert=3, slash=3, min_seq=0, gather=True)))

    # M4: per-head pattern assignment. An offline search (below, per layer, on the DENSE stream's
    # input — the calibration forward) routes each query head to the CHEAPEST selector kind whose
    # per-head attention-output error vs dense is within ``ph_tol``; ``head_selectors`` then dispatches
    # per head. Candidates are ONE config per kind, ordered cheap→accurate by KERNEL cost (ashape:
    # static positional, no probe < vslash: one global probe < xattn: per-block antidiagonal nucleus);
    # fallback = the last/most-accurate (xattn t=0.9, the best uniform per M1). Heads sharing a kind
    # share that kind's params (``ph_params`` == the candidates' params). Two anchors guard it: a MIXED
    # keep-all assignment (== dense) and a gather twin of the derived assignment.
    ph_params = dict(threshold=0.9, local=2, vert=2, slash=2)
    cand = [
        ("ashape", XAttnConfig(block=128, stride=16, selector="ashape",
                               local=ph_params["local"], min_seq=0, gather=False)),
        ("vslash", XAttnConfig(block=128, stride=16, selector="vslash", vert=ph_params["vert"],
                               slash=ph_params["slash"], min_seq=0, gather=False)),
        ("xattn", XAttnConfig(block=128, stride=16, selector="xattn",
                              threshold=ph_params["threshold"], min_seq=0, gather=False)),
    ]
    cand_kinds = [c for c, _ in cand]
    ph_tol = 0.02                                                # per-head output relerr budget vs dense
    ph_mix = tuple(cand_kinds[h % len(cand_kinds)] for h in range(cfg.num_attention_heads))
    ph_anchor = XAttnConfig(block=128, stride=16, head_selectors=ph_mix, threshold=1.0,
                            local=nblk, vert=nblk, slash=nblk, min_seq=0, gather=False)
    ph_labels = ["perhead anchor", "perhead", "perhead gat"]

    # M5/M6: per-head PARAMS. The candidate grid spans multiple param-points per kind (ashape L2/L4,
    # vslash v2s2/v3s3, xattn t0.90/t0.95), so two heads of the same kind can land on DIFFERENT params —
    # including two vslash params (M6: the probe returns param-independent masses, so a derived head_specs
    # may mix v2s2 and v3s3 freely — no pin). Per layer (below) each candidate's per-head error vs dense
    # AND its kernel-aware FLOP cost (mean kept blocks + per-kind kernel constant) are measured on the
    # dense calibration forward; candidates are sorted cheap→accurate; ``assign_head_specs`` picks per head
    # the most accurate one that fits ``ph2_budget`` blocks (else the cheapest). A MIXED keep-all
    # per-head-specs assignment is the parity anchor (== dense); a gather twin checks the invariant. The
    # derived Δppl + realized (kind,params) mix are the measured result (not pass/fail).
    ph2_cand = [
        HeadSpec("ashape", local=2),
        HeadSpec("ashape", local=4),
        HeadSpec("vslash", vert=2, slash=2),
        HeadSpec("vslash", vert=3, slash=3),     # M6: a 2nd vslash param-point — enables per-head vslash params
        HeadSpec("xattn", threshold=0.90),
        HeadSpec("xattn", threshold=0.95),
    ]
    ph2_budget = 4.0                                             # FLOP budget in mean-kept-blocks/query
    ph2_anchor_specs = tuple(
        [HeadSpec("xattn", threshold=1.0), HeadSpec("ashape", local=nblk),
         HeadSpec("vslash", vert=nblk, slash=nblk)][h % 3]
        for h in range(cfg.num_attention_heads))
    ph2_anchor = XAttnConfig(block=128, stride=16, head_specs=ph2_anchor_specs, vert=nblk, slash=nblk,
                             min_seq=0, gather=False)
    ph2_labels = ["perhd-p anchor", "perhd-p", "perhd-p gat"]
    dist2: dict[str, int] = {_spec_label(sp): 0 for sp in ph2_cand}   # realized (kind,params) mix

    emb = art.embed()
    h0 = emb[arr][None]                                   # [1, T, H] bf16
    del emb
    art.release()
    mx.clear_cache()
    labels = [lab for lab, _ in variants] + ph_labels + ph2_labels
    hs = [h0 for _ in labels]                             # diverge after the embedding (shared h0)
    nv = len(variants)
    ph_anchor_i, ph_mask_i, ph_gat_i = nv, nv + 1, nv + 2
    ph2_anchor_i, ph2_mask_i, ph2_gat_i = nv + 3, nv + 4, nv + 5
    dist = {k: 0 for k in cand_kinds}                     # realized M4 assignment mix (Σ over layers×heads)

    for i in range(n):
        layer = _DecoderLayer(cfg)
        _load_decoder_layer(layer, art, i, mx.bfloat16)

        # M4: derive THIS layer's per-head assignment from the DENSE stream's input (hs[0] is still the
        # layer input here — no stream has advanced yet), the standard offline calibration on dense
        # q/k/v. Per-head L2 error of each candidate's attention output vs dense → cheapest within tol.
        xn = layer.attention_norm(hs[0])
        layer.attention.sparse = None
        dense_heads = layer.attention._attn_heads(xn, use_fast=True)         # [1, nh, T, hd]
        den = mx.sqrt(mx.sum(dense_heads.astype(mx.float32) ** 2, axis=(0, 2, 3))) + 1e-6   # [nh]
        errs = []
        for _, cc in cand:
            layer.attention.sparse = cc
            ch = layer.attention._attn_heads(xn, use_fast=True)
            num = mx.sqrt(mx.sum((ch - dense_heads).astype(mx.float32) ** 2, axis=(0, 2, 3)))
            errs.append(num / den)                                           # per-head relerr [nh]
        assign = assign_head_selectors(mx.stack(errs, axis=0), cand_kinds, ph_tol)   # tuple[str]*nh
        for s in assign:
            dist[s] += 1
        ph_mask_cfg = XAttnConfig(block=128, stride=16, head_selectors=assign, min_seq=0,
                                  gather=False, **ph_params)
        ph_gat_cfg = XAttnConfig(block=128, stride=16, head_selectors=assign, min_seq=0,
                                 gather=True, **ph_params)

        # M5/M6: per-head PARAMS — reuse the SAME dense calibration (dense_heads/den). Per candidate
        # measure per-head output relerr AND kernel-aware cost (mean kept blocks + per-kind kernel
        # constant), sort cheap→accurate (so policy ties break to the cheaper), then budget-search the
        # per-head (kind, params). M6: two vslash param-points (v2s2/v3s3) may be mixed per head — the
        # probe masses are param-independent, so the derived config needs no config-level vert/slash.
        errs2, costs2 = [], []
        for sp2 in ph2_cand:
            uc = _uni(sp2)
            layer.attention.sparse = uc
            ch2 = layer.attention._attn_heads(xn, use_fast=True)
            errs2.append(mx.sqrt(mx.sum((ch2 - dense_heads).astype(mx.float32) ** 2, axis=(0, 2, 3))) / den)
            costs2.append(float(mx.mean(layer.attention._attn_keep_counts(xn, uc))) + _KERNEL_OVH[sp2.kind])
        order = mx.argsort(mx.array(costs2)).tolist()            # cheap → accurate (tie → cheaper)
        cand_o = [ph2_cand[o] for o in order]
        errs_o = mx.stack([errs2[o] for o in order], axis=0)     # [C, nh] sorted
        costs_o = mx.array([costs2[o] for o in order])
        assign2 = assign_head_specs(errs_o, costs_o, cand_o, ph2_budget)   # tuple[HeadSpec]*nh
        for sp2 in assign2:
            dist2[_spec_label(sp2)] += 1
        # M6: no config vert/slash — the probe masses are param-independent, so a derived head_specs may
        # mix v2s2 and v3s3 and each vslash head cuts its own params from the one shared probe.
        ph2_mask_cfg = XAttnConfig(block=128, stride=16, head_specs=assign2, min_seq=0, gather=False)
        ph2_gat_cfg = replace(ph2_mask_cfg, gather=True)

        for vi, (_, sp) in enumerate(variants):
            layer.attention.sparse = sp                  # M0 hook; None ⇒ dense, byte-unchanged
            hs[vi] = layer(hs[vi], use_fast=True)        # offset 0, t == kv_len ⇒ from-scratch prefill
        # M4 perhead streams: anchor (fixed mixed keep-all); mask/gather use this layer's assignment.
        layer.attention.sparse = ph_anchor
        hs[ph_anchor_i] = layer(hs[ph_anchor_i], use_fast=True)
        layer.attention.sparse = ph_mask_cfg
        hs[ph_mask_i] = layer(hs[ph_mask_i], use_fast=True)
        layer.attention.sparse = ph_gat_cfg
        hs[ph_gat_i] = layer(hs[ph_gat_i], use_fast=True)
        # M5 perhead-params streams: anchor (fixed mixed keep-all); mask/gather use this layer's search.
        layer.attention.sparse = ph2_anchor
        hs[ph2_anchor_i] = layer(hs[ph2_anchor_i], use_fast=True)
        layer.attention.sparse = ph2_mask_cfg
        hs[ph2_mask_i] = layer(hs[ph2_mask_i], use_fast=True)
        layer.attention.sparse = ph2_gat_cfg
        hs[ph2_gat_i] = layer(hs[ph2_gat_i], use_fast=True)
        mx.eval(hs)
        del layer
        art.release()
        mx.clear_cache()

    # final norm + LM head in fp32 (matches internlm2_logits' ppl-precision convention)
    norm_w = art.final_norm().astype(mx.float32)
    lm_head = art.lm_head().astype(mx.float32)

    print(f"\n=== InternLM2.5 long-doc teacher-forced ppl (int8-g64 bake, layers={n}, "
          f"tokens={len(ids)}, blocks={(len(ids) + 127) // 128}) ===")
    print(f"{'variant':<18} {'ppl':>10} {'top-1':>8} {'Δppl%':>8}")
    ppls: dict[str, float] = {}
    base_ppl: float | None = None
    for label, h in zip(labels, hs):
        hf = mx.fast.rms_norm(h.astype(mx.float32), norm_w, cfg.norm_eps)
        logits = (hf @ lm_head.T)[0]
        mx.eval(logits)
        ppl, acc = _metrics(logits, arr)
        ppls[label] = ppl
        if base_ppl is None:
            base_ppl = ppl
            dpct = "—"
        else:
            dpct = f"{100 * (ppl - base_ppl) / base_ppl:+.2f}"
        print(f"{label:<18} {ppl:>10.4f} {acc:>8.3f} {dpct:>8}")

    # --- gates ---------------------------------------------------------------------------------
    dense = ppls["dense"]
    mask90 = ppls["xattn t=0.90"]
    gat90 = ppls["xattn t=0.90 gat"]
    d90 = 100 * (mask90 - dense) / dense

    healthy = dense < 20.0                         # int8 bake on long mixed prose: low single-digit ppl
    twin = abs(gat90 - mask90) / mask90 < 0.01     # M1 invariant: gather speed-path == mask quality-path
    free = d90 <= 2.0                              # "free" bar: ≤2% Δppl at threshold 0.9
    sane = d90 <= 8.0                              # ceiling: a blown-up selection on real weights fails

    # M2 A-shape gates: keep-all == dense (the real-model parity anchor) and gather==mask (the M2
    # invariant) are hard correctness checks; the tight-window Δppl is the *measured* cost (recorded,
    # not pass/fail). A-shape is a STATIC selector — meant to be cheaper-but-lossier and assigned
    # per-head in MInference, NOT applied at every head — so a "free" bar would be the wrong test;
    # a_sane only catches a blown-up integration (tightest window must not double ppl).
    a_keepall = ppls["ashape keepall"]
    a4_mask, a4_gat, a2_mask = ppls["ashape L=4"], ppls["ashape L=4 gat"], ppls["ashape L=2"]
    a_anchor = abs(a_keepall - dense) / dense < 0.01     # keep-all (local=n_blocks) == dense
    a_twin = abs(a4_gat - a4_mask) / a4_mask < 0.01      # A-shape gather speed-path == mask path
    a_sane = a2_mask < 2.0 * dense                       # tightest window not blown up (bug catch)
    d_a4 = 100 * (a4_mask - dense) / dense
    d_a2 = 100 * (a2_mask - dense) / dense

    # M3 vertical-slash gates: keep-all == dense (the real-model parity anchor) and gather==mask (the
    # M3 invariant — both executions read the ONE global index) are hard correctness checks; the
    # tight-pattern Δppl is the *measured* cost (recorded, not pass/fail — vertical-slash is one of a
    # per-head menu, not forced on every head). v_sane only catches a blown-up integration.
    v_keepall = ppls["vslash keepall"]
    v3_mask, v3_gat, v2_mask = ppls["vslash v3s3"], ppls["vslash v3s3 gat"], ppls["vslash v2s2"]
    v_anchor = abs(v_keepall - dense) / dense < 0.01     # keep-all (vert=slash=n_blocks) == dense
    v_twin = abs(v3_gat - v3_mask) / v3_mask < 0.01      # vertical-slash gather speed-path == mask path
    v_sane = v2_mask < 2.0 * dense                       # tightest pattern not blown up (bug catch)
    d_v3 = 100 * (v3_mask - dense) / dense
    d_v2 = 100 * (v2_mask - dense) / dense

    # M4 per-head gates: a MIXED keep-all assignment == dense (the per-head parity anchor — routing is
    # exact regardless of which kinds mix) and the derived assignment's gather==mask (the M4 twin) are
    # hard correctness checks; the derived perhead Δppl + its selector mix are the *measured* result
    # (recorded, not pass/fail — the win is cheaper kernels at bounded quality, not beating the best
    # uniform). ph_sane only catches a blown-up integration.
    ph_anchor_ppl = ppls["perhead anchor"]
    ph_mask, ph_gat = ppls["perhead"], ppls["perhead gat"]
    ph_anchor_ok = abs(ph_anchor_ppl - dense) / dense < 0.01    # mixed keep-all == dense
    ph_twin = abs(ph_gat - ph_mask) / ph_mask < 0.01           # perhead gather speed-path == mask path
    ph_sane = ph_mask < 2.0 * dense                            # derived assignment not blown up
    d_ph = 100 * (ph_mask - dense) / dense
    n_assign = sum(dist.values())
    mix = "  ".join(f"{k} {dist[k]}/{n_assign} ({100 * dist[k] / n_assign:.0f}%)" for k in cand_kinds)

    # M5/M6 per-head-params gates: a MIXED keep-all per-head-specs assignment == dense (the parity anchor)
    # and the derived assignment's gather==mask (the twin) are hard checks; the derived Δppl + the realized
    # (kind,params) mix — which under M6 can include two vslash params — are the measured result.
    ph2_anchor_ppl = ppls["perhd-p anchor"]
    ph2_mask, ph2_gat = ppls["perhd-p"], ppls["perhd-p gat"]
    ph2_anchor_ok = abs(ph2_anchor_ppl - dense) / dense < 0.01    # mixed keep-all == dense
    ph2_twin = abs(ph2_gat - ph2_mask) / ph2_mask < 0.01         # perhd-p gather speed-path == mask path
    ph2_sane = ph2_mask < 2.0 * dense                            # derived assignment not blown up
    d_ph2 = 100 * (ph2_mask - dense) / dense
    n_assign2 = sum(dist2.values())
    mix2 = "  ".join(f"{lab} {dist2[lab]} ({100 * dist2[lab] / n_assign2:.0f}%)"
                     for lab in dist2 if dist2[lab])

    print(f"\n  dense ppl<20: {healthy} ({dense:.3f})   "
          f"gather==mask @0.9 (<1%): {twin} (Δ={abs(gat90 - mask90):.2e})")
    print(f"  xattn  t=0.90 Δppl = {d90:+.2f}%   free(≤2%): {free}   sane(≤8%): {sane}")
    print(f"  ashape keep-all==dense (<1%): {a_anchor} (Δ={abs(a_keepall - dense):.2e})   "
          f"gather==mask @L=4 (<1%): {a_twin} (Δ={abs(a4_gat - a4_mask):.2e})")
    print(f"  ashape L=4 Δppl = {d_a4:+.2f}%   L=2 Δppl = {d_a2:+.2f}%   "
          f"(static selector — cost recorded; cf. xattn t=0.9 {d90:+.2f}%)")
    print(f"  vslash keep-all==dense (<1%): {v_anchor} (Δ={abs(v_keepall - dense):.2e})   "
          f"gather==mask @v3s3 (<1%): {v_twin} (Δ={abs(v3_gat - v3_mask):.2e})")
    print(f"  vslash v3s3 Δppl = {d_v3:+.2f}%   v2s2 Δppl = {d_v2:+.2f}%   "
          f"(global probe pattern — cost recorded; cf. xattn t=0.9 {d90:+.2f}%)")
    print(f"  perhead mixed keep-all==dense (<1%): {ph_anchor_ok} (Δ={abs(ph_anchor_ppl - dense):.2e})"
          f"   gather==mask (<1%): {ph_twin} (Δ={abs(ph_gat - ph_mask):.2e})")
    print(f"  perhead Δppl = {d_ph:+.2f}%  (tol={ph_tol}; assign Σlayers×heads: {mix})")
    print(f"           cf. uniform ingredients — ashape L=2 {d_a2:+.2f}%   vslash v2s2 {d_v2:+.2f}%   "
          f"xattn t=0.9 {d90:+.2f}%")
    print(f"  perhd-p mixed keep-all==dense (<1%): {ph2_anchor_ok} (Δ={abs(ph2_anchor_ppl - dense):.2e})"
          f"   gather==mask (<1%): {ph2_twin} (Δ={abs(ph2_gat - ph2_mask):.2e})")
    print(f"  perhd-p Δppl = {d_ph2:+.2f}%  (budget={ph2_budget} blocks; assign Σlayers×heads: {mix2})")
    ok = (healthy and twin and sane and a_anchor and a_twin and a_sane
          and v_anchor and v_twin and v_sane and ph_anchor_ok and ph_twin and ph_sane
          and ph2_anchor_ok and ph2_twin and ph2_sane)
    print(f"\n{'PASS' if ok else 'FAIL'}  (xattn t=0.90 {'FREE' if free else 'NOT free'}; A-shape + "
          f"vertical-slash + per-head kind + per-head params (incl. vslash) integrated — keep-all "
          f"anchored, gather==mask, cost recorded)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    args = sys.argv[1:]
    nl = int(args[0]) if len(args) > 0 else None
    mt = int(args[1]) if len(args) > 1 else 1024
    run(nl, mt)
