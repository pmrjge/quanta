"""InternLM2.5 long-doc sparse-prefill ppl gate — M1 of the MInference sparse-prefill track.

Measures the **quality cost** of XAttention block-sparse prefill on the real int8-g64
InternLM2.5-7B bake across a threshold sweep: dense vs XAttention (the additive-mask path for
the quality number, plus the block-gather speed-path twin), reporting ppl / top-1 / Δppl%.

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

import mlx.core as mx

from quanta.internlm2.artifact import InternLM2Artifact
from quanta.internlm2.model import _DecoderLayer, _load_decoder_layer
from quanta.internlm2.tokenizer import InternLM2Tokenizer
from quanta.modeling.xattention import XAttnConfig

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

    emb = art.embed()
    h0 = emb[arr][None]                                   # [1, T, H] bf16
    del emb
    art.release()
    mx.clear_cache()
    hs = [h0 for _ in variants]                           # diverge after the embedding (shared h0)

    for i in range(n):
        layer = _DecoderLayer(cfg)
        _load_decoder_layer(layer, art, i, mx.bfloat16)
        for vi, (_, sp) in enumerate(variants):
            layer.attention.sparse = sp                  # M0 hook; None ⇒ dense, byte-unchanged
            hs[vi] = layer(hs[vi], use_fast=True)        # offset 0, t == kv_len ⇒ from-scratch prefill
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
    for (label, _), h in zip(variants, hs):
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

    print(f"\n  dense ppl<20: {healthy} ({dense:.3f})   "
          f"gather==mask @0.9 (<1%): {twin} (Δ={abs(gat90 - mask90):.2e})")
    print(f"  t=0.90 Δppl = {d90:+.2f}%   free(≤2%): {free}   sane(≤8%): {sane}")
    ok = healthy and twin and sane
    print(f"\n{'PASS' if ok else 'FAIL'}  "
          f"(quality baseline recorded; t=0.90 is {'FREE' if free else 'NOT free'})")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    args = sys.argv[1:]
    nl = int(args[0]) if len(args) > 0 else None
    mt = int(args[1]) if len(args) > 1 else 1024
    run(nl, mt)
