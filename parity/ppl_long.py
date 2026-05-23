"""Long-document teacher-forced ppl — the XAttention quality gate.

The 142-token `parity.ppl` is too short to exercise sparsity (everything is local).
This runs a multi-block document and compares **dense** prefill vs **XAttention**
block-sparse prefill across a threshold sweep, reporting ppl / top-1 and the Δ vs
dense — the lossy lever's quality cost (CLAUDE.md: sparse attention is ppl-gated,
not parity-gated).

All variants share **one** weight-streaming pass: each layer is loaded once and every
variant's hidden state is pushed through it (one layer resident at a time, as always).
NOTE: this measures *quality* only. With fast SDPA + an additive block mask MLX still
computes the full QKᵀ, so there is no prefill *speedup* here — realizing the FLOP/memory
win (cheap strided scoring without full QKᵀ + gathered block execution) is a later phase.

    uv run --with tiktoken python -m parity.ppl_long            # full 61 layers, 1024 tok
    uv run --with tiktoken python -m parity.ppl_long 12 768     # n_layers, max_tokens
"""

from __future__ import annotations

import sys

import mlx.core as mx

from quanta.config import KimiTextConfig
from quanta.loader import SourceCheckpoint
from quanta.model import (
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    build_runtime_layer,
    load_layer_raw,
)
from quanta.modeling.xattention import XAttnConfig
from quanta.tokenizer import KimiTokenizer

MODEL = "/Users/pmrj/models/Kimi-K2.6"
THRESHOLDS = (0.95, 0.9, 0.8)

LONG_TEXT = (
    "The Earth's climate is governed by a delicate balance between the energy it receives "
    "from the Sun and the energy it radiates back into space. Sunlight arrives mostly as "
    "visible and ultraviolet radiation, passes through the atmosphere, and warms the land "
    "and oceans. The warmed surface re-emits that energy as longer-wavelength infrared "
    "radiation. Certain gases in the atmosphere, notably carbon dioxide, methane, and water "
    "vapour, absorb part of this outgoing infrared and re-radiate it in all directions, "
    "trapping heat near the surface. This natural greenhouse effect keeps the planet roughly "
    "thirty degrees warmer than it would otherwise be, and without it the oceans would freeze "
    "and complex life as we know it could not exist.\n\n"
    "Photosynthesis sits at the heart of the carbon cycle that regulates these gases. Green "
    "plants, algae, and cyanobacteria capture light energy and use it to split water and fix "
    "carbon dioxide into sugars, releasing oxygen as a by-product. The carbon locked into "
    "those sugars flows through food webs as animals eat plants and other animals, and it "
    "returns to the atmosphere through respiration, decay, and combustion. Over short "
    "timescales this exchange is nearly balanced: the carbon dioxide drawn down by growing "
    "forests in spring and summer is released again as leaves fall and decompose. The famous "
    "sawtooth record measured at Mauna Loa shows this annual breathing of the biosphere "
    "superimposed on a relentless upward climb.\n\n"
    "That upward climb is the signature of human activity. Since the industrial revolution, "
    "the burning of coal, oil, and natural gas has transferred carbon that was buried for "
    "hundreds of millions of years back into the air in the span of a few generations. "
    "Deforestation compounds the problem, because clearing and burning forests both releases "
    "their stored carbon and removes the very organisms that would otherwise draw it down. "
    "The concentration of carbon dioxide in the atmosphere has risen from about two hundred "
    "and eighty parts per million before industrialization to well over four hundred today, a "
    "level not seen in several million years.\n\n"
    "The oceans have softened the blow. They have absorbed roughly a third of the carbon "
    "dioxide emitted by human beings and more than nine tenths of the additional heat trapped "
    "by the thickening blanket of greenhouse gases. This service comes at a cost. As carbon "
    "dioxide dissolves in seawater it forms carbonic acid, lowering the ocean's pH in a "
    "process known as ocean acidification. More acidic water makes it harder for corals, "
    "shellfish, and tiny planktonic organisms to build the calcium carbonate skeletons and "
    "shells on which entire marine food webs depend. Warmer water also holds less dissolved "
    "oxygen and drives the bleaching of coral reefs that shelter a quarter of all marine "
    "species.\n\n"
    "Feedback loops make the system harder to predict. As bright sea ice and snow melt, they "
    "expose darker ocean and land that absorb more sunlight, accelerating the warming that "
    "melted them in the first place. Thawing permafrost releases methane and carbon dioxide "
    "from long-frozen soils, adding to the greenhouse burden. Warmer air holds more water "
    "vapour, itself a greenhouse gas, amplifying the initial heating. Against these "
    "amplifying feedbacks stand a few stabilizing ones, such as the way additional plant "
    "growth can draw down some extra carbon, but the balance of evidence suggests the "
    "amplifiers dominate over the timescales that matter to human society.\n\n"
    "Understanding this balance is why scientists build elaborate models of the atmosphere "
    "and oceans, and why the historical record of carbon, temperature, and ice is studied so "
    "carefully. The same photosynthesis that first filled the air with oxygen billions of "
    "years ago still regulates the carbon dioxide that warms the planet today, binding the "
    "story of life to the story of climate."
)


def _metrics(logits: mx.array, arr: mx.array) -> tuple[float, float]:
    lg = logits[:-1].astype(mx.float32)
    targets = arr[1:]
    ce = mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, targets[:, None], axis=-1)[:, 0]
    ppl = mx.exp(ce.mean()).item()
    acc = (mx.argmax(lg, axis=-1) == targets).astype(mx.float32).mean().item()
    return ppl, acc


def run(n_layers: int | None = None, max_tokens: int = 1024) -> None:
    cfg = KimiTextConfig.from_pretrained(MODEL)
    ck = SourceCheckpoint(MODEL)
    tok = KimiTokenizer(MODEL, bos_id=cfg.bos_token_id)
    n = cfg.num_hidden_layers if n_layers is None else n_layers

    ids = tok.encode(LONG_TEXT, add_bos=True)[:max_tokens]
    arr = mx.array(ids)
    pos = mx.arange(0, arr.shape[0])

    variants: list[tuple[str, XAttnConfig | None]] = [("dense", None)]
    for t in THRESHOLDS:
        variants.append((f"xattn t={t}", XAttnConfig(block=128, stride=16, threshold=t, min_seq=0)))
    # gather (speed-path) variant: must match its mask-path twin (xattn t=0.9) in ppl
    variants.append(("xattn 0.9 gather", XAttnConfig(block=128, stride=16, threshold=0.9, min_seq=0, gather=True)))
    # chunked gather (tiny max_alloc_gb forces many small chunks): must match the unchunked gather
    variants.append(("xattn 0.9 gа/chunk", XAttnConfig(block=128, stride=16, threshold=0.9, min_seq=0, gather=True, max_alloc_gb=0.1)))

    h0 = ck.embed_tokens(arr)[None].astype(mx.bfloat16)
    hs = [h0 for _ in variants]  # diverge after layer 0

    for i in range(n):
        raw = load_layer_raw(ck, cfg, i, mx.bfloat16)
        layer = build_runtime_layer(cfg, raw)
        for vi, (_, sp) in enumerate(variants):
            layer.self_attn.sparse = sp
            hs[vi] = layer(hs[vi], pos, use_fast=True)
        mx.eval(hs)
        del layer, raw
        ck.release()

    norm_w = ck.read(FINAL_NORM_KEY).astype(mx.bfloat16)
    lm_head = ck.read(LM_HEAD_KEY).astype(mx.bfloat16)

    print(f"\n=== Long-doc teacher-forced ppl (bf16, layers={n}, tokens={len(ids)}, "
          f"blocks={(len(ids) + 127) // 128}) ===")
    print(f"{'variant':<14} {'ppl':>10} {'top-1':>8} {'Δppl%':>8}")
    base_ppl = None
    for (label, _), h in zip(variants, hs):
        hf = mx.fast.rms_norm(h, norm_w, cfg.rms_norm_eps)
        logits = (hf @ lm_head.T)[0]
        mx.eval(logits)
        ppl, acc = _metrics(logits, arr)
        if base_ppl is None:
            base_ppl = ppl
            dpct = "—"
        else:
            dpct = f"{100 * (ppl - base_ppl) / base_ppl:+.1f}"
        print(f"{label:<14} {ppl:>10.3f} {acc:>8.3f} {dpct:>8}")


if __name__ == "__main__":
    args = sys.argv[1:]
    nl = int(args[0]) if len(args) > 0 else None
    mt = int(args[1]) if len(args) > 1 else 1024
    run(nl, mt)
