"""Real-prose teacher-forced perplexity smoke for DeepSeek-V4-Flash (the quality arbiter).

Ties the parity-gated BPE tokenizer (#75) to the validated 43-layer forward (#74): tokenizes a fixed
coherent English passage (with BOS, per the methodology) and runs the streamed bf16 reference forward
once, reporting both teacher-forced perplexity and top-1 next-token accuracy. Unlike the #74 random-id
smoke (ppl ~3e7 by construction), these are *quality* numbers on natural language — a correct forward
should give low single/low-double-digit ppl and high top-1 agreement; a subtly broken one would not.

Streams every layer's experts (fp4->bf16), one layer resident (rule-8). Runtime-only (no torch oracle):

    uv run python -m parity.dsv4_ppl
"""

from __future__ import annotations

import mlx.core as mx

from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.loader import DeepSeekV4SourceCheckpoint
from quanta.dsv4.model import dsv4_logits
from quanta.dsv4.tokenizer import DeepSeekV4Tokenizer

ART = "/Users/pmrj/models/DeepSeek-V4-Flash"

PROSE = (
    "The history of the printing press is often told as a single moment of invention, but in truth "
    "it was the result of many smaller advances accumulating over centuries. Long before movable type "
    "appeared in Europe, craftsmen in East Asia had experimented with carved wooden blocks and even "
    "with individual ceramic characters. What changed in the middle of the fifteenth century was not a "
    "single idea but a practical combination of ideas: a durable metal alloy for the type, an oil-based "
    "ink that adhered to metal, and a press adapted from the kind already used to crush grapes and "
    "olives. Together these allowed a single workshop to produce hundreds of identical pages in the "
    "time it had once taken to copy a single book by hand. The consequences were enormous, reshaping "
    "religion, science, and politics across the following two centuries."
)


def run() -> None:
    cfg = DeepSeekV4Config.from_pretrained(ART)
    tok = DeepSeekV4Tokenizer.from_pretrained(ART)
    ids = tok.encode(PROSE, add_bos=True)
    print(f"[dsv4 ppl] tokenized real prose: {len(ids)} tokens (BOS-first={ids[0]==tok.bos_id}); "
          f"first 12={ids[:12]}", flush=True)
    print("[dsv4 ppl] streaming bf16 forward (43 layers, one resident) ...", flush=True)

    ck = DeepSeekV4SourceCheckpoint(ART, cfg)
    logits = dsv4_logits(ck, mx.array([ids]), cfg, dtype=mx.bfloat16).astype(mx.float32)[0]  # [S, V]
    tgt = mx.array(ids[1:])
    pred = logits[:-1]
    lse = mx.logsumexp(pred, axis=-1)
    tru = mx.take_along_axis(pred, tgt[:, None], axis=-1)[:, 0]
    ce = mx.mean(lse - tru)
    ppl = float(mx.exp(ce).item())
    top1 = float(mx.mean((mx.argmax(pred, axis=-1) == tgt).astype(mx.float32)).item())

    print(f"[dsv4 ppl] teacher-forced perplexity = {ppl:.3f}")
    print(f"[dsv4 ppl] top-1 next-token accuracy = {100*top1:.1f}%  ({len(ids)-1} positions)")
    print("PASS (coherent prose ppl)" if ppl < 30.0 else f"HIGH ppl={ppl:.1f} — investigate")


if __name__ == "__main__":
    run()
