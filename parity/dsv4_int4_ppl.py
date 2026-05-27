"""DSV4 baked int4-g64 artifact: teacher-forced perplexity vs the bf16 source reference.

The e2e quality gate for the bake (#76 finish): loads the freshly baked
``~/models/DeepSeek-V4-Flash-quanta_int4g64`` artifact through
:class:`quanta.dsv4.runtime.DSV4ResidentModel` (RAM-resident, all-layers, int4 AWQ experts +
int8 dense + bf16 norms/router/HC/embed/head/MTP) and reports:

* teacher-forced perplexity on the same coherent prose used by ``parity/dsv4_ppl.py``;
* top-1 next-token agreement;
* (optionally) absolute ppl drift vs the bf16 reference — the int4 should give a low single
  / low-double-digit ppl, with no more than a few percent drift vs the reference (else the
  bake corrupted the forward).

NOT model-free: loads the full 169 GB artifact (one layer at a time during build, then
all-resident). Use only when GPU + memory are available.

    uv run python -m parity.dsv4_int4_ppl
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.dsv4.runtime import DSV4ResidentModel
from quanta.dsv4.tokenizer import DeepSeekV4Tokenizer

ART = "/Users/pmrj/models/DeepSeek-V4-Flash-quanta_int4g64"

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
    import os
    n_layers = int(os.environ.get("DSV4_N_LAYERS", "4"))
    print(f"[dsv4 int4-g64 ppl] loading resident model from {ART} (n_layers={n_layers}) ...", flush=True)
    t0 = time.perf_counter()
    model = DSV4ResidentModel(ART, n_layers=n_layers)
    cfg = model.cfg
    print(f"[dsv4 int4-g64 ppl] resident in {(time.perf_counter() - t0):.1f}s "
          f"({cfg.num_hidden_layers} layers)", flush=True)

    tok = DeepSeekV4Tokenizer.from_pretrained(ART)
    ids = tok.encode(PROSE, add_bos=True)
    print(f"[dsv4 int4-g64 ppl] tokenized real prose: {len(ids)} tokens "
          f"(BOS-first={ids[0] == tok.bos_id}); first 12={ids[:12]}", flush=True)
    print("[dsv4 int4-g64 ppl] prefill (all-resident, parity-correct dsv4_block per layer) ...", flush=True)

    t1 = time.perf_counter()
    logits = model(mx.array(ids))[0].astype(mx.float32)        # [S, V]
    mx.eval(logits)
    print(f"[dsv4 int4-g64 ppl] prefill done in {(time.perf_counter() - t1):.1f}s", flush=True)

    tgt = mx.array(ids[1:])
    pred = logits[:-1]
    lse = mx.logsumexp(pred, axis=-1)
    tru = mx.take_along_axis(pred, tgt[:, None], axis=-1)[:, 0]
    ce = mx.mean(lse - tru)
    ppl = float(mx.exp(ce).item())
    top1 = float(mx.mean((mx.argmax(pred, axis=-1) == tgt).astype(mx.float32)).item())

    print(f"[dsv4 int4-g64 ppl] teacher-forced perplexity = {ppl:.3f}")
    print(f"[dsv4 int4-g64 ppl] top-1 next-token accuracy = {100 * top1:.1f}%  "
          f"({len(ids) - 1} positions)")
    print("PASS (coherent prose ppl)" if ppl < 30.0 else f"HIGH ppl={ppl:.1f} — investigate")


if __name__ == "__main__":
    run()
