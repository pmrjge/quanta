"""Nemotron-H int4 e2e ppl gate (#38) — the quantized bake vs the bf16 reference.

Runs the **same** streamed teacher-forced forward as ``parity.nemotron_ppl`` (one layer
resident; rule-8) but over the baked int4/int8 artifact, with every packed weight
dequantized back to bf16 by :class:`quanta.nemotron.artifact.NemotronArtifact`. Because the
artifact duck-types the source checkpoint, ``streamed_logits`` is reused verbatim — only the
weights differ, so the ppl delta isolates quantization error. Same PROSE, same 109 tokens,
same fp32 head as the reference, so the numbers are directly comparable.

Gate: int4 ppl should sit close to the bf16 reference (3.379, post the U1 group-wise mamba-norm
fix). A large gap ⇒ re-bake the experts at g64 (#53) and re-gate.

    uv run --with tokenizers python -m parity.nemotron_int4_ppl
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

from parity.nemotron_ppl import PROSE, streamed_logits
from quanta.nemotron.artifact import NemotronArtifact
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.tokenizer import NemotronTokenizer

ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"  # default; override via argv[1]
BF16_PPL = 3.379  # parity.nemotron_ppl reference, post U1 group-wise-norm fix (was 5.981 buggy); 109 tokens


def run() -> None:
    art_dir = sys.argv[1] if len(sys.argv) > 1 else ART
    cfg = NemotronHConfig.from_pretrained(art_dir)
    art = NemotronArtifact(art_dir)
    tok = NemotronTokenizer(art_dir)
    ids = tok.encode(PROSE, add_bos=False)[:192]
    arr = mx.array(ids)
    t0 = time.perf_counter()
    logits = streamed_logits(art, cfg, arr)
    lg = logits[:-1].astype(mx.float32)
    targets = arr[1:]
    ce = mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, targets[:, None], axis=-1)[:, 0]
    ppl = mx.exp(ce.mean()).item()
    acc = (mx.argmax(lg, axis=-1) == targets).astype(mx.float32).mean().item()
    print(f"\n=== Nemotron-H int4 artifact (streamed, tokens={len(ids)}) ===")
    print(f"perplexity           : {ppl:.3f}   (bf16 ref {BF16_PPL:.3f}, "
          f"Δ {ppl - BF16_PPL:+.3f} / {100 * (ppl - BF16_PPL) / BF16_PPL:+.1f}%)")
    print(f"top-1 next-token acc : {acc:.3f}")
    print(f"elapsed              : {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    run()
