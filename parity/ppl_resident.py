"""The int3-floor: teacher-forced perplexity on the resident quantized model vs bf16 3.31.

Loads the baked artifact RAM-resident (int3/int4 experts + int8 non-experts) and runs the
same teacher-forced ppl as parity.ppl. sparse=None for a pure quantization measurement (no
sparse-prefill approximation). This is the project's central answer: does int3 GPTQ preserve
coherence? Run after run_bake completes.

    uv run --with tiktoken python -m parity.ppl_resident
"""

from __future__ import annotations

import mlx.core as mx

from quanta.runtime import ResidentModel
from quanta.tokenizer import KimiTokenizer

ARTIFACT = "/Users/pmrj/models/Kimi-K2.6-quanta_int3"
PROSE = (
    "Photosynthesis is the process by which green plants, algae, and some bacteria convert "
    "light energy into chemical energy stored in sugars. Inside the chloroplasts, chlorophyll "
    "absorbs sunlight, which drives the splitting of water molecules into oxygen, protons, and "
    "electrons. The oxygen is released into the atmosphere as a byproduct, while the energy "
    "captured is used to fix carbon dioxide from the air into glucose. This remarkable reaction "
    "sustains nearly all life on Earth, forming the base of the food chain and regulating the "
    "balance of oxygen and carbon dioxide in the atmosphere. Without photosynthesis, the planet "
    "would be unable to support the diversity of organisms that depend, directly or indirectly, "
    "on plants for food and breathable air."
)


def run() -> None:
    mx.set_wired_limit(490 * 1024**3)  # pin the resident weight set
    rm = ResidentModel(ARTIFACT)  # all 61 layers resident (~427 GB)
    tok = KimiTokenizer(ARTIFACT, bos_id=rm.cfg.bos_token_id)
    ids = tok.encode(PROSE, add_bos=True)[:192]
    arr = mx.array(ids)
    logits = rm(arr, sparse=None)[0]  # pure quant (no sparse) for the int3-floor

    lg = logits[:-1].astype(mx.float32)
    targets = arr[1:]
    ce = mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, targets[:, None], axis=-1)[:, 0]
    ppl = mx.exp(ce.mean()).item()
    acc = (mx.argmax(lg, axis=-1) == targets).astype(mx.float32).mean().item()
    print(f"\n=== int3-floor (resident int3/int4 + int8, tokens={len(ids)}) ===")
    print(f"perplexity           : {ppl:.3f}   (bf16 reference: 3.31)")
    print(f"top-1 next-token acc : {acc:.3f}")


if __name__ == "__main__":
    run()
