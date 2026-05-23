"""Teacher-forced perplexity gate — the end-to-end coherence arbiter.

Tokenizes real prose (BOS-led), runs the streamed runtime, and reports
teacher-forced perplexity + top-1 next-token accuracy. This is the metric the
prior project failed (~165 ppl on trivial text); a parity-correct bf16 runtime
should give low single digits on ordinary English. Later this same harness, with
int3-quantized experts, answers the int3-floor question.

    uv run --with tiktoken python -m parity.ppl          # full 61 layers
    uv run --with tiktoken python -m parity.ppl 8 160    # n_layers, max_tokens
"""

from __future__ import annotations

import sys

import mlx.core as mx

from quanta.config import KimiTextConfig
from quanta.loader import SourceCheckpoint
from quanta.model import KimiModel
from quanta.tokenizer import KimiTokenizer

MODEL = "/Users/pmrj/models/Kimi-K2.6"

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


def run(n_layers: int | None = None, max_tokens: int = 192) -> None:
    cfg = KimiTextConfig.from_pretrained(MODEL)
    model = KimiModel(cfg, SourceCheckpoint(MODEL), mx.bfloat16)
    tok = KimiTokenizer(MODEL, bos_id=cfg.bos_token_id)

    ids = tok.encode(PROSE, add_bos=True)[:max_tokens]
    arr = mx.array(ids)
    logits = model(arr, n_layers=n_layers)[0]  # [T, V]

    lg = logits[:-1].astype(mx.float32)  # predict positions 0..T-2
    targets = arr[1:]
    ce = mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, targets[:, None], axis=-1)[:, 0]
    ppl = mx.exp(ce.mean()).item()
    acc = (mx.argmax(lg, axis=-1) == targets).astype(mx.float32).mean().item()

    n = cfg.num_hidden_layers if n_layers is None else n_layers
    print(f"\n=== Teacher-forced perplexity (bf16, layers={n}, tokens={len(ids)}) ===")
    print(f"perplexity            : {ppl:.3f}")
    print(f"top-1 next-token acc  : {acc:.3f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    n = int(args[0]) if len(args) > 0 else None
    mt = int(args[1]) if len(args) > 1 else 192
    run(n, mt)
