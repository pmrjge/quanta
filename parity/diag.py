"""One-forward diagnostic for the catastrophic-perplexity failure.

Runs the same teacher-forced setup as ``parity.ppl`` but, from a single forward,
also dumps the *signature* of the failure: per-position top-5 predictions
(decoded) vs the true next token, how many distinct argmax tokens occur (a
collapse detector), and logit/hidden-state stats incl. NaN/Inf. This tells us
whether the runtime is collapsing to one token, blowing up, or predicting
plausible-but-misaligned tokens — which the parity gate (runtime vs self-authored
reference) cannot reveal.

    uv run --with tiktoken python -m parity.diag           # full 61 layers
    uv run --with tiktoken python -m parity.diag 8 96      # n_layers, max_tokens
"""

from __future__ import annotations

import sys
from collections import Counter

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
    "electrons."
)


def run(n_layers: int | None = None, max_tokens: int = 96, show: int = 24) -> None:
    cfg = KimiTextConfig.from_pretrained(MODEL)
    model = KimiModel(cfg, SourceCheckpoint(MODEL), mx.bfloat16)
    tok = KimiTokenizer(MODEL, bos_id=cfg.bos_token_id)

    ids = tok.encode(PROSE, add_bos=True)[:max_tokens]
    arr = mx.array(ids)
    logits = model(arr, n_layers=n_layers)[0]  # [T, V]
    mx.eval(logits)

    n = cfg.num_hidden_layers if n_layers is None else n_layers
    lg = logits[:-1].astype(mx.float32)
    targets = arr[1:]
    ce = mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, targets[:, None], axis=-1)[:, 0]
    ppl = mx.exp(ce.mean()).item()
    argmax = mx.argmax(lg, axis=-1)
    acc = (argmax == targets).astype(mx.float32).mean().item()

    print(f"\n=== Diagnostic (bf16, layers={n}, tokens={len(ids)}) ===")
    print(f"perplexity            : {ppl:.3f}")
    print(f"top-1 next-token acc  : {acc:.3f}")

    # Health of the logits / final hidden.
    lf = logits.astype(mx.float32)
    print(
        f"logits  min/mean/max  : {mx.min(lf).item():.3f} / "
        f"{mx.mean(lf).item():.3f} / {mx.max(lf).item():.3f}"
    )
    print(f"any NaN / Inf         : {bool(mx.any(mx.isnan(lf)).item())} / "
          f"{bool(mx.any(mx.isinf(lf)).item())}")

    # Collapse detector: distinct argmax tokens across all predicted positions.
    am = argmax.tolist()
    cnt = Counter(am)
    print(f"distinct argmax tokens: {len(cnt)} / {len(am)} positions")
    top = cnt.most_common(3)
    print("most-common argmax    : " + ", ".join(
        f"{tok.decode([t])!r}×{c}" for t, c in top))

    # Per-position: prev token, true next, top-5 predicted (decoded + prob).
    probs = mx.softmax(lg, axis=-1)
    k = 5
    order = mx.argsort(-lg, axis=-1)[:, :k]
    print(f"\npos | prev -> TRUE next | top-{k} predicted (prob)")
    idl = ids
    for i in range(min(show, lg.shape[0])):
        prev = tok.decode([idl[i]])
        true_next = tok.decode([idl[i + 1]])
        preds = []
        for j in range(k):
            tid = int(order[i, j].item())
            p = float(probs[i, tid].item())
            preds.append(f"{tok.decode([tid])!r}={p:.3f}")
        hit = "OK" if int(argmax[i].item()) == idl[i + 1] else "  "
        print(f"{i:3d} | {prev!r} -> {true_next!r} {hit} | " + ", ".join(preds))


if __name__ == "__main__":
    args = sys.argv[1:]
    n = int(args[0]) if len(args) > 0 else None
    mt = int(args[1]) if len(args) > 1 else 96
    run(n, mt)
