"""REAL bf16 teacher-forced perplexity gate for InternLM2.5-7B-Chat-1M (the e2e correctness arbiter).

Streams the real ~7B bf16 source one layer at a time (rule-8) and scores a clean-prose passage with
the correct BOS (=1, ``<s>``). This is the foundational parity gate: a parity-correct forward gives
**low single-digit** ppl; a localized forward bug yields catastrophic ppl (the Kimi lesson — ~165
with the offset-binary dequant bug vs ~3.3 fixed). Secondary signal: top-1 next-token agreement (the
fraction of positions whose argmax logit is the actual next token), which sits well above chance for
a healthy 7B on fluent English.

Heavy — needs the checkpoint + memory. NOT model-free; run only in a GPU/memory-available session:

    uv run --extra reference --with numpy python -m parity.internlm2_bf16_ppl
"""

from __future__ import annotations

import mlx.core as mx

from quanta.internlm2.config import InternLM2Config
from quanta.internlm2.loader import InternLM2SourceCheckpoint
from quanta.internlm2.model import internlm2_logits
from quanta.internlm2.tokenizer import InternLM2Tokenizer

SOURCE = "/Users/pmrj/models/internlm2_5-7b-chat-1m"

# Three passages spanning the predictability spectrum (all self-authored / public structure, no
# copyright). A parity-correct forward must (a) nail the highly-repetitive text at very low ppl
# (CLAUDE.md bug signature: a broken forward "wrecks even literal repetition/counting"), and
# (b) score fluent prose in the low single-to-low-double digits. A scrambled-head or wrong-scale
# forward fails (a) outright.
PASSAGES: dict[str, str] = {
    # (a) repetition — the strongest forward-soundness probe; after one cycle the model should
    #     predict the repeated tokens almost perfectly → ppl near 1.
    "repeat": ("The quick brown fox jumps over the lazy dog. " * 8).strip(),
    # (b) counting — structured, highly predictable continuation.
    "count": "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30",
    # (c) fluent original expository prose — the realistic ppl point.
    "prose": (
        "Good engineering begins with a clear understanding of the problem at hand. Before writing "
        "a single line of code, a careful developer studies how the existing system behaves, "
        "identifies the precise point at which it fails, and considers the smallest change that "
        "would restore correct behavior. This discipline saves time in the long run, because broad "
        "rewrites tend to introduce new defects faster than they remove old ones. When a fix is "
        "ready, it should be tested first against the narrow case that motivated it, and then "
        "against the wider suite of behaviors that the surrounding code depends upon."
    ),
}


def _score(ck, ids: mx.array, cfg: InternLM2Config) -> tuple[float, float]:
    """(ppl, top-1 agreement) for ``ids`` ``[1,S]`` — one streamed bf16 forward."""
    logits = internlm2_logits(ck, ids, cfg).astype(mx.float32)[0]      # [S, vocab]
    tgt = ids[0, 1:]
    lse = mx.logsumexp(logits[:-1], axis=-1)
    tokv = mx.take_along_axis(logits[:-1], tgt[:, None], axis=-1)[:, 0]
    ppl = float(mx.exp(mx.mean(lse - tokv)).item())
    pred = mx.argmax(logits[:-1], axis=-1)
    top1 = float(mx.mean((pred == tgt).astype(mx.float32)).item())
    return ppl, top1


def run() -> None:
    cfg = InternLM2Config.from_pretrained(SOURCE)
    tok = InternLM2Tokenizer.from_pretrained(SOURCE)
    print(f"source={SOURCE}  layers={cfg.num_hidden_layers}\n")

    results: dict[str, tuple[int, float, float]] = {}
    for name, text in PASSAGES.items():
        ids_list = tok.encode(text, add_bos=True)
        ids = mx.array([ids_list])
        ck = InternLM2SourceCheckpoint(SOURCE, cfg)        # fresh streamed reader per passage
        ppl, top1 = _score(ck, ids, cfg)
        results[name] = (ids.shape[1], ppl, top1)
        print(f"  {name:7s}  tokens={ids.shape[1]:3d}  ppl={ppl:8.4f}  top-1={top1*100:5.1f}%")

    # Soundness gates: repetition/counting must be very low ppl (forward not scrambled);
    # fluent prose in a sane range.
    rep_ok = results["repeat"][1] < 3.0
    count_ok = results["count"][1] < 4.0
    prose_ok = results["prose"][1] < 15.0
    healthy = rep_ok and count_ok and prose_ok
    print(f"\n  repeat<3.0: {rep_ok}   count<4.0: {count_ok}   prose<15: {prose_ok}")
    print(f"\n{'PASS' if healthy else 'FAIL'}  (forward parity-correct iff repetition/counting are low ppl)")


if __name__ == "__main__":
    run()
