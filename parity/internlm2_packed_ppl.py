"""e2e int8 ppl gate for InternLM2.5-7B-Chat-1M: packed runtime vs the bf16 source reference.

The end-to-end quant arbiter (CLAUDE.md methodology #4): teacher-forced ppl of the int8-packed
resident runtime must track the bf16 source within a small delta on the same prose — int8 affine
RTN is ~lossless, so a large gap means a bake/decode bug, not "quantization is hard". Compares on a
fluent-prose passage (realistic ppl) and a repetition passage (the forward-soundness probe — both
paths must nail it near ppl 1).

Loads the bf16 source (streamed, one layer resident) AND the int8 artifact (packed). Heavy — run in
a GPU/memory-available session:

    uv run --with numpy python -m parity.internlm2_packed_ppl
"""

from __future__ import annotations

import mlx.core as mx

from quanta.internlm2.config import InternLM2Config
from quanta.internlm2.loader import InternLM2SourceCheckpoint
from quanta.internlm2.model import internlm2_logits
from quanta.internlm2.runtime import InternLM2ResidentModel
from quanta.internlm2.tokenizer import InternLM2Tokenizer

SOURCE = "/Users/pmrj/models/internlm2_5-7b-chat-1m"
ARTIFACT = "/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64"

PASSAGES: dict[str, str] = {
    "repeat": ("The quick brown fox jumps over the lazy dog. " * 8).strip(),
    "prose": (
        "Good engineering begins with a clear understanding of the problem at hand. Before writing "
        "a single line of code, a careful developer studies how the existing system behaves, "
        "identifies the precise point at which it fails, and considers the smallest change that "
        "would restore correct behavior. This discipline saves time in the long run, because broad "
        "rewrites tend to introduce new defects faster than they remove old ones."
    ),
}


def _ppl_top1(logits: mx.array, ids: mx.array) -> tuple[float, float]:
    logits = logits.astype(mx.float32)[0]
    tgt = ids[0, 1:]
    lse = mx.logsumexp(logits[:-1], axis=-1)
    tok = mx.take_along_axis(logits[:-1], tgt[:, None], axis=-1)[:, 0]
    ppl = float(mx.exp(mx.mean(lse - tok)).item())
    top1 = float(mx.mean((mx.argmax(logits[:-1], -1) == tgt).astype(mx.float32)).item())
    return ppl, top1


def run() -> None:
    cfg = InternLM2Config.from_pretrained(SOURCE)
    tok = InternLM2Tokenizer.from_pretrained(SOURCE)
    enc = {n: mx.array([tok.encode(t, add_bos=True)]) for n, t in PASSAGES.items()}

    # int8 packed resident runtime (the serving path)
    packed = InternLM2ResidentModel(ARTIFACT, packed=True)
    int8 = {n: _ppl_top1(packed(ids), ids) for n, ids in enc.items()}
    del packed
    mx.clear_cache()

    # bf16 source reference (streamed)
    bf16: dict[str, tuple[float, float]] = {}
    for n, ids in enc.items():
        ck = InternLM2SourceCheckpoint(SOURCE, cfg)
        bf16[n] = _ppl_top1(internlm2_logits(ck, ids, cfg), ids)

    print(f"{'passage':8s}  {'bf16 ppl':>9s}  {'int8 ppl':>9s}  {'Δppl%':>7s}  "
          f"{'bf16 t1':>7s}  {'int8 t1':>7s}")
    ok = True
    for n in PASSAGES:
        bp, bt = bf16[n]
        ip, it = int8[n]
        dpct = 100.0 * (ip - bp) / bp
        print(f"{n:8s}  {bp:9.4f}  {ip:9.4f}  {dpct:+6.1f}%  {bt*100:6.1f}%  {it*100:6.1f}%")
        ok = ok and abs(dpct) < 5.0          # int8 ~lossless: within 5% of bf16 ppl
    rep_ok = int8["repeat"][0] < 3.0          # int8 still nails repetition
    print(f"\n  int8 within 5% of bf16: {ok}   int8 repeat<3.0: {rep_ok}")
    print(f"\n{'PASS' if ok and rep_ok else 'FAIL'}")
    assert ok and rep_ok


if __name__ == "__main__":
    run()
