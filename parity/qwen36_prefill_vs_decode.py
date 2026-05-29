"""Discriminate prefill-vs-decode + measure teacher-forced top-1 on the baked Qwen3.6 reference.

generate() was garbage, but its residual stream (prefill capture) looked healthy. generate() drives
the DECODE path (single-token stepping + KV/recurrent state), NOT prefill. This probe separates them
on real prose (one model load):

  (A) PREFILL teacher-forced top-1: feed a real text once (caches=None), predict token i+1 from the
      prefix; agreement with the actual next token. High ⇒ the shared forward MATH is correct.
      ~0 ⇒ the forward itself is broken (not a decode-only bug).
  (B) DECODE path: step the SAME text one token at a time through the cached decode path; compare its
      per-position top-1 to the prefill top-1. Divergence ⇒ a decode-only bug (recurrent-state seeding
      / KV cache / yarn_seq), exonerating the shared math.

Real prose with the correct (no-)BOS. ~65 GB resident — run SOLO.

    uv run python -u -m parity.qwen36_prefill_vs_decode
"""

from __future__ import annotations

import mlx.core as mx

from quanta.qwen35.runtime import Qwen35ResidentModel
from quanta.qwen35.tokenizer import Qwen35Tokenizer

ART = "/Users/pmrj/models/Qwen3.6-35B-A3B-quanta_int4g64"
TEXT = ("The quick brown fox jumps over the lazy dog. The capital of France is Paris, and the "
        "capital of Japan is Tokyo. Water is made of hydrogen and oxygen.")


def run() -> None:
    mx.set_wired_limit(int(120 * 1024 ** 3))
    model = Qwen35ResidentModel(ART)
    tok = Qwen35Tokenizer.from_pretrained(ART)
    ids = tok.encode(TEXT, add_bos=False)
    T = len(ids)
    print(f"loaded; text={T} tokens, resident≈{mx.get_active_memory()/1024**3:.1f} GiB", flush=True)

    # (A) PREFILL teacher-forced top-1
    lg_pf = model(mx.array(ids))                      # [1,T,vocab]
    mx.eval(lg_pf)
    pred_pf = mx.argmax(lg_pf[0, :-1], axis=-1)       # predict ids[1:]
    tgt = mx.array(ids[1:])
    agree_pf = float(mx.mean((pred_pf == tgt).astype(mx.float32)).item())
    print(f"\n(A) PREFILL teacher-forced top-1 agreement = {agree_pf:.3f}  ({int(agree_pf*(T-1))}/{T-1})",
          flush=True)
    # show a few (prefix -> predicted vs actual)
    pp = pred_pf.tolist()
    for i in (0, 3, 6, 9, 12):
        if i < T - 1:
            print(f"    ctx={tok.decode(ids[:i+1])!r:50.50}  pred={tok.decode([pp[i]])!r:12}  "
                  f"actual={tok.decode([ids[i+1]])!r}", flush=True)

    # (B) DECODE path: step the same ids one at a time, collect per-position top-1
    cache = model.make_caches()
    pred_dec = []
    for pos, tid in enumerate(ids[:-1]):
        lg = model(mx.array([tid]), caches=cache, offset=pos)
        mx.eval(lg)
        pred_dec.append(int(mx.argmax(lg[0, -1]).item()))
    agree_dec = sum(int(p == t) for p, t in zip(pred_dec, ids[1:])) / (T - 1)
    match_pf = sum(int(p == q) for p, q in zip(pred_dec, pp)) / (T - 1)
    print(f"\n(B) DECODE teacher-forced top-1 agreement = {agree_dec:.3f}", flush=True)
    print(f"    DECODE vs PREFILL top-1 match           = {match_pf:.3f}  "
          f"(1.0 ⇒ decode==prefill; <1 ⇒ decode-only bug)", flush=True)


if __name__ == "__main__":
    run()
