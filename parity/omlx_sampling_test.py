"""Gate: oMLX-shim sampling controls + hard stops — model-free, ~0 GB.

Answers "do the generation kwargs actually apply, so the model can't loop forever?" The sampling /
stop loop is shared by both model classes (only the decode stepper differs), so exercising it once
covers Kimi and Nemotron. Checks:
  (a) ``_apply_penalties`` math — repetition (CTRL/HF multiplicative), frequency (count-scaled
      subtraction), presence (flat subtraction), and the no-op fast paths;
  (b) end-to-end through the engine: an eos-less degenerate runtime is bounded by ``max_tokens``
      (a loop can never run unbounded), and ``frequency_penalty`` breaks a single-token loop.

    uv run python -m parity.omlx_sampling_test
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import mlx.core as mx

from quanta.shim.omlx import QuantaOmlxEngine, _apply_penalties

NEM_ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"
A, B, V = 50, 51, 200  # token A (logit 10) is greedily preferred over B (9); both non-eos


class _FixedRuntime:
    """Returns the SAME logits every call (A > B >> rest), never emits eos — a degenerate looper.
    Hybrid ``(logits, ssm, conv)`` signature + ``cfg.layers_block_type`` so the Nemotron stepper drives it."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(layers_block_type=["mamba"])
        self.num_layers = 1

    def __call__(self, token_ids, *, caches=None, ssm=None, conv=None, **kw):
        t = int(token_ids.shape[0])
        row = mx.full((V,), -50.0).at[A].add(60.0).at[B].add(59.0)  # logit[A]=10, logit[B]=9
        return mx.broadcast_to(row, (1, t, V)), [mx.zeros((1,))], [mx.zeros((1,))]


class _Tok:
    eos_id = 11
    stop_ids = (11,)

    def encode(self, text, *, add_bos=False, allow_special=False):
        return [1, 2, 3]

    def decode(self, ids, *, skip_special=True):
        return "".join("ab"[int(i) - A] if int(i) in (A, B) else "?" for i in ids)


async def _collect(eng, **kw):
    return [o async for o in eng.stream_generate("x", **kw)]


def run() -> None:
    ok = True
    base = mx.zeros((10,)) + 2.0  # every logit = +2
    prev = [3, 3, 5]              # token 3 emitted twice, token 5 once

    # (a) frequency: subtract count*freq (so t3 penalized 2x t5); unseen untouched
    f = _apply_penalties(base, prev, 1.0, 0.5, 0.0)
    good = (abs(float(f[3]) - 1.0) < 1e-5 and abs(float(f[5]) - 1.5) < 1e-5 and abs(float(f[0]) - 2.0) < 1e-5)
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] frequency(count-scaled): t3={float(f[3]):.2f} t5={float(f[5]):.2f} t0={float(f[0]):.2f}")

    # (a) presence: flat penalty for any seen token, independent of count
    p = _apply_penalties(base, prev, 1.0, 0.0, 0.7)
    good = (abs(float(p[3]) - 1.3) < 1e-5 and abs(float(p[5]) - 1.3) < 1e-5 and abs(float(p[0]) - 2.0) < 1e-5)
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] presence(flat): t3={float(p[3]):.2f} t5={float(p[5]):.2f} t0={float(p[0]):.2f}")

    # (a) repetition: multiplicative on seen (logit>0 -> /rep)
    r = _apply_penalties(base, prev, 2.0, 0.0, 0.0)
    good = abs(float(r[3]) - 1.0) < 1e-5 and abs(float(r[0]) - 2.0) < 1e-5
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] repetition(mult): t3={float(r[3]):.2f} t0={float(r[0]):.2f}")

    # (a) fast paths: empty prev OR all-neutral -> unchanged
    good = (bool(mx.all(_apply_penalties(base, [], 2.0, 0.5, 0.7) == base).item())
            and bool(mx.all(_apply_penalties(base, prev, 1.0, 0.0, 0.0) == base).item()))
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] no-op fast paths (empty prev / all-neutral) unchanged={good}")

    # (b) hard cap: an eos-less greedy loop is bounded by max_tokens (loops on A, stops at 'length')
    eng = QuantaOmlxEngine(NEM_ART, runtime=_FixedRuntime(), tokenizer=_Tok(), eos_token_ids={11})
    last = asyncio.run(_collect(eng, max_tokens=8, temperature=0.0))[-1]
    good = len(last.tokens) == 8 and last.finish_reason == "length" and all(t == A for t in last.tokens)
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] max_tokens bounds eos-less loop: n={len(last.tokens)} "
          f"finish={last.finish_reason!r} all_A={all(t == A for t in last.tokens)}")

    # (b) frequency_penalty breaks the single-token loop (B gets selected once A is penalized enough)
    eng2 = QuantaOmlxEngine(NEM_ART, runtime=_FixedRuntime(), tokenizer=_Tok(), eos_token_ids={11})
    last = asyncio.run(_collect(eng2, max_tokens=8, temperature=0.0, frequency_penalty=0.6))[-1]
    good = B in last.tokens and last.finish_reason == "length"
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] frequency_penalty breaks loop: tokens={last.tokens}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
