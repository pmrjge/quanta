"""Lossless gate for EAGLE spec-decode (#51): spec_generate reproduces greedy decode.

Losslessness must hold for ANY drafter (the verify step guarantees it), so this validates the
draft/verify/rollback LOGIC cheaply: an 8-layer Kimi prefix + a RANDOM drafter (which yields ~0
accept) must reproduce greedy decode. Spec-decode verifies k+1 tokens in ONE batched forward, so at
a near-tie its argmax can differ from Sq=1 greedy decode by a bf16 ULP — inherent and accepted. So
the gate is: spec == greedy, OR the first divergence is a genuine near-tie (the greedy logit margin
between its token and spec's token is tiny), proving spec still tracks the model's own argmax.

    uv run python -m parity.eagle_spec_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.cache import MLACache
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.spec import spec_generate
from quanta.eagle.train import load_frozen_embed_head
from quanta.runtime import ResidentModel

ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int2g64"
NL = 8
LAYERS = (2, 4, 6)
MAXN = 24
TIE = 0.5  # bf16 near-tie logit margin


def _greedy_with_logits(model, prompt, max_new):
    caches = [MLACache(quantized=False) for _ in range(NL)]
    logits = model(mx.array(prompt), n_layers=NL, caches=caches, absorbed=False, sparse=None)
    mx.eval(logits, [c.c_kv for c in caches], [c.k_pe for c in caches])
    toks, logs = [], []
    last = logits[0, -1]
    cur = int(mx.argmax(last).item())
    for _ in range(max_new):
        toks.append(cur)
        logs.append(last)
        logits = model(mx.array([cur]), n_layers=NL, caches=caches, offset=caches[0].offset,
                       absorbed=False, sparse=None)
        mx.eval(logits, [c.c_kv for c in caches], [c.k_pe for c in caches])
        last = logits[0, -1]
        cur = int(mx.argmax(last).item())
    return toks, logs


def run() -> None:
    mx.set_wired_limit(int(120 * 1024**3))
    model = ResidentModel(ART, n_layers=NL)
    embed, head = load_frozen_embed_head(ART)
    mx.random.seed(0)
    drafter = EagleDrafter(hidden=model.cfg.hidden_size, n_heads=56, head_dim=128,
                           intermediate=14336, rope_base=50000.0)
    mx.eval(drafter.parameters())

    prompt = list(range(16))
    greedy, logs = _greedy_with_logits(model, prompt, MAXN)
    spec, stats = spec_generate(model, drafter, embed, head, prompt, max_new=MAXN, k=4,
                                layers=LAYERS, quantized_kv=False, sparse=None)

    nmatch = sum(a == b for a, b in zip(spec, greedy))
    print("=== EAGLE spec-decode lossless gate (8-layer prefix, random drafter) ===")
    print(f"  greedy[:12]: {greedy[:12]}")
    print(f"  spec  [:12]: {spec[:12]}")
    print(f"  exact match: {nmatch}/{min(len(spec), len(greedy))} | mean_accept={stats['mean_accept']:.2f}")

    ok = spec == greedy
    if not ok:
        i = next(d for d in range(min(len(spec), len(greedy))) if spec[d] != greedy[d])
        gap = float((logs[i][greedy[i]] - logs[i][spec[i]]).item())
        tie = gap < TIE
        ok = tie
        print(f"  first divergence @ {i}: greedy={greedy[i]} spec={spec[i]} | greedy logit margin={gap:.4f} "
              f"-> {'bf16 near-tie (lossless)' if tie else 'REAL DIVERGENCE'}")
    print("PASS (lossless)" if ok else "FAIL")


if __name__ == "__main__":
    run()
