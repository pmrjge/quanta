"""Gate: GLM native-MTP spec-decode is lossless (== greedy) — model-free, tiny tensors, ~0 GB.

Drives :func:`quanta.glm.spec.spec_generate` with a STUB main model + STUB MTP (fixed logits, like the
DSV4 / EAGLE fake-runtime tests). The stub model's greedy step is ``g(t) = (t+3) mod V`` independent of
position, so the plain-greedy stream is an arithmetic chain; a *perfect* MTP predicts ``g(cur)`` (every
draft accepted → mean_accept 2, ~half the main forwards), a *wrong* MTP never matches (mean_accept 1).
Verifies the emitted stream is **bit-identical to greedy** in all cases (incl. eos stop), proving the
verify/accept/rollback is lossless (rule 4). No model is loaded.

    uv run --with numpy python -m parity.glm_mtp_spec_test
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from quanta.glm.spec import spec_generate

V, STEP, D = 64, 3, 4
LAST = 1  # cfg.num_hidden_layers == 2 -> last layer index 1
EMBED = mx.zeros((V, D))
HEAD = mx.zeros((V, D))


def g(t: int) -> int:
    return (t + STEP) % V


def _row(tok: int) -> mx.array:
    return (mx.arange(V) == tok).astype(mx.float32) * 60.0 - 30.0


class _StubCache:
    def __init__(self) -> None:
        self._len = 0

    @property
    def offset(self) -> int:
        return self._len

    def truncate(self, length: int) -> None:
        self._len = length


class _Model:
    """Greedy step == g(input token), position-independent; returns the final-layer feature on capture."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(num_hidden_layers=2)
        self.num_layers = 2
        self.forwards = 0

    def make_caches(self) -> _StubCache:
        return _StubCache()

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        self.forwards += 1
        flat = [int(t) for t in mx.array(token_ids).reshape(-1).tolist()]
        logits = mx.stack([_row(g(t)) for t in flat])[None]  # [1,T,V]
        if capture_layers:
            caps = {layer: mx.stack([mx.full((D,), float(t)) for t in flat]) for layer in capture_layers}
            return logits, caps
        return logits


def _mtp(perfect: bool):
    def mtp(prev_hidden, next_ids, embed, head):
        cur = int(mx.array(next_ids).reshape(-1)[0].item())
        pred = g(cur) if perfect else (g(cur) + 1) % V
        return _row(pred)[None, None]  # [1,1,V]
    return mtp


def ref_greedy(prompt, max_new: int, eos=None) -> list[int]:
    out = [g(prompt[-1])]
    while len(out) < max_new:
        out.append(g(out[-1]))
    out = out[:max_new]
    if eos is not None:
        for k, t in enumerate(out):
            if t == eos:
                return out[: k + 1]
    return out


def run() -> None:
    ok = True
    PROMPT, NEW = [7], 16

    # (1) perfect MTP: spec == greedy, every draft accepted (mean_accept 2)
    m = _Model()
    toks, st = spec_generate(m, _mtp(True), EMBED, HEAD, PROMPT, max_new=NEW)
    gold = ref_greedy(PROMPT, NEW)
    good = toks == gold and st["mean_accept"] == 2.0 and st["rounds"] == 8
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] perfect MTP: spec==greedy={toks == gold} mean_accept={st['mean_accept']} rounds={st['rounds']}")
    print(f"             greedy[:8]={gold[:8]}")
    print(f"             spec  [:8]={toks[:8]}")

    # (2) perfect MTP makes fewer main forwards than tokens emitted (the speedup)
    good = m.forwards < len(toks)
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] fewer forwards than tokens: forwards={m.forwards} tokens={len(toks)}")

    # (3) wrong MTP: still == greedy, no draft ever accepted (mean_accept 1)
    toks_w, st_w = spec_generate(_Model(), _mtp(False), EMBED, HEAD, PROMPT, max_new=NEW)
    good = toks_w == gold and abs(st_w["mean_accept"] - 1.0) < 1e-9
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] wrong MTP: spec==greedy={toks_w == gold} mean_accept={st_w['mean_accept']:.2f}")

    # (4) eos stops, inclusive, matching greedy's eos stop
    EOS = 22
    toks_e, _ = spec_generate(_Model(), _mtp(True), EMBED, HEAD, PROMPT, max_new=NEW, eos_id=EOS)
    gold_e = ref_greedy(PROMPT, NEW, eos=EOS)
    good = toks_e == gold_e and toks_e[-1] == EOS
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] eos stops: spec==greedy={toks_e == gold_e} ends_with_eos={toks_e[-1] == EOS} spec={toks_e}")

    # (5) wrong MTP + eos (as a set) still == greedy
    toks_we, _ = spec_generate(_Model(), _mtp(False), EMBED, HEAD, PROMPT, max_new=NEW, eos_id={EOS})
    good = toks_we == gold_e
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] wrong MTP + eos set: spec==greedy={toks_we == gold_e}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
