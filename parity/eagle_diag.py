"""Diagnostic: is the EAGLE drafter's ~4.5% holdout top-1 undertraining or a deeper bug?

Cheap (captured features + frozen embed/head only — no 389 GB model):
  (1) the trained drafter's TRAIN vs HOLDOUT step-1 top-1 — is it even fitting the training set?
  (2) a quick linear-probe ceiling (feat3 -> Linear -> RMSNorm -> the *frozen* target head): can the
      captured features be linearly decoded to the target's next token at all?

If the probe lands well above 4.5%, the signal is present and the drafter/training is the lever
(missing feature-regression loss, lr/epochs, data); if it's also ~4.5%, the feature/head setup is off.

    uv run python -m parity.eagle_diag
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from quanta.eagle.capture import load_features
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.train import _holdout_multistep, load_drafter, load_frozen_embed_head

ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int2g64"
FEATS = "/Users/pmrj/models/kimi_eagle/features.safetensors"
DRAFTERS = [("general", "/Users/pmrj/models/kimi_eagle/drafter_general.safetensors"),
            ("ttt", "/Users/pmrj/models/kimi_eagle/drafter_ttt.safetensors")]


def _mk(embed):
    return EagleDrafter(hidden=embed.shape[1], n_heads=56, head_dim=128, intermediate=14336,
                        rope_base=50000.0)


def run() -> None:
    d = load_features(FEATS)
    f3, it, tg = d["feat3"], d["in_tokens"], d["targets"]
    embed, head = load_frozen_embed_head(ART)
    mx.eval(embed, head)
    H, htf = embed.shape[1], head.T
    chunk = 2048
    nch = f3.shape[0] // chunk
    F = f3[:nch * chunk].reshape(nch, chunk, -1)
    T = tg[:nch * chunk].reshape(nch, chunk)
    tok_in = it[:nch * chunk].reshape(nch, chunk)
    tr, ho = slice(0, nch - 2), slice(nch - 2, nch)
    print(f"[diag] {nch} chunks ({nch * chunk} toks)  train={nch - 2} holdout=2  H={H}", flush=True)

    # (1) trained drafter: TRAIN vs HOLDOUT step-1 top-1
    for name, path in DRAFTERS:
        try:
            dr = load_drafter(path, _mk(embed))
            mx.eval(dr.parameters())
        except Exception as e:  # missing checkpoint — skip
            print(f"  drafter[{name}]: load failed ({type(e).__name__}: {e})", flush=True)
            continue
        a_tr = _holdout_multistep(dr, F[tr][:2], tok_in[tr][:2], T[tr][:2], embed, head, 1)[0]
        a_ho = _holdout_multistep(dr, F[ho], tok_in[ho], T[ho], embed, head, 1)[0]
        print(f"  drafter[{name:8s}] step-1 top1:  train={100 * a_tr:5.1f}%   holdout={100 * a_ho:5.1f}%",
              flush=True)

    # (2) linear-probe ceiling on the captured features (frozen head)
    class Probe(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.w = nn.Linear(3 * h, h, bias=False)
            self.n = nn.RMSNorm(h, eps=1e-6)

        def __call__(self, x):
            return self.n(self.w(x))

    probe = Probe(H)
    mx.eval(probe.parameters())
    opt = optim.Adam(learning_rate=1e-3)

    def loss_fn(p, fb, tb):
        lg = p(fb) @ htf
        return mx.mean(mx.logsumexp(lg, -1) - mx.take_along_axis(lg, tb[..., None], -1)[..., 0])

    lvg = nn.value_and_grad(probe, loss_fn)
    print("[diag] training linear probe (feat3 -> H -> frozen head) ...", flush=True)
    for step in range(400):
        i = int(mx.random.randint(0, nch - 2, [1]).item())
        loss, g = lvg(probe, F[i].astype(mx.float32), T[i])
        opt.update(probe, g)
        mx.eval(probe.parameters(), opt.state)
        if step % 100 == 99:
            pr = mx.argmax(probe(F[ho].astype(mx.float32)) @ htf, -1)
            acc = float(mx.mean((pr == T[ho]).astype(mx.float32)).item())
            print(f"  probe step {step + 1:3d}  loss {float(loss.item()):.3f}  holdout top1 {100 * acc:5.1f}%",
                  flush=True)
    print("[diag] done", flush=True)


if __name__ == "__main__":
    run()
