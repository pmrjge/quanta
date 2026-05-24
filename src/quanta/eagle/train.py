"""EAGLE-3 drafter training (MLX) — fit the drafter to the quantized target's next-token predictions.

Loads captured ``(feat3, in_tokens, targets)`` and the target's **frozen** embedding + LM head
(dequantized from the int2g64 artifact, so the drafter predicts in the target's logit space). Trains
the drafter single-step over fixed-length chunks with causal self-attention (each position: reduce
target feature → combine with frozen embedding → 1 layer → frozen head → CE vs the target's argmax
next token). Reports train loss and held-out top-1 accept (drafter argmax == target's next token) —
the single-step proxy for accept-length; the multi-step "training-time-test" rollout is the quality
follow-up. Embedding/head are not module params, so the optimizer touches only the drafter.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from quanta.eagle.drafter import EagleDrafter
from quanta.runtime import EMBED_KEY, LM_HEAD_KEY, ResidentArtifact


def load_frozen_embed_head(art_dir: str | Path) -> tuple[mx.array, mx.array]:
    """Dequantize the artifact's int8 embedding + LM head to bf16 ``[V,H]`` (frozen, shared)."""
    art = ResidentArtifact(art_dir)

    def deq(key: str) -> mx.array:
        m = art.manifest[key]
        return mx.dequantize(art.get(f"{key}.weight_packed"), art.get(f"{key}.weight_scale"),
                             art.get(f"{key}.weight_bias"), group_size=m["group_size"], bits=m["bits"])

    embed, head = deq(EMBED_KEY).astype(mx.bfloat16), deq(LM_HEAD_KEY).astype(mx.bfloat16)
    mx.eval(embed, head)
    art.release()
    return embed, head


def _ce(drafter: EagleDrafter, f3b: mx.array, itb: mx.array, tgb: mx.array,
        embed: mx.array, head: mx.array) -> mx.array:
    r = drafter.reduce_target_features(f3b)
    _, normed = drafter.step(r, embed[itb], offset=0, mask="causal")
    logits = normed @ head.T  # [B,T,V]
    lse = mx.logsumexp(logits, axis=-1)
    tgt = mx.take_along_axis(logits, tgb[..., None], axis=-1)[..., 0]
    return mx.mean(lse - tgt)


def _holdout_accept(drafter: EagleDrafter, f3: mx.array, it: mx.array, tg: mx.array,
                    embed: mx.array, head: mx.array) -> float:
    r = drafter.reduce_target_features(f3)
    _, normed = drafter.step(r, embed[it], offset=0, mask="causal")
    pred = mx.argmax(normed @ head.T, axis=-1)
    return float(mx.mean((pred == tg).astype(mx.float32)).item())


def _ce_multistep(drafter: EagleDrafter, f3b: mx.array, itb: mx.array, tgb: mx.array,
                  embed: mx.array, head: mx.array, steps: int) -> mx.array:
    """Multi-step "training-time test" CE. Step 1 uses the real reduced target feature; steps 2..S
    **self-feed** the drafter's own ``x`` as the next feature (teacher-forced token embeddings +
    labels), so ``x`` is trained to be a valid recurrent feature — the regime spec-decode runs in.
    ``steps==1`` is exactly the single-step loss. Each step shrinks the valid length by one."""
    feat = drafter.reduce_target_features(f3b)   # [B,T,H]
    emb = embed[itb]                             # [B,T,H]
    t = feat.shape[1]
    total = mx.array(0.0)
    for s in range(steps):
        ts = t - s
        x_, normed = drafter.step(feat[:, :ts], emb[:, s:s + ts], offset=0, mask="causal")
        logits = normed @ head.T
        tgt = tgb[:, s:s + ts]
        total = total + mx.mean(mx.logsumexp(logits, axis=-1)
                                - mx.take_along_axis(logits, tgt[..., None], axis=-1)[..., 0])
        feat = x_                                # self-feed for the next step
    return total / steps


def _holdout_multistep(drafter: EagleDrafter, f3: mx.array, it: mx.array, tg: mx.array,
                       embed: mx.array, head: mx.array, steps: int) -> tuple[float, ...]:
    """Per-step held-out top-1 accept (step 1 = single-step accuracy; steps 2..S = self-fed)."""
    feat = drafter.reduce_target_features(f3)
    emb = embed[it]
    t = feat.shape[1]
    accs = []
    for s in range(steps):
        ts = t - s
        x_, normed = drafter.step(feat[:, :ts], emb[:, s:s + ts], offset=0, mask="causal")
        pred = mx.argmax(normed @ head.T, axis=-1)
        accs.append(float(mx.mean((pred == tg[:, s:s + ts]).astype(mx.float32)).item()))
        feat = x_
    return tuple(accs)


def train_drafter(drafter: EagleDrafter, feat3: mx.array, in_tokens: mx.array, targets: mx.array,
                  embed: mx.array, head: mx.array, *, chunk: int = 2048, batch: int = 2,
                  epochs: int = 60, lr: float = 2e-4, holdout: int = 2, steps: int = 1) -> dict:
    """Train over chunks; return final loss + per-step held-out top-1 accept. ``steps>1`` runs the
    multi-step "training-time test" (self-fed features) — required for spec-decode (steps 2..S are
    the self-fed accept that single-step training leaves at ~0)."""
    nch = feat3.shape[0] // chunk
    f3 = feat3[:nch * chunk].reshape(nch, chunk, -1)
    it = in_tokens[:nch * chunk].reshape(nch, chunk)
    tg = targets[:nch * chunk].reshape(nch, chunk)
    n_train = nch - holdout
    assert n_train >= batch, f"need >= {batch} train chunks, have {n_train} (corpus too small)"
    hf3, hit, htg = f3[n_train:], it[n_train:], tg[n_train:]

    opt = optim.Adam(learning_rate=lr)
    lvg = nn.value_and_grad(drafter, lambda d, a, b, c: _ce_multistep(d, a, b, c, embed, head, steps))
    base = _holdout_multistep(drafter, hf3, hit, htg, embed, head, steps)
    hist = []
    last_loss = 0.0
    for ep in range(epochs):
        order = [int(i) for i in mx.random.permutation(n_train)]
        for s in range(0, n_train - batch + 1, batch):
            idx = mx.array(order[s:s + batch])
            loss, grads = lvg(drafter, f3[idx], it[idx], tg[idx])
            opt.update(drafter, grads)
            mx.eval(drafter.parameters(), opt.state)
            last_loss = float(loss.item())
        if ep % 10 == 9 or ep == epochs - 1:
            acc = _holdout_multistep(drafter, hf3, hit, htg, embed, head, steps)
            hist.append((ep + 1, last_loss, acc))
            accs = " ".join(f"{a:.3f}" for a in acc)
            print(f"  epoch {ep + 1:3d}  loss {last_loss:.4f}  holdout top1/step [{accs}]", flush=True)
    final = hist[-1][2] if hist else base
    return {"base_holdout": base, "final_holdout": final, "final_loss": last_loss, "history": hist,
            "base_holdout_top1": base[0], "final_holdout_top1": final[0]}


def save_drafter(path: str | Path, drafter: EagleDrafter) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), dict(tree_flatten(drafter.parameters())))


def load_drafter(path: str | Path, drafter: EagleDrafter) -> EagleDrafter:
    drafter.update(tree_unflatten(list(mx.load(str(path)).items())))
    return drafter
