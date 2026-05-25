"""EAGLE-3 drafter training (MLX) — fit the drafter to the quantized target's next feature + token.

Loads captured ``(feat3, in_tokens, targets)`` and the target's **frozen** embedding + LM head
(dequantized from the int2g64 artifact, so the drafter predicts in the target's logit space).

Canonical EAGLE-3 loss (:func:`_ce_multistep`): at position ``p`` the drafter consumes the previous
feature ``f_{p-1}`` and the next token embedding ``e_p``, and is trained to (a) predict ``token_{p+1}``
via CE through the frozen head on its **head-space** output ``normed_p`` **and** (b) regress its
*separate* **feature-space** output ``recur_p`` onto the next reduced target feature ``f_p`` (smooth-L1,
stop-grad). ``recur`` (not ``normed``) is what ``steps>1`` self-feeds for the multi-step "training-time
test" rollout spec-decode runs in, so the recurrence stays in the reduced-feature space. (Two earlier
bugs both collapsed self-fed accept: first, **no** feature term at all — the recurrent feature was
untrained (~0 / 0.39x); then a feature term that regressed/self-fed the **head-space** output ``normed``
onto the **feature-space** ``f_p`` — incompatible bases, so the regression fought CE and pinned accept
at ~3%. A linear probe on the same features beat the drafter both times: training bugs, not the data.)

Embedding/head are not module params, so the optimizer touches only the drafter.
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


def _smooth_l1(pred: mx.array, target: mx.array, beta: float = 1.0) -> mx.array:
    """Mean smooth-L1 (Huber) — EAGLE's feature-regression loss between predicted and target feature."""
    d = mx.abs(pred - target)
    return mx.mean(mx.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta))


def _ce_multistep(drafter: EagleDrafter, f3b: mx.array, itb: mx.array, tgb: mx.array,
                  embed: mx.array, head: mx.array, steps: int, feat_w: float) -> mx.array:
    """Canonical EAGLE-3 multi-step loss: **decoupled** head-space CE + feature-space regression.

    At position ``p`` the drafter consumes the *previous* feature ``f_{p-1}`` (the real reduced target
    feature on step 1, its own ``recur`` thereafter) and the *next* token embedding ``e_p``, and is
    trained to (a) predict ``token_{p+1}`` through the frozen head (CE on the head-space output
    ``normed``) **and** (b) regress its separate feature-space output ``recur_p`` onto the next reduced
    target feature ``f_p`` (smooth-L1, stop-grad). ``recur`` lives in the reduced-feature space, so it
    can match ``f_p`` *and* be self-fed without disturbing ``normed``'s head space — the two regimes are
    decoupled. ``steps>1`` self-feeds ``recur`` for the multi-step "training-time test" rollout
    spec-decode runs in. Each step shifts the embedding/labels by one and shrinks the length by one.
    ``feat_w`` weights the regression vs the CE."""
    red = drafter.reduce_target_features(f3b)    # [B,T,H] reduced target features (grad flows via input)
    emb = embed[itb]                             # [B,T,H]
    t = red.shape[1]
    feat = red                                   # step-1 feature source; self-fed `normed` afterwards
    ce_sum = mx.array(0.0)
    reg_sum = mx.array(0.0)
    n = 0
    for s in range(steps):
        ts = t - 1 - s
        if ts <= 0:
            break
        # `recur` (feature space) is the recurrent output we regress/self-feed; `normed` (head space) is
        # the CE output. Decoupling them is the fix — a single shared output forced into both bases made
        # the regression fight CE and pinned accept ~3%.
        recur, normed = drafter.step(feat[:, :ts], emb[:, s + 1:s + 1 + ts], offset=0, mask="causal")
        logits = normed @ head.T
        tgt = tgb[:, s + 1:s + 1 + ts]
        ce_sum = ce_sum + mx.mean(mx.logsumexp(logits, axis=-1)
                                  - mx.take_along_axis(logits, tgt[..., None], axis=-1)[..., 0])
        # scale-invariant feature regression: RMS-normalize recur + target by the target's per-token
        # RMS so the smooth-L1 is O(1) and isn't dominated by the large raw reduced features (absmax
        # ~760 once feat_norm is gone), which otherwise starves the CE and collapses token accuracy.
        tgt_feat = mx.stop_gradient(red[:, s + 1:s + 1 + ts])
        inv = mx.stop_gradient(mx.rsqrt(mx.mean(tgt_feat * tgt_feat, axis=-1, keepdims=True) + 1e-6))
        reg_sum = reg_sum + _smooth_l1(recur * inv, tgt_feat * inv)
        feat = recur                             # self-feed the feature-space recurrent output
        n += 1
    n = max(n, 1)
    return (ce_sum + feat_w * reg_sum) / n


def _holdout_multistep(drafter: EagleDrafter, f3: mx.array, it: mx.array, tg: mx.array,
                       embed: mx.array, head: mx.array, steps: int) -> tuple[float, ...]:
    """Per-step held-out top-1 accept under the canonical alignment (matches :func:`_ce_multistep`):
    step ``s`` predicts ``token_{p+1}`` from ``(f_{p-1}, e_p)`` over the positions still valid after
    ``s`` self-feeds."""
    red = drafter.reduce_target_features(f3)
    emb = embed[it]
    t = red.shape[1]
    feat = red
    accs = []
    for s in range(steps):
        ts = t - 1 - s
        if ts <= 0:
            break
        recur, normed = drafter.step(feat[:, :ts], emb[:, s + 1:s + 1 + ts], offset=0, mask="causal")
        pred = mx.argmax(normed @ head.T, axis=-1)
        accs.append(float(mx.mean((pred == tg[:, s + 1:s + 1 + ts]).astype(mx.float32)).item()))
        feat = recur
    return tuple(accs)


def train_drafter(drafter: EagleDrafter, feat3: mx.array, in_tokens: mx.array, targets: mx.array,
                  embed: mx.array, head: mx.array, *, chunk: int = 2048, batch: int = 2,
                  epochs: int = 60, lr: float = 2e-4, holdout: int = 2, steps: int = 1,
                  feat_w: float = 1.0, patience: int = 8, save_path: str | Path | None = None) -> dict:
    """Train over chunks with **early stopping + best-epoch restore**; return the final loss + the
    *best* per-step held-out top-1 accept. Uses the canonical EAGLE-3 loss (next-token CE +
    ``feat_w``-weighted next-feature regression, :func:`_ce_multistep`); ``steps>1`` self-feeds the
    predicted feature for the multi-step "training-time test" rollout spec-decode runs in. Set
    ``feat_w=0`` to ablate the feature regression.

    The holdout is evaluated **every epoch** and scored by mean per-step accept; the params are
    snapshot (and, if ``save_path`` is given, checkpointed to disk) whenever that score improves;
    training stops after ``patience`` epochs without improvement; the **best** params are restored
    before returning. ``epochs`` is the cap, not a fixed count."""
    nch = feat3.shape[0] // chunk
    f3 = feat3[:nch * chunk].reshape(nch, chunk, -1)
    it = in_tokens[:nch * chunk].reshape(nch, chunk)
    tg = targets[:nch * chunk].reshape(nch, chunk)
    n_train = nch - holdout
    assert n_train >= batch, f"need >= {batch} train chunks, have {n_train} (corpus too small)"
    hf3, hit, htg = f3[n_train:], it[n_train:], tg[n_train:]

    opt = optim.Adam(learning_rate=lr)
    lvg = nn.value_and_grad(drafter, lambda d, a, b, c: _ce_multistep(d, a, b, c, embed, head, steps, feat_w))
    base = _holdout_multistep(drafter, hf3, hit, htg, embed, head, steps)
    hist: list = []
    last_loss = 0.0
    best_score, best_acc, best_epoch, since = -1.0, base, 0, 0
    best_params = None
    for ep in range(epochs):
        order = [int(i) for i in mx.random.permutation(n_train)]
        for s in range(0, n_train - batch + 1, batch):
            idx = mx.array(order[s:s + batch])
            loss, grads = lvg(drafter, f3[idx], it[idx], tg[idx])
            opt.update(drafter, grads)
            mx.eval(drafter.parameters(), opt.state)
            last_loss = float(loss.item())
        acc = _holdout_multistep(drafter, hf3, hit, htg, embed, head, steps)
        score = sum(acc) / len(acc)
        hist.append((ep + 1, last_loss, acc))
        improved = score > best_score + 1e-4
        if improved:
            best_score, best_acc, best_epoch, since = score, acc, ep + 1, 0
            best_params = [(k, mx.array(v)) for k, v in tree_flatten(drafter.parameters())]
            mx.eval([v for _, v in best_params])  # materialize the snapshot off the live graph
            if save_path is not None:
                save_drafter(save_path, drafter)   # checkpoint the best epoch (crash-safe)
        else:
            since += 1
        accs = " ".join(f"{a:.3f}" for a in acc)
        print(f"  epoch {ep + 1:3d}  loss {last_loss:.4f}  holdout top1/step [{accs}]  "
              f"mean {score:.3f}{'  *best' if improved else ''}", flush=True)
        if since >= patience:
            print(f"  early stop @ epoch {ep + 1}: no gain in {patience} (best epoch {best_epoch}, "
                  f"mean {best_score:.3f})", flush=True)
            break
    if best_params is not None:
        drafter.update(tree_unflatten(best_params))  # restore best-epoch params
        mx.eval(drafter.parameters())
    return {"base_holdout": base, "final_holdout": best_acc, "final_loss": last_loss, "history": hist,
            "base_holdout_top1": base[0], "final_holdout_top1": best_acc[0],
            "best_epoch": best_epoch, "best_mean_accept": best_score}


def save_drafter(path: str | Path, drafter: EagleDrafter) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), dict(tree_flatten(drafter.parameters())))


def load_drafter(path: str | Path, drafter: EagleDrafter) -> EagleDrafter:
    drafter.update(tree_unflatten(list(mx.load(str(path)).items())))
    return drafter
