"""Phase 1 synthetic ablation: recall@K_meta for a sigmoid-linear meta-router.

Model-free. Generates Dirichlet-distributed synthetic routing scores for a
hypothetical ``E=128, topk=6`` router (sized down 2x from DSV4 to keep the run
under 2 min on a CPU stream), trains a tiny meta-router ``[hidden, E]`` with
sigmoid + BCE, and reports recall@K_meta for ``K_meta in {16, 24, 32, 48, 64,
96}``.

Why this answers Phase 2's question
-----------------------------------

The bandwidth-cut story relies on the meta-router achieving high recall on the
existing top-k *at the smallest possible K_meta*. The synthetic data is by
construction *easier* than real routing (a single low-rank signal + a small
noise floor); recall on synthetic is therefore an *upper bound* on real-model
recall. If synthetic recall at K_meta=48 (a 2.67x bandwidth cut at scale
``E=128``, equivalent to ``K_meta≈96`` at the real DSV4 ``E=256``) doesn't
clear 0.98, then real-model recall won't either — Phase 2 isn't viable.

Determinism
-----------

``mx.random.seed(0)`` + ``numpy.random.default_rng(0)`` + ``mx.random.key`` per
operation; reruns reproduce bit-identical recall (within fp tolerance of the
final summation order). Sized for CPU stream (``mx.set_default_device``) so the
script runs without GPU contention.

    uv run --with numpy python -m parity.hierarchical_routing_ablation
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class Config:
    """Hyperparameters of the synthetic ablation. Sized for CPU-stream <2 min.

    Defaults tuned for a *learnable* synthetic problem: a moderate-noise routing
    signal where a sigmoid-linear meta-router with Adam reaches the recall
    plateau the design doc projects (top-K_meta=64 → ≥98% recall on E=128).
    Sufficient epochs to converge on CPU within ~30s.
    """

    hidden: int = 128
    n_experts: int = 128            # E
    topk: int = 6
    n_tokens: int = 4096            # N
    n_topics: int = 8               # rank of the latent "subject" signal
    train_frac: float = 0.75
    epochs: int = 200
    batch: int = 256
    lr: float = 0.05                # Adam learning rate (small model + BCE)
    beta1: float = 0.9              # Adam momentum
    beta2: float = 0.999            # Adam variance
    eps: float = 1e-8               # Adam denominator stabilization
    pos_weight: float = 8.0         # (E-topk)/topk = 122/6 ≈ 20; soften to 8 for stability
    noise_scale: float = 0.2        # added gaussian to the synthetic logits
    k_meta_sweep: tuple[int, ...] = (16, 24, 32, 48, 64, 96)
    seed: int = 0


def synthesize(cfg: Config) -> tuple[mx.array, mx.array, mx.array]:
    """Generate ``(hidden_inputs, true_top_k_idx, true_score)`` of synthetic routing data.

    Model: each token has a latent "topic" mixture ``z [n_topics]``; the
    routing score is a structured ``E x n_topics`` projection of ``z`` plus
    iid noise. Skew is Dirichlet over topics so a few topics dominate (sparse
    expert hot-set per token).

    Returns:
        ``x [N, hidden]`` (bf16-castable f32 inputs);
        ``y_idx [N, topk] int32`` (true top-k expert ids = supervision signal);
        ``true_scores [N, E] f32`` (kept for diagnostics; not used by the
        meta-router during training).
    """
    rng = np.random.default_rng(cfg.seed)

    # Topic mixture per token: Dirichlet with alpha=0.3 (sparse-favored).
    topic_mix = rng.dirichlet([0.3] * cfg.n_topics, size=cfg.n_tokens)  # [N, n_topics]
    # Each topic activates a sparse subset of ~10% of experts strongly + small bg.
    topic_to_expert = rng.standard_normal((cfg.n_topics, cfg.n_experts)) * 0.2
    # Make ~10% of (topic, expert) entries large (the "hot" experts per topic)
    hot_mask = (rng.random((cfg.n_topics, cfg.n_experts)) < 0.10).astype(np.float32)
    topic_to_expert = topic_to_expert + hot_mask * rng.standard_normal((cfg.n_topics, cfg.n_experts)) * 2.0

    # The routing logits = topic_mix @ topic_to_expert + noise
    logits = topic_mix @ topic_to_expert + rng.standard_normal((cfg.n_tokens, cfg.n_experts)) * cfg.noise_scale

    # The hidden input is a linear projection of topic_mix to ``hidden`` dim
    # (so the meta-router has *something* to learn from). The linear is
    # rng-derived but fixed, simulating the prior layers' hidden output. We add
    # noise so it's not bit-exact recoverable.
    topic_to_hidden = rng.standard_normal((cfg.n_topics, cfg.hidden)) * (1.0 / np.sqrt(cfg.n_topics))
    x = topic_mix @ topic_to_hidden + rng.standard_normal((cfg.n_tokens, cfg.hidden)) * 0.1

    # Existing top-k router selects from logits (this is what the meta-router
    # must recall).
    y_idx = np.argpartition(-logits, kth=cfg.topk - 1, axis=-1)[:, : cfg.topk].astype(np.int32)

    return (
        mx.array(x.astype(np.float32)),
        mx.array(y_idx),
        mx.array(logits.astype(np.float32)),
    )


def _bce_loss(score_meta: mx.array, y_onehot: mx.array, pos_weight: float) -> mx.array:
    """Per-element BCE with positive-class reweight. ``score_meta`` are LOGITS.

    BCE(p=sigmoid(z), y) = -[ pw*y*log p + (1-y)*log(1-p) ]
                        = -[ pw*y*(z - softplus(z)) + (1-y)*(-softplus(z)) ]
                        = pw*y*softplus(z) - pw*y*z + (1-y)*softplus(z)

    The ``pos_weight`` term reweights positives by ``pos_weight`` to compensate
    the sparse positive class (topk/E ~ 2-4%). Stable in logit form (no log of
    a sigmoid).
    """
    sp = mx.logaddexp(score_meta, mx.zeros_like(score_meta))      # softplus(z)
    pos = pos_weight * y_onehot * (sp - score_meta)                # for y=1 → pw*softplus(-z)
    neg = (1.0 - y_onehot) * sp                                    # for y=0 → softplus(z)
    return mx.mean(pos + neg)


def train_meta_router(
    x_train: mx.array, y_train: mx.array, cfg: Config
) -> tuple[mx.array, mx.array]:
    """Train ``W_meta [E, hidden] + b_meta [E]`` via Adam on BCE loss.

    Returns the trained ``(W, b)``. Loss is BCE on per-expert membership of the
    existing top-k. Adam is used over plain SGD because the sparse-positive BCE
    has very different gradient magnitudes per expert (rarely-positive experts
    barely move under SGD); Adam's per-param adaptive lr converges much faster
    on this loss shape.
    """
    mx.random.seed(cfg.seed)
    E, h = cfg.n_experts, cfg.hidden
    n = x_train.shape[0]

    # init: small gaussian (similar to nn.Linear default scale)
    W = mx.random.normal((E, h), scale=1.0 / float(np.sqrt(h)))
    b = mx.zeros((E,))

    # Adam moments
    mW = mx.zeros_like(W)
    vW = mx.zeros_like(W)
    mb = mx.zeros_like(b)
    vb = mx.zeros_like(b)

    # Build a one-hot label matrix once: y_onehot [N, E]
    # Equivalent to scatter; numpy-built for clarity then handed to mlx.
    y_np = np.asarray(y_train)
    y_onehot_np = np.zeros((n, E), dtype=np.float32)
    rows = np.repeat(np.arange(n), cfg.topk)
    y_onehot_np[rows, y_np.reshape(-1)] = 1.0
    y_onehot = mx.array(y_onehot_np)

    def loss_fn(W_in: mx.array, b_in: mx.array, x_batch: mx.array, y_batch: mx.array) -> mx.array:
        scores = x_batch @ W_in.T + b_in[None]
        return _bce_loss(scores, y_batch, cfg.pos_weight)

    grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1))

    rng = np.random.default_rng(cfg.seed + 1)
    n_batches = (n + cfg.batch - 1) // cfg.batch
    step = 0
    for epoch in range(cfg.epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        for b_i in range(n_batches):
            sl = order[b_i * cfg.batch : (b_i + 1) * cfg.batch]
            x_b = x_train[mx.array(sl)]
            y_b = y_onehot[mx.array(sl)]
            loss_v, (gW, gb) = grad_fn(W, b, x_b, y_b)
            step += 1
            # Adam update with bias correction
            mW = cfg.beta1 * mW + (1.0 - cfg.beta1) * gW
            vW = cfg.beta2 * vW + (1.0 - cfg.beta2) * (gW * gW)
            mb = cfg.beta1 * mb + (1.0 - cfg.beta1) * gb
            vb = cfg.beta2 * vb + (1.0 - cfg.beta2) * (gb * gb)
            bc1 = 1.0 - cfg.beta1 ** step
            bc2 = 1.0 - cfg.beta2 ** step
            W = W - cfg.lr * (mW / bc1) / (mx.sqrt(vW / bc2) + cfg.eps)
            b = b - cfg.lr * (mb / bc1) / (mx.sqrt(vb / bc2) + cfg.eps)
            mx.eval(W, b, mW, vW, mb, vb)
            epoch_loss += float(loss_v.item())
        final_loss = epoch_loss / n_batches
        if epoch % 25 == 0 or epoch == cfg.epochs - 1:
            print(f"  epoch {epoch:3d}/{cfg.epochs}  loss={final_loss:.4f}")
    return W, b


def recall_at_k_meta(score_meta: mx.array, y_idx: mx.array, k_meta: int) -> float:
    """Recall: for each token, ``|intersect(meta_subset, top_k)| / topk``."""
    n, _topk = y_idx.shape
    # top-K_meta subset by descending score
    subset = mx.argpartition(-score_meta, kth=k_meta - 1, axis=-1)[:, :k_meta]  # [N, K_meta]
    # For each (n, j) in y_idx, check membership in subset[n].
    # Approach: build a [N, E] mask of subset membership, then gather rows by y_idx.
    E = score_meta.shape[1]
    subset_mask = mx.zeros((n, E), dtype=mx.float32)
    rows = mx.repeat(mx.arange(n, dtype=mx.int32), k_meta)
    cols = subset.reshape(-1)
    subset_mask_np = np.zeros((n, E), dtype=np.float32)
    subset_mask_np[np.asarray(rows), np.asarray(cols)] = 1.0
    subset_mask = mx.array(subset_mask_np)
    hits = mx.take_along_axis(subset_mask, y_idx, axis=-1)  # [N, topk] in {0,1}
    return float(mx.mean(hits).item())


def random_recall(k_meta: int, cfg: Config) -> float:
    """Closed-form recall of a uniform-random K_meta subset: K_meta / E."""
    return float(k_meta) / float(cfg.n_experts)


def run() -> None:
    t0 = time.perf_counter()
    cfg = Config()
    # Force CPU stream so the ablation never contends with GPU.
    mx.set_default_device(mx.cpu)
    mx.random.seed(cfg.seed)

    print("=== Hierarchical MoE routing — Phase 1 synthetic ablation ===")
    print(
        f"config: hidden={cfg.hidden} E={cfg.n_experts} topk={cfg.topk} "
        f"N={cfg.n_tokens} train_frac={cfg.train_frac} epochs={cfg.epochs}\n"
    )

    x, y_idx, _ = synthesize(cfg)
    n = x.shape[0]
    n_train = int(round(cfg.train_frac * n))
    perm = np.random.default_rng(cfg.seed + 2).permutation(n)
    perm_arr = mx.array(perm.astype(np.int32))
    x_perm = x[perm_arr]
    y_perm = y_idx[perm_arr]
    x_train, x_eval = x_perm[:n_train], x_perm[n_train:]
    y_train, y_eval = y_perm[:n_train], y_perm[n_train:]
    print(f"train={x_train.shape[0]} eval={x_eval.shape[0]}\n")

    print("random-baseline recall (no model):")
    for k_meta in cfg.k_meta_sweep:
        r = random_recall(k_meta, cfg)
        print(
            f"  K_meta={k_meta:<4d} recall={r:.3f}  bandwidth cut: {cfg.n_experts / k_meta:.2f}x"
        )
    print()

    print("training meta-router (linear + sigmoid, BCE loss):")
    W, b = train_meta_router(x_train, y_train, cfg)
    print()

    # eval recall
    eval_scores = x_eval @ W.T + b[None]
    print("trained meta-router recall (sigmoid linear, BCE loss):")
    crossings = []
    for k_meta in cfg.k_meta_sweep:
        r = recall_at_k_meta(eval_scores, y_eval, k_meta)
        cut = cfg.n_experts / k_meta
        marker = ""
        if r >= 0.99:
            marker = "  ✅ lossless (≥0.995 in design)"
        elif r >= 0.98:
            marker = "  ✅ ≥98% (Phase 2 ship target)"
        elif r >= 0.95:
            marker = "  ⚠ ≥95% (hard floor)"
        else:
            marker = "  ❌ below 95% (subset is ~random)"
        print(f"  K_meta={k_meta:<4d} recall={r:.3f}  bandwidth cut: {cut:.2f}x{marker}")
        crossings.append((k_meta, r, cut))
    print()

    # Verdict line
    pass_98 = [k for (k, r, _c) in crossings if r >= 0.98]
    pass_99 = [k for (k, r, _c) in crossings if r >= 0.99]
    k98 = min(pass_98) if pass_98 else None
    k99 = min(pass_99) if pass_99 else None
    print(
        f"meta-router params: W=[{cfg.n_experts}, {cfg.hidden}] b=[{cfg.n_experts}] = "
        f"{cfg.n_experts * cfg.hidden + cfg.n_experts} f32"
    )
    print(
        f"first K_meta reaching recall≥0.98: {k98} "
        f"(bandwidth cut: {cfg.n_experts / k98:.2f}x)" if k98 else
        "  ❌ NO K_meta reaches recall≥0.98 — Phase 2 not viable with this loss"
    )
    print(
        f"first K_meta reaching recall≥0.99: {k99} "
        f"(bandwidth cut: {cfg.n_experts / k99:.2f}x)" if k99 else
        "  ⚠ NO K_meta reaches recall≥0.99 — lossless target not met (98% mode only)"
    )
    print(f"\nelapsed: {(time.perf_counter() - t0):.1f}s")
    print("PASS" if k98 is not None else "FAIL")
    if k98 is None:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
