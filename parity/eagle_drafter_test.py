"""EAGLE-3 drafter module — mechanical sanity: it overfits a tiny fixed batch.

No model load. A small drafter (frozen random embed/head) is trained single-step (reduce target
features → combine with embedding → 1 layer → frozen head → CE vs next token) on one fixed random
batch. If forward + backprop + optimizer are wired correctly the loss collapses toward 0 (the model
memorizes the batch). This gates the architecture before the real (expensive) capture/train stages.

    uv run python -m parity.eagle_drafter_test
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from quanta.eagle.drafter import EagleDrafter

H, V, NH, HD, INTER = 64, 128, 4, 16, 128
B, T, STEPS = 2, 16, 2000


def run() -> None:
    mx.random.seed(0)
    model = EagleDrafter(hidden=H, n_heads=NH, head_dim=HD, intermediate=INTER, eps=1e-6, n_feature_layers=3)
    embed = mx.random.normal((V, H)) * 0.02   # frozen target embedding (not a module param)
    head = mx.random.normal((V, H)) * 0.02     # frozen target lm_head
    feat3 = mx.random.normal((B, T, 3 * H))    # fixed batch: target low/mid/high fused features
    tokens = mx.random.randint(0, V, (B, T))
    targets = mx.random.randint(0, V, (B, T))

    def loss_fn(model: EagleDrafter) -> mx.array:
        r = model.reduce_target_features(feat3)
        _, normed = model.step(r, embed[tokens], offset=0, mask="causal")
        logits = normed @ head.T  # [B,T,V] via frozen head
        lse = mx.logsumexp(logits, axis=-1)
        tgt = mx.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
        return mx.mean(lse - tgt)

    lvg = nn.value_and_grad(model, loss_fn)
    opt = optim.Adam(learning_rate=5e-3)
    loss0 = float(loss_fn(model).item())
    loss = None
    for i in range(STEPS):
        loss, grads = lvg(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        if i % 250 == 249:
            print(f"  step {i + 1:4d}  loss {float(loss.item()):.4f}")
    lossN = float(loss.item())

    print("\n=== EAGLE-3 drafter overfit sanity ===")
    print(f"loss: {loss0:.3f} (init, ~ln{V}={mx.log(mx.array(float(V))).item():.2f}) -> {lossN:.4f} (final)")
    ok = lossN < 0.1 and lossN < 0.3 * loss0
    print(f"overfits fixed batch -> {ok}")
    assert ok
    print("EAGLE-3 drafter OK (forward + backprop + optimizer learn; embed/head stay frozen)")


if __name__ == "__main__":
    run()
