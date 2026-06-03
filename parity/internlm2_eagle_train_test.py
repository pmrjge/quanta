"""Model-free gate for InternLM2.5's EAGLE-3 drafter TRAINING loop (M2).

M0/M1 proved the substrate (capture + lossless truncate) and the adapter (spec==greedy). M2 adds the
two deferred GPU-job scripts that train the drafter on the real int8g64 bake
(``parity.eagle_capture_internlm2`` → ``parity.eagle_train_internlm2``). This gate exercises that
exact pipeline END-TO-END on a tiny random model (NO checkpoint, NO GPU), so the wiring is proven
before the multi-hour real run, reusing the M1 gate's tiny-model stand-ins:

A. **capture → shards → reload** — drive :func:`internlm2_capture_forward` through
   :func:`capture_features_to_shards_fn` over a tiny token corpus, flush to a temp dir (multi-shard),
   and reload with :func:`load_feature_shards`; assert the feature/label shapes line up
   (``feat3 = [N, n_layers*H]``, one in-token + one target argmax per position).
B. **the drafter LEARNS** — :func:`train_drafter` on those captured features drives the training loss
   strictly DOWN: Adam + the canonical CE + feature-regression loss (:func:`_ce_multistep`) are wired
   correctly, not merely shape-correct. (Loss decrease is the robust signal — on this deterministic
   feat3→argmax map a random-init drafter's holdout accept can already be near the modal token, so
   accept is reported but not asserted.)
C. **train + reload stays lossless** — :func:`save_drafter` → :func:`load_drafter` into a fresh
   drafter, then drive :func:`il2_spec_generate`; the emitted stream still equals plain greedy, so
   training + serialization do not break the EAGLE guarantee (which holds for any drafter — here the
   *trained* one, exercising the real save/load roundtrip through the adapter).

    uv run python -m parity.internlm2_eagle_train_test
"""

from __future__ import annotations

import tempfile

import mlx.core as mx

from parity.internlm2_eagle_spec_test import (
    _PROMPT,
    _build_bf16,
    _greedy,
    _resident,
    _tiny_cfg,
    _tiny_drafter,
)
from quanta.eagle.capture import capture_features_to_shards_fn, load_feature_shards
from quanta.eagle.train import load_drafter, save_drafter, train_drafter
from quanta.internlm2.eagle import internlm2_capture_forward
from quanta.internlm2.eagle import spec_generate as il2_spec_generate

_LAYERS: tuple[int, int, int] = (0, 1, 2)   # low / mid / high of the 4-layer toy model
_N_TOKENS = 256                              # 32 train-chunks of 8 — enough for a real train loop
_MAX_NEW = 12
_K = 4


def _capture(rm, layers: tuple[int, ...], n_tokens: int) -> dict[str, mx.array]:
    """Capture features over a deterministic toy id stream through the production resident wrapper +
    :func:`capture_features_to_shards_fn` (shard + flush), then reload the shards — the same path the
    real ``parity.eagle_capture_internlm2`` job runs, in miniature."""
    ids = mx.array([(i * 7 + 3) % rm.cfg.vocab_size for i in range(n_tokens)], dtype=mx.int32)
    fwd = internlm2_capture_forward(rm)
    with tempfile.TemporaryDirectory() as d:
        capture_features_to_shards_fn(fwd, ids, layers, d, chunk=64, shard_tokens=128)
        return load_feature_shards(d)


def test_capture_shards() -> None:
    cfg = _tiny_cfg()
    rm = _resident(_build_bf16(cfg), cfg)
    g = _capture(rm, _LAYERS, _N_TOKENS)
    n, f = g["feat3"].shape
    assert f == len(_LAYERS) * cfg.hidden_size, f"feat3 width {f} != {len(_LAYERS)}*{cfg.hidden_size}"
    assert n == _N_TOKENS, f"captured {n} rows, expected {_N_TOKENS}"
    assert g["in_tokens"].shape == (n,) and g["targets"].shape == (n,), "in/target shape mismatch"
    print(f"A capture: feat3 {tuple(g['feat3'].shape)} = {len(_LAYERS)}×[N,H], "
          f"in/targets {tuple(g['in_tokens'].shape)}  ok")


def test_drafter_learns() -> None:
    cfg = _tiny_cfg()
    rm = _resident(_build_bf16(cfg), cfg)
    embed, head = rm.embed_head()
    g = _capture(rm, _LAYERS, _N_TOKENS)
    drafter = _tiny_drafter(cfg)
    res = train_drafter(drafter, g["feat3"], g["in_tokens"], g["targets"], embed, head,
                        chunk=8, batch=2, epochs=30, lr=2e-3, holdout=2, steps=2, feat_w=1.0,
                        patience=30, seed=0)
    losses = [h[1] for h in res["history"]]
    assert min(losses) < losses[0], f"training loss did not decrease: {losses[0]:.4f} -> min "
    base = sum(res["base_holdout"]) / len(res["base_holdout"])
    best = sum(res["final_holdout"]) / len(res["final_holdout"])
    print(f"B learns: loss {losses[0]:.3f} -> {min(losses):.3f} over {len(losses)} ep | "
          f"holdout accept base {base:.3f} -> best {best:.3f}  ok")


def test_train_reload_lossless() -> None:
    cfg = _tiny_cfg()
    rm = _resident(_build_bf16(cfg), cfg)
    embed, head = rm.embed_head()
    g = _capture(rm, _LAYERS, _N_TOKENS)
    drafter = _tiny_drafter(cfg)
    train_drafter(drafter, g["feat3"], g["in_tokens"], g["targets"], embed, head,
                  chunk=8, batch=2, epochs=10, lr=2e-3, holdout=2, steps=2, feat_w=1.0,
                  patience=10, seed=0)
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/drafter.safetensors"
        save_drafter(p, drafter)
        fresh = load_drafter(p, _tiny_drafter(cfg))           # random params overwritten by trained
    spec_out, stats = il2_spec_generate(rm, fresh, embed, head, _PROMPT,
                                        max_new=_MAX_NEW, k=_K, layers=_LAYERS, eos_id=None)
    greedy_out = _greedy(rm, _PROMPT, _MAX_NEW)
    assert spec_out == greedy_out, f"trained+reloaded spec != greedy:\n  spec  ={spec_out}\n  greedy={greedy_out}"
    print(f"C train+reload spec==greedy: {len(spec_out)} toks, mean_accept={stats['mean_accept']:.2f}  ok")


def run() -> None:
    test_capture_shards()
    test_drafter_learns()
    test_train_reload_lossless()
    print("PASS — InternLM2.5 EAGLE-3 training loop (capture→shards→train→save/reload→lossless spec)")


if __name__ == "__main__":
    run()
