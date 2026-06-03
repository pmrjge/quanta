"""Model-free gate for InternLM2.5's EAGLE-3 spec-decode substrate (M0).

The lossless EAGLE loop (:func:`quanta.eagle.spec_core.spec_generate`) needs two primitives from the
target's forward; this proves both on a tiny random-init bf16 :class:`InternLM2Model` (NO checkpoint,
NO GPU):

A. **capture** — ``InternLM2Model(ids, capture_layers=...)`` returns ``(logits, {layer: [T, hidden]})``
   of each named layer's **post-layer residual** (the low/mid/high target features the drafter fuses),
   with the logits **BIT-identical** to the no-capture forward (the shipped path is unchanged), and
   the last captured residual reconstructing the logits through the final norm + head (so the capture
   point is the true pre-norm residual, not an off-by-one layer).
B. **truncate** — :meth:`InternLM2Cache.truncate` already exists; this pins the property the EAGLE
   rollback relies on: ``truncate(M)`` leaves the cache byte-identical to having consumed only ``M``
   tokens, so a rejected draft rolls back losslessly (forward-N → truncate-M → continue == prefill-M →
   continue). Causal KV at position ``p < M`` is independent of later tokens, so this is exact.

    uv run python -m parity.internlm2_eagle_capture_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.internlm2.config import InternLM2Config
from quanta.internlm2.decode import InternLM2Cache
from quanta.internlm2.model import InternLM2Model


def _tiny_cfg() -> InternLM2Config:
    """A 4-layer toy InternLM2.5 (random init) — same field set as the other internlm2 model-free
    gates, bumped to 4 layers so capture spans low/mid/high."""
    return InternLM2Config(
        vocab_size=64, hidden_size=32, num_hidden_layers=4, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, attention_bias=False,
        rope_theta=1.0e4, rope_scaling_type="dynamic", rope_scaling_factor=2.5,
        max_position_embeddings=4096, hidden_act="silu", norm_eps=1e-5, tie_word_embeddings=False,
        eos_token_id=2, eos_token_ids=(2,), pad_token_id=2, bos_token_id=1, add_bos_token=True,
    )


def _apply_head(model: InternLM2Model, residual: mx.array) -> mx.array:
    """Final norm + output head on a ``[1, T, H]`` residual → ``[1, T, V]`` (the model's own head)."""
    h = model.norm(residual)
    if model.cfg.tie_word_embeddings:
        return h @ model.tok_embeddings.weight.T
    return model.output(h)


def test_capture() -> None:
    cfg = _tiny_cfg()
    model = InternLM2Model(cfg)
    mx.eval(model.parameters())
    ids = mx.array([[3, 9, 1, 40, 22, 5, 18]])          # [1, T]
    t = ids.shape[1]
    layers = tuple(range(cfg.num_hidden_layers))         # capture every layer

    logits_plain = model(ids)
    logits_cap, caps = model(ids, capture_layers=layers)

    # (1) capture must NOT perturb the shipped logits (it only reads the residual).
    d_logits = float(mx.max(mx.abs(logits_plain - logits_cap)))
    assert d_logits == 0.0, f"capture changed logits: max|Δ|={d_logits}"

    # (2) each captured residual is [T, hidden].
    for layer in layers:
        assert caps[layer].shape == (t, cfg.hidden_size), f"caps[{layer}] shape {caps[layer].shape}"

    # (3) the last captured residual is the true pre-final-norm stream: norm+head(caps[-1]) == logits.
    recon = _apply_head(model, caps[cfg.num_hidden_layers - 1][None])
    d_recon = float(mx.max(mx.abs(recon - logits_cap)))
    assert d_recon < 1e-4, f"last-layer cap is not the pre-norm residual: max|Δ|={d_recon}"

    print(f"A capture: logits Δ={d_logits}, caps {len(caps)}×[T={t},H={cfg.hidden_size}], "
          f"recon Δ={d_recon:.2e}  ok")


def test_truncate_lossless() -> None:
    cfg = _tiny_cfg()
    model = InternLM2Model(cfg)
    mx.eval(model.parameters())
    ids = mx.array([[1, 7, 13, 2, 40, 33, 8, 19, 4]])    # [1, N]
    n, m = ids.shape[1], 5

    # forward all N, then roll the cache back to M (drop a "rejected draft" of length N-M)
    cache_a = InternLM2Cache(cfg)
    model(ids, caches=cache_a.as_list())
    cache_a.truncate(m)
    out_a = model(ids[:, m:n], caches=cache_a.as_list())     # continue from offset M (cache.offset)

    # reference: a cache that only ever consumed M, then the same tail
    cache_b = InternLM2Cache(cfg)
    model(ids[:, :m], caches=cache_b.as_list())
    out_b = model(ids[:, m:n], caches=cache_b.as_list())

    d = float(mx.max(mx.abs(out_a - out_b)))
    assert d < 1e-4, f"truncate not lossless: forward-{n}→truncate-{m} != prefill-{m}: max|Δ|={d}"
    print(f"B truncate: forward-{n}→truncate-{m}→continue == prefill-{m}→continue → Δ={d:.2e}  ok")


def run() -> None:
    test_capture()
    test_truncate_lossless()
    print("PASS — InternLM2.5 EAGLE substrate (capture + lossless truncate) green")


if __name__ == "__main__":
    run()
