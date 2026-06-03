"""InternLM2.5 XAttention integration parity gate — M0 of the MInference sparse-prefill track.

InternLM2.5 is the only keeper still paying full O(T²) dense prefill. The sparse-prefill execution
substrate (bounded-memory chunked block-gather) already exists, validated, in
:mod:`quanta.modeling.xattention`; M0 wires it into
:class:`quanta.internlm2.attention.InternLM2Attention` behind a ``self.sparse`` flag (None = dense,
byte-unchanged) — the smallest reuse that gives InternLM2 a sparse-prefill lever before any
MInference-specific selector is added on top.

This gate proves the *integration* is output-equivalent, on the real attention module (so it covers
the GQA ``mx.repeat``, the ``attn_scale``, and the post-RoPE q/k the substrate scores), not just the
bare substrate functions (:mod:`parity.xattention_parity` covers those):

  1. ``threshold=1.0`` additive-mask path  == dense causal attention,
  2. ``threshold=1.0`` block-gather path    == dense causal attention (the long-context speed path),
  3. a lower threshold drops blocks ⇒ the output changes (the lossy lever actually engages),
  4. ``min_seq > T`` gates sparsity off entirely (dense even with a sparse config set),
  5. perturbing the final token cannot change an earlier query block's output (no future leakage).

Model-free (tiny random ``InternLM2Attention``, fp32 for a tight numeric bound). The *quality* cost
of the lever — how much ppl it trades for the speedup — is measured separately on the real bake by
the long-doc ppl gate (CLAUDE.md: sparse attention is ppl-gated, never numeric-parity-gated).

    uv run python -m parity.internlm2_xattn_test
"""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_map

from quanta.internlm2.attention import InternLM2Attention
from quanta.internlm2.config import InternLM2Config
from quanta.modeling.xattention import XAttnConfig

T = 500       # not a multiple of block=128 → exercises the substrate's padding; spans 4 blocks
BLOCK = 128


def _cfg() -> InternLM2Config:
    return InternLM2Config(
        vocab_size=64, hidden_size=128, num_hidden_layers=1, intermediate_size=256,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, attention_bias=False,
        rope_theta=1.0e4, rope_scaling_type="dynamic", rope_scaling_factor=2.5,
        max_position_embeddings=4096, hidden_act="silu", norm_eps=1e-5, tie_word_embeddings=False,
        eos_token_id=2, eos_token_ids=(2,), pad_token_id=2, bos_token_id=1, add_bos_token=True,
    )


def _attn(cfg: InternLM2Config) -> InternLM2Attention:
    a = InternLM2Attention(cfg)
    a.update(tree_map(lambda p: p.astype(mx.float32), a.parameters()))  # fp32 → tight parity bound
    mx.eval(a.parameters())
    return a


def _prefill(a: InternLM2Attention, x: mx.array, sparse: XAttnConfig | None, *, use_fast: bool):
    a.sparse = sparse
    out = a(x, cache=None, use_fast=use_fast)   # cache=None ⇒ offset 0, kv_len == t ⇒ prefill
    mx.eval(out)
    return out


def _xc(threshold: float, *, gather: bool, min_seq: int = 0) -> XAttnConfig:
    return XAttnConfig(block=BLOCK, stride=16, threshold=threshold, min_seq=min_seq, gather=gather)


def run() -> None:
    mx.random.seed(0)
    cfg = _cfg()
    a = _attn(cfg)
    x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.float32)

    dense_fast = _prefill(a, x, None, use_fast=True)
    dense_naive = _prefill(a, x, None, use_fast=False)
    d0 = float(mx.max(mx.abs(dense_fast - dense_naive)))
    print(f"dense fast vs naive             max|Δ| = {d0:.2e}")
    assert d0 < 5e-3, f"dense fast/naive disagree ({d0}) — a pre-existing issue, not sparse"

    # 1. keep-all additive-mask path == dense
    d1 = float(mx.max(mx.abs(_prefill(a, x, _xc(1.0, gather=False), use_fast=True) - dense_fast)))
    print(f"xattn t=1.0 mask   vs dense     max|Δ| = {d1:.2e}")
    assert d1 < 1e-4, f"keep-all mask path != dense ({d1}) — sparse integration bug"

    # 2. keep-all block-gather path == dense (the long-context speed path)
    d2 = float(mx.max(mx.abs(_prefill(a, x, _xc(1.0, gather=True), use_fast=True) - dense_fast)))
    print(f"xattn t=1.0 gather vs dense     max|Δ| = {d2:.2e}")
    assert d2 < 1e-3, f"keep-all gather path != dense ({d2}) — sparse integration bug"

    # 3. a lower threshold drops blocks ⇒ output changes (the lossy lever engages)
    low = _xc(0.5, gather=True)
    base_low = _prefill(a, x, low, use_fast=True)
    d3 = float(mx.max(mx.abs(base_low - dense_fast)))
    print(f"xattn t=0.5 gather vs dense     max|Δ| = {d3:.2e}  (expected > 0: sparsity active)")
    assert d3 > 1e-2, f"threshold=0.5 did not change the output ({d3}) — sparsity not engaging"

    # 4. min_seq gate: T below min_seq ⇒ dense even with a sparse config set
    d4 = float(mx.max(mx.abs(_prefill(a, x, _xc(0.5, gather=True, min_seq=T + 1), use_fast=True)
                             - dense_fast)))
    print(f"xattn t=0.5 min_seq>T vs dense  max|Δ| = {d4:.2e}  (expected 0: gated off)")
    assert d4 < 1e-6, f"min_seq gate failed ({d4}) — sparse engaged below min_seq"

    # 5. causal safety: perturb the FINAL token; block 0 (rows [0,BLOCK)) is strictly causally
    #    upstream of it (earlier query blocks can neither attend to nor select on the last block),
    #    so its output must be bit-unchanged — proves no future key leaks through the integration.
    x2 = mx.array(x)
    x2[:, -1, :] = mx.random.normal((cfg.hidden_size,)).astype(mx.float32)
    pert_low = _prefill(a, x2, low, use_fast=True)
    d5 = float(mx.max(mx.abs(base_low[:, :BLOCK] - pert_low[:, :BLOCK])))
    print(f"causal: perturb last tok, Δ blk0 max|Δ| = {d5:.2e}  (expected 0)")
    assert d5 < 1e-6, f"perturbing the final token changed block 0 ({d5}) — future leakage"

    print("PASS — InternLM2 XAttention prefill integration is dense-equivalent at t=1.0, "
          "sparsity-active below it, min_seq-gated, and causal-safe.")


if __name__ == "__main__":
    run()
