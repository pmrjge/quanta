"""Parity: Qwen3.5 incremental (decode) == prefill, for BOTH hybrid layer types, + lossless rollback.

THE core decode gate for :mod:`quanta.qwen35.decode` + :mod:`quanta.qwen35.runtime` (the cache state
threading + rollback). MODEL-FREE per the safety rule: tiny random ``Qwen35Config`` (small hidden, few
heads/experts) + a few-MB of random tensors — NEVER load the real checkpoint (a 398 GB capture may be
GPU-resident). It drives the SAME per-layer decode threading the runtime's ``_decode_block`` uses
(GatedDeltaNet recurrent state seeded from zero on token 0; the gated-GQA ``KVCache``), via the real
:class:`quanta.qwen35.decode.Qwen35Cache`.

For a **Gated-DeltaNet** (linear) layer AND a **gated-GQA** (full) layer:

  1. run the PREFILL path over the full ``T`` tokens at once (``seq_hint=T`` pins the dynamic-YaRN
     factor so prefill == decode);
  2. run the incremental decode one token at a time, threading ``Qwen35Cache`` (recurrent ``commit`` on
     linear; KV ``update`` on full);
  3. assert (2) matches (1) per position to a tight fp tolerance.

Then the rollback (speculative-decode drop), for BOTH state types: decode to ``T``, ``truncate(t)``
with ``t < T``, continue decoding ``[t, T)`` — and assert it matches a fresh decode that only ever fed
``[0, T)`` then sliced ``[t:]``. For the recurrent state this is the hard case (it cannot be sliced):
``truncate`` must restore the exact snapshot. Also asserts a too-deep recurrent rollback (past the
retained snapshots) RAISES (rule 6 — never silently keep a diverged recurrent state).

    uv run --with numpy python -m parity.qwen35_decode_attn_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.qwen35.attention import KVCache, Qwen35Attention
from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.decode import DEFAULT_SNAPSHOT_DEPTH, Qwen35Cache, _GDNLayerState
from quanta.qwen35.gated_deltanet import GatedDeltaNet

# tiny geometry — a few KB of params, model-free
LAYER_TYPES = ("linear_attention", "full_attention")


def _cfg() -> Qwen35Config:
    return Qwen35Config(
        vocab_size=64, hidden_size=32, num_hidden_layers=len(LAYER_TYPES),
        layer_types=LAYER_TYPES, full_attention_interval=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        attn_output_gate=True, partial_rotary_factor=0.25, rope_theta=1e7,
        mrope_section=(), mrope_interleaved=False, use_qk_norm=True,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=4, mamba_ssm_dtype="float32",
        num_experts=8, num_experts_per_tok=3, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, scoring_func="softmax", norm_topk_prob=True,
        router_aux_loss_coef=0.001, num_mtp_modules=1, mtp_use_dedicated_embeddings=False,
        hidden_act="silu", norm_eps=1e-6, max_position_embeddings=4096,
        eos_token_id=248046, eos_token_ids=(248046, 248044), pad_token_id=248044,
        tie_word_embeddings=False,
    )


def _rel(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))
                 / (mx.max(mx.abs(b.astype(mx.float32))) + 1e-6))


def _rand_gdn(cfg: Qwen35Config) -> GatedDeltaNet:
    mx.random.seed(11)
    m = GatedDeltaNet(cfg)
    m.in_proj_qkv.weight = mx.random.normal(m.in_proj_qkv.weight.shape) * 0.1
    m.in_proj_a.weight = mx.random.normal(m.in_proj_a.weight.shape) * 0.1
    m.in_proj_b.weight = mx.random.normal(m.in_proj_b.weight.shape) * 0.1
    m.in_proj_z.weight = mx.random.normal(m.in_proj_z.weight.shape) * 0.1
    m.out_proj.weight = mx.random.normal(m.out_proj.weight.shape) * 0.1
    m.conv_weight = mx.random.normal(m.conv_weight.shape) * 0.2
    m.A_log = mx.random.normal((cfg.linear_num_value_heads,)) * 0.5
    m.dt_bias = mx.random.normal((cfg.linear_num_value_heads,)) * 0.1
    m.norm = mx.random.uniform(0.5, 1.5, (cfg.linear_value_head_dim,))
    m.chunk = 4
    return m


def _rand_attn(cfg: Qwen35Config) -> Qwen35Attention:
    mx.random.seed(12)
    a = Qwen35Attention(cfg)
    a.q_proj.weight = mx.random.normal(a.q_proj.weight.shape) * 0.1
    a.k_proj.weight = mx.random.normal(a.k_proj.weight.shape) * 0.1
    a.v_proj.weight = mx.random.normal(a.v_proj.weight.shape) * 0.1
    a.o_proj.weight = mx.random.normal(a.o_proj.weight.shape) * 0.1
    a.q_norm = mx.random.uniform(0.5, 1.5, (cfg.head_dim,))
    a.k_norm = mx.random.uniform(0.5, 1.5, (cfg.head_dim,))
    return a


# --- per-layer decode threading, mirroring runtime._decode_block ---------------------------------
def _gdn_step(m: GatedDeltaNet, lc: _GDNLayerState, x_t: mx.array) -> mx.array:
    """One linear-layer decode step through the cache, exactly as the runtime threads it (seed zero
    state + zero conv on the first token so the O(1) recurrence engages from offset 0)."""
    state, conv = lc.recurrent_state, lc.conv_state
    if conv is None:
        state = mx.zeros((1, m.hv, m.dk, m.dv), dtype=mx.float32)
        conv = mx.zeros((1, m.k - 1, m.conv_dim), dtype=x_t.dtype)
    out, rec, conv = m(x_t, state=state, conv_state=conv)
    lc.commit(conv, rec)
    return out


def _attn_step(a: Qwen35Attention, kv: KVCache, x_t: mx.array, seq_hint: int) -> mx.array:
    """One full-layer decode step through its KV cache (YaRN factor pinned via seq_hint)."""
    return a(x_t, cache=kv, use_fast=True, seq_hint=seq_hint)


def _decode_linear(m: GatedDeltaNet, x, cache: Qwen35Cache, layer: int, lo: int, hi: int):
    return mx.concatenate([_gdn_step(m, cache[layer], x[:, t:t + 1]) for t in range(lo, hi)], axis=1)


def _decode_full(a: Qwen35Attention, x, cache: Qwen35Cache, layer: int, lo: int, hi: int, seq_hint):
    return mx.concatenate([_attn_step(a, cache[layer], x[:, t:t + 1], seq_hint)
                           for t in range(lo, hi)], axis=1)


def _run_linear(cfg: Qwen35Config, T: int, rb: int) -> bool:
    m = _rand_gdn(cfg)
    x = mx.random.normal((1, T, cfg.hidden_size))
    ref, _, _ = m(x)                                          # prefill (chunked), conv/state None

    cache = Qwen35Cache(cfg.num_hidden_layers, cfg)
    inc = _decode_linear(m, x, cache, 0, 0, T)
    d_inc = _rel(inc, ref)
    off_ok = cache.offset == T

    # rollback: truncate to rb, re-decode [rb,T), compare against fresh decode [0,T) sliced [rb:]
    cache.truncate(rb)
    trunc_off_ok = cache.offset == rb
    a_cont = _decode_linear(m, x, cache, 0, rb, T)
    fresh = Qwen35Cache(cfg.num_hidden_layers, cfg)
    b_cont = _decode_linear(m, x, fresh, 0, 0, T)[:, rb:]
    d_roll = _rel(a_cont, b_cont)

    good = (inc.shape == ref.shape and d_inc < 1e-3 and d_roll < 1e-3 and off_ok and trunc_off_ok)
    print(f"  [{'OK' if good else 'FAIL'}] GatedDeltaNet (linear)  T={T} rb={rb}  "
          f"|Δprefill|={d_inc:.2e} |Δrollback|={d_roll:.2e} off={cache.offset}"
          f"{'' if off_ok and trunc_off_ok else ' STATE-BAD'}")
    return good


def _run_full(cfg: Qwen35Config, T: int, rb: int) -> bool:
    a = _rand_attn(cfg)
    x = mx.random.normal((1, T, cfg.hidden_size))
    ref = a(x, use_fast=True, seq_hint=T)                     # prefill (KV from offset 0)

    cache = Qwen35Cache(cfg.num_hidden_layers, cfg)
    inc = _decode_full(a, x, cache, 1, 0, T, T)
    d_inc = _rel(inc, ref)
    off_ok = cache.offset == T

    cache.truncate(rb)
    trunc_off_ok = cache.offset == rb
    a_cont = _decode_full(a, x, cache, 1, rb, T, T)
    fresh = Qwen35Cache(cfg.num_hidden_layers, cfg)
    b_cont = _decode_full(a, x, fresh, 1, 0, T, T)[:, rb:]
    d_roll = _rel(a_cont, b_cont)

    good = (inc.shape == ref.shape and d_inc < 2e-2 and d_roll < 2e-2 and off_ok and trunc_off_ok)
    print(f"  [{'OK' if good else 'FAIL'}] gated-GQA (full)        T={T} rb={rb}  "
          f"|Δprefill|={d_inc:.2e} |Δrollback|={d_roll:.2e} off={cache.offset}"
          f"{'' if off_ok and trunc_off_ok else ' STATE-BAD'}")
    return good


def _run_truncate_safety(cfg: Qwen35Config) -> bool:
    """A recurrent rollback deeper than the retained snapshots must RAISE (rule 6), and the cache
    ``truncate`` rolls BOTH layer types together (KV slice + recurrent restore) in one call."""
    m = _rand_gdn(cfg)
    T = DEFAULT_SNAPSHOT_DEPTH + 4                            # decode past the snapshot window
    x = mx.random.normal((1, T, cfg.hidden_size))
    cache = Qwen35Cache(cfg.num_hidden_layers, cfg)
    _decode_linear(m, x, cache, 0, 0, T)
    raised = False
    try:
        cache.truncate(1)                                    # far older than the retained snapshots
    except ValueError:
        raised = True
    # a within-window truncate still works (no false positive)
    ok_within = True
    try:
        cache2 = Qwen35Cache(cfg.num_hidden_layers, cfg)
        _decode_linear(m, x, cache2, 0, 0, T)
        cache2.truncate(T - 1)                               # 1-token rollback (the k=1 spec case)
    except ValueError:
        ok_within = False
    good = raised and ok_within
    print(f"  [{'OK' if good else 'FAIL'}] recurrent rollback safety: too-deep raises={raised} "
          f"1-token rollback ok={ok_within}")
    return good


def _run_cache_joint(cfg: Qwen35Config) -> bool:
    """One ``Qwen35Cache.truncate`` rolls BOTH a linear and a full layer back losslessly in lock-step."""
    m = _rand_gdn(cfg)
    a = _rand_attn(cfg)
    T, rb = 9, 6
    xl = mx.random.normal((1, T, cfg.hidden_size))
    xf = mx.random.normal((1, T, cfg.hidden_size))
    cache = Qwen35Cache(cfg.num_hidden_layers, cfg)
    # interleave both layers' steps as the runtime does (every token completes all layers)
    for t in range(T):
        _gdn_step(m, cache[0], xl[:, t:t + 1])
        _attn_step(a, cache[1], xf[:, t:t + 1], T)
    off_full = cache.offset == T and cache[0].offset == T and cache[1].offset == T
    cache.truncate(rb)
    off_rb = cache.offset == rb and cache[0].offset == rb and cache[1].offset == rb
    # continue both, compare to fresh
    lin_a = _decode_linear(m, xl, cache, 0, rb, T)
    full_a = _decode_full(a, xf, cache, 1, rb, T, T)
    fresh = Qwen35Cache(cfg.num_hidden_layers, cfg)
    lin_b = _decode_linear(m, xl, fresh, 0, 0, T)[:, rb:]
    full_b = _decode_full(a, xf, fresh, 1, 0, T, T)[:, rb:]
    d = max(_rel(lin_a, lin_b), _rel(full_a, full_b))
    good = off_full and off_rb and d < 2e-2
    print(f"  [{'OK' if good else 'FAIL'}] joint cache truncate (both types) rb={rb}  "
          f"|Δ|={d:.2e} off_full={off_full} off_rb={off_rb}")
    return good


def run() -> None:
    cfg = _cfg()
    ok = True
    print("\n=== Qwen3.5 decode == prefill + lossless rollback (tiny, model-free) ===")
    ok &= _run_linear(cfg, 10, 7)
    ok &= _run_full(cfg, 9, 6)
    ok &= _run_truncate_safety(cfg)
    ok &= _run_cache_joint(cfg)
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    run()
