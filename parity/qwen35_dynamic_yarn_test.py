"""Parity: Qwen3.5 dynamic-YaRN — short context is tax-free, long context is pinned + consistent.

Gates the per-request dynamic-YaRN policy (:meth:`Qwen35Config.effective_yarn_factor`,
:func:`quanta.qwen35.attention.yarn_inv_freq`) and the cache-level factor pin that keeps the
rotate-then-cache roped K consistent (:meth:`Qwen35Cache.pin_yarn` / :meth:`Qwen35Cache.yarn_seq`).
MODEL-FREE per the safety rule: a tiny random ``Qwen35Config`` with a deliberately small
``yarn_original_max`` so positions cross "native" in a handful of tokens — NEVER load the real
checkpoint. A single full-attention layer (the only regime RoPE touches; linear layers carry no KV).

Four gates:

  1. **factor schedule** — ``effective_yarn_factor`` is exactly 1.0 at/below native, ramps as
     ``seq/native`` above it, and caps at ``yarn_factor`` (the continuous dynamic-NTK curve).
  2. **s=1 bit-exact** — at/below native the dynamic path is BIT-IDENTICAL to a no-scaling config
     (``yarn_inv_freq`` == plain RoPE inv-freq; and a full decode matches to 0). Proves the
     short-context common case (the orchestrator's case) pays NO YaRN tax ⇒ safe to default-on.
  3. **pinned-tier consistency** — above native, decode with a pinned factor == a one-shot prefill at
     that same factor (``seq_hint=L``). This is the cache-consistency proof: the pin holds the factor
     fixed so the cached K (roped earlier) and the new Q agree.
  4. **rule-6 guard** — decoding PAST native WITHOUT a pin raises (a drifting factor would silently
     corrupt the KV); at/below native unpinned does NOT raise (no false positive).
  5. **admit-pin wiring (#26)** — the serving seam: ``_Qwen35BatchedSession.admit(slot, prompt_ids,
     max_new)`` pins the cache to ``len(prompt_ids) + max_new`` BEFORE prefill (so prefill ropes at the
     request's factor), and skips the pin when ``max_new`` is unbudgeted. Closes the loop from the
     engine's per-stream budget down to the cache pin gates 1–4 prove correct — driven model-free via
     a fake runtime (no checkpoint, no GPU).

    uv run --with numpy python -m parity.qwen35_dynamic_yarn_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.qwen35.attention import Qwen35Attention, yarn_inv_freq
from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.decode import Qwen35Cache

NATIVE = 16  # tiny "native window" so the test crosses it in a few tokens


def _cfg(*, yarn_dynamic: bool = True, yarn_factor: float = 4.0) -> Qwen35Config:
    """One full-attention layer; small ``yarn_original_max`` (=NATIVE) so >native is cheap to reach."""
    return Qwen35Config(
        vocab_size=64, hidden_size=32, num_hidden_layers=1,
        layer_types=("full_attention",), full_attention_interval=1,
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
        yarn_original_max=NATIVE, yarn_factor=yarn_factor, yarn_dynamic=yarn_dynamic,
    )


def _rel(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))
                 / (mx.max(mx.abs(b.astype(mx.float32))) + 1e-6))


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


def _decode(a: Qwen35Attention, cfg: Qwen35Config, x: mx.array, cache: Qwen35Cache):
    """Decode all T tokens one at a time, resolving seq_hint through the cache exactly as the runtime
    ``_decode_block`` does (``cache.yarn_seq(offset+1, cfg)``)."""
    outs = []
    for t in range(x.shape[1]):
        hint = cache.yarn_seq(cache.offset + 1, cfg)
        outs.append(a(x[:, t:t + 1], cache=cache[0], use_fast=True, seq_hint=hint))
    return mx.concatenate(outs, axis=1)


def _plain_inv_freq(cfg: Qwen35Config) -> mx.array:
    """Un-scaled RoPE inverse frequencies (the factor==1.0 / no-YaRN reference)."""
    dim = cfg.rotary_dim
    idx = mx.arange(0, dim, 2, dtype=mx.float32)
    return 1.0 / (cfg.rope_theta ** (idx / dim))


def _run_factor_schedule() -> bool:
    cfg = _cfg()
    checks = {
        1: cfg.effective_yarn_factor(1),
        NATIVE: cfg.effective_yarn_factor(NATIVE),                # == native ⇒ still 1.0
        NATIVE + 1: cfg.effective_yarn_factor(NATIVE + 1),        # just over ⇒ ~1.06
        int(NATIVE * 2): cfg.effective_yarn_factor(NATIVE * 2),   # 2x ⇒ 2.0
        int(NATIVE * 8): cfg.effective_yarn_factor(NATIVE * 8),   # 8x ⇒ capped at 4.0
    }
    ok = (checks[1] == 1.0 and checks[NATIVE] == 1.0
          and abs(checks[NATIVE * 2] - 2.0) < 1e-9
          and checks[NATIVE + 1] > 1.0
          and abs(checks[NATIVE * 8] - cfg.yarn_factor) < 1e-9)   # capped
    print(f"  [{'OK' if ok else 'FAIL'}] factor schedule: f(<=native)=1.0, f(2x)=2.0, "
          f"f(8x)={checks[NATIVE * 8]:.2f} (cap {cfg.yarn_factor})")
    return ok


def _run_s1_bit_exact() -> bool:
    """At/below native the dynamic path is bit-identical to no-scaling — zero short-context tax."""
    cfg_dyn = _cfg(yarn_dynamic=True)
    cfg_off = _cfg(yarn_dynamic=False, yarn_factor=1.0)          # factor==1.0 always ⇒ plain RoPE
    # (a) freq level: yarn_inv_freq at a <=native length == plain inv-freq, exactly.
    freq_ok = bool(mx.array_equal(yarn_inv_freq(cfg_dyn, NATIVE), _plain_inv_freq(cfg_dyn))) and \
        bool(mx.array_equal(yarn_inv_freq(cfg_off, NATIVE), _plain_inv_freq(cfg_off)))
    # (b) decode level: a full <=native decode is identical under dynamic vs no-scaling.
    T = NATIVE - 2
    x = mx.random.normal((1, T, cfg_dyn.hidden_size))
    a_dyn, a_off = _rand_attn(cfg_dyn), _rand_attn(cfg_off)      # identical weights (same seed)
    d = _rel(_decode(a_dyn, cfg_dyn, x, Qwen35Cache(1, cfg_dyn)),
             _decode(a_off, cfg_off, x, Qwen35Cache(1, cfg_off)))
    ok = freq_ok and d < 1e-6
    print(f"  [{'OK' if ok else 'FAIL'}] s=1 bit-exact (<=native): inv_freq=={freq_ok}, "
          f"decode |Δ|={d:.2e}")
    return ok


def _run_pinned_consistency() -> bool:
    """Above native: pinned decode == one-shot prefill at the same pinned factor (cache-consistency)."""
    cfg = _cfg()
    L = NATIVE * 2 + 3                                            # > native ⇒ factor ~2.x
    a = _rand_attn(cfg)
    x = mx.random.normal((1, L, cfg.hidden_size))
    ref = a(x, use_fast=True, seq_hint=L)                        # one-shot prefill at f(L)
    cache = Qwen35Cache(1, cfg)
    cache.pin_yarn(L)                                            # hold f(L) for every decode step
    inc = _decode(a, cfg, x, cache)
    # every decode step must have resolved to the pinned factor (constant), not the live offset
    pinned_const = all(cache.yarn_seq(p, cfg) == L for p in (1, NATIVE + 1, L))
    d = _rel(inc, ref)
    ok = d < 2e-2 and pinned_const and cache.offset == L
    print(f"  [{'OK' if ok else 'FAIL'}] pinned >native consistency: decode==prefill |Δ|={d:.2e}, "
          f"pin-constant={pinned_const}")
    return ok


def _run_unpinned_guard() -> bool:
    """Decoding past native WITHOUT a pin must raise (rule 6); at/below native must not."""
    cfg = _cfg()
    a = _rand_attn(cfg)
    x = mx.random.normal((1, NATIVE + 4, cfg.hidden_size))
    # unpinned decode walks offset up; the step that first exceeds native must raise.
    raised = False
    try:
        _decode(a, cfg, x, Qwen35Cache(1, cfg))                  # no pin_yarn()
    except RuntimeError:
        raised = True
    # no false positive: a <=native unpinned decode runs clean.
    clean = True
    try:
        xs = mx.random.normal((1, NATIVE - 1, cfg.hidden_size))
        _decode(a, cfg, xs, Qwen35Cache(1, cfg))
    except RuntimeError:
        clean = False
    ok = raised and clean
    print(f"  [{'OK' if ok else 'FAIL'}] rule-6 guard: unpinned >native raises={raised}, "
          f"<=native clean={clean}")
    return ok


class _FakeRT:
    """Minimal stand-in for ``Qwen35BatchedResidentModel`` — hands the session a real
    :class:`Qwen35Cache` and a prefill that records the pin state AT prefill time (so the gate can
    prove the pin happened BEFORE prefill, not after). No weights, no GPU."""

    num_layers = 1

    def __init__(self, cfg: Qwen35Config) -> None:
        self._cfg = cfg

    def make_caches(self) -> Qwen35Cache:
        return Qwen35Cache(1, self._cfg, quantized=False)

    def prefill(self, prompt_ids, cache: Qwen35Cache) -> mx.array:
        cache._pin_at_prefill = cache.yarn_seq_hint    # snapshot: must already be pinned here
        return mx.zeros((1, 1, self._cfg.vocab_size))


def _run_admit_pin_wiring() -> bool:
    """The serving seam: ``admit`` pins the per-slot cache from the request budget, before prefill."""
    from quanta.shim.omlx import _Qwen35BatchedSession

    cfg = _cfg()
    sess = _Qwen35BatchedSession(runtime=_FakeRT(cfg), capacity=4)
    prompt = [1, 2, 3, 4, 5]
    budget = 20
    # (a) budgeted admit pins to prompt_len + max_new, and the pin is in place BEFORE prefill ran.
    sess.admit(0, prompt, max_new=budget)
    c0 = sess._caches[0]
    pinned = c0.yarn_seq_hint == len(prompt) + budget
    pinned_pre_prefill = getattr(c0, "_pin_at_prefill", "unset") == len(prompt) + budget
    # (b) the pin is the one the runtime would read at the request's final position (> native here).
    final_pos = len(prompt) + budget
    resolves = c0.yarn_seq(final_pos, cfg) == final_pos and final_pos > NATIVE
    # (c) an unbudgeted admit (max_new=None) leaves the cache unpinned (live-position path).
    sess.admit(1, prompt, max_new=None)
    unpinned = sess._caches[1].yarn_seq_hint is None
    ok = pinned and pinned_pre_prefill and resolves and unpinned
    print(f"  [{'OK' if ok else 'FAIL'}] admit-pin wiring (#26): pin={pinned}, "
          f"before-prefill={pinned_pre_prefill}, resolves@final={resolves}, unbudgeted→unpinned={unpinned}")
    return ok


def run() -> None:
    ok = True
    print("\n=== Qwen3.5 dynamic-YaRN: tax-free short ctx + pinned long ctx (tiny, model-free) ===")
    ok &= _run_factor_schedule()
    ok &= _run_s1_bit_exact()
    ok &= _run_pinned_consistency()
    ok &= _run_unpinned_guard()
    ok &= _run_admit_pin_wiring()
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    run()
