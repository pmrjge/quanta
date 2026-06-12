"""Parity-gate the Qwen3.5 long-context chunked-prefill substrate (tiny random weights, NO model).

The 1M-window feasibility substrate (Nex-N2-Pro N3): the per-token serving prefill is O(T) full
forwards and the single-shot prefill path holds the whole window with no decode cache — so long
context needs (a) a **chunk-parallel Gated-DeltaNet prefill** (the WY/UT representation — the
sequential within-chunk scan is O(L) tiny kernel launches per layer, infeasible at 100K+), (b)
**prefill continuation** across chunks (conv-window carry + recurrent-state carry on the 45 linear
layers; KV append + bottom-right-aligned causal SDPA on the 15 full layers), and (c) a driver
(:func:`quanta.qwen35.runtime.chunked_prefill`) that consumes a prompt into a
:class:`quanta.qwen35.decode.Qwen35Cache` one bounded chunk at a time. Model-free gates:

* **WY kernel == sequential oracle**: :func:`gdn_chunked_wy` == :func:`gdn_recurrence` ==
  :func:`gdn_chunked` (fp32) — chunk-multiple + ragged lengths, B=2, state carry across blocks,
  and an extreme-decay stress (log-decay −100s; the log-space form never rounds through
  ``exp→0→log``).
* **conv continuation is bit-exact**: ``causal_conv1d(state=...)`` split at any boundary ==
  the full-sequence conv, exactly (identical terms, identical bounded-K summation order).
* **mixer continuation**: ``GatedDeltaNet`` two-chunk prefill (T>1 with ``conv_state``) == the
  single-shot prefill — sequential and WY arms — and a 1-token mid-prefill chunk routes through
  the (unchanged) decode step.
* **driver e2e** (tiny ``Qwen35Model`` blocks): ``chunked_prefill`` == single-shot prefill ==
  the per-token decode path (``chunk_tokens=1`` IS the per-token serving prefill semantics),
  ragged chunking included, for sequential AND WY arms — logits-close + greedy-equal, then the
  greedy *continuation* (decode from the chunk-built cache) matches token-for-token.
* **two-call continuation**: ``chunked_prefill`` twice (prefix extension on a non-empty cache)
  == once.
* **int8 KV**: chunked prefill over a quantized cache == the per-token path (same per-token
  codes by construction).
* **dynamic YaRN (rule 6)**: past the native window the driver raises without a pinned factor;
  with ``pin_yarn`` it matches single-shot at the pinned factor.

    uv run python -m parity.qwen35_prefill_chunked_test
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx

from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.decode import Qwen35Cache
from quanta.qwen35.gated_deltanet import (
    GatedDeltaNet,
    causal_conv1d,
    gdn_chunked,
    gdn_chunked_wy,
    gdn_recurrence,
)
from quanta.qwen35.model import Qwen35Model
from quanta.qwen35.runtime import chunked_prefill

CHECKS = 0


def _ok(name: str, cond: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    print(f"  [{CHECKS:>2}] {name:<58} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        raise AssertionError(f"{name}: {detail}")


def _maxdiff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def _rel(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))
                 / (float(mx.max(mx.abs(b.astype(mx.float32)))) + 1e-12))


def _tiny_cfg() -> Qwen35Config:
    """The forward-test tiny config: 3 layers (linear, full, linear), MoE every layer."""
    return Qwen35Config(
        vocab_size=64, hidden_size=32, num_hidden_layers=3,
        layer_types=("linear_attention", "full_attention", "linear_attention"),
        full_attention_interval=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        attn_output_gate=True, partial_rotary_factor=0.25, rope_theta=1e7, mrope_section=(),
        mrope_interleaved=False, use_qk_norm=True, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, mamba_ssm_dtype="float32", num_experts=8,
        num_experts_per_tok=3, moe_intermediate_size=16, shared_expert_intermediate_size=16,
        scoring_func="softmax", norm_topk_prob=True, router_aux_loss_coef=0.001,
        num_mtp_modules=1, mtp_use_dedicated_embeddings=False, hidden_act="silu", norm_eps=1e-6,
        max_position_embeddings=4096, eos_token_id=248046, eos_token_ids=(248046, 248044),
        pad_token_id=248044, tie_word_embeddings=False,
    )


def _randomize_model(model: Qwen35Model) -> None:
    """The forward-test randomizer (small scales keep the tiny forward well-conditioned)."""
    from quanta.qwen35.attention import Qwen35Attention
    cfg = model.cfg
    for blk in model.layers:
        m = blk.mixer
        blk.mlp.gate = mx.random.normal(blk.mlp.gate.shape)
        blk.mlp.experts_gate_up = mx.random.normal(blk.mlp.experts_gate_up.shape) * 0.1
        blk.mlp.experts_down = mx.random.normal(blk.mlp.experts_down.shape) * 0.1
        blk.mlp.shared_gate_proj = mx.random.normal(blk.mlp.shared_gate_proj.shape) * 0.1
        blk.mlp.shared_up_proj = mx.random.normal(blk.mlp.shared_up_proj.shape) * 0.1
        blk.mlp.shared_down_proj = mx.random.normal(blk.mlp.shared_down_proj.shape) * 0.1
        blk.mlp.shared_expert_gate = mx.random.normal(blk.mlp.shared_expert_gate.shape)
        if isinstance(m, GatedDeltaNet):
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
        elif isinstance(m, Qwen35Attention):
            m.q_proj.weight = mx.random.normal(m.q_proj.weight.shape) * 0.1
            m.k_proj.weight = mx.random.normal(m.k_proj.weight.shape) * 0.1
            m.v_proj.weight = mx.random.normal(m.v_proj.weight.shape) * 0.1
            m.o_proj.weight = mx.random.normal(m.o_proj.weight.shape) * 0.1
            m.q_norm = mx.random.uniform(0.5, 1.5, (cfg.head_dim,))
            m.k_norm = mx.random.uniform(0.5, 1.5, (cfg.head_dim,))
    model.lm_head.weight = mx.random.normal(model.lm_head.weight.shape) * 0.1


# --- 1. WY kernel == sequential oracle ----------------------------------------------------------
def _wy_inputs(b: int, length: int, hk: int = 2, hv: int = 4, dk: int = 8, dv: int = 8,
               log_lo: float = -0.10, log_hi: float = -0.001):
    """Random delta-rule inputs in the mixer's convention (q scaled, q/k l2-normalized upstream is
    NOT required for kernel equivalence — the kernels are algebra over whatever q/k they get)."""
    q = mx.random.normal((b, length, hk, dk)) * (dk ** -0.5)
    k = mx.random.normal((b, length, hk, dk))
    log_g = mx.random.uniform(log_lo, log_hi, (b, length, hv))     # log decay (≤ 0)
    beta = mx.random.uniform(0.0, 1.0, (b, length, hv))
    v = mx.random.normal((b, length, hv, dv))
    return q, k, v, log_g, beta


def _gate_wy_kernel() -> None:
    mx.random.seed(0)
    for tag, (b, length, c) in {"aligned": (1, 64, 8), "ragged": (2, 53, 8),
                                "chunk64": (1, 192, 64)}.items():
        q, k, v, lg, bt = _wy_inputs(b, length)
        g = mx.exp(lg)
        y_ref, s_ref = gdn_recurrence(q, k, v, g, bt)
        y_wy, s_wy = gdn_chunked_wy(q, k, v, lg, bt, c)
        y_ch, s_ch = gdn_chunked(q, k, v, g, bt, c if length % c == 0 else 1)
        d_y, d_s = _rel(y_wy, y_ref), _rel(s_wy, s_ref)
        _ok(f"WY == recurrence ({tag} L={length} C={c})", d_y < 1e-4 and d_s < 1e-4,
            f"rel_y={d_y:.2e} rel_s={d_s:.2e}")
        d_c = _rel(y_wy, y_ch)
        _ok(f"WY == sequential-chunked ({tag})", d_c < 1e-4, f"rel={d_c:.2e}")

    # state carry: two WY blocks (ragged cut) == one block == recurrence
    q, k, v, lg, bt = _wy_inputs(1, 100)
    g = mx.exp(lg)
    y_ref, s_ref = gdn_recurrence(q, k, v, g, bt)
    cut = 37
    y1, s1 = gdn_chunked_wy(q[:, :cut], k[:, :cut], v[:, :cut], lg[:, :cut], bt[:, :cut], 8)
    y2, s2 = gdn_chunked_wy(q[:, cut:], k[:, cut:], v[:, cut:], lg[:, cut:], bt[:, cut:], 8,
                            state_in=s1)
    y_cat = mx.concatenate([y1, y2], axis=1)
    _ok("WY state carry (ragged two-block) == recurrence",
        _rel(y_cat, y_ref) < 1e-4 and _rel(s2, s_ref) < 1e-4,
        f"rel_y={_rel(y_cat, y_ref):.2e} rel_s={_rel(s2, s_ref):.2e}")

    # extreme-decay stress: log decays of −100s (g underflows to exactly 0 in the exp'd form);
    # the WY log-space cumulative differences stay finite — outputs match the oracle.
    q, k, v, lg, bt = _wy_inputs(1, 32, log_lo=-150.0, log_hi=-0.001)
    g = mx.exp(lg)                                     # several entries are exactly 0.0
    y_ref, s_ref = gdn_recurrence(q, k, v, g, bt)
    y_wy, s_wy = gdn_chunked_wy(q, k, v, lg, bt, 8)
    finite = bool(mx.all(mx.isfinite(y_wy)).item() and mx.all(mx.isfinite(s_wy)).item())
    _ok("WY extreme decay (g underflows to 0) finite + == oracle",
        finite and _maxdiff(y_wy, y_ref) < 1e-4 and _maxdiff(s_wy, s_ref) < 1e-4,
        f"d_y={_maxdiff(y_wy, y_ref):.2e} d_s={_maxdiff(s_wy, s_ref):.2e} "
        f"zeros={int(mx.sum(g == 0).item())}")


# --- 2. conv continuation is bit-exact ----------------------------------------------------------
def _gate_conv_continuation() -> None:
    mx.random.seed(1)
    u = mx.random.normal((2, 23, 12))
    w = mx.random.normal((12, 4)) * 0.3
    bias = mx.random.normal((12,)) * 0.1
    full = causal_conv1d(u, w, bias)
    worst = 0.0
    for cut in (1, 2, 3, 7, 20):                       # incl. cuts shorter than K-1
        a = causal_conv1d(u[:, :cut], w, bias)
        # the continuation state is the last K-1 pre-activation rows (zero-extended when short)
        tail = mx.pad(u[:, :cut], [(0, 0), (max(0, 3 - cut), 0), (0, 0)])[:, -3:]
        b2 = causal_conv1d(u[:, cut:], w, bias, state=tail)
        worst = max(worst, _maxdiff(mx.concatenate([a, b2], axis=1), full))
    _ok("conv split-at-any-boundary == full (BIT-exact)", worst == 0.0, f"max|d|={worst:.1e}")
    try:
        causal_conv1d(u, w, bias, state=mx.zeros((2, 2, 12)))
        bad = False
    except ValueError:
        bad = True
    _ok("conv wrong-shape state fails loud (rule 6)", bad)


# --- 3. mixer continuation: two-chunk prefill == single-shot ------------------------------------
def _gate_mixer_continuation(cfg: Qwen35Config) -> None:
    mx.random.seed(2)
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
    length = 18
    x = mx.random.normal((1, length, cfg.hidden_size))
    y_full, s_full, c_full = m(x)                                  # single-shot prefill

    for wy in (False, True):
        for cut in (8, 17):                                        # chunk-aligned + 1-token tail
            y1, s1, c1 = m(x[:, :cut], wy=wy)
            y2, s2, c2 = m(x[:, cut:], state=s1, conv_state=c1, wy=wy)  # T>1 (or T=1) continuation
            y_cat = mx.concatenate([y1, y2], axis=1)
            # conv state is the PRE-conv projection output: recomputed at the chunk's own
            # matmul-M (a 1-token chunk is a gemv), so across a cut it matches at projection-ULP
            # (the conv *algebra* itself is bit-exact — gate 9), not bit-for-bit.
            tol = 1e-5 if not wy else 1e-4
            _ok(f"mixer two-chunk (cut={cut}, wy={wy}) == single prefill",
                _rel(y_cat, y_full) < tol and _rel(s2, s_full) < tol
                and _maxdiff(c2, c_full) < 1e-5,
                f"rel_y={_rel(y_cat, y_full):.2e} rel_s={_rel(s2, s_full):.2e} "
                f"d_conv={_maxdiff(c2, c_full):.1e}")


# --- 4./5./6./7. driver e2e on tiny model blocks -------------------------------------------------
def _greedy_decode(layers, embed_w, norm_w, lm_head_w, cfg, caches: Qwen35Cache, first_tok: int,
                   n: int, hint: int) -> list[int]:
    """Greedy continuation through the per-token decode path (the runtime's `_decode_block`
    semantics, driven via chunked_prefill with 1-token chunks — identical machinery)."""
    out, tok = [], first_tok
    for _ in range(n):
        lg = chunked_prefill(layers, embed_w, norm_w, lm_head_w, cfg, [tok], caches,
                             chunk_tokens=1)
        tok = int(mx.argmax(lg[0, -1]).item())
        out.append(tok)
    return out


def _gate_driver(cfg: Qwen35Config) -> None:
    mx.random.seed(3)
    model = Qwen35Model(cfg)
    _randomize_model(model)
    layers = model.layers
    embed_w = model.embed_tokens.weight
    norm_w = model.norm.weight
    lm_head_w = model.lm_head.weight
    length = 22
    ids = mx.random.randint(0, cfg.vocab_size, (length,))
    ids_l = [int(t) for t in ids]

    # single-shot reference (whole window, fp32 tiny weights)
    lg_full, _, _ = model(ids, use_fast=True, seq_hint=length)
    ref_last = lg_full[:, -1:]
    ref_tok = int(mx.argmax(ref_last[0, -1]).item())

    # per-token decode-path reference (chunk_tokens=1 IS the per-token serving prefill semantics)
    c_tok = Qwen35Cache(len(layers), cfg, quantized=False)
    lg_tok = chunked_prefill(layers, embed_w, norm_w, lm_head_w, cfg, ids_l, c_tok,
                             chunk_tokens=1)
    _ok("driver per-token (chunk=1) == single-shot prefill",
        _rel(lg_tok, ref_last) < 1e-4
        and int(mx.argmax(lg_tok[0, -1]).item()) == ref_tok and c_tok.offset == length,
        f"rel={_rel(lg_tok, ref_last):.2e} offset={c_tok.offset}")
    ref_cont = _greedy_decode(layers, embed_w, norm_w, lm_head_w, cfg, c_tok, ref_tok, 8, length)

    for wy in (False, True):
        for ct in (8, 7):                              # aligned (22=8+8+6) + ragged (22=7+7+7+1)
            cc = Qwen35Cache(len(layers), cfg, quantized=False)
            lg_c = chunked_prefill(layers, embed_w, norm_w, lm_head_w, cfg, ids_l, cc,
                                   chunk_tokens=ct, wy=wy)
            tok_c = int(mx.argmax(lg_c[0, -1]).item())
            _ok(f"driver chunked (ct={ct}, wy={wy}) == single-shot",
                _rel(lg_c, ref_last) < 2e-4 and tok_c == ref_tok and cc.offset == length,
                f"rel={_rel(lg_c, ref_last):.2e} tok={tok_c}=={ref_tok}")
            cont = _greedy_decode(layers, embed_w, norm_w, lm_head_w, cfg, cc, tok_c, 8, length)
            _ok(f"driver chunked (ct={ct}, wy={wy}) greedy continuation == per-token ref",
                cont == ref_cont, f"{cont} vs {ref_cont}")

    # two-call continuation: prefix extension onto a non-empty cache == one call
    c2 = Qwen35Cache(len(layers), cfg, quantized=False)
    chunked_prefill(layers, embed_w, norm_w, lm_head_w, cfg, ids_l[:10], c2, chunk_tokens=8)
    lg_2 = chunked_prefill(layers, embed_w, norm_w, lm_head_w, cfg, ids_l[10:], c2,
                           chunk_tokens=8)
    _ok("driver two-call continuation == one call",
        _rel(lg_2, ref_last) < 2e-4 and int(mx.argmax(lg_2[0, -1]).item()) == ref_tok
        and c2.offset == length,
        f"rel={_rel(lg_2, ref_last):.2e}")

    # int8 KV: chunked == per-token over a quantized cache (same per-token codes by construction).
    # The shared tiny head_dim=8 is below mx.quantize's g32 floor — run this one check on a
    # head_dim=32 variant (rotary_dim = 0.25*32 = 8) with the serving-style g32 int8 KV.
    cfg_q = dataclasses.replace(cfg, head_dim=32)
    model_q = Qwen35Model(cfg_q)
    _randomize_model(model_q)
    lay_q, emb_q = model_q.layers, model_q.embed_tokens.weight
    nrm_q, head_q = model_q.norm.weight, model_q.lm_head.weight
    cq_tok = Qwen35Cache(len(lay_q), cfg_q, quantized=True, group_size=32)
    lgq_tok = chunked_prefill(lay_q, emb_q, nrm_q, head_q, cfg_q, ids_l, cq_tok, chunk_tokens=1)
    cq = Qwen35Cache(len(lay_q), cfg_q, quantized=True, group_size=32)
    lgq = chunked_prefill(lay_q, emb_q, nrm_q, head_q, cfg_q, ids_l, cq, chunk_tokens=8, wy=True)
    _ok("driver int8-KV chunked == int8-KV per-token",
        _rel(lgq, lgq_tok) < 2e-4
        and int(mx.argmax(lgq[0, -1]).item()) == int(mx.argmax(lgq_tok[0, -1]).item()),
        f"rel={_rel(lgq, lgq_tok):.2e}")

    # validation failures (rule 6)
    bad = 0
    try:
        chunked_prefill(layers, embed_w, norm_w, lm_head_w, cfg, [], c2)
    except ValueError:
        bad += 1
    try:
        chunked_prefill(layers, embed_w, norm_w, lm_head_w, cfg, ids_l,
                        Qwen35Cache(len(layers), cfg, quantized=False), chunk_tokens=0)
    except ValueError:
        bad += 1
    _ok("driver empty-prompt / chunk_tokens<1 fail loud (rule 6)", bad == 2)


# --- 8. dynamic YaRN past the native window (rule 6) ---------------------------------------------
def _gate_yarn(cfg: Qwen35Config) -> None:
    mx.random.seed(4)
    cfg_y = dataclasses.replace(cfg, yarn_original_max=16, yarn_dynamic=True, yarn_factor=4.0)
    model = Qwen35Model(cfg_y)
    _randomize_model(model)
    layers, embed_w = model.layers, model.embed_tokens.weight
    norm_w, lm_head_w = model.norm.weight, model.lm_head.weight
    length = 24                                                     # > the 16-token native window
    ids = mx.random.randint(0, cfg_y.vocab_size, (length,))
    ids_l = [int(t) for t in ids]

    c0 = Qwen35Cache(len(layers), cfg_y, quantized=False)
    try:
        chunked_prefill(layers, embed_w, norm_w, lm_head_w, cfg_y, ids_l, c0, chunk_tokens=8)
        raised = False
    except RuntimeError:
        raised = True
    _ok("driver past-native without pin_yarn raises (rule 6)", raised)

    pin = length + 4
    c1 = Qwen35Cache(len(layers), cfg_y, quantized=False)
    c1.pin_yarn(pin)
    lg_c = chunked_prefill(layers, embed_w, norm_w, lm_head_w, cfg_y, ids_l, c1, chunk_tokens=8)
    lg_full, _, _ = model(ids, use_fast=True, seq_hint=pin)          # single-shot @ pinned factor
    _ok("driver pinned-YaRN chunked == single-shot @ pinned factor",
        _rel(lg_c, lg_full[:, -1:]) < 2e-4
        and int(mx.argmax(lg_c[0, -1]).item()) == int(mx.argmax(lg_full[0, -1]).item()),
        f"rel={_rel(lg_c, lg_full[:, -1:]):.2e}")


def run() -> None:
    print("=== Qwen3.5 long-context chunked-prefill substrate (tiny, model-free) ===")
    cfg = _tiny_cfg()
    _gate_wy_kernel()
    _gate_conv_continuation()
    _gate_mixer_continuation(cfg)
    _gate_driver(cfg)
    _gate_yarn(cfg)
    print("\nAll green — WY/UT chunk-parallel GDN == oracle (incl. extreme decay + state carry); "
          "conv continuation bit-exact; mixer/driver chunked == single-shot == per-token (seq + "
          "WY, ragged + int8 KV + two-call continuation); YaRN pin enforced past native.")
    print(f"PARITY-CHECKS: {CHECKS}")


if __name__ == "__main__":
    run()
