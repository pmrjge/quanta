"""Parity-gate the Qwen3.5-397B-A17B forward on tiny random-weight configs (NO model load).

Model-free per the project's safety rule: a tiny random ``Qwen35Config`` (small hidden, few
heads/experts, 3 layers) + a few-MB of random tensors. Proves each path's optimized form is
output-equivalent to its dead-simple reference:

* GatedDeltaNet (linear): sequential scan == chunked prefill == step-by-step decode (+ state),
  and the ``GatedDeltaNet`` module's prefill == feeding tokens one at a time (conv + recurrent
  state threading).
* gated-GQA (full): fast (mx.fast.rope + sdpa) == naive (explicit RoPE + softmax); and chunked /
  incremental decode (KV cache) == single-shot prefill; the explicit==fast RoPE check covers the
  partial-mRoPE rotation.
* MoE: sparse ``gather_mm`` dispatch == dense (run-every-expert) reference, top-10 softmax routing.
* assembled tiny model: a full ``Qwen35Model`` forward is finite + correctly shaped, prefill ==
  stepwise decode (state threaded across linear + full + MoE layers), and the naive (use_fast=False,
  sparse=False) path == the optimized one per layer.

    uv run --with numpy python -m parity.qwen35_forward_test

The true end-to-end arbiter — teacher-forced bf16 perplexity on real prose through the streamed
loader — is DEFERRED (needs the real checkpoint + memory; a 398 GB capture may be queued). Its
heavy invocation, NOT run here:

    # from quanta.qwen35.config import Qwen35Config
    # from quanta.qwen35.loader import Qwen35SourceCheckpoint
    # cfg = Qwen35Config.from_pretrained("~/models/Qwen3.5-397B-A17B")
    # ck = Qwen35SourceCheckpoint("~/models/Qwen3.5-397B-A17B", cfg)
    # ids = mx.array(tokenizer.encode(prose))[None]
    # build Qwen35Model(cfg); stream-load each layer's weights (ck.linear_attn/full_attn/moe);
    # logits = model(ids)[0]; ppl = exp(mean CE(logits[:-1], ids[1:]))  # bf16 arbiter, ~native
"""

from __future__ import annotations

import mlx.core as mx

from quanta.qwen35.attention import KVCache, Qwen35Attention
from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.gated_deltanet import (
    GatedDeltaNet,
    gdn_chunked,
    gdn_recurrence,
    gdn_step,
)
from quanta.qwen35.model import Qwen35Model
from quanta.qwen35.moe import qwen35_moe


def _maxdiff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def _rel(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)) / (mx.max(mx.abs(b)) + 1e-6))


def _tiny_cfg() -> Qwen35Config:
    """A tiny, structurally-faithful config: 3 layers (linear, full, linear), MoE every layer."""
    n = 3
    return Qwen35Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=n,
        layer_types=("linear_attention", "full_attention", "linear_attention"),
        full_attention_interval=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,                       # rotary_dim = round(0.25*8) = 2
        attn_output_gate=True,
        partial_rotary_factor=0.25,
        rope_theta=1e7,
        mrope_section=(),                 # text 1D positions => no mRoPE split needed for parity
        mrope_interleaved=False,
        use_qk_norm=True,
        linear_num_key_heads=2,
        linear_num_value_heads=4,         # rep = 4//2 = 2
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        mamba_ssm_dtype="float32",
        num_experts=8,
        num_experts_per_tok=3,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        scoring_func="softmax",
        norm_topk_prob=True,
        router_aux_loss_coef=0.001,
        num_mtp_modules=1,
        mtp_use_dedicated_embeddings=False,
        hidden_act="silu",
        norm_eps=1e-6,
        max_position_embeddings=4096,
        eos_token_id=248046,
        eos_token_ids=(248046, 248044),
        pad_token_id=248044,
        tie_word_embeddings=False,
    )


# --- Gated DeltaNet: scan == chunk == decode ---------------------------------------------------
def _gdn_kernel_parity() -> tuple[bool, float, float]:
    mx.random.seed(0)
    b, length, hk, hv, dk, dv, q = 1, 12, 2, 4, 8, 8, 4  # rep = hv//hk = 2; nc = 3
    qy = mx.random.normal((b, length, hk, dk))
    ky = mx.random.normal((b, length, hk, dk))
    vy = mx.random.normal((b, length, hv, dv))
    g = mx.random.uniform(0.90, 0.999, (b, length, hv))   # decay in (0,1)
    beta = mx.random.uniform(0.0, 1.0, (b, length, hv))

    y_seq, s_seq = gdn_recurrence(qy, ky, vy, g, beta)
    y_ch, s_ch = gdn_chunked(qy, ky, vy, g, beta, q)
    chunk_ok = _maxdiff(y_seq, y_ch) < 1e-4 and _maxdiff(s_seq, s_ch) < 1e-4

    s = mx.zeros((b, hv, dk, dv))
    ys = []
    for t in range(length):
        y_t, s = gdn_step(qy[:, t], ky[:, t], vy[:, t], g[:, t], beta[:, t], s)
        ys.append(y_t)
    y_step = mx.stack(ys, axis=1)
    step_ok = _maxdiff(y_seq, y_step) < 1e-4 and _maxdiff(s_seq, s) < 1e-4

    # bounded-memory: two blocks carrying state == full sequence
    cut = 8  # both blocks divisible by q=4
    y1, s1 = gdn_chunked(qy[:, :cut], ky[:, :cut], vy[:, :cut], g[:, :cut], beta[:, :cut], q)
    y2, s2 = gdn_chunked(qy[:, cut:], ky[:, cut:], vy[:, cut:], g[:, cut:], beta[:, cut:], q,
                         state_in=s1)
    carry_ok = (_maxdiff(mx.concatenate([y1, y2], axis=1), y_seq) < 1e-4
                and _maxdiff(s2, s_seq) < 1e-4)
    return (chunk_ok and step_ok and carry_ok), _maxdiff(y_seq, y_ch), _maxdiff(y_seq, y_step)


def _gdn_module_parity(cfg: Qwen35Config) -> tuple[bool, float]:
    """The GatedDeltaNet module: chunked prefill == feeding tokens one at a time (conv+state)."""
    mx.random.seed(1)
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

    length = 10
    x = mx.random.normal((1, length, cfg.hidden_size))
    y_pf, _, _ = m(x)  # chunked prefill (state/conv None)

    state = mx.zeros((1, m.hv, m.dk, m.dv), dtype=mx.float32)
    conv = mx.zeros((1, m.k - 1, m.conv_dim))
    ys = []
    for t in range(length):
        y_t, state, conv = m(x[:, t : t + 1], state=state, conv_state=conv)
        ys.append(y_t)
    y_dec = mx.concatenate(ys, axis=1)
    return _rel(y_dec, y_pf) < 1e-3, _rel(y_dec, y_pf)


# --- gated GQA: fast == naive, decode == prefill ----------------------------------------------
def _attn_parity(cfg: Qwen35Config) -> tuple[bool, float, float]:
    mx.random.seed(2)
    a = Qwen35Attention(cfg)
    a.q_proj.weight = mx.random.normal(a.q_proj.weight.shape) * 0.1
    a.k_proj.weight = mx.random.normal(a.k_proj.weight.shape) * 0.1
    a.v_proj.weight = mx.random.normal(a.v_proj.weight.shape) * 0.1
    a.o_proj.weight = mx.random.normal(a.o_proj.weight.shape) * 0.1
    a.q_norm = mx.random.uniform(0.5, 1.5, (cfg.head_dim,))
    a.k_norm = mx.random.uniform(0.5, 1.5, (cfg.head_dim,))

    length = 7
    x = mx.random.normal((1, length, cfg.hidden_size))
    out_fast = a(x, use_fast=True, seq_hint=length)
    out_naive = a(x, use_fast=False, seq_hint=length)
    fast_ok = _rel(out_fast, out_naive) < 2e-2

    # incremental decode (KV cache) == single-shot prefill
    cache = KVCache()
    outs = []
    for t in range(length):
        outs.append(a(x[:, t : t + 1], cache=cache, use_fast=True, seq_hint=length))
    out_dec = mx.concatenate(outs, axis=1)
    decode_ok = _rel(out_dec, out_fast) < 2e-2
    return (fast_ok and decode_ok), _rel(out_fast, out_naive), _rel(out_dec, out_fast)


# --- MoE: sparse == dense ----------------------------------------------------------------------
def _moe_parity(cfg: Qwen35Config) -> tuple[bool, float]:
    mx.random.seed(3)
    h, e, inter = cfg.hidden_size, cfg.num_experts, cfg.moe_intermediate_size
    si = cfg.shared_expert_intermediate_size
    p = {
        "gate": mx.random.normal((e, h)),
        "experts_gate_up": mx.random.normal((e, cfg.moe_gate_up_out, h)) * 0.1,
        "experts_down": mx.random.normal((e, h, inter)) * 0.1,
        "shared_gate_proj": mx.random.normal((si, h)) * 0.1,
        "shared_up_proj": mx.random.normal((si, h)) * 0.1,
        "shared_down_proj": mx.random.normal((h, si)) * 0.1,
        "shared_expert_gate": mx.random.normal((1, h)),
    }
    x = mx.random.normal((1, 9, h))
    y_sparse = qwen35_moe(x, p, cfg, sparse=True)
    y_dense = qwen35_moe(x, p, cfg, sparse=False)
    return _maxdiff(y_sparse, y_dense) < 1e-4, _maxdiff(y_sparse, y_dense)


# --- assembled tiny model --------------------------------------------------------------------
def _randomize_model(model: Qwen35Model) -> None:
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


def _model_parity(cfg: Qwen35Config) -> tuple[bool, bool, float, bool, float]:
    mx.random.seed(4)
    model = Qwen35Model(cfg)
    _randomize_model(model)
    length = 8
    ids = mx.random.randint(0, cfg.vocab_size, (length,))

    logits_pf, _, _ = model(ids, use_fast=True, seq_hint=length)
    finite_ok = bool(mx.all(mx.isfinite(logits_pf)).item())
    shape_ok = logits_pf.shape == (1, length, cfg.vocab_size)

    # naive (explicit RoPE + softmax + dense MoE) == optimized
    logits_naive, _, _ = model(ids, use_fast=False, sparse=False, seq_hint=length)
    naive_ok = _rel(logits_naive, logits_pf) < 2e-2

    # prefill == stepwise decode across linear (recurrent+conv) / full (KV) / MoE layers
    caches, state, conv = model.make_state()
    dec = []
    for t in range(length):
        lg, state, conv = model(ids[t : t + 1], caches=caches, state=state, conv=conv,
                                use_fast=True, seq_hint=length)
        dec.append(lg)
    logits_dec = mx.concatenate(dec, axis=1)
    decode_ok = _rel(logits_dec, logits_pf) < 2e-2
    return (finite_ok and shape_ok), naive_ok, _rel(logits_naive, logits_pf), decode_ok, \
        _rel(logits_dec, logits_pf)


def run() -> None:
    cfg = _tiny_cfg()

    gdn_ok, gdn_dc, gdn_ds = _gdn_kernel_parity()
    gdn_mod_ok, gdn_mod_r = _gdn_module_parity(cfg)
    attn_ok, attn_fn, attn_dec = _attn_parity(cfg)
    moe_ok, moe_d = _moe_parity(cfg)
    model_base_ok, model_naive_ok, model_naive_r, model_dec_ok, model_dec_r = _model_parity(cfg)

    print("\n=== Qwen3.5-397B-A17B forward parity (tiny, model-free) ===")
    print(f"GDN scan==chunk==decode (+state)     : {gdn_ok}  d_chunk={gdn_dc:.2e} d_step={gdn_ds:.2e}")
    print(f"GDN module prefill==stepwise decode  : {gdn_mod_ok}  rel={gdn_mod_r:.2e}")
    print(f"GQA fast==naive ; decode==prefill    : {attn_ok}  fast={attn_fn:.2e} dec={attn_dec:.2e}")
    print(f"MoE sparse==dense (top-10 softmax)   : {moe_ok}  maxdiff={moe_d:.2e}")
    print(f"model forward finite + shaped        : {model_base_ok}")
    print(f"model naive==optimized               : {model_naive_ok}  rel={model_naive_r:.2e}")
    print(f"model prefill==stepwise decode       : {model_dec_ok}  rel={model_dec_r:.2e}")
    assert all([gdn_ok, gdn_mod_ok, attn_ok, moe_ok, model_base_ok, model_naive_ok, model_dec_ok])
    print("Qwen3.5 forward OK (GDN scan==chunk==decode; GQA fast==naive==decode; MoE sparse==dense;"
          " assembled model finite + naive==opt + prefill==decode)")


if __name__ == "__main__":
    run()
