"""Nex-N2-Pro (Qwen3.5-397B-A17B) N1: per-layer numeric parity vs a transformers Qwen3_5Moe reference.

Layer-streamed (rule 8): one real Nex layer resident at a time, loaded bf16 from the 739 GiB source.
For the first linear (L0, Gated-DeltaNet), first full (L3, gated-GQA) and an MoE block, diff our MLX
modules against an **independent** transformers ``Qwen3_5Moe*`` reference at full 397B scale (hidden
4096, 32/2 GQA head_dim 256, 16/64 linear key/value heads, 512 experts top-10 + shared). The
``qwen35`` module is generic + already a fleet keeper at 35B (Qwen3.6-35B-A3B); this re-gates the
SAME code at 397B (the Nemotron Super→Ultra pattern). ``transformers``/``torch`` are reference-only
(offline) — never on the runtime path (rule 5).

  * deltanet : our :class:`GatedDeltaNet` prefill vs transformers ``Qwen3_5MoeGatedDeltaNet`` (the
    pure-torch ``torch_chunk_gated_delta_rule`` fallback path — no FLA/CUDA); fp32. +
    self-consistency: chunked prefill == token-by-token recurrent decode.
  * attn     : our :class:`Qwen35Attention` (naive) vs transformers ``Qwen3_5MoeAttention`` (eager
    softmax + its own partial-mRoPE rope + the doubled-``q_proj`` sigmoid output gate + per-head
    ``(1+w)`` q/k RMSNorm); fp32. + self-consistency: fast == naive, prefill == incremental decode.
  * moe      : router top-10 **set + weights** (softmax, ``norm_topk_prob`` renorm — NOT DeepSeek
    sigmoid/noaux_tc) vs transformers ``Qwen3_5MoeTopKRouter`` (built on ``meta`` so only the tiny
    ``gate`` materializes — the 512 experts never allocate); experts + sigmoid-gated shared expert
    vs an inline fp32 dense reference; the dispatch is token-chunk invariant.
  * block    : our full :class:`Qwen35Block` (``x + mixer(in_norm(x))`` then ``x + moe(post_norm
    (x))``) vs transformers ``Qwen3_5MoeDecoderLayer`` — the end-to-end gate that exercises the
    ``Qwen3_5MoeRMSNorm`` **``(1+w)``** convention on the input/post-attention norms + the residual
    wiring + mixer-kind dispatch, for a LINEAR (L0) AND a FULL (L3) layer; fp32.

The full 739 GiB bf16 model is never loaded — only one layer's tensors, streamed + released. Peak
residency is one block's bf16/fp32 expert stacks (≈26 GiB fp32) << 490 GiB. A green diff at 397B is
the N1 gate that precedes the bits-decision ppl arbiter (N2).

    uv run --extra reference python -m parity.nex_n2_pro_layer_parity
"""

from __future__ import annotations

import warnings

import mlx.core as mx
import numpy as np

from quanta.qwen35.attention import Qwen35Attention
from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.gated_deltanet import GatedDeltaNet
from quanta.qwen35.loader import Qwen35SourceCheckpoint
from quanta.qwen35.model import Qwen35Block, Qwen35MoEModule
from quanta.qwen35.moe import qwen35_route
from quanta.qwen35.runtime import _one_plus

NEX = "/Users/pmrj/models/Nex-N2-Pro"
T = 16        # tokens for deltanet/attn/block (causal structure)
T_MOE = 8     # fewer tokens for moe (bounded inline-dense reference loop)


def _rel(a: mx.array, b: mx.array) -> float:
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float(mx.max(mx.abs(a - b)) / (mx.max(mx.abs(b)) + 1e-6))


def _to_torch(arr: mx.array):
    import torch
    return torch.from_numpy(np.array(arr.astype(mx.float32)))


def _to_mx(t) -> mx.array:
    return mx.array(np.asarray(t.detach().to("cpu"), dtype=np.float32))


def _clear() -> None:
    try:
        mx.clear_cache()
    except AttributeError:
        pass


def _conv_CK(w: mx.array) -> mx.array:
    """Source depthwise conv weight → ``[C,K]`` (the mixer layout; squeeze a ``[C,1,K]`` ship)."""
    return w.reshape(w.shape[0], w.shape[-1]) if w.ndim == 3 else w


def _conv_C1K(w: mx.array):
    """Source depthwise conv weight → torch ``[C,1,K]`` (the nn.Conv1d layout)."""
    ck = _conv_CK(w)
    return _to_torch(ck.reshape(ck.shape[0], 1, ck.shape[1]))


# ----------------------------------------------------------------------------- deltanet (linear)
def _deltanet_parity(cfg: Qwen35Config, tcfg, ck: Qwen35SourceCheckpoint, idx: int) -> dict:
    import torch
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as Q

    t = ck.linear_attn(idx)
    m = GatedDeltaNet(cfg)
    for proj in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj"):
        getattr(m, proj).weight = t[f"{proj}.weight"].astype(mx.float32)
    m.conv_weight = _conv_CK(t["conv1d.weight"]).astype(mx.float32)
    m.conv_bias = mx.zeros((m.conv_dim,), mx.float32)            # Qwen3.5 GDN conv is bias-free
    m.A_log = t["A_log"].astype(mx.float32)
    m.dt_bias = t["dt_bias"].astype(mx.float32)
    m.norm = t["norm.weight"].astype(mx.float32)                # gated RMSNorm: plain weight (NOT 1+w)

    x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.float32)
    y_pf, _, _ = m(x)
    mx.eval(y_pf)

    # self-consistency: chunked prefill == token-by-token recurrent decode
    state, cstate = None, mx.zeros((1, m.k - 1, m.conv_dim), mx.float32)
    ys = []
    for ti in range(T):
        y_t, state, cstate = m(x[:, ti:ti + 1], state=state, conv_state=cstate)
        ys.append(y_t)
    sc = _rel(mx.concatenate(ys, axis=1), y_pf)

    # transformers reference (pure-torch chunk path), weights onto a meta-instantiated module
    with torch.device("meta"):
        ref = Q.Qwen3_5MoeGatedDeltaNet(tcfg, idx)
    sd = {
        "in_proj_qkv.weight": _to_torch(t["in_proj_qkv.weight"]),
        "in_proj_a.weight": _to_torch(t["in_proj_a.weight"]),
        "in_proj_b.weight": _to_torch(t["in_proj_b.weight"]),
        "in_proj_z.weight": _to_torch(t["in_proj_z.weight"]),
        "out_proj.weight": _to_torch(t["out_proj.weight"]),
        "conv1d.weight": _conv_C1K(t["conv1d.weight"]),
        "A_log": _to_torch(t["A_log"]),
        "dt_bias": _to_torch(t["dt_bias"]),
        "norm.weight": _to_torch(t["norm.weight"]),
    }
    ref.load_state_dict(sd, assign=True, strict=True)
    ref.eval()
    with torch.no_grad():
        yr = ref(_to_torch(x))
        yr = yr[0] if isinstance(yr, tuple) else yr
    par = _rel(y_pf, _to_mx(yr))

    del m, ref, sd
    ck.release()
    _clear()
    return {"par": par, "prefill_decode": sc}


# ----------------------------------------------------------------------------- attention (full)
def _attn_parity(cfg: Qwen35Config, tcfg, ck: Qwen35SourceCheckpoint, idx: int) -> dict:
    import torch
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as Q

    t = ck.full_attn(idx)
    attn = Qwen35Attention(cfg)
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(attn, proj).weight = t[f"{proj}.weight"].astype(mx.float32)
    attn.q_norm = _one_plus(t["q_norm.weight"].astype(mx.float32))   # (1+w) Qwen3_5MoeRMSNorm
    attn.k_norm = _one_plus(t["k_norm.weight"].astype(mx.float32))

    x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.float32)
    y_naive = attn(x, use_fast=False)
    y_fast = attn(x, use_fast=True)
    fast_naive = _rel(y_fast, y_naive)

    from quanta.qwen35.attention import KVCache as QKV
    cache = QKV(quantized=False)
    od = [attn(x[:, ti:ti + 1], cache=cache, use_fast=True) for ti in range(T)]
    dec = _rel(mx.concatenate(od, axis=1), y_fast)

    # transformers reference: Qwen3_5MoeAttention (eager) with its OWN partial-mRoPE rope + the gate
    with torch.device("meta"):
        ref = Q.Qwen3_5MoeAttention(tcfg, idx)
    ref.config._attn_implementation = "eager"
    ref.load_state_dict({
        "q_proj.weight": _to_torch(t["q_proj.weight"]),
        "k_proj.weight": _to_torch(t["k_proj.weight"]),
        "v_proj.weight": _to_torch(t["v_proj.weight"]),
        "o_proj.weight": _to_torch(t["o_proj.weight"]),
        "q_norm.weight": _to_torch(t["q_norm.weight"]),
        "k_norm.weight": _to_torch(t["k_norm.weight"]),
    }, assign=True, strict=True)
    ref.eval()
    rot = Q.Qwen3_5MoeTextRotaryEmbedding(tcfg)
    with torch.no_grad():
        xt = _to_torch(x)
        pos = torch.arange(T)[None]
        cos, sin = rot(xt, pos)
        mask = torch.triu(torch.full((T, T), float("-inf")), 1)[None, None]
        yr, _ = ref(xt, position_embeddings=(cos, sin), attention_mask=mask)
    par = _rel(y_naive, _to_mx(yr))

    del attn, ref
    ck.release()
    _clear()
    return {"par": par, "fast_naive": fast_naive, "prefill_decode": dec}


# ----------------------------------------------------------------------------- moe
def _moe_parity(cfg: Qwen35Config, tcfg, ck: Qwen35SourceCheckpoint, idx: int) -> dict:
    import torch
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as Q

    inter, topk = cfg.moe_intermediate_size, cfg.num_experts_per_tok
    t = ck.moe(idx)
    mod = Qwen35MoEModule(cfg)
    mod.gate = t["gate"]
    mod.set_experts(t["experts_gate_up"], t["experts_down"])
    mod.shared_gate_proj = t["shared_gate_proj"]
    mod.shared_up_proj = t["shared_up_proj"]
    mod.shared_down_proj = t["shared_down_proj"]
    mod.shared_expert_gate = t["shared_expert_gate"]

    x = mx.random.normal((1, T_MOE, cfg.hidden_size)).astype(mx.bfloat16)
    out = mod(x)
    mx.eval(out)
    n = T_MOE
    xf = x.reshape(n, cfg.hidden_size)
    idx_ours, w_ours = qwen35_route(xf, mod.gate, cfg)

    # (a) router parity vs transformers (meta-MoE: only gate materialized, 512 experts stay meta)
    with torch.device("meta"):
        rmoe = Q.Qwen3_5MoeSparseMoeBlock(tcfg)
    rmoe.gate.load_state_dict({"weight": _to_torch(t["gate"])}, assign=True, strict=True)
    rmoe.eval()
    with torch.no_grad():
        _, tk_w, tk_idx = rmoe.gate(_to_torch(x).reshape(n, cfg.hidden_size))
    oi = np.array(idx_ours)
    ow = np.array(w_ours.astype(mx.float32))
    ti = tk_idx.reshape(n, -1).cpu().numpy()
    tw = tk_w.reshape(n, -1).float().cpu().numpy()
    set_ok = all(set(oi[r].tolist()) == set(ti[r].tolist()) for r in range(n))
    wdiff = 0.0
    for r in range(n):
        od = dict(zip(oi[r].tolist(), ow[r].tolist()))
        td = dict(zip(ti[r].tolist(), tw[r].tolist()))
        for e, wv in od.items():
            if e in td:
                wdiff = max(wdiff, abs(wv - td[e]))

    # (b) experts + sigmoid-gated shared expert vs an inline fp32 dense reference (only the selected
    #     experts upcast transiently — memory stays bounded to the bf16 stacks)
    gu_stack, dn_stack = t["experts_gate_up"], t["experts_down"]
    rows = []
    for tok in range(n):
        acc = mx.zeros((cfg.hidden_size,), mx.float32)
        xt = xf[tok].astype(mx.float32)
        for s in range(topk):
            e = int(idx_ours[tok, s].item())
            gu = (xt @ gu_stack[e].astype(mx.float32).T)           # [2*inter]
            g, u = gu[:inter], gu[inter:]
            h = (g * mx.sigmoid(g)) * u                            # silu(gate)*up
            acc = acc + float(w_ours[tok, s]) * (h @ dn_stack[e].astype(mx.float32).T)
        mx.eval(acc)
        rows.append(acc)
    sg = mx.sigmoid(xf.astype(mx.float32) @ mod.shared_expert_gate.astype(mx.float32).T)   # [n,1]
    sh_g = xf.astype(mx.float32) @ mod.shared_gate_proj.astype(mx.float32).T
    sh_u = xf.astype(mx.float32) @ mod.shared_up_proj.astype(mx.float32).T
    shared = ((sh_g * mx.sigmoid(sh_g)) * sh_u) @ mod.shared_down_proj.astype(mx.float32).T
    ref = mx.stack(rows, 0) + shared * sg
    dense = _rel(out.reshape(n, cfg.hidden_size), ref)

    # (c) token-chunking is output-equivalent
    mod.token_chunk = max(1, n // 3)
    out_c = mod(x)
    chunk = _rel(out_c, out)

    del mod, rmoe, gu_stack, dn_stack
    ck.release()
    _clear()
    return {"router_set": set_ok, "router_wdiff": wdiff, "dense": dense, "chunk": chunk}


# ----------------------------------------------------------------------------- full block (end-to-end)
def _block_parity(cfg: Qwen35Config, tcfg, ck: Qwen35SourceCheckpoint, idx: int) -> dict:
    """Our :class:`Qwen35Block` vs transformers ``Qwen3_5MoeDecoderLayer`` (fp32) — exercises the
    ``(1+w)`` input/post norms + residual wiring + mixer dispatch + MoE end-to-end."""
    import torch
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as Q

    is_lin = cfg.is_linear_attention(idx)
    norms = ck.block_norms(idx)
    moe = ck.moe(idx)

    # --- our block (fp32) ---
    blk = Qwen35Block(cfg, idx)
    blk.input_layernorm.weight = _one_plus(norms["input_layernorm"].astype(mx.float32))
    blk.post_attention_layernorm.weight = _one_plus(norms["post_attention_layernorm"].astype(mx.float32))
    mm = blk.mixer
    sd: dict = {
        "input_layernorm.weight": _to_torch(norms["input_layernorm"]),
        "post_attention_layernorm.weight": _to_torch(norms["post_attention_layernorm"]),
        "mlp.gate.weight": _to_torch(moe["gate"]),
        "mlp.experts.gate_up_proj": _to_torch(moe["experts_gate_up"]),
        "mlp.experts.down_proj": _to_torch(moe["experts_down"]),
        "mlp.shared_expert.gate_proj.weight": _to_torch(moe["shared_gate_proj"]),
        "mlp.shared_expert.up_proj.weight": _to_torch(moe["shared_up_proj"]),
        "mlp.shared_expert.down_proj.weight": _to_torch(moe["shared_down_proj"]),
        "mlp.shared_expert_gate.weight": _to_torch(moe["shared_expert_gate"]),
    }
    blk.mlp.gate = moe["gate"].astype(mx.float32)
    blk.mlp.set_experts(moe["experts_gate_up"].astype(mx.float32), moe["experts_down"].astype(mx.float32))
    blk.mlp.shared_gate_proj = moe["shared_gate_proj"].astype(mx.float32)
    blk.mlp.shared_up_proj = moe["shared_up_proj"].astype(mx.float32)
    blk.mlp.shared_down_proj = moe["shared_down_proj"].astype(mx.float32)
    blk.mlp.shared_expert_gate = moe["shared_expert_gate"].astype(mx.float32)

    if is_lin:
        la = ck.linear_attn(idx)
        for proj in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj"):
            getattr(mm, proj).weight = la[f"{proj}.weight"].astype(mx.float32)
            sd[f"linear_attn.{proj}.weight"] = _to_torch(la[f"{proj}.weight"])
        mm.conv_weight = _conv_CK(la["conv1d.weight"]).astype(mx.float32)
        mm.conv_bias = mx.zeros((mm.conv_dim,), mx.float32)
        mm.A_log = la["A_log"].astype(mx.float32)
        mm.dt_bias = la["dt_bias"].astype(mx.float32)
        mm.norm = la["norm.weight"].astype(mx.float32)
        sd["linear_attn.conv1d.weight"] = _conv_C1K(la["conv1d.weight"])
        sd["linear_attn.A_log"] = _to_torch(la["A_log"])
        sd["linear_attn.dt_bias"] = _to_torch(la["dt_bias"])
        sd["linear_attn.norm.weight"] = _to_torch(la["norm.weight"])
    else:
        fa = ck.full_attn(idx)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(mm, proj).weight = fa[f"{proj}.weight"].astype(mx.float32)
            sd[f"self_attn.{proj}.weight"] = _to_torch(fa[f"{proj}.weight"])
        mm.q_norm = _one_plus(fa["q_norm.weight"].astype(mx.float32))
        mm.k_norm = _one_plus(fa["k_norm.weight"].astype(mx.float32))
        sd["self_attn.q_norm.weight"] = _to_torch(fa["q_norm.weight"])
        sd["self_attn.k_norm.weight"] = _to_torch(fa["k_norm.weight"])

    x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.float32)
    if is_lin:
        y_ours, _, _ = blk(x)
    else:
        y_ours, _, _ = blk(x, seq_hint=T)
    mx.eval(y_ours)

    # --- transformers reference DecoderLayer (fp32) ---
    with torch.device("meta"):
        ref = Q.Qwen3_5MoeDecoderLayer(tcfg, idx)   # inner self_attn captures tcfg (eager) via self.config
    ref.load_state_dict(sd, assign=True, strict=True)
    ref.eval()
    rot = Q.Qwen3_5MoeTextRotaryEmbedding(tcfg)
    with torch.no_grad():
        xt = _to_torch(x)
        pos = torch.arange(T)[None]
        cos, sin = rot(xt, pos)
        if is_lin:
            yr = ref(xt, position_embeddings=(cos, sin), attention_mask=None)
        else:
            mask = torch.triu(torch.full((T, T), float("-inf")), 1)[None, None]
            yr = ref(xt, position_embeddings=(cos, sin), attention_mask=mask)
        yr = yr[0] if isinstance(yr, tuple) else yr
    par = _rel(y_ours, _to_mx(yr))

    del blk, ref, sd, moe
    ck.release()
    _clear()
    return {"par": par, "kind": "linear" if is_lin else "full"}


def run() -> None:
    warnings.filterwarnings("ignore")
    mx.random.seed(0)
    from transformers import AutoConfig

    cfg = Qwen35Config.from_pretrained(NEX)
    tcfg = AutoConfig.from_pretrained(NEX).get_text_config()
    tcfg._attn_implementation = "eager"
    ck = Qwen35SourceCheckpoint(NEX, cfg)

    i_lin = next(i for i in range(cfg.num_hidden_layers) if cfg.is_linear_attention(i))
    i_full = next(i for i in range(cfg.num_hidden_layers) if cfg.is_full_attention(i))

    d = _deltanet_parity(cfg, tcfg, ck, i_lin)
    a = _attn_parity(cfg, tcfg, ck, i_full)
    e = _moe_parity(cfg, tcfg, ck, i_lin)
    bl = _block_parity(cfg, tcfg, ck, i_lin)     # linear block (L0)
    bf = _block_parity(cfg, tcfg, ck, i_full)    # full block  (L3)

    # tolerances: fp32 cross-impl (deltanet/attn/block) is machine-precision here (~1e-6 measured, even
    # the chunked-WY-vs-scalar-scan deltanet recurrence), so 1e-4 keeps a ~50x margin while a real
    # forward bug (e.g. the Nemotron group-norm class) lands at 1e-2+; bf16-module-vs-fp32-inline (moe
    # experts) is inherently ~1e-3 so 5e-3; router/chunk-invariance are bit-exact-class. All catch O(1)
    # forward bugs — and the (1+w) input/post-norm convention is exercised only at the block level.
    deltanet_ok = d["par"] < 1e-4 and d["prefill_decode"] < 1e-4
    attn_ok = a["par"] < 1e-4 and a["fast_naive"] < 1e-4 and a["prefill_decode"] < 1e-4
    moe_ok = e["router_set"] and e["router_wdiff"] < 1e-3 and e["dense"] < 5e-3 and e["chunk"] < 1e-4
    block_ok = bl["par"] < 1e-4 and bf["par"] < 1e-4

    print("\n=== Nex-N2-Pro N1 (layer parity vs transformers Qwen3_5Moe @ 397B) ===")
    print(f"layers: deltanet L{i_lin} / attention L{i_full} / moe L{i_lin} / block L{i_lin}+L{i_full}  "
          f"(hidden {cfg.hidden_size}, {cfg.num_experts}e top-{cfg.num_experts_per_tok}, "
          f"linear {cfg.linear_num_key_heads}/{cfg.linear_num_value_heads}h, full {cfg.num_attention_heads}/"
          f"{cfg.num_key_value_heads}h hd{cfg.head_dim})")
    print(f"deltanet: ref Δ {d['par']:.2e}  | prefill==decode {d['prefill_decode']:.2e}   -> {deltanet_ok}")
    print(f"attn    : ref Δ {a['par']:.2e}  | fast==naive {a['fast_naive']:.2e} | "
          f"prefill==decode {a['prefill_decode']:.2e}   -> {attn_ok}")
    print(f"moe     : router set-match {e['router_set']} (w Δ {e['router_wdiff']:.2e}) | "
          f"experts-vs-dense Δ {e['dense']:.2e} | chunk Δ {e['chunk']:.2e}   -> {moe_ok}")
    print(f"block   : linear Δ {bl['par']:.2e} | full Δ {bf['par']:.2e}   -> {block_ok}")
    assert deltanet_ok, f"deltanet parity failed: {d}"
    assert attn_ok, f"attention parity failed: {a}"
    assert moe_ok, f"moe parity failed: {e}"
    assert block_ok, f"block parity failed: linear={bl} full={bf}"
    print("PARITY-CHECKS: 11")
    print("PASS — Nex deltanet/attention/moe/block match the transformers Qwen3_5Moe reference @ 397B.")


if __name__ == "__main__":
    run()
