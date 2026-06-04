"""Nemotron-3-Ultra-550B U1: per-layer numeric parity vs a transformers NemotronH reference.

Layer-streamed (rule 8): one real Ultra layer resident at a time, loaded from the bf16 source. For
the first mamba (L0), attention (L7) and moe (L1) layers, diff our MLX module forward against an
**independent** transformers ``NemotronH*`` reference at Ultra scale (hidden 8192, GQA 64/2, 512
experts top-22, mamba 256 heads, latent 2048). ``transformers``/``torch`` are reference-only
(offline) — never on the runtime path (rule #5).

  * mamba: our :class:`MambaMixer` prefill vs transformers ``NemotronHMamba2Mixer`` (naive CPU
    path); fp32. + self-consistency: prefill == token-by-token decode.
  * attn : our :class:`NemotronAttention` (naive) vs a reference composed from transformers'
    own ``apply_rotary_pos_emb`` + ``eager_attention_forward`` (rope θ=10000, GQA, scale d^-1/2);
    fp32. + self-consistency: fast == naive, prefill == incremental decode.
  * moe  : router top-22 **set + weights** vs transformers ``route_tokens_to_experts`` (the MoE is
    built on the ``meta`` device, so only the tiny ``gate`` is materialized — the 512 experts never
    allocate); experts/latent/shared vs an inline dense per-token/per-expert reference; the dispatch
    is token-chunk invariant.

The full 1023 GiB bf16 model is never loaded — only one layer's tensors, streamed and released. Peak
residency is one real moe layer's bf16 expert stacks (~21.5 GiB) << 490 GiB. A green diff at Ultra
scale is the U1 gate that must precede the multi-hour U2 bake.

    uv run python -m parity.nemotron_ultra_layer_parity
"""

from __future__ import annotations

import warnings

import mlx.core as mx
import numpy as np

from quanta.nemotron.attention import KVCache, NemotronAttention
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.loader import NemotronSourceCheckpoint
from quanta.nemotron.mamba_mixer import MambaMixer
from quanta.nemotron.moe import NemotronLatentMoE, relu2

ULTRA = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
T = 16        # tokens for mamba/attn (causal structure)
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


# ----------------------------------------------------------------------------- mamba
def _mamba_parity(cfg: NemotronHConfig, tcfg, ck: NemotronSourceCheckpoint, idx: int) -> dict:
    import torch
    from transformers.models.nemotron_h import modeling_nemotron_h as MH

    t = ck.mamba_tensors(idx)
    mix = MambaMixer(cfg)
    mix.in_proj.weight = t["in_proj.weight"].astype(mx.float32)
    mix.out_proj.weight = t["out_proj.weight"].astype(mx.float32)
    mix.conv_weight = t["conv1d.weight"].astype(mx.float32)   # (conv_dim, k)
    mix.conv_bias = t["conv1d.bias"].astype(mx.float32)
    mix.A_log = t["A_log"].astype(mx.float32)
    mix.D = t["D"].astype(mx.float32)
    mix.dt_bias = t["dt_bias"].astype(mx.float32)
    mix.norm.weight = t["norm.weight"].astype(mx.float32)

    x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.float32)
    y_pf, _, _ = mix(x)
    mx.eval(y_pf)

    # self-consistency: chunked prefill == token-by-token recurrence
    state, cstate = None, mx.zeros((1, cfg.conv_kernel - 1, cfg.mamba_conv_dim), mx.float32)
    ys = []
    for ti in range(T):
        y_t, state, cstate = mix(x[:, ti:ti + 1], state=state, conv_state=cstate)
        ys.append(y_t)
    sc = _rel(mx.concatenate(ys, axis=1), y_pf)

    # transformers reference (naive CPU path), weights loaded onto a meta-instantiated module
    with torch.device("meta"):
        ref = MH.NemotronHMamba2Mixer(tcfg, idx)
    conv_w = t["conv1d.weight"].reshape(cfg.mamba_conv_dim, 1, cfg.conv_kernel)
    sd = {
        "in_proj.weight": _to_torch(t["in_proj.weight"]),
        "out_proj.weight": _to_torch(t["out_proj.weight"]),
        "conv1d.weight": _to_torch(conv_w),
        "conv1d.bias": _to_torch(t["conv1d.bias"]),
        "A_log": _to_torch(t["A_log"]),
        "D": _to_torch(t["D"]),
        "dt_bias": _to_torch(t["dt_bias"]),
        "norm.weight": _to_torch(t["norm.weight"]),
    }
    ref.load_state_dict(sd, assign=True, strict=True)
    ref.eval()
    with torch.no_grad():
        yr = ref(_to_torch(x))
        yr = yr[0] if isinstance(yr, tuple) else yr
    par = _rel(y_pf, _to_mx(yr))

    del mix, ref, sd
    ck.release()
    _clear()
    return {"par": par, "prefill_decode": sc}


# ----------------------------------------------------------------------------- attention
def _attn_parity(cfg: NemotronHConfig, ck: NemotronSourceCheckpoint, idx: int) -> dict:
    import torch
    from types import SimpleNamespace

    from transformers.models.nemotron_h import modeling_nemotron_h as MH

    t = ck.attention_tensors(idx)
    attn = NemotronAttention(cfg)
    attn.q_proj.weight = t["q_proj.weight"].astype(mx.float32)
    attn.k_proj.weight = t["k_proj.weight"].astype(mx.float32)
    attn.v_proj.weight = t["v_proj.weight"].astype(mx.float32)
    attn.o_proj.weight = t["o_proj.weight"].astype(mx.float32)

    x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.float32)
    y_naive = attn(x, use_fast=False)
    y_fast = attn(x, use_fast=True)
    fast_naive = _rel(y_fast, y_naive)

    cache = KVCache(quantized=False)
    od = [attn(x[:, ti:ti + 1], cache=cache, use_fast=True) for ti in range(T)]
    dec = _rel(mx.concatenate(od, axis=1), y_fast)

    # transformers reference: q/k/v projections (torch), then transformers' OWN rope + eager softmax
    hd, nh, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
    xt = _to_torch(x)
    qw, kw = _to_torch(t["q_proj.weight"]), _to_torch(t["k_proj.weight"])
    vw, ow = _to_torch(t["v_proj.weight"]), _to_torch(t["o_proj.weight"])
    with torch.no_grad():
        q = (xt @ qw.T).view(1, T, nh, hd).transpose(1, 2)
        k = (xt @ kw.T).view(1, T, nkv, hd).transpose(1, 2)
        v = (xt @ vw.T).view(1, T, nkv, hd).transpose(1, 2)
        pos = torch.arange(T).float()
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2).float() / hd))
        fr = pos[:, None] * inv[None, :]
        emb = torch.cat([fr, fr], dim=-1)
        cos, sin = emb.cos()[None], emb.sin()[None]
        qr, kr = MH.apply_rotary_pos_emb(q, k, cos, sin)
        mask = torch.triu(torch.full((T, T), float("-inf")), 1)[None, None]
        mod = SimpleNamespace(num_key_value_groups=nh // nkv, training=False)
        ao, _ = MH.eager_attention_forward(mod, qr, kr, v, mask, scaling=hd ** -0.5)
        yr = ao.reshape(1, T, nh * hd) @ ow.T
    par = _rel(y_naive, _to_mx(yr))

    del attn
    ck.release()
    _clear()
    return {"par": par, "fast_naive": fast_naive, "prefill_decode": dec}


# ----------------------------------------------------------------------------- moe
def _moe_parity(cfg: NemotronHConfig, tcfg, ck: NemotronSourceCheckpoint, idx: int) -> dict:
    import torch
    from transformers.models.nemotron_h import modeling_nemotron_h as MH

    ne, lat, topk = cfg.n_routed_experts, cfg.moe_latent_size, cfg.num_experts_per_tok
    t = ck.moe_nonexpert_tensors(idx)
    st = ck.expert_stacks(idx, ne)  # {"up":[E,inter,lat], "down":[E,lat,inter]} bf16

    moe = NemotronLatentMoE(cfg)
    moe.gate_weight = t["gate.weight"]
    moe.e_score_correction_bias = t["gate.e_score_correction_bias"]
    moe.fc1_latent_proj.weight = t["fc1_latent_proj.weight"]
    moe.fc2_latent_proj.weight = t["fc2_latent_proj.weight"]
    moe.shared_up.weight = t["shared_experts.up_proj.weight"]
    moe.shared_down.weight = t["shared_experts.down_proj.weight"]
    moe.set_experts(st["up"], st["down"])

    x = mx.random.normal((1, T_MOE, cfg.hidden_size)).astype(mx.bfloat16)
    out = moe(x)
    mx.eval(out)
    n = T_MOE
    xf = x.reshape(n, cfg.hidden_size)
    idx_ours, w_ours = moe._route(xf)

    # (a) router parity vs transformers (meta-MoE: only gate materialized, 512 experts stay meta)
    with torch.device("meta"):
        rmoe = MH.NemotronHMoE(tcfg, idx)
    rmoe.load_state_dict(
        {"gate.weight": _to_torch(t["gate.weight"]),
         "gate.e_score_correction_bias": _to_torch(t["gate.e_score_correction_bias"])},
        assign=True, strict=False)
    rmoe.eval()
    with torch.no_grad():
        logits = rmoe.gate(_to_torch(x).reshape(1, n, cfg.hidden_size))
        tk_idx, tk_w = rmoe.route_tokens_to_experts(logits)
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

    # (b) experts/latent/shared vs an inline dense per-token/per-expert reference (selected experts
    #     upcast to fp32 transiently — memory stays bounded to the bf16 stacks)
    latent = moe.fc1_latent_proj(xf)
    rows = []
    for tok in range(n):
        acc = mx.zeros((lat,), mx.float32)
        lt = latent[tok].astype(mx.float32)
        for s in range(topk):
            e = int(idx_ours[tok, s].item())
            ue = st["up"][e].astype(mx.float32)
            de = st["down"][e].astype(mx.float32)
            acc = acc + float(w_ours[tok, s]) * (de @ relu2(ue @ lt))
        mx.eval(acc)
        rows.append(acc)
    ref = (moe.fc2_latent_proj(mx.stack(rows, 0).astype(x.dtype))
           + moe.shared_down(relu2(moe.shared_up(xf))))
    dense = _rel(out.reshape(n, cfg.hidden_size), ref)

    # (c) token-chunking is output-equivalent
    moe.token_chunk = max(1, n // 3)
    out_c = moe(x)
    chunk = _rel(out_c, out)

    del moe, st, rmoe
    ck.release()
    _clear()
    return {"router_set": set_ok, "router_wdiff": wdiff, "dense": dense, "chunk": chunk}


def run() -> None:
    warnings.filterwarnings("ignore")
    mx.random.seed(0)
    from transformers import AutoConfig

    cfg = NemotronHConfig.from_pretrained(ULTRA)
    tcfg = AutoConfig.from_pretrained(ULTRA)
    ck = NemotronSourceCheckpoint(ULTRA)

    i_m = cfg.layers_block_type.index("mamba")
    i_a = cfg.layers_block_type.index("attention")
    i_e = cfg.layers_block_type.index("moe")

    m = _mamba_parity(cfg, tcfg, ck, i_m)
    a = _attn_parity(cfg, ck, i_a)
    e = _moe_parity(cfg, tcfg, ck, i_e)

    # tolerances: fp32 cross-impl (mamba/attn) tight; bf16-module-vs-fp32-inline (moe experts) loose;
    # self-consistency / chunk-invariance (same kernel family) tight. All catch O(1) forward bugs.
    mamba_ok = m["par"] < 5e-3 and m["prefill_decode"] < 3e-3
    attn_ok = a["par"] < 5e-3 and a["fast_naive"] < 2e-3 and a["prefill_decode"] < 2e-3
    moe_ok = e["router_set"] and e["router_wdiff"] < 5e-3 and e["dense"] < 5e-2 and e["chunk"] < 1e-4

    print("\n=== Nemotron-3-Ultra-550B U1 (layer parity vs transformers) ===")
    print(f"layers: mamba L{i_m} / attention L{i_a} / moe L{i_e}  (hidden {cfg.hidden_size}, "
          f"{cfg.n_routed_experts}e top-{cfg.num_experts_per_tok}, mamba {cfg.mamba_num_heads}h)")
    print(f"mamba   : ref Δ {m['par']:.2e}  | prefill==decode {m['prefill_decode']:.2e}   -> {mamba_ok}")
    print(f"attn    : ref Δ {a['par']:.2e}  | fast==naive {a['fast_naive']:.2e} | "
          f"prefill==decode {a['prefill_decode']:.2e}   -> {attn_ok}")
    print(f"moe     : router set-match {e['router_set']} (w Δ {e['router_wdiff']:.2e}) | "
          f"experts-vs-dense Δ {e['dense']:.2e} | chunk Δ {e['chunk']:.2e}   -> {moe_ok}")
    assert mamba_ok, f"mamba parity failed: {m}"
    assert attn_ok, f"attention parity failed: {a}"
    assert moe_ok, f"moe parity failed: {e}"
    print("PASS — Ultra mamba/attention/moe match the transformers reference at full scale.")


if __name__ == "__main__":
    run()
