"""Bisect layer-0 sub-ops: quanta vs HF for input_norm → GDN mixer → post_norm → MoE.

Layer 0 still diverges (rel~1.26) after the GDN decay-order + gated-norm fixes, though GDN and MoE
both match HF by inspection. This isolates the exact sub-op: capture HF layer-0 submodule outputs
(``input_layernorm`` / ``linear_attn`` / ``post_attention_layernorm`` / ``mlp``) and replicate the
same sub-steps in quanta on the bit-aligned embedding. ``norm1`` depends only on the (aligned) embed,
so it must match; the FIRST sub-op that diverges is the bug (GDN mixer vs the rest).

    uv run --with transformers --with torch python -u -m parity.qwen36_l0_internals hf    # → npz
    uv run python -u -m parity.qwen36_l0_internals diff                                   # quanta + diff
"""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np

from quanta.qwen35.tokenizer import Qwen35Tokenizer

SRC = "/Users/pmrj/models/Qwen3.6-35B-A3B"
ART = "/Users/pmrj/models/Qwen3.6-35B-A3B-quanta_int4g64"
NPZ = "/tmp/qwen36_l0_caps.npz"
TEXT = "The capital of France is Paris. Water is made of hydrogen and oxygen."


def _ids() -> list[int]:
    return Qwen35Tokenizer.from_pretrained(ART).encode(TEXT, add_bos=False)


def phase_hf() -> None:
    import torch
    import transformers as tf

    ids = _ids()
    cfg = tf.AutoConfig.from_pretrained(SRC)
    model = getattr(tf, cfg.architectures[0]).from_pretrained(
        SRC, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager")
    model.eval()
    text = model.model.language_model
    l0 = text.layers[0]
    caps: dict[str, np.ndarray] = {}

    def cap(name):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            caps[name] = h.detach().float().cpu().numpy()[0]
        return hook

    def cap_readout(_m, inp):  # forward_pre_hook: (module, args); args = (core_attn_out, z) both [N,Dv]
        caps["gdn_readout"] = inp[0].detach().float().cpu().numpy()   # recurrence readout, pre gated-norm

    hs = [l0.input_layernorm.register_forward_hook(cap("norm1")),
          l0.linear_attn.register_forward_hook(cap("mixer")),
          l0.linear_attn.norm.register_forward_pre_hook(cap_readout),
          l0.post_attention_layernorm.register_forward_hook(cap("norm2")),
          l0.mlp.register_forward_hook(cap("moe"))]

    class _Stop(Exception):
        pass

    hs.append(text.layers[1].register_forward_hook(lambda *_: (_ for _ in ()).throw(_Stop())))
    inp = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        emb = text.embed_tokens(inp).float().cpu().numpy()[0]
        try:
            text(input_ids=inp, use_cache=False)
        except _Stop:
            pass
    for h in hs:
        h.remove()
    np.savez(NPZ, ids=np.array(ids), embed=emb, **caps)
    print(f"saved HF layer-0 internals: {list(caps)} to {NPZ}", flush=True)


def phase_diff() -> None:
    from quanta.qwen35.runtime import Qwen35ResidentModel

    d = np.load(NPZ)
    ids = [int(x) for x in d["ids"]]
    mx.set_wired_limit(int(120 * 1024 ** 3))
    model = Qwen35ResidentModel(ART)
    blk = model.layers[0]

    from quanta.qwen35.gated_deltanet import _l2norm, causal_conv1d, silu

    x = model.embed_w[mx.array(ids)][None].astype(mx.bfloat16)   # [1,T,hidden]
    norm1 = blk.input_layernorm(x)
    # replicate the GDN forward up to the recurrence readout `o` (pre gated-norm) to isolate
    # conv+recurrence from the gated-norm/out_proj
    m = blk.mixer
    b, t, _ = norm1.shape
    a = -mx.exp(m.A_log.astype(mx.float32))
    qkv = silu(causal_conv1d(m.in_proj_qkv(norm1), m.conv_weight, m.conv_bias))
    dt = (mx.maximum(m.in_proj_a(norm1).astype(mx.float32) + m.dt_bias.astype(mx.float32), 0)
          + mx.log1p(mx.exp(-mx.abs(m.in_proj_a(norm1).astype(mx.float32) + m.dt_bias.astype(mx.float32)))))
    g = mx.exp(dt * a[None, None, :])
    beta = mx.sigmoid(m.in_proj_b(norm1).astype(mx.float32))
    q, k, v = m._split_qkv(qkv, b, t)
    q, k = _l2norm(q) * (m.dk ** -0.5), _l2norm(k)
    gdn_readout, _ = m._prefill(q, k, v, g, beta, None)          # [b,t,hv,dv] pre gated-norm
    mixer, _, _ = blk.mixer(norm1, state=None, conv_state=None)
    x1 = x + mixer
    norm2 = blk.post_attention_layernorm(x1)
    moe = blk.mlp(norm2)
    mx.eval(norm1, mixer, norm2, moe, gdn_readout)

    # GDN readout (pre gated-norm): HF stored it flattened [N, Dv]; reshape to [T, Hv, Dv]
    hf_ro = mx.array(np.asarray(d["gdn_readout"], dtype=np.float32)).reshape(t, m.hv, m.dv)
    ro_rel = float((mx.max(mx.abs(hf_ro - gdn_readout[0].astype(mx.float32)))
                    / (mx.max(mx.abs(hf_ro)) + 1e-6)).item())
    print(f"\n  GDN readout (pre gated-norm)  rel={ro_rel:.3e}{'   <== conv/recurrence bug' if ro_rel > 0.05 else '   (conv+recurrence OK ⇒ bug is gated-norm/out_proj)'}", flush=True)

    def rel(a_np, b_mx) -> float:
        a = mx.array(np.asarray(a_np, dtype=np.float32))
        b = b_mx[0].astype(mx.float32)
        return float((mx.max(mx.abs(a - b)) / (mx.max(mx.abs(a)) + 1e-6)).item())

    print("\nlayer-0 sub-op rel-err vs HF (first divergence = the bug):", flush=True)
    for name, q in (("norm1 (input_layernorm)", norm1), ("mixer (GatedDeltaNet)", mixer),
                    ("norm2 (post_attn_norm)", norm2), ("moe (MoE block)", moe)):
        r = rel(d[name.split()[0]], q)
        print(f"  {name:28} rel={r:.3e}{'   <== DIVERGES' if r > 0.05 else ''}", flush=True)


def run() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "diff"
    {"hf": phase_hf, "diff": phase_diff}[mode]()


if __name__ == "__main__":
    run()
