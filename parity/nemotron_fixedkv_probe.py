"""Probe: fixed-size KV cache + full-step mx.compile for Nemotron decode (#41 perf).

Collapses the whole 88-layer decode step into ONE compiled graph (vs the current 80 per-mixer
compiled calls + eager attention/glue). Attention uses a static [1,nkv,MAX,hd] KV buffer written
at an array offset (slice_update) with an array-offset causal mask — so shapes are static and the
step compiles. Validates greedy-token parity vs the eager growing-KV path, then times tok/s.

    uv run --with tokenizers python -m parity.nemotron_fixedkv_probe
"""

from __future__ import annotations

import time

import mlx.core as mx

import quanta.nemotron.mamba_mixer as mm
from quanta.nemotron.generate import attn_caches
from quanta.nemotron.runtime import NemotronResidentModel

mm.FUSED_SSD_STEP = False
ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"
MAX = 2048


def _make_step(m):
    cfg = m.cfg

    def step(token, kbufs, vbufs, offset, ssms, convs):
        h = m.embed_w[token][None].astype(mx.bfloat16)
        ai = mi = 0
        for blk in m.layers:
            if blk.kind == "mamba":
                y, ssms[mi], convs[mi] = blk.mixer(blk.norm(h), state=ssms[mi], conv_state=convs[mi])
                h = h + y
                mi += 1
            elif blk.kind == "moe":
                h = h + blk.mixer(blk.norm(h))
            else:
                a, hn = blk.mixer, blk.norm(h)
                q = a.q_proj(hn).reshape(1, 1, a.nh, a.hd).transpose(0, 2, 1, 3)
                k = a.k_proj(hn).reshape(1, 1, a.nkv, a.hd).transpose(0, 2, 1, 3)
                v = a.v_proj(hn).reshape(1, 1, a.nkv, a.hd).transpose(0, 2, 1, 3)
                q = mx.fast.rope(q, a.hd, traditional=False, base=a.theta, scale=1.0, offset=offset)
                k = mx.fast.rope(k, a.hd, traditional=False, base=a.theta, scale=1.0, offset=offset)
                z = mx.array(0, mx.uint32)
                starts = mx.stack([z, z, offset.astype(mx.uint32), z])
                kbufs[ai] = mx.slice_update(kbufs[ai], k, starts, axes=(0, 1, 2, 3))
                vbufs[ai] = mx.slice_update(vbufs[ai], v, starts, axes=(0, 1, 2, 3))
                kr, vr = mx.repeat(kbufs[ai], a.rep, axis=1), mx.repeat(vbufs[ai], a.rep, axis=1)
                mask = mx.where(mx.arange(MAX) <= offset, mx.array(0.0, mx.bfloat16),
                                mx.array(-mx.inf, mx.bfloat16))[None, None, None]
                o = mx.fast.scaled_dot_product_attention(q, kr, vr, scale=a.scale, mask=mask)
                h = h + a.o_proj(o.transpose(0, 2, 1, 3).reshape(1, 1, a.nh * a.hd))
                ai += 1
        h = mx.fast.rms_norm(h, m.norm_f.astype(h.dtype), cfg.norm_eps)
        return (h @ m.lm_head_w.T), kbufs, vbufs, ssms, convs
    return step


def _fixed_from_prefill(m, ids):
    """Eager prefill (growing KV) → copy into static [1,nkv,MAX,hd] buffers; return decode state."""
    caches = attn_caches(m)
    logits, ssm, conv = m(mx.array(ids), caches=caches)
    plen = len(ids)
    a = next(b.mixer for b in m.layers if b.kind == "attention")
    kbufs, vbufs = [], []
    for c in caches:
        if c is None:
            continue
        kb = mx.zeros((1, a.nkv, MAX, a.hd), mx.bfloat16)
        vb = mx.zeros((1, a.nkv, MAX, a.hd), mx.bfloat16)
        kb[:, :, :plen] = c.k.astype(mx.bfloat16)
        vb[:, :, :plen] = c.v.astype(mx.bfloat16)
        kbufs.append(kb)
        vbufs.append(vb)
    ssms = [s for s, b in zip(ssm, m.layers) if b.kind == "mamba"]
    convs = [cv for cv, b in zip(conv, m.layers) if b.kind == "mamba"]
    return logits, kbufs, vbufs, mx.array(plen, mx.int32), ssms, convs


def _eager(m, ids, K):
    caches = attn_caches(m)
    logits, ssm, conv = m(mx.array(ids), caches=caches)
    cur = int(mx.argmax(logits[0, -1]).item())
    out = [cur]
    for _ in range(K):
        logits, ssm, conv = m(mx.array([cur]), caches=caches, ssm=ssm, conv=conv)
        cur = int(mx.argmax(logits[0, -1]).item())
        out.append(cur)
    return out


def run() -> None:
    mx.set_wired_limit(int(120 * 1024**3))
    m = NemotronResidentModel(ART)
    ids = list(range(20))
    step = mx.compile(_make_step(m))

    # parity: compiled fixed-KV greedy tokens vs eager growing-KV
    def fixed(K):
        logits, kb, vb, off, ssms, convs = _fixed_from_prefill(m, ids)
        cur = mx.argmax(logits[0, -1])[None].astype(mx.int32)
        out = [int(cur.item())]
        for _ in range(K):
            logits, kb, vb, ssms, convs = step(cur, kb, vb, off, ssms, convs)
            off = off + 1
            cur = mx.argmax(logits[0, -1])[None].astype(mx.int32)
            out.append(int(cur.item()))
        return out
    ref = _eager(m, ids, 24)
    fx = fixed(24)
    print("fixed-KV compiled == eager:", fx == ref)
    print("  eager:", ref[:10])
    print("  fixed:", fx[:10])

    # bench
    logits, kb, vb, off, ssms, convs = _fixed_from_prefill(m, ids)
    cur = mx.argmax(logits[0, -1])[None].astype(mx.int32)
    for _ in range(8):
        logits, kb, vb, ssms, convs = step(cur, kb, vb, off, ssms, convs)
        off = off + 1
        cur = mx.argmax(logits[0, -1])[None].astype(mx.int32)
        mx.eval(cur, off)
    N = 128
    t0 = time.perf_counter()
    for _ in range(N):
        logits, kb, vb, ssms, convs = step(cur, kb, vb, off, ssms, convs)
        off = off + 1
        cur = mx.argmax(logits[0, -1])[None].astype(mx.int32)
        mx.eval(cur, off)
    print(f"fixed-KV full-step compiled: {N / (time.perf_counter() - t0):.1f} tok/s  "
          f"[eager 30 | per-mixer compiled 35]")


if __name__ == "__main__":
    run()
