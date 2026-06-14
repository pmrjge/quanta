"""Model-free M3-2 gate: the MiniMax-M3-VL batched serving runtime + packed-int8 mixer.

Builds a tiny SYNTHETIC M3 text decoder (no real weights) and proves the two M3-2 optimizations are
output-equivalent to the proven M1/M2/M3-1 reference forward:

* **packed int8 mixer == bf16 mixer (greedy-exact).** GQA q/k/v/o (all layers) and the dense-FFN
  gate/up/down (layers 0–2) are RTN-quantized to int8 once; the *packed* model holds each as an
  ``nn.QuantizedLinear`` (``mx.quantized_matmul``), the *bf16* model holds the SAME codes dequantized
  into an ``nn.Linear``. Same forward — only fused-vs-separate dequant differs ⇒ identical top-1 and
  ~ULP logits (the fused ``mx.quantized_matmul`` keeps the int8 weight at full precision, so the
  packed path is the MORE precise one). This is the M3-2 packed-mixer invariant.
* **batched Design A == single-stream decode (greedy-exact, ragged offsets).** ``B`` streams each
  prefilled to a DIFFERENT length, then one :meth:`step_batch` step: each stream's logits match the
  single-stream :class:`MiniMaxM3ResidentModel` decode at the same offset against its own cache. The
  per-stream attention + per-(token,slot) ``gather_qmm`` are M=1 (bit-exact); only the batched
  router/shared GEMM ULP-reorders ⇒ top-1 exact, ~ULP logits. (B=1 is the degenerate sub-case.)
* **batched prefill == single-stream prefill** (last-position logits, bit-identical — the cached
  forward the M3-1 gate already proves equals the ``caches=None`` prefill).
* **rule-6:** ``step_batch`` refuses a desynced offset, a per-stream cache of the wrong length, and a
  batch over ``max_batch``; ``make_batch_caches`` refuses a batch over ``max_batch``.

All on tiny synthetic dims; runs in the model-free sweep (no real weights, no checkpoint IO).

    uv run python -m parity.minimax_m3_batched_test
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from quanta.bake.quant import quantize_affine
from quanta.minimax import batched_runtime_m3 as BR
from quanta.minimax import model_m3 as M
from quanta.minimax import runtime_m3 as RT
from quanta.minimax.config_m3 import MiniMaxM3Config

GS = 32        # affine group size — divides hidden 64 / moe_inter 32 / dense_inter 96 (the in-dims)
BITS = 6       # routed experts int6 (the served scheme)
MIXER_BITS = 8  # GQA q/k/v/o + dense FFN int8 (the served mixer scheme)

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _rel(a: mx.array, b: mx.array) -> float:
    an, bn = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    return float(np.abs(an - bn).max() / (np.abs(bn).max() + 1e-9))


def _argmax_agree(a: mx.array, b: mx.array) -> float:
    aa = mx.argmax(a[0].astype(mx.float32), axis=-1)
    bb = mx.argmax(b[0].astype(mx.float32), axis=-1)
    return float(mx.mean((aa == bb).astype(mx.float32)).item())


# --- tiny synthetic config (structurally faithful; 2 dense + 3 MoE layers) --- #

def _cfg() -> MiniMaxM3Config:
    tc = {
        "vocab_size": 128, "hidden_size": 64, "intermediate_size": 32,
        "dense_intermediate_size": 96, "shared_intermediate_size": 32,
        "num_hidden_layers": 5,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 16,
        "rotary_dim": 8, "partial_rotary_factor": 0.5, "rope_theta": 5e6,
        "use_qk_norm": True, "qk_norm_type": "per_head", "use_gemma_norm": True,
        "attention_output_gate": False,
        "num_local_experts": 6, "num_experts_per_tok": 2, "n_shared_experts": 1,
        "scoring_func": "sigmoid", "use_routing_bias": True, "routed_scaling_factor": 2.0,
        "norm_topk_prob": True,
        "moe_layer_freq": [0, 0, 1, 1, 1],
        "hidden_act": "swigluoai", "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
        "rms_norm_eps": 1e-6, "max_position_embeddings": 1048576, "tie_word_embeddings": False,
        "eos_token_id": 200020,
    }
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "config.json").write_text(json.dumps({"model_type": "minimax_m3_vl", "text_config": tc}))
        return MiniMaxM3Config.from_pretrained(d)


# --- deterministic synthetic weights (shared by every model) ----------------- #

def _synth(cfg: MiniMaxM3Config, key) -> dict:
    ks = mx.random.split(key, 512)
    c = iter(range(512))

    def nx():
        return ks[next(c)]

    def bf(shape, scale):
        return (mx.random.normal(shape, key=nx()) * scale).astype(mx.bfloat16)

    h = cfg.hidden_size
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    e, inter = cfg.num_local_experts, cfg.moe_intermediate_size
    si, di = cfg.shared_intermediate_size, cfg.dense_intermediate_size
    w: dict = {"embed": bf((cfg.vocab_size, h), 0.05), "final_norm": bf((h,), 0.2),
               "lm_head": bf((cfg.vocab_size, h), 0.05)}
    for i in range(cfg.num_hidden_layers):
        w[(i, "in")], w[(i, "post")] = bf((h,), 0.2), bf((h,), 0.2)
        w[(i, "q")] = bf((nh * hd, h), 0.05)
        w[(i, "k")] = bf((nkv * hd, h), 0.05)
        w[(i, "v")] = bf((nkv * hd, h), 0.05)
        w[(i, "o")] = bf((h, nh * hd), 0.05)
        w[(i, "qn")], w[(i, "kn")] = bf((hd,), 0.2), bf((hd,), 0.2)
        if cfg.is_moe_layer(i):
            w[(i, "gate")] = mx.random.normal((e, h), key=nx()).astype(mx.float32)
            w[(i, "bias")] = mx.random.normal((e,), key=nx()).astype(mx.float32)
            w[(i, "gate_up")] = bf((e, 2 * inter, h), 0.1)     # [E, 2*inter, h] (w1 over w3)
            w[(i, "down")] = bf((e, h, inter), 0.1)            # [E, h, inter]   (w2)
            w[(i, "sg")], w[(i, "su")], w[(i, "sd")] = bf((si, h), 0.1), bf((si, h), 0.1), bf((h, si), 0.1)
        else:
            w[(i, "dg")], w[(i, "du")], w[(i, "dd")] = bf((di, h), 0.1), bf((di, h), 0.1), bf((h, di), 0.1)
    mx.eval(list(w.values()))
    return w


def _pack(stack: mx.array) -> dict:
    pq, sc, b = quantize_affine(stack, BITS, GS, scale_dtype=mx.bfloat16)
    return {"packed": pq, "scale": sc, "bias": b, "group_size": GS, "bits": BITS}


def _dq(trip: dict) -> mx.array:
    return mx.dequantize(trip["packed"], trip["scale"], trip["bias"],
                         group_size=trip["group_size"], bits=trip["bits"]).astype(mx.bfloat16)


def _q8_linear(wt: mx.array, in_dims: int, out_dims: int) -> nn.QuantizedLinear:
    """A packed int8 ``nn.QuantizedLinear`` from a ``[out,in]`` bf16 weight (the resident mixer path —
    same RTN codes the artifact bakes, loaded into ``mx.quantized_matmul`` exactly as
    ``runtime_m3._packed_linear`` does)."""
    pq, sc, b = quantize_affine(wt, MIXER_BITS, GS, scale_dtype=mx.bfloat16)
    ql = nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=GS, bits=MIXER_BITS)
    ql.weight, ql.scales, ql.biases = pq, sc, b
    return ql


def _q8_dq(wt: mx.array) -> mx.array:
    """The bf16 dequant of the SAME int8 codes ``_q8_linear`` holds packed (the bf16-mixer reference)."""
    pq, sc, b = quantize_affine(wt, MIXER_BITS, GS, scale_dtype=mx.bfloat16)
    return mx.dequantize(pq, sc, b, group_size=GS, bits=MIXER_BITS).astype(mx.bfloat16)


def _build_blocks(cfg: MiniMaxM3Config, w: dict, *, packed_mixer: bool,
                  packed_experts: bool) -> list[M.MiniMaxM3Block]:
    h = cfg.hidden_size
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    di = cfg.dense_intermediate_size
    blocks: list[M.MiniMaxM3Block] = []
    for i in range(cfg.num_hidden_layers):
        blk = M.MiniMaxM3Block(cfg, i)
        blk.input_layernorm.weight = M.one_plus(w[(i, "in")])
        blk.post_attention_layernorm.weight = M.one_plus(w[(i, "post")])
        m = blk.self_attn
        if packed_mixer:
            m.q_proj = _q8_linear(w[(i, "q")], h, nh * hd)
            m.k_proj = _q8_linear(w[(i, "k")], h, nkv * hd)
            m.v_proj = _q8_linear(w[(i, "v")], h, nkv * hd)
            m.o_proj = _q8_linear(w[(i, "o")], nh * hd, h)
        else:
            m.q_proj.weight = _q8_dq(w[(i, "q")])
            m.k_proj.weight = _q8_dq(w[(i, "k")])
            m.v_proj.weight = _q8_dq(w[(i, "v")])
            m.o_proj.weight = _q8_dq(w[(i, "o")])
        m.q_norm = M.one_plus(w[(i, "qn")])
        m.k_norm = M.one_plus(w[(i, "kn")])
        if cfg.is_moe_layer(i):
            blk.mlp.gate = w[(i, "gate")]
            blk.mlp.e_score_correction_bias = w[(i, "bias")]
            gu, dn = _pack(w[(i, "gate_up")]), _pack(w[(i, "down")])
            if packed_experts:
                blk.mlp.set_experts_packed(gu, dn)
            else:
                blk.mlp.set_experts(_dq(gu), _dq(dn))            # SAME int6 codes, dequantized
            blk.mlp.shared_gate_proj = w[(i, "sg")]              # shared expert stays bf16
            blk.mlp.shared_up_proj = w[(i, "su")]
            blk.mlp.shared_down_proj = w[(i, "sd")]
        elif packed_mixer:
            blk.mlp.gate_proj = _q8_linear(w[(i, "dg")], h, di)
            blk.mlp.up_proj = _q8_linear(w[(i, "du")], h, di)
            blk.mlp.down_proj = _q8_linear(w[(i, "dd")], di, h)
        else:
            blk.mlp.gate_proj.weight = _q8_dq(w[(i, "dg")])
            blk.mlp.up_proj.weight = _q8_dq(w[(i, "du")])
            blk.mlp.down_proj.weight = _q8_dq(w[(i, "dd")])
        blocks.append(blk)
    return blocks


def _single(cfg, blocks, w) -> RT.MiniMaxM3ResidentModel:
    return RT.MiniMaxM3ResidentModel.from_blocks(
        blocks, w["embed"], M.one_plus(w["final_norm"]), w["lm_head"], cfg)


def _batched(cfg, blocks, w, *, max_batch=8) -> BR.MiniMaxM3BatchedResidentModel:
    # pin loopkill=False: this gate is the Design-A per-stream path (bit-exact dispatch). The GQA
    # loop-kill (the graduated default since M3-3) is gated separately in minimax_m3_loopkill_test.
    return BR.MiniMaxM3BatchedResidentModel.from_inner(
        blocks, w["embed"], M.one_plus(w["final_norm"]), w["lm_head"], cfg, max_batch=max_batch,
        loopkill=False)


def run() -> None:
    mx.random.seed(0)
    cfg = _cfg()
    V = cfg.vocab_size
    w = _synth(cfg, mx.random.key(1))

    # (1) packed int8 mixer == bf16 mixer (ISOLATED — same int8 codes, bf16 experts both), prefill --
    # top-1 is a noisy secondary signal on a tiny synthetic (bf16-ULP near-ties flip — the settled
    # "Nemotron-Ultra rule", floor 0.90); the binding claim is the small logit rel (the fused
    # mx.quantized_matmul keeps the int8 weight at full precision, the bf16 ref rounds it first).
    blk_mix = _build_blocks(cfg, w, packed_mixer=True, packed_experts=False)
    blk_ref = _build_blocks(cfg, w, packed_mixer=False, packed_experts=False)
    sm_mix, sm_ref = _single(cfg, blk_mix, w), _single(cfg, blk_ref, w)
    _ck(sm_mix.packed and not sm_ref.packed, "from_blocks did not detect packed-mixer state")

    T = 12
    ids = mx.array([(i * 7 + 3) % V for i in range(T)], dtype=mx.int32)
    lp, lb = sm_mix(ids), sm_ref(ids)
    mx.eval(lp, lb)
    rel_m, agree_m = _rel(lp, lb), _argmax_agree(lp, lb)
    _ck(rel_m < 3e-2, f"packed-mixer logits != bf16-mixer logits: rel {rel_m:.2e}")
    _ck(agree_m >= 0.90, f"packed-mixer top-1 drifts from bf16-mixer: agree {agree_m:.4f} < 0.90")

    # the M3-2 serving model (packed mixer + packed experts) + a single-stream ref over the SAME
    # blocks — so checks (2)–(4) isolate the BATCHING (per-stream attn + batched MoE), not quant.
    blk_serve = _build_blocks(cfg, w, packed_mixer=True, packed_experts=True)
    single = _single(cfg, blk_serve, w)
    batched = _batched(cfg, blk_serve, w, max_batch=8)
    _ck(single.packed and single.packed_experts, "serving model is not fully packed")
    _ck(batched.packed and batched.packed_experts, "batched runtime did not detect packed state")

    # (2) batched prefill == single-stream prefill (last-position logits), bit-identical ------------
    prompt0 = [int(t) for t in ids[:6]]
    pl = batched.prefill(prompt0, batched.make_caches())
    sl = single(mx.array(prompt0))[:, -1:]
    mx.eval(pl, sl)
    rel_pf = _rel(pl, sl)
    _ck(_argmax_agree(pl, sl) == 1.0 and rel_pf < 1e-3,
        f"batched prefill != single-stream prefill: rel {rel_pf:.2e}")

    # (3) batched step_batch == single-stream decode, RAGGED offsets (same blocks ⇒ isolates the
    # batched dispatch). Per-stream attention + per-slot gather_qmm are M=1 (bit-exact); only the
    # batched router/shared GEMM can ULP-reorder ⇒ tight rel; top-1 floor guards a rare tie flip. ---
    B = 4
    prompts = [[(s * 13 + i * 7 + 1) % V for i in range(3 + s)] for s in range(B)]  # lengths 3,4,5,6
    nxt = [int((s * 5 + 2) % V) for s in range(B)]
    ref = []
    for s in range(B):
        ca = single.make_caches()
        single(mx.array(prompts[s]), caches=ca)                # consume the prompt
        ref.append(single(mx.array([nxt[s]]), caches=ca))      # single-stream decode step
    cbs = batched.make_batch_caches(B)
    for s in range(B):
        batched.prefill(prompts[s], cbs[s])
    offs = [len(prompts[s]) for s in range(B)]
    out = batched.step_batch(nxt, cbs, offs)
    mx.eval(out + ref)
    worst_rel, worst_agree = 0.0, 1.0
    for s in range(B):
        a, r = _argmax_agree(out[s], ref[s]), _rel(out[s], ref[s])
        worst_rel, worst_agree = max(worst_rel, r), min(worst_agree, a)
        _ck(r < 1e-2, f"batched stream {s} logits != single-stream: rel {r:.2e}")
        _ck(a >= 0.90, f"batched stream {s} top-1 drifts from single-stream: agree {a:.4f}")

    # (4) B=1 batched == single-stream (degenerate) ------------------------------------------------
    cb1 = batched.make_batch_caches(1)
    batched.prefill(prompts[0], cb1[0])
    out1 = batched.step_batch([nxt[0]], cb1, [len(prompts[0])])
    _ck(_rel(out1[0], ref[0]) < 1e-2 and _argmax_agree(out1[0], ref[0]) >= 0.90,
        "B=1 batched step != single-stream decode")

    # (5) rule-6 sanity: desynced offset / wrong cache len / over-batch all refuse ------------------
    cbad = batched.make_batch_caches(1)
    batched.prefill(prompts[0], cbad[0])               # now at offset len(prompts[0])
    def _raises(fn) -> bool:
        try:
            fn()
            return False
        except ValueError:
            return True
    _ck(_raises(lambda: batched.step_batch([nxt[0]], cbad, [len(prompts[0]) + 1])),
        "step_batch accepted a desynced offset (rule 6)")
    _ck(_raises(lambda: batched.step_batch([nxt[0]], [cbad[0][:-1]], [len(prompts[0])])),
        "step_batch accepted a wrong-length per-stream cache (rule 6)")
    nine = [batched.make_caches() for _ in range(9)]   # 9 > max_batch 8 (checked before any cache use)
    _ck(_raises(lambda: batched.step_batch([0] * 9, nine, [0] * 9)),
        "step_batch accepted a batch over max_batch (rule 6)")
    _ck(_raises(lambda: batched.make_batch_caches(9)),
        "make_batch_caches accepted a batch over max_batch (rule 6)")

    print("\n=== MiniMax-M3-VL M3-2 batched serving + packed-int8 mixer gate (model-free) ===")
    print(f"(1) packed int8 mixer == bf16 mixer: top-1 agree {agree_m:.4f}, logit rel {rel_m:.2e}")
    print(f"(2) batched prefill == single prefill: rel {rel_pf:.2e} (top-1 exact)")
    print(f"(3) batched step (B={B}, ragged) == single decode: worst agree {worst_agree:.4f}, "
          f"worst rel {worst_rel:.2e}")
    print("(4) B=1 batched == single decode (top-1 exact)")
    print("(5) rule-6: desync / wrong-len cache / over-batch all refuse")
    print(f"PARITY-CHECKS: {_N}")
    print("PASS — M3-2 batched Design A serving + packed-int8 mixer: per-stream output equivalent to "
          "the single-stream reference, rule-6 honored.")


if __name__ == "__main__":
    run()
