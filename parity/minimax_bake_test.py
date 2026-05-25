"""Model-free gate for the MiniMax-M2.7 bake (#100) — tiny synthetic tensors only.

Exercises the bake's quant paths on a few-KB of random data (NO checkpoint, NO GPU, no big
allocations), per the host-OOM safety rule:

(a) **int6 GPTQ expert path** — GPTQ a small stacked SwiGLU expert (gate ``w1`` on routed rows, down
    ``w2`` on the SwiGLU intermediate), pack via the project packer, ``mx.dequantize`` back, and assert
    the activation-weighted reconstruction is within an int6 bound AND no worse than plain RTN (error
    feedback must help).
(b) **int6 pack round-trip (the #100 caveat)** — the project certifies the affine packer
    ``== mx.quantize`` only for bits 3/4/8; ``32 % 6 == 2`` straddles word boundaries. Assert, on real
    GPTQ codes, that ``unpack(pack(codes)) == codes`` AND ``dequant(pack(codes))`` lies on the
    ``mx.quantize`` grid bit-exactly — via the bake's own :func:`_assert_int6_packs` (the fail-loud
    guard the real bake runs before writing any int6 codes, rule 6). Also checks the real-bake dims
    (hidden=3072, inter=1536, g128) satisfy the packer's ``in·bits % 32 == 0`` constraint.
(c) **int8 dense path** — int8-affine a small non-expert weight; recon must be tight.
(d) **manifest round-trip** — a tiny :class:`ArtifactWriter` bake (1 layer, 4 experts: warm + cold)
    through :class:`quanta.minimax.artifact.MiniMaxArtifact`: experts/attention tagged ``affine_packed``
    at the right bits, router/norms ``dense``, dequant matches the writer's grid, keys/shapes/params
    correct, NO shared expert, NO ``awq_scale``.

    uv run --with numpy python -m parity.minimax_bake_test

deferred (run later on GPU, #100): the real bake + e2e teacher-forced ppl — see the docstring of
:mod:`quanta.minimax.bake` for the exact (un-run) invocation.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx

from quanta.bake.gptq import gptq_quantize_batch
from quanta.bake.quant import dequantize_affine, pack_affine, quantize_affine, unpack_affine
from quanta.minimax.artifact import MiniMaxArtifact
from quanta.minimax.bake import (
    _assert_int6_packs,
    _bake_experts_layer,
    _down_input,
    _write_int8,
)
from quanta.minimax.config import MiniMaxConfig

GS = 32       # tiny-tensor group size (real bake uses 128); divides the synthetic in-dims 32/64
BITS = 6      # the MiniMax routed-expert width
INTER, DIM, NE, NTOK = 64, 32, 4, 20   # one layer: 4 experts, FFN width 64, hidden 32, 20 calib rows


def _rel(num: mx.array, den: mx.array) -> float:
    return float((mx.linalg.norm(num.astype(mx.float32))
                  / (mx.linalg.norm(den.astype(mx.float32)) + 1e-12)).item())


def _awl(w: mx.array, wq: mx.array, x: mx.array) -> float:
    """Activation-weighted error ``‖(Ŵ−W)Xᵀ‖/‖WXᵀ‖`` — the quantity GPTQ minimizes."""
    wf, xt = w.astype(mx.float32), x.astype(mx.float32).T
    return float((mx.linalg.norm((wq.astype(mx.float32) - wf) @ xt)
                  / (mx.linalg.norm(wf @ xt) + 1e-9)).item())


def _tiny_cfg() -> MiniMaxConfig:
    """A few-KB MiniMax-M2.7 config: real flags (sigmoid routing, no shared expert), tiny dims."""
    return MiniMaxConfig(
        vocab_size=64, hidden_size=DIM, moe_intermediate_size=INTER, num_hidden_layers=1,
        num_attention_heads=6, num_key_value_heads=2, head_dim=8, rotary_dim=4,
        use_qk_norm=True, qk_norm_type="per_layer", attn_type_list=(1,),
        num_local_experts=NE, num_experts_per_tok=2, shared_intermediate_size=0,
        scoring_func="sigmoid", use_routing_bias=True, norm_topk_prob=True, routed_scaling_factor=1.0,
        use_mtp=False, num_mtp_modules=0, mtp_transformer_layers=1,
        hidden_act="silu", norm_eps=1e-6, rope_theta=5e6, max_position_embeddings=204800,
        bos_token_id=200019, eos_token_id=200020, eos_token_ids=(200020,),
        tie_word_embeddings=False, quantization_config={},
    )


def _gptq_expert_path() -> tuple[bool, float, float, float]:
    """(a) int6 GPTQ on a small SwiGLU expert: gate ``w1`` on routed rows + down ``w2`` on the SwiGLU
    intermediate; GPTQ must beat RTN (activation-weighted) and reconstruct within an int6 bound."""
    rng = mx.random.key(0)
    k1, k3, kx = mx.random.split(rng, 3)
    w1 = mx.random.normal((INTER, DIM), key=k1) * 0.1   # gate [inter, dim]
    w3 = mx.random.normal((INTER, DIM), key=k3) * 0.1   # up   [inter, dim]
    xe = mx.random.normal((NTOK, DIM), key=kx)          # routed post-norm rows [n, dim]

    # gate w1 (GPTQ input = xe): batched solver (1 expert) -> pack -> dequantize
    codes, sc, b = gptq_quantize_batch(w1[None], [xe], BITS, group_size=GS)
    wq_g = mx.dequantize(pack_affine(codes[0].astype(mx.uint32), BITS), sc[0], b[0],
                         group_size=GS, bits=BITS)
    wr_g = dequantize_affine(*quantize_affine(w1, BITS, GS), BITS, GS)  # plain RTN baseline
    err_g, rtn_g = _awl(w1, wq_g, xe), _awl(w1, wr_g, xe)

    # down w2 (GPTQ input = SwiGLU intermediate) — the bake's exact down-proj calibration input
    w2 = mx.random.normal((DIM, INTER), key=mx.random.key(1)) * 0.1     # down [dim, inter]
    xd = _down_input(w1, w3, xe)                                        # [n, inter]
    codes2, sc2, b2 = gptq_quantize_batch(w2[None], [xd], BITS, group_size=GS)
    wq_d = mx.dequantize(pack_affine(codes2[0].astype(mx.uint32), BITS), sc2[0], b2[0],
                         group_size=GS, bits=BITS)
    err_d = _awl(w2, wq_d, xd)

    ok = err_g < 0.05 and err_d < 0.05 and err_g <= rtn_g * 1.02
    return ok, err_g, rtn_g, err_d


def _int6_pack_roundtrip() -> tuple[bool, bool]:
    """(b) The #100 int6 packing caveat. On real GPTQ codes: the bake's fail-loud guard must pass
    (``unpack(pack)==codes`` + grid-exact), and the real-bake dims must satisfy ``in·bits % 32 == 0``."""
    rng = mx.random.key(7)
    kw, kx = mx.random.split(rng, 2)
    w = mx.random.normal((INTER, DIM), key=kw) * 0.1
    x = mx.random.normal((NTOK, DIM), key=kx)
    codes, sc, b = gptq_quantize_batch(w[None], [x], BITS, group_size=GS)

    # Direct round-trip assertion (independent of the guard): bitstream reversible AND the packed
    # words decode (via mx.dequantize) to exactly the affine formula codes·scale+bias.
    packed = pack_affine(codes[0].astype(mx.uint32), BITS)
    rt = unpack_affine(packed, DIM, BITS).astype(mx.int32)
    codes_exact = bool(mx.all(rt == codes[0]).item())
    deq = mx.dequantize(packed, sc[0], b[0], group_size=GS, bits=BITS).astype(mx.float32)
    scb = mx.repeat(sc[0].astype(mx.float32), GS, axis=1)[:, :DIM]
    bib = mx.repeat(b[0].astype(mx.float32), GS, axis=1)[:, :DIM]
    manual = codes[0].astype(mx.float32) * scb + bib
    grid_exact = mx.max(mx.abs(deq - manual)).item() < 1e-3

    # The bake's own guard must NOT raise on valid int6 codes (and would FAIL LOUD on bad bits).
    guard_ok = True
    try:
        _assert_int6_packs(codes[0], sc[0], b[0], DIM, BITS, GS)
    except ValueError:
        guard_ok = False

    # Real-bake dims (hidden=3072 for w1/w3, inter=1536 for w2; group 128) must pack cleanly.
    dims_ok = all((d * BITS) % 32 == 0 and d % 128 == 0 for d in (3072, 1536))

    return (codes_exact and grid_exact and guard_ok), dims_ok


def _int8_path() -> tuple[bool, float]:
    """(c) int8 affine on a small non-expert (GQA projection) weight; recon must be tight."""
    w = mx.random.normal((48, 64), key=mx.random.key(2))
    p, sc, b = quantize_affine(w, 8, GS)
    err = _rel(w - mx.dequantize(p, sc, b, group_size=GS, bits=8), w)
    return err < 0.02, err


def _manifest_roundtrip() -> tuple[bool, bool, dict]:
    """(d) Tiny bake into a tempdir via the bake helpers, then round-trip through MiniMaxArtifact:
    formats/bits correct, dequant matches the writer grid, shapes/params right, no shared expert/AWQ."""
    cfg = _tiny_cfg()
    out = tempfile.mkdtemp(suffix="_minimax_bake_gate")
    try:
        from quanta.bake.artifact import ArtifactWriter
        (Path(out) / "config.json").write_text(json.dumps({"model_type": "minimax_m2"}))
        writer = ArtifactWriter(out, Path(out) / "config.json")

        rng = mx.random.key(3)
        kq, kn, kg, kb, k1, k3, k2, kx = mx.random.split(rng, 8)
        pre = "model.layers.0."
        # int8 GQA q/k/v/o projections + dense q_norm/k_norm + dense norms + dense router. Shapes are
        # GS-divisible (not architecturally exact) — the gate validates the quant/reader paths, not geometry.
        for proj, (oh, ih) in (("q_proj", (48, DIM)), ("k_proj", (16, DIM)),
                               ("v_proj", (16, DIM)), ("o_proj", (DIM, 64))):
            _write_int8(writer, f"{pre}self_attn.{proj}", mx.random.normal((oh, ih), key=kq), GS, mx.bfloat16)
        writer.add_dense(f"{pre}self_attn.q_norm.weight", mx.random.normal((8,), key=kn).astype(mx.bfloat16))
        writer.add_dense(f"{pre}self_attn.k_norm.weight", mx.random.normal((8,), key=kn).astype(mx.bfloat16))
        writer.add_dense(f"{pre}input_layernorm.weight", mx.random.normal((DIM,), key=kn).astype(mx.bfloat16))
        writer.add_dense(f"{pre}post_attention_layernorm.weight", mx.random.normal((DIM,), key=kn).astype(mx.bfloat16))
        writer.add_dense(f"{pre}block_sparse_moe.gate.weight", mx.random.normal((NE, DIM), key=kg).astype(mx.bfloat16))
        writer.add_dense(f"{pre}block_sparse_moe.e_score_correction_bias",
                         (mx.random.normal((NE,), key=kb) * 0.5).astype(mx.bfloat16))

        # int6-GPTQ experts via the bake's real per-layer path: warm experts (rows) + a cold one (none).
        es = {
            "w1": mx.random.normal((NE, INTER, DIM), key=k1) * 0.1,
            "w3": mx.random.normal((NE, INTER, DIM), key=k3) * 0.1,
            "w2": mx.random.normal((NE, DIM, INTER), key=k2) * 0.1,
        }
        # routing idx (deterministic): experts 0,1,2 each get rows (warm); expert 3 never -> cold RTN.
        # Slot 0 cycles 0,1,2,0,...; slot 1 cycles 1,2,0,... so all of {0,1,2} appear, never 3.
        col0 = mx.arange(NTOK, dtype=mx.int32) % (NE - 1)
        col1 = (mx.arange(NTOK, dtype=mx.int32) + 1) % (NE - 1)
        idx = mx.stack([col0, col1], axis=1)  # [NTOK, 2], values in {0,1,2}
        ln2 = mx.random.normal((NTOK, DIM), key=kx).astype(mx.bfloat16)
        in_dims = {"w1": DIM, "w3": DIM, "w2": INTER}
        warm = _bake_experts_layer(writer, f"{pre}block_sparse_moe.experts.", es, ln2, idx,
                                   list(range(NE)), BITS, GS, "gptq", in_dims, mx.bfloat16)
        writer.finalize({"experts": f"int6 gptq g{GS}", "non_experts": f"int8 g{GS}",
                         "shared_expert": "none"})

        man = json.loads((Path(out) / "manifest.json").read_text())["tensors"]
        wmap = json.loads((Path(out) / "model.safetensors.index.json").read_text())["weight_map"]

        q8 = man[f"{pre}self_attn.q_proj"]
        e_w1 = man[f"{pre}block_sparse_moe.experts.0.w1"]
        e_w2_cold = man[f"{pre}block_sparse_moe.experts.3.w2"]
        norm = man[f"{pre}input_layernorm.weight"]
        gate = man[f"{pre}block_sparse_moe.gate.weight"]

        suffix_ok = (all(f"{pre}self_attn.q_proj{s}" in wmap
                         for s in (".weight_packed", ".weight_scale", ".weight_bias"))
                     and all(f"{pre}block_sparse_moe.experts.0.w1{s}" in wmap
                             for s in (".weight_packed", ".weight_scale", ".weight_bias")))
        no_awq = not any(k.endswith(".awq_scale") for k in wmap)  # GPTQ path: never an AWQ scale
        fmt_ok = (q8["format"] == "affine_packed" and q8["bits"] == 8 and q8["group_size"] == GS
                  and e_w1["format"] == "affine_packed" and e_w1["bits"] == BITS and e_w1["group_size"] == GS
                  and e_w2_cold["format"] == "affine_packed" and e_w2_cold["bits"] == BITS  # cold = RTN, same fmt
                  and norm["format"] == "dense" and gate["format"] == "dense"
                  and suffix_ok and no_awq and warm == 3)

        # Round-trip through the artifact reader: dequant must match the writer's stored grid.
        art = MiniMaxArtifact(out, cfg)
        # int8 GQA projection
        q_ref = mx.dequantize(art.get(f"{pre}self_attn.q_proj.weight_packed"),
                              art.get(f"{pre}self_attn.q_proj.weight_scale"),
                              art.get(f"{pre}self_attn.q_proj.weight_bias"), group_size=GS, bits=8)
        attn = art.attention(0)
        q_match = _rel(attn["q_proj"].astype(mx.float32) - q_ref.astype(mx.float32), q_ref) < 1e-3
        shape_ok = (attn["q_proj"].shape == (48, DIM) and attn["q_norm"].shape == (8,))
        # int6 expert stacks: dequant via reader == dequant of stored packed codes
        stacks = art.expert_stacks(0, NE)
        e0 = mx.dequantize(art.get(f"{pre}block_sparse_moe.experts.0.w1.weight_packed"),
                           art.get(f"{pre}block_sparse_moe.experts.0.w1.weight_scale"),
                           art.get(f"{pre}block_sparse_moe.experts.0.w1.weight_bias"),
                           group_size=GS, bits=BITS)
        e_match = _rel(stacks["w1"][0].astype(mx.float32) - e0.astype(mx.float32), e0) < 1e-3
        stacks_ok = (stacks["w1"].shape == (NE, INTER, DIM) and stacks["w2"].shape == (NE, DIM, INTER))
        # router dense round-trips; full MoE refuses a (non-existent) shared expert cleanly
        router = art.moe_router(0)
        router_ok = (router["weight"].shape == (NE, DIM)
                     and router["e_score_correction_bias"].shape == (NE,))
        moe_ok = "experts" in art.moe(0) and not cfg.has_shared_expert
        rt_ok = bool(q_match and e_match and shape_ok and stacks_ok and router_ok and moe_ok)
        return fmt_ok, rt_ok, {"q8": q8, "expert_w1": e_w1, "cold_w2": e_w2_cold,
                               "norm": norm["format"], "warm": warm}
    finally:
        shutil.rmtree(out, ignore_errors=True)


def run() -> None:
    mx.random.seed(0)
    gptq_ok, eg, rg, ed = _gptq_expert_path()
    (pack_ok, dims_ok) = _int6_pack_roundtrip()
    int8_ok, e8 = _int8_path()
    fmt_ok, rt_ok, formats = _manifest_roundtrip()

    print("\n=== MiniMax-M2.7 bake gate (#100, model-free, tiny synthetic) ===")
    print(f"(a) int6 GPTQ expert recon   : w1 {eg:.4f}<0.05 (RTN {rg:.4f})  w2 {ed:.4f}<0.05 -> {gptq_ok}")
    print(f"(b) int6 pack round-trip     : codes/grid exact -> {pack_ok}   real-dims in*6%32==0 -> {dims_ok}")
    print(f"(c) int8 dense non-expert    : {e8:.5f}<0.02 -> {int8_ok}")
    print(f"(d) manifest round-trip      : fmt {fmt_ok}  reader {rt_ok}  {formats}")
    ok = gptq_ok and pack_ok and dims_ok and int8_ok and fmt_ok and rt_ok
    assert ok, "MiniMax bake gate FAILED"
    print("PASS — int6 GPTQ reconstructs + packs bit-exactly; int8 tight; manifest round-trips")


if __name__ == "__main__":
    run()
