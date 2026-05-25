"""MiniMax-M2.7 bake -> a self-contained int6(GPTQ)/int8/bf16 artifact (parity-first).

Streamed, one layer resident at a time (rule 8), mirroring the DSV4 / Nemotron bakes
(:mod:`quanta.dsv4.bake`, :mod:`quanta.nemotron.bake`) but with MiniMax's recipe:

* **routed experts** (Mixtral ``w1`` gate / ``w3`` up / ``w2`` down) -> **int6 GPTQ g128**. The
  block-fp8 source dequantizes to bf16 (via :class:`quanta.minimax.loader.MiniMaxSourceCheckpoint`),
  so it has full sub-int6-grid headroom for GPTQ's error feedback. ``w1``/``w3`` calibrate on the
  expert's routed post-attn-norm rows; ``w2`` on the SwiGLU intermediate ``silu(gate·x)·(up·x)`` of
  those rows. Cold experts (no routed rows) fall back to plain int6 RTN. GPTQ codes are packed into
  the MLX affine layout (``affine_packed``) the resident ``gather_qmm`` consumes — the same path as
  the int8 dense weights, so the runtime is uniform.
* **non-experts** (GQA ``q/k/v/o`` projections) -> **int8** affine g128.
* **control tensors** (router ``gate`` + ``e_score_correction_bias``, all norms incl. per-layer
  ``q_norm``/``k_norm``, embeddings, ``lm_head``, final norm) -> **bf16** dense, verbatim.
* **NO shared expert** (``shared_intermediate_size == 0``) — refuse to invent one (rule 6).

⚠️ **int6 packing is validated at runtime here.** CLAUDE.md only certifies the affine pack/unpack
``== mx.quantize`` for bits 3/4/8; int6 straddles 32-bit word boundaries (``32 % 6 == 2``). For
group-128 weights ``in`` is a multiple of 128 so ``in·6`` is a multiple of 32 and the contiguous
bitstream packs cleanly — :func:`_assert_int6_packs` proves ``unpack(pack(codes)) == codes`` and
``dequant(pack(codes)) == mx.dequantize(mx.quantize(...))`` bit-exactly before any int6 codes are
written, and **fails loud** (rule 6) if a future shape ever breaks that. The model-free gate
(:mod:`parity.minimax_bake_test`) runs the same check.

Two streamed passes mirror the sibling bakes: (1) capture per-layer post-attn-norm activations +
routing for GPTQ; (2) write the artifact (one block's bf16 source resident at a time). Runnable on a
slice (``n_layers``, ``expert_subset``) for bounded validation; the full call is the real bake.

DEFERRED real bake (GPU + memory heavy — DO NOT run here; host OOM rebooted the box once):

    from quanta.minimax.bake import bake_minimax
    import mlx.core as mx
    bake_minimax("/Users/pmrj/models/MiniMax-M2.7",
                 "/Users/pmrj/models/MiniMax-M2.7-quanta_int6",
                 calib_ids, group_size=128, expert_method="gptq", scale_dtype=mx.bfloat16)
    # then teacher_forced_ppl over the resident int6/int8 runtime vs the bf16 reference.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from quanta.bake.artifact import ArtifactWriter
from quanta.bake.calibrate import expert_rows
from quanta.bake.gptq import gptq_quantize_batch
from quanta.bake.quant import pack_affine, quantize_affine, unpack_affine
from quanta.minimax.calibrate import capture_calibration
from quanta.minimax.config import MiniMaxConfig
from quanta.minimax.loader import MiniMaxSourceCheckpoint

EMBED, NORMF, HEAD = "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"
_EXPERT_BITS = 6
_INT8_BITS = 8
_ATTN_MATMULS = ("q_proj", "k_proj", "v_proj", "o_proj")  # GQA projections -> int8
_EXPERT_PROJS = ("w1", "w3", "w2")                        # Mixtral: gate, up, down
EXPERT_CHUNK = 16  # experts batched per GPTQ call (bounds the per-expert R [chunk,in,in] memory)


def _assert_int6_packs(codes: mx.array, scales: mx.array, biases: mx.array, in_: int,
                       bits: int, group_size: int) -> None:
    """Fail loud (rule 6) unless the affine packer round-trips these ``bits`` codes bit-exactly.

    Proves, on the *actual* code tensor about to be written, that (1) ``unpack(pack(codes)) == codes``
    (the bitstream is reversible) and (2) ``mx.dequantize(pack(codes), scales, biases)`` reproduces the
    affine formula ``codes·scale + bias`` (group-broadcast) — i.e. the resident ``mx.dequantize`` /
    ``gather_qmm`` decodes *exactly* the GPTQ codes from the packed words. The project only certifies
    the packer ``== mx.quantize`` for bits 3/4/8; int6 (``32 % 6 == 2`` straddles word boundaries) is
    re-validated here on real codes so wrong bits can never be emitted silently."""
    n_groups = (in_ + group_size - 1) // group_size
    if scales.shape[1] != n_groups or biases.shape[1] != n_groups:
        raise ValueError(f"int{bits} scale/bias grid {tuple(scales.shape)} != expected n_groups "
                         f"{n_groups} (in={in_}, group_size={group_size}) — refusing to emit (rule 6)")
    c = codes.astype(mx.uint32)
    packed = pack_affine(c, bits)
    rt = unpack_affine(packed, in_, bits).astype(mx.uint32)
    if not bool(mx.all(rt == c).item()):
        raise ValueError(f"int{bits} pack round-trip FAILED (in={in_}, group_size={group_size}): "
                         f"unpack(pack(codes)) != codes — refusing to emit wrong bits (rule 6)")
    deq = mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits).astype(mx.float32)
    sc = mx.repeat(scales.astype(mx.float32), group_size, axis=1)[:, :in_]   # group-broadcast scale
    bi = mx.repeat(biases.astype(mx.float32), group_size, axis=1)[:, :in_]
    manual = c.astype(mx.float32) * sc + bi                                  # affine formula, codes·s+b
    if mx.max(mx.abs(deq - manual)).item() > 1e-3:
        raise ValueError(f"int{bits} mx.dequantize(pack(codes)) != codes·scale+bias (in={in_}) — the "
                         f"packed bitstream does not decode to the GPTQ codes; refusing (rule 6)")


def _write_int8(writer: ArtifactWriter, key: str, w: mx.array, gs: int,
                scale_dtype: mx.Dtype | None) -> None:
    """int8 affine-quantize a 2-D weight ``[out,in]`` and add it under ``key`` (``affine_packed``)."""
    writer.add_quantized(key, *quantize_affine(w, _INT8_BITS, gs, scale_dtype=scale_dtype), _INT8_BITS, gs)


def _write_affine_codes(writer: ArtifactWriter, key: str, codes: mx.array, scales: mx.array,
                        biases: mx.array, bits: int, gs: int, in_: int,
                        scale_dtype: mx.Dtype | None) -> None:
    """Pack int codes (validated for ``bits``) + scales/biases and add under ``key`` (``affine_packed``)."""
    _assert_int6_packs(codes, scales, biases, in_, bits, gs)
    if scale_dtype is not None:
        scales, biases = scales.astype(scale_dtype), biases.astype(scale_dtype)
    writer.add_quantized(key, pack_affine(codes.astype(mx.uint32), bits), scales, biases, bits, gs)


def _write_rtn(writer: ArtifactWriter, key: str, w: mx.array, bits: int, gs: int,
               scale_dtype: mx.Dtype | None) -> None:
    """Plain RTN affine fallback (cold experts): ``mx.quantize`` already packs at ``bits`` g128."""
    writer.add_quantized(key, *quantize_affine(w, bits, gs, scale_dtype=scale_dtype), bits, gs)


def _down_input(g: mx.array, u: mx.array, xe: mx.array) -> mx.array:
    """``w2`` (down) GPTQ calibration input: ``silu(gate·x)·(up·x)`` -> ``[n, inter]``.

    Matches :func:`quanta.minimax.moe._swiglu_stack` (plain SwiGLU, no clamp). ``g``/``u`` are the
    expert's ``w1``/``w3`` ``[inter, hidden]``; ``xe`` its routed rows ``[n, hidden]``."""
    return (nn.silu(g.astype(mx.float32) @ xe.astype(mx.float32).T)
            * (u.astype(mx.float32) @ xe.astype(mx.float32).T)).T


def _bake_experts_layer(writer: ArtifactWriter, pre: str, es: dict, ln2: mx.array | None,
                        idx: mx.array | None, experts: list[int], bits: int, gs: int,
                        method: str, in_dims: dict[str, int], scale_dtype: mx.Dtype | None) -> int:
    """GPTQ (or RTN) the routed experts of one layer. Returns the warm-expert count.

    ``es`` is ``{w1,w3:[E,inter,hidden], w2:[E,hidden,inter]}`` (bf16). Per projection, GPTQ runs in
    expert chunks so the ordered column loop is shared across the chunk (one batched trailing GEMM per
    block); cold experts (no routed rows) fall back to RTN. ``w1``/``w3`` calibrate on the routed rows
    ``xe``; ``w2`` on ``silu(gate·x)·(up·x)`` of those rows.
    """
    warm: set[int] = set()
    if method == "rtn" or ln2 is None:  # scale-only: no calibration solve, all-GPU
        for e in experts:
            for proj in _EXPERT_PROJS:
                _write_rtn(writer, f"{pre}{e}.{proj}", es[proj][e], bits, gs, scale_dtype)
        return 0

    # Per-expert calibration inputs (one layer resident): xe for gate/up, SwiGLU intermediate for down.
    xe_of: dict[int, mx.array] = {e: expert_rows(ln2, idx, e) for e in experts}
    xd_of: dict[int, mx.array] = {}
    for e in experts:
        xe = xe_of[e]
        xd_of[e] = _down_input(es["w1"][e], es["w3"][e], xe) if xe.shape[0] > 0 else xe
        if xe.shape[0] > 0:
            warm.add(e)

    for proj in _EXPERT_PROJS:
        x_of = xd_of if proj == "w2" else xe_of
        in_ = in_dims[proj]
        cold = [e for e in experts if x_of[e].shape[0] == 0]
        for e in cold:  # cold expert -> RTN at the same bits (still affine_packed, one runtime path)
            _write_rtn(writer, f"{pre}{e}.{proj}", es[proj][e], bits, gs, scale_dtype)
        grp = [e for e in experts if x_of[e].shape[0] > 0]
        for c0 in range(0, len(grp), EXPERT_CHUNK):  # batched GPTQ over a chunk of experts
            chunk = grp[c0:c0 + EXPERT_CHUNK]
            ws = mx.stack([es[proj][e] for e in chunk])
            codes, scales, biases = gptq_quantize_batch(
                ws, [x_of[e] for e in chunk], bits, group_size=gs)
            for ci, e in enumerate(chunk):
                _write_affine_codes(writer, f"{pre}{e}.{proj}", codes[ci], scales[ci], biases[ci],
                                    bits, gs, in_, scale_dtype)
        del x_of
    return len(warm)


def bake_minimax(
    source: str | Path,
    out_dir: str | Path,
    calib_ids: mx.array,
    *,
    n_layers: int | None = None,
    expert_subset: Iterable[int] | None = None,
    include_head: bool = True,
    group_size: int = 128,
    expert_method: str = "gptq",
    scale_dtype: mx.Dtype | None = None,
) -> dict:
    """Bake the MiniMax-M2.7 source into a self-contained int6(GPTQ)/int8/bf16 artifact at ``out_dir``.

    Returns a summary ``dict`` (counts, warm experts, bytes). ``n_layers`` / ``expert_subset`` slice
    the bake for bounded validation; ``include_head`` toggles embed/final-norm/lm_head. The full call
    (no slicing) is the real bake — heavy, deferred to a GPU session (see module docstring).
    """
    assert expert_method in ("gptq", "rtn"), f"expert_method must be 'gptq'|'rtn', got {expert_method!r}"
    cfg = MiniMaxConfig.from_pretrained(source)
    if cfg.has_shared_expert:  # MiniMax-M2.7 has none; refuse to bake one we can't read (rule 6)
        raise ValueError(f"config reports a shared expert (shared_intermediate_size="
                         f"{cfg.shared_intermediate_size}) but the MiniMax bake assumes none")
    ck = MiniMaxSourceCheckpoint(source, cfg)
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    experts = list(range(cfg.num_local_experts)) if expert_subset is None else list(expert_subset)
    in_dims = {"w1": cfg.hidden_size, "w3": cfg.hidden_size, "w2": cfg.moe_intermediate_size}

    caps = capture_calibration(ck, cfg, calib_ids, n_layers=n) if expert_method == "gptq" else {}

    writer = ArtifactWriter(out_dir, Path(source) / "config.json")
    if include_head:
        writer.add_dense(EMBED, ck.embed())              # bf16 (logit-sensitive; policy)
        writer.add_dense(NORMF, ck.final_norm())
        if not cfg.tie_word_embeddings:
            writer.add_dense(HEAD, ck.lm_head())
        ck.release()

    warm = 0
    for i in range(n):
        pre = f"model.layers.{i}."
        a = ck.attention(i)  # GQA q/k/v/o (fp8->bf16) int8; q_norm/k_norm bf16 dense
        for name in _ATTN_MATMULS:
            _write_int8(writer, f"{pre}self_attn.{name}", a[name], group_size, scale_dtype)
        for name in ("q_norm", "k_norm"):
            writer.add_dense(f"{pre}self_attn.{name}.weight", a[name])
        norms = ck.block_norms(i)
        writer.add_dense(f"{pre}input_layernorm.weight", norms["input_layernorm"])
        writer.add_dense(f"{pre}post_attention_layernorm.weight", norms["post_attention_layernorm"])

        router = ck.moe_router(i)  # gate.weight + e_score_correction_bias: both bf16 dense (control)
        writer.add_dense(f"{pre}block_sparse_moe.gate.weight", router["weight"])
        writer.add_dense(f"{pre}block_sparse_moe.e_score_correction_bias",
                         router["e_score_correction_bias"])

        es = ck.expert_stacks(i, cfg.num_local_experts)  # {w1,w3:[E,inter,hidden], w2:[E,hidden,inter]}
        ln2, idx = caps.get(i, (None, None))
        warm += _bake_experts_layer(writer, f"{pre}block_sparse_moe.experts.", es, ln2, idx,
                                    experts, _EXPERT_BITS, group_size, expert_method, in_dims,
                                    scale_dtype)
        del a, norms, router, es
        ck.release()
        mx.clear_cache()

    counts = {"int8": 0, "dense": 0, "expert_int6": 0}
    for entry in writer.manifest.values():
        if entry["format"] == "dense":
            counts["dense"] += 1
        elif entry.get("bits") == _EXPERT_BITS:
            counts["expert_int6"] += 1
        else:
            counts["int8"] += 1

    scale_tag = "bf16" if scale_dtype == mx.bfloat16 else "fp32"
    policy = {"experts": f"int6 {expert_method} g{group_size}", "non_experts": f"int8 g{group_size}",
              "router_norms_embed_head": "bf16", "shared_expert": "none", "scales": scale_tag}
    writer.finalize(policy)

    out = Path(out_dir)
    total_bytes = sum(p.stat().st_size for p in out.glob("model-*.safetensors"))
    return {"layers": n, "experts_per_layer": len(experts), "warm_experts": warm,
            "expert_method": expert_method, "counts": counts, "bytes": total_bytes}
