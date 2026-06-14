"""MiniMax-M3-VL vision V2: real-weight standalone ViT forward re-gate (SOLO, ~1.6 GiB).

Loads the **dense CLIP-ViT vision tower** from the real int4-g64 artifact (the 523 vision tensors,
~1.6 GiB bf16 — :func:`quanta.minimax.model_vision_m3.load_vision_model`), feeds a real factor-aligned
image through the native image processor (:class:`quanta.minimax.image_m3.MiniMaxM3ImageProcessor`,
resize == identity ⇒ the exactly-reproducible path), and runs the full tower → merged LLM tokens
``[N//merge², 6144]``.

There is **no shipped M3 vision forward** to diff against (the checkpoint ships only
``configuration_*.py``; even the V1 model-free gate pins only the CLIP *encoder layer* vs transformers,
with the 3-D RoPE pinned to the Qwen2.5-VL convention). So this gate validates the real-weight
**mechanics + structural invariants**, NOT a numeric reference (the decisive arbiter for the
[PINNED-pending-e2e] ``rope_section`` is the V3 multimodal e2e, which splices these embeddings into the
233 GiB text model and reads the caption):

  * **load coverage** — every model parameter assigned and every source vision key consumed (the
    loader's two-way rule-6 assertions fire on any mismatch);
  * **shape** — one image of grid ``(1,h,w)`` → ``h·w`` patches → ``(h·w)//4`` merged tokens of width
    ``projection_dim`` (6144);
  * **finite + sane magnitude** — outputs all finite, non-degenerate (nonzero std), not exploded;
  * **rule 4** — ``use_fast`` SDPA == naive softmax attention on the real weights;
  * **2-D-degenerate property** (real weights) — for an image (``grid_t=1``) the 3-D rope leaves the
    t-section dims of a real layer-0 q/k unchanged (the property that makes ``rope_section`` affect only
    video; the h/w split is what an image uses);
  * **per-image isolation** — encoding image-0 alone == its slice of a joint two-image forward
    (attention never crosses images).

    uv run python -m parity.minimax_m3_vision_real

# parity-gate: real-weight
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from quanta.minimax import model_vision_m3 as V
from quanta.minimax.image_m3 import MiniMaxM3ImageProcessor

ART = "/Users/pmrj/models/MiniMax-M3-quanta_int4g64"

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _grad_image(h: int, w: int, seed: int) -> np.ndarray:
    """A smooth structured RGB image (a per-channel gradient + ripple) — a 'real' image, not noise, so
    the tower sees spatial structure the rope acts on."""
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    r = (255 * (0.5 + 0.5 * np.sin(6.0 * xx + seed))).astype(np.uint8)
    g = (255 * yy).astype(np.uint8)
    b = (255 * (0.5 + 0.5 * np.cos(4.0 * yy * xx + seed))).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def main() -> None:
    mx.set_wired_limit(int(8 * 1024**3))   # pin the small resident ViT (~1.6 GiB)

    print(f"loading vision tower from {ART} …")
    model = V.load_vision_model(ART)
    cfg = model.cfg
    hd = cfg.hidden_size // cfg.num_attention_heads
    merge = cfg.spatial_merge_size
    g = merge * merge
    print(f"  ViT: {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, {cfg.num_attention_heads} "
          f"heads (head_dim {hd}), rope_section {model.rope_section}")

    proc = MiniMaxM3ImageProcessor()

    # --- single image: 56x56 -> grid (1,4,4) = 16 patches -> 4 merged tokens ----------------------
    out0 = proc.preprocess(_grad_image(56, 56, 1))
    _ck(out0.grid_thw == [(1, 4, 4)], f"grid_thw {out0.grid_thw}")
    pv0 = mx.array(out0.pixel_values).astype(mx.bfloat16)
    grids0 = out0.grid_thw
    tok = model(pv0, grids0)
    mx.eval(tok)
    npix = grids0[0][0] * grids0[0][1] * grids0[0][2]
    _ck(tok.shape == (npix // g, cfg.projection_dim), f"merged token shape {tok.shape}")
    tnp = np.asarray(tok.astype(mx.float32), dtype=np.float64)
    _ck(bool(np.all(np.isfinite(tnp))), "tower output has non-finite values")
    std = float(tnp.std())
    mx_abs = float(np.max(np.abs(tnp)))
    _ck(std > 1e-4, f"tower output degenerate (std {std:.2e})")
    _ck(mx_abs < 1e4, f"tower output exploded (max|x| {mx_abs:.2e})")
    print(f"  single 56x56 -> {tok.shape[0]} tokens, std {std:.3f}, max|x| {mx_abs:.2f}")

    # shared layer-0 substrate for the op-level checks (pre-norm input + image rope).
    pos = V.vision_position_ids(grids0, merge)
    cos, sin = V.vision_rope_3d(pos, hd, cfg.rope_theta, model.rope_section)
    h_in = model.pre_layrnorm(model.patch_embed(pv0))
    xln0 = model.layers[0].layer_norm1(h_in)

    # --- rule 4: fast == naive attention, op-level on the real layer-0 weights (no multi-layer ----
    #     bf16 compounding — the whole-stack drift is ~2.5% over 32 layers, not a kernel bug). ------
    of = model.layers[0].self_attn(xln0, cos, sin, use_fast=True)
    on = model.layers[0].self_attn(xln0, cos, sin, use_fast=False)
    rel = float(np.max(np.abs(np.asarray(of.astype(mx.float32), np.float64)
                              - np.asarray(on.astype(mx.float32), np.float64)))
                / (np.max(np.abs(np.asarray(on.astype(mx.float32), np.float64))) + 1e-9))
    _ck(rel < 1e-2, f"fast != naive vision attention on real layer-0 weights (rel {rel:.2e})")
    print(f"  rule-4 fast==naive (layer-0 op): rel {rel:.2e}")

    # --- 2-D-degenerate: for an image, rope leaves the t-section dims of a real q/k unchanged ------
    attn0 = model.layers[0].self_attn
    npat = h_in.shape[0]
    q = attn0.q_proj(xln0).reshape(npat, cfg.num_attention_heads, hd)
    q_rot = V.apply_rope_vision(q, cos, sin)
    st = model.rope_section[0]
    half = hd // 2
    qn = np.asarray(q.astype(mx.float32), np.float64)
    rn = np.asarray(q_rot.astype(mx.float32), np.float64)
    _ck(np.allclose(rn[..., :st], qn[..., :st], atol=1e-2)
        and np.allclose(rn[..., half:half + st], qn[..., half:half + st], atol=1e-2),
        "image rope must leave the t-section dims unchanged (real q)")
    print(f"  2-D-degenerate: t-section ({st} pairs) inert on real q ✓")

    # --- per-image isolation: image-0 alone == its slice of a joint two-image forward -------------
    out1 = proc.preprocess([_grad_image(56, 56, 1), _grad_image(84, 56, 2)])
    pv_all = mx.array(out1.pixel_values).astype(mx.bfloat16)
    enc_all = model.encode(pv_all, out1.grid_thw)
    n0 = out1.grid_thw[0][0] * out1.grid_thw[0][1] * out1.grid_thw[0][2]
    enc_solo = model.encode(pv_all[:n0], [out1.grid_thw[0]])
    iso = float(np.max(np.abs(np.asarray(enc_all[:n0].astype(mx.float32), np.float64)
                             - np.asarray(enc_solo.astype(mx.float32), np.float64)))
                / (np.max(np.abs(np.asarray(enc_solo.astype(mx.float32), np.float64))) + 1e-9))
    _ck(iso < 1e-3, f"per-image attention not isolated (rel {iso:.2e})")
    print(f"  per-image isolation: rel {iso:.2e}")

    print(f"PARITY-CHECKS: {_N}")
    print(f"PASS — MiniMax-M3-VL vision V2 real-weight ViT: loaded 523 dense tensors, image→{tok.shape[0]} "
          f"tokens (finite/sane), fast==naive, 2-D-degenerate, per-image isolation ({_N} checks). "
          f"[rope_section {model.rope_section} is PINNED-pending the V3 multimodal e2e]")


if __name__ == "__main__":
    main()
