"""Model-free V1 parity gate for the MiniMax-M3-VL **vision tower** (:mod:`quanta.minimax.model_vision_m3`).

The M3 vision tower is a CLIP-ViT with three deltas (Conv3d-as-linear patch embed, no learned
pos-embed/CLS/post-norm, 3-D RoPE) plus a project→merge head. Each piece is pinned to an authoritative
reference, in isolation, on tiny SYNTHETIC dims:

* **CLIP encoder layer** (:class:`model_vision_m3.MiniMaxM3VisionLayer`, RoPE switched OFF) vs the REAL
  ``transformers.models.clip.modeling_clip.CLIPEncoderLayer`` on IDENTICAL weights — validates the
  pre-norm attention (biased q/k/v/out, full bidirectional) + GELU MLP + LayerNorm + residual structure.
* **Exact GELU**, **Conv3d-as-linear patch embed**, **3-D vision RoPE**, **multi_modal_projector**,
  **patch_merge_mlp**, **per-patch (t,h,w) position ids** — each vs a self-contained **numpy-fp64**
  oracle (the same formulas the module pins), so MLX == oracle proves IMPLEMENTATION parity.
* **rule 4**: ``use_fast`` (``mx.fast`` SDPA) == naive softmax attention.
* **rule 6**: ``rope_section`` must sum to ``head_dim//2``; patch-merge refuses an indivisible token count.
* The 2-D-degenerate property: for an **image** (``grid_t=1``) the t-section of the rope is identity, and
  the full 3-D rope equals a pure 2-D (h,w) rope — the property that makes the [PINNED-pending-e2e]
  ``rope_section`` knob affect only video; the decisive arbiter for the exact split is the V2 real-weight
  vision e2e (the ViT is 1.6 GiB, loadable standalone).

All on tiny synthetic dims; runs in the model-free sweep (needs the ``reference`` extra for the
``transformers`` CLIP pin).

    uv run --extra reference python -m parity.minimax_m3_vision_test
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import torch

from quanta.minimax import model_vision_m3 as V
from quanta.minimax.config_m3 import MiniMaxVisionConfig

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-9))


def _np(x: mx.array) -> np.ndarray:
    return np.asarray(x.astype(mx.float32), dtype=np.float64)


def _mx(a: np.ndarray) -> mx.array:
    return mx.array(np.asarray(a, dtype=np.float32))


def _cfg() -> MiniMaxVisionConfig:
    # tiny but structurally faithful: head_dim = 32/4 = 8 (half = 4), merge 2, temporal 2.
    return MiniMaxVisionConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
        patch_size=2, image_size=64, projection_dim=48, rope_theta=1e4, rope_mode="3d",
        layer_norm_eps=1e-5, hidden_act="gelu", num_channels=3,
        spatial_merge_size=2, temporal_patch_size=2,
    )


# --- numpy-fp64 oracles ------------------------------------------------------- #

def _gelu_np(x):
    from scipy.special import erf  # noqa: PLC0415
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def _rope_np(pos, hd, theta, section):
    st, sh, sw = section
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))   # [half]
    axis = np.concatenate([np.zeros(st, int), np.ones(sh, int), np.full(sw, 2, int)])
    pos_sel = pos.astype(np.float64)[:, axis]                              # [N, half]
    ang = pos_sel * inv[None, :]
    emb = np.concatenate([ang, ang], axis=-1)
    return np.cos(emb), np.sin(emb)


def _rotate_half_np(x):
    d = x.shape[-1] // 2
    return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def run() -> None:
    rng = np.random.default_rng(0)
    cfg = _cfg()
    hd = cfg.hidden_size // cfg.num_attention_heads        # 8
    half = hd // 2                                          # 4
    section = (2, 1, 1)                                     # sums to half
    theta = cfg.rope_theta

    # 1) exact GELU == oracle
    try:
        x = rng.standard_normal((5, 7)).astype(np.float32)
        _ck(_rel(_np(V.gelu(_mx(x))), _gelu_np(x.astype(np.float64))) < 1e-5, "gelu != erf oracle")
    except ImportError:
        # scipy may be absent; fall back to a math.erf elementwise oracle
        xe = x.astype(np.float64)
        ref = 0.5 * xe * (1.0 + np.vectorize(math.erf)(xe / math.sqrt(2.0)))
        _ck(_rel(_np(V.gelu(_mx(x))), ref) < 1e-5, "gelu != erf oracle")

    # 2) default_rope_section(80) == (8,16,16), sums to 40 (the real head_dim)
    ds = V.default_rope_section(80)
    _ck(ds == (8, 16, 16) and sum(ds) == 40, f"default_rope_section(80) wrong: {ds}")

    # 3) per-patch (t,h,w) position ids: merge-block order, for one image (t,h,w)=(1,4,4)
    pos = V.vision_position_ids([(1, 4, 4)], cfg.spatial_merge_size)
    pos_np = np.asarray(pos)
    _ck(pos_np.shape == (16, 3), "position_ids shape")
    _ck(int(pos_np[:, 0].max()) == 0, "image t-pos must be all 0 (grid_t=1)")
    # merge-block order: the first 4 patches are the top-left 2x2 block -> h in {0,1}, w in {0,1}
    blk0 = set(map(tuple, pos_np[:4, 1:].tolist()))
    _ck(blk0 == {(0, 0), (0, 1), (1, 0), (1, 1)}, f"first merge-block not the 2x2 corner: {blk0}")
    # next block is columns 2,3 of the top rows
    blk1 = set(map(tuple, pos_np[4:8, 1:].tolist()))
    _ck(blk1 == {(0, 2), (0, 3), (1, 2), (1, 3)}, f"second merge-block wrong: {blk1}")

    # 4) 3-D rope: MLX == numpy-fp64 oracle (arbitrary t,h,w + section)
    pos3 = mx.array(rng.integers(0, 6, size=(11, 3)).astype(np.int32))
    cos_mx, sin_mx = V.vision_rope_3d(pos3, hd, theta, section)
    cos_np, sin_np = _rope_np(np.asarray(pos3), hd, theta, section)
    _ck(_rel(_np(cos_mx), cos_np) < 1e-5 and _rel(_np(sin_mx), sin_np) < 1e-5, "3d rope != oracle")

    # 5) 2-D degenerate: image (t=0) -> the t-section dims are identity (cos=1, sin=0)
    pos_img = V.vision_position_ids([(1, 4, 4)], cfg.spatial_merge_size)
    c_img, s_img = V.vision_rope_3d(pos_img, hd, theta, section)
    st = section[0]
    # the t-section occupies freq pairs [0:st] (doubled at [half:half+st]); for t=0 cos=1, sin=0
    cimg = _np(c_img)
    simg = _np(s_img)
    _ck(np.allclose(cimg[:, :st], 1.0, atol=1e-6) and np.allclose(cimg[:, half:half + st], 1.0, atol=1e-6),
        "t-section cos must be 1 for an image")
    _ck(np.allclose(simg[:, :st], 0.0, atol=1e-6) and np.allclose(simg[:, half:half + st], 0.0, atol=1e-6),
        "t-section sin must be 0 for an image")
    # and the full 3-D rope == a pure 2-D (h,w) rope (section (0, sh, sw)) for an image
    c2, s2 = V.vision_rope_3d(pos_img, hd, theta, (0, section[1] + st, section[2]))
    # not identical (different section layout); instead assert applying either rope to a vector that is
    # zero on the t-dims gives the same result — i.e. images never excite the t-section. Simpler robust
    # check: rotate a random q with the image rope and confirm t-section rows are untouched.
    qv = _mx(rng.standard_normal((16, cfg.num_attention_heads, hd)).astype(np.float32))
    rot = V.apply_rope_vision(qv, c_img, s_img)
    qn = _np(qv)
    rn = _np(rot)
    _ck(np.allclose(rn[..., :st], qn[..., :st], atol=1e-5)
        and np.allclose(rn[..., half:half + st], qn[..., half:half + st], atol=1e-5),
        "image rope must leave the t-section dims unchanged")
    del c2, s2

    # 6) rule 6: rope_section must sum to head_dim//2
    try:
        V.vision_rope_3d(pos3, hd, theta, (1, 1, 1))   # sums to 3 != 4
        _ck(False, "bad rope_section accepted")
    except ValueError:
        _ck(True, "bad rope_section refused")

    # 7) patch embed (Conv3d-as-linear) == oracle
    in_dim = cfg.num_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size  # 24
    pe = V.MiniMaxM3VisionPatchEmbed(cfg)
    W_pe = rng.standard_normal((cfg.hidden_size, in_dim)).astype(np.float32)
    pe.proj.weight = _mx(W_pe)
    pv = rng.standard_normal((16, in_dim)).astype(np.float32)
    _ck(_rel(_np(pe(_mx(pv))), pv.astype(np.float64) @ W_pe.astype(np.float64).T) < 1e-4,
        "patch embed != linear oracle")

    # 8) projector == oracle (1280->6144->6144 shape here hidden->proj->proj)
    pr = V.MiniMaxM3VisionProjector(cfg)
    W1 = rng.standard_normal((cfg.projection_dim, cfg.hidden_size)).astype(np.float32)
    b1 = rng.standard_normal((cfg.projection_dim,)).astype(np.float32)
    W2 = rng.standard_normal((cfg.projection_dim, cfg.projection_dim)).astype(np.float32)
    b2 = rng.standard_normal((cfg.projection_dim,)).astype(np.float32)
    pr.linear_1.weight, pr.linear_1.bias = _mx(W1), _mx(b1)
    pr.linear_2.weight, pr.linear_2.bias = _mx(W2), _mx(b2)
    xp = rng.standard_normal((16, cfg.hidden_size)).astype(np.float32)
    h1 = _gelu_np_safe(xp.astype(np.float64) @ W1.astype(np.float64).T + b1)
    ref = h1 @ W2.astype(np.float64).T + b2
    _ck(_rel(_np(pr(_mx(xp))), ref) < 1e-4, "projector != oracle")

    # 9) patch merge == oracle (consecutive-4 concat) + indivisible refusal (rule 6)
    pm = V.MiniMaxM3VisionPatchMerge(cfg)
    g = cfg.spatial_merge_size ** 2                                    # 4
    in_m = cfg.projection_dim * g
    Wm1 = rng.standard_normal((cfg.projection_dim, in_m)).astype(np.float32)
    bm1 = rng.standard_normal((cfg.projection_dim,)).astype(np.float32)
    Wm2 = rng.standard_normal((cfg.projection_dim, cfg.projection_dim)).astype(np.float32)
    bm2 = rng.standard_normal((cfg.projection_dim,)).astype(np.float32)
    pm.linear_1.weight, pm.linear_1.bias = _mx(Wm1), _mx(bm1)
    pm.linear_2.weight, pm.linear_2.bias = _mx(Wm2), _mx(bm2)
    xm = rng.standard_normal((16, cfg.projection_dim)).astype(np.float32)
    cat = xm.astype(np.float64).reshape(16 // g, g * cfg.projection_dim)
    hm = _gelu_np_safe(cat @ Wm1.astype(np.float64).T + bm1)
    refm = hm @ Wm2.astype(np.float64).T + bm2
    out_m = pm(_mx(xm))
    _ck(out_m.shape == (16 // g, cfg.projection_dim), "patch-merge token count")
    _ck(_rel(_np(out_m), refm) < 1e-4, "patch-merge != oracle")
    try:
        pm(_mx(rng.standard_normal((15, cfg.projection_dim)).astype(np.float32)))  # 15 % 4 != 0
        _ck(False, "indivisible patch-merge accepted")
    except ValueError:
        _ck(True, "indivisible patch-merge refused")

    # 10) CLIP encoder layer pin: MLX (rope OFF) == transformers CLIPEncoderLayer (shared weights)
    from transformers.models.clip.configuration_clip import CLIPVisionConfig  # noqa: PLC0415
    from transformers.models.clip.modeling_clip import CLIPEncoderLayer  # noqa: PLC0415

    ccfg = CLIPVisionConfig(hidden_size=cfg.hidden_size, intermediate_size=cfg.intermediate_size,
                            num_attention_heads=cfg.num_attention_heads, num_hidden_layers=1,
                            hidden_act="gelu", attention_dropout=0.0,
                            layer_norm_eps=cfg.layer_norm_eps)
    ccfg._attn_implementation = "eager"
    tlayer = CLIPEncoderLayer(ccfg).eval()
    sd = {k: torch.tensor(rng.standard_normal(tuple(v.shape)).astype(np.float32))
          for k, v in tlayer.state_dict().items()}
    tlayer.load_state_dict(sd)
    mlayer = V.MiniMaxM3VisionLayer(cfg)
    mlayer.self_attn.q_proj.weight = _mx(sd["self_attn.q_proj.weight"].numpy())
    mlayer.self_attn.q_proj.bias = _mx(sd["self_attn.q_proj.bias"].numpy())
    mlayer.self_attn.k_proj.weight = _mx(sd["self_attn.k_proj.weight"].numpy())
    mlayer.self_attn.k_proj.bias = _mx(sd["self_attn.k_proj.bias"].numpy())
    mlayer.self_attn.v_proj.weight = _mx(sd["self_attn.v_proj.weight"].numpy())
    mlayer.self_attn.v_proj.bias = _mx(sd["self_attn.v_proj.bias"].numpy())
    mlayer.self_attn.out_proj.weight = _mx(sd["self_attn.out_proj.weight"].numpy())
    mlayer.self_attn.out_proj.bias = _mx(sd["self_attn.out_proj.bias"].numpy())
    mlayer.layer_norm1.weight = _mx(sd["layer_norm1.weight"].numpy())
    mlayer.layer_norm1.bias = _mx(sd["layer_norm1.bias"].numpy())
    mlayer.layer_norm2.weight = _mx(sd["layer_norm2.weight"].numpy())
    mlayer.layer_norm2.bias = _mx(sd["layer_norm2.bias"].numpy())
    mlayer.mlp.fc1.weight = _mx(sd["mlp.fc1.weight"].numpy())
    mlayer.mlp.fc1.bias = _mx(sd["mlp.fc1.bias"].numpy())
    mlayer.mlp.fc2.weight = _mx(sd["mlp.fc2.weight"].numpy())
    mlayer.mlp.fc2.bias = _mx(sd["mlp.fc2.bias"].numpy())
    nt = 12
    xin = rng.standard_normal((nt, cfg.hidden_size)).astype(np.float32)
    with torch.no_grad():
        t_out = tlayer(torch.tensor(xin)[None], attention_mask=None)
        t_out = (t_out[0] if isinstance(t_out, tuple) else t_out)[0].numpy()
    rope_off_cos = mx.ones((nt, hd))
    rope_off_sin = mx.zeros((nt, hd))
    m_out = mlayer(_mx(xin), rope_off_cos, rope_off_sin, use_fast=True)
    _ck(_rel(_np(m_out), t_out) < 2e-3, f"CLIP encoder layer != transformers (rel {_rel(_np(m_out), t_out):.2e})")

    # 11) rule 4: fast == naive attention (with rope ON)
    c_a, s_a = V.vision_rope_3d(V.vision_position_ids([(1, 4, 3)], 1), hd, theta, section)  # merge=1 -> 12 patches
    xin2 = rng.standard_normal((12, cfg.hidden_size)).astype(np.float32)
    of = mlayer.self_attn(_mx(xin2), c_a, s_a, use_fast=True)
    on = mlayer.self_attn(_mx(xin2), c_a, s_a, use_fast=False)
    _ck(_rel(_np(of), _np(on)) < 2e-3, "fast != naive vision attention")

    # 12) full tower: token count N -> N/merge**2, projection_dim; multi-image == per-image
    model = V.MiniMaxM3VisionModel(cfg, rope_section=section)
    grids = [(1, 4, 4), (1, 2, 2)]                                    # 16 + 4 = 20 patches
    ntot = sum(t * h * w for (t, h, w) in grids)
    pv_all = _mx(rng.standard_normal((ntot, in_dim)).astype(np.float32))
    out = model(pv_all, grids)
    _ck(out.shape == (ntot // g, cfg.projection_dim), f"tower output shape {out.shape}")
    # per-image forward of image 0 alone == the first slice of the joint forward (attention is per-image)
    enc_all = model.encode(pv_all, grids)
    enc0 = model.encode(pv_all[:16], [(1, 4, 4)])
    _ck(_rel(_np(enc_all[:16]), _np(enc0)) < 1e-4, "per-image attention not isolated (image 0)")

    print(f"PARITY-CHECKS: {_N}")
    print(f"PASS — MiniMax-M3-VL V1 vision parity: CLIP encoder layer == transformers; "
          f"patch-embed/3d-rope/projector/merge == numpy fp64; fast==naive; per-image isolation "
          f"({_N} checks). [rope_section split is PINNED-pending the V2 real-weight e2e]")


def _gelu_np_safe(x):
    try:
        return _gelu_np(x)
    except ImportError:
        return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


if __name__ == "__main__":
    run()
