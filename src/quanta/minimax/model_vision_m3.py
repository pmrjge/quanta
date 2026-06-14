"""MiniMax-M3-VL **vision tower** — MLX-native ``mlx.nn`` modules (V1 vision parity).

The runnable reference forward for the M3 vision track (full-VL, per user). It is **additive**:
the text backbone (:mod:`quanta.minimax.model_m3`) and the M2.7 siblings are untouched. The vision
weights are already baked dense bf16 in the int4 artifact (the 523 vision tensors); this module is
the forward that consumes them.

Architecture (empirically grounded — ``vision_config`` + the real checkpoint headers, see the
``parity/minimax_m3_vision_test`` gate). M3's vision tower is a **CLIP-ViT encoder** (``model_type
clip_vision_model``, 32 layers, hidden 1280, 16 heads, intermediate 5120, GELU, LayerNorm eps 1e-5,
the CLIP ``pre_layrnorm`` typo preserved) with **three deltas from a stock transformers CLIP**, each
pinned against an authoritative sibling or forced by the on-disk shapes:

* **Conv3d patch embed as a linear** (``embeddings.patch_embedding.weight`` ``[1280,3,2,14,14]`` — a
  3-D conv, ``temporal_patch_size=2``, Qwen2-VL-style). The image processor (the shipped
  ``image_processor.py``) flattens each patch to a ``3·2·14·14 = 1176``-vector ordered
  ``[channel,temporal,h,w]`` and the conv weight reshaped ``[1280,1176]`` in the SAME order applies
  as a plain ``[1176→1280]`` linear (:class:`MiniMaxM3VisionPatchEmbed`). NOT a stock CLIP Conv2d.
* **No learned position embedding / no class token / no post-layernorm** (none ship). Position comes
  from **3-D RoPE** (``rope_mode="3d"``, ``rope_theta=1e4``); "full" feature select keeps every patch
  token; the feature layer is the last encoder layer's output verbatim.
* **3-D vision RoPE** over ``(t,h,w)`` patch coords — :func:`vision_rope_3d`. There is **no shipped
  reference** (transformers CLIP has learned pos-embeds; even Qwen3-VL's *vision* tower is 2-D h/w),
  so the construction follows the **Qwen2.5-VL M-RoPE convention** (ONE shared ``inv_freq`` ladder
  over the head_dim, the freq dims *sectioned* across the three axes by ``rope_section``) — it
  degenerates exactly to the Qwen2-VL 2-D (h,w) vision rope for an **image** (``grid_t=1`` ⇒ the
  t-position is 0 ⇒ the t-section rotates by 0 ⇒ identity on those dims). [PINNED-pending-e2e: the
  exact ``rope_section`` split of the 40 freq pairs across t/h/w is the one knob no on-disk artifact
  fixes; the V1 gate proves IMPLEMENTATION parity (MLX == numpy-fp64 oracle for any section) + the
  2-D-degenerate property, and the decisive arbiter is the cheap real-weight vision e2e (V2 — the ViT
  is 1.6 GiB, loadable standalone) which settles the section against a real image.]

Then the vision features are projected and merged into LLM tokens (the order is **forced by the
on-disk input dims**, NOT a guess):

* **multi_modal_projector** (``linear_1`` input 1280 = ViT hidden) maps each patch ``1280 → 6144``
  (GELU between) — runs FIRST, per patch (:class:`MiniMaxM3VisionProjector`).
* **patch_merge_mlp** (``linear_1`` input ``24576 = 4·6144``) groups each 2×2 spatial neighbourhood
  of 4 projected tokens (the image processor already orders patches in merge-blocks, so the merge is
  a consecutive-4 concat), ``24576 → 6144`` (GELU) ``→ 6144`` (:class:`MiniMaxM3VisionPatchMerge`).

So one image of ``grid=(t,h,w)`` patches → ``t·h·w`` ViT tokens → projector → ``(t·h·w)/4`` merged
LLM tokens, matching the processor's ``num_tokens = grid.prod() // merge_size**2`` placeholder count
(``image_token_index`` 200025). The splice into the text stream is the V2 milestone.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from quanta.minimax.config_m3 import MiniMaxVisionConfig

# ----------------------------------------------------------------------------- #
# Activation — exact (erf) GELU (config ``hidden_act`` / ``projector_hidden_act`` == "gelu").
# ----------------------------------------------------------------------------- #


def gelu(x: mx.array) -> mx.array:
    """Exact erf GELU (transformers ``ACT2FN["gelu"]`` == ``nn.GELU()``), computed in fp32 then cast
    back — matches the numpy-fp64 oracle. NOT ``quick_gelu`` (M3's config says plain ``gelu``)."""
    xf = x.astype(mx.float32)
    out = 0.5 * xf * (1.0 + mx.erf(xf / math.sqrt(2.0)))
    return out.astype(x.dtype)


# ----------------------------------------------------------------------------- #
# 3-D vision RoPE (Qwen2.5-VL M-RoPE convention; degenerates to 2-D for images).
# ----------------------------------------------------------------------------- #


def vision_position_ids(grid_thw: list[tuple[int, int, int]], merge: int) -> mx.array:
    """Per-patch ``(t,h,w)`` position ids ``[N,3]`` in the **merge-block** patch order — the exact
    order the shipped ``image_processor.py`` emits (the ``[h//m,m,w//m,m].transpose(1,2).flatten()``
    layout, == ``transformers.vision_utils.get_vision_position_ids`` extended with the temporal axis).

    Consecutive groups of ``merge**2`` patches form one 2×2 spatial block (what
    :class:`MiniMaxM3VisionPatchMerge` later concatenates), so the position ids and the merge stay in
    lock-step. ``grid_thw`` is one ``(t,h,w)`` per image; images are concatenated along ``N``."""
    rows = []
    for (t, h, w) in grid_thw:
        t, h, w = int(t), int(h), int(w)
        # h / w ids laid out in merge-block order (matches the processor permute).
        hpos = mx.broadcast_to(mx.arange(h)[:, None], (h, w))
        hpos = hpos.reshape(h // merge, merge, w // merge, merge)
        hpos = mx.transpose(hpos, (0, 2, 1, 3)).reshape(-1)
        wpos = mx.broadcast_to(mx.arange(w)[None, :], (h, w))
        wpos = wpos.reshape(h // merge, merge, w // merge, merge)
        wpos = mx.transpose(wpos, (0, 2, 1, 3)).reshape(-1)
        tpos = mx.broadcast_to(mx.arange(t)[:, None], (t, h * w)).reshape(-1)
        hw = mx.stack([hpos, wpos], axis=-1)                 # [h*w, 2]
        thw = mx.concatenate([tpos[:, None], mx.broadcast_to(hw[None], (t, h * w, 2)).reshape(-1, 2)],
                             axis=-1)                          # [t*h*w, 3]
        rows.append(thw)
    return mx.concatenate(rows, axis=0).astype(mx.int32)      # [N,3]


def default_rope_section(head_dim: int) -> tuple[int, int, int]:
    """Default ``(t,h,w)`` freq-pair split of ``head_dim//2`` for the 3-D vision RoPE.

    [PINNED-pending-e2e] No on-disk artifact fixes this; the V2 real-weight vision e2e settles it. The
    default keeps **h == w** (spatial symmetry) and gives the temporal axis the remainder: for
    ``head_dim=80`` ⇒ ``head_dim//2=40`` pairs ⇒ ``(8,16,16)``. For an image (``grid_t=1``) the
    t-section rotates by 0 regardless, so this only affects video; the h/w split (``16/16`` here) is
    what an image actually uses, and is the symmetric choice."""
    half = head_dim // 2
    sh = half // 5 * 2                                        # 16 for 40
    sw = sh
    st = half - sh - sw                                       # 8 for 40
    return (st, sh, sw)


def vision_rope_3d(position_ids: mx.array, head_dim: int, theta: float,
                   section: tuple[int, int, int]) -> tuple[mx.array, mx.array]:
    """3-D vision RoPE ``(cos, sin)`` ``[N, head_dim]`` from per-patch ``(t,h,w)`` ``position_ids``
    ``[N,3]`` (Qwen2.5-VL M-RoPE convention).

    ONE shared ``inv_freq`` ladder ``1/theta**(2i/head_dim)`` over ``head_dim//2`` pairs; the pairs are
    **sectioned** across the three axes by ``section=(st,sh,sw)`` (``st+sh+sw == head_dim//2``): pair
    ``j`` reads the t-position for ``j<st``, the h-position for ``st<=j<st+sh``, else the w-position.
    ``emb = cat([freqs, freqs])`` doubles to ``head_dim`` (rotate-half pairs ``j`` with ``j+head_dim/2``,
    same axis), then ``cos``/``sin``. For an image (``t==0``) the t-section's angle is 0 ⇒ ``cos=1``,
    ``sin=0`` ⇒ identity on those dims, so this is the 2-D (h,w) vision rope plus inert t-dims."""
    half = head_dim // 2
    st, sh, sw = section
    if st + sh + sw != half:
        raise ValueError(f"rope_section {section} must sum to head_dim//2 = {half}")
    inv = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))   # [half]
    axis = mx.concatenate([mx.zeros((st,), mx.int32), mx.ones((sh,), mx.int32),
                           mx.full((sw,), 2, mx.int32)])      # [half] axis per pair
    pos = position_ids.astype(mx.float32)                     # [N,3]
    # per pair, gather the chosen axis's position: ang[n,j] = pos[n, axis[j]] * inv[j]
    pos_sel = mx.take(pos, axis, axis=1)                      # [N, half]
    ang = pos_sel * inv[None, :]                              # [N, half]
    emb = mx.concatenate([ang, ang], axis=-1)                 # [N, head_dim]
    return mx.cos(emb), mx.sin(emb)


def rotate_half(x: mx.array) -> mx.array:
    """Split-half rotate (HF ``rotate_half``): ``[-x2, x1]`` over the last axis."""
    d = x.shape[-1] // 2
    return mx.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def apply_rope_vision(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply vision RoPE to ``x`` ``[N, H, head_dim]`` with ``cos``/``sin`` ``[N, head_dim]`` (one
    rotation per patch, shared across heads). fp32 internally, cast back (the ``apply_rotary_pos_emb_vision``
    convention)."""
    xf = x.astype(mx.float32)
    c = cos[:, None, :]
    s = sin[:, None, :]
    out = xf * c + rotate_half(xf) * s
    return out.astype(x.dtype)


# ----------------------------------------------------------------------------- #
# Patch embedding (Conv3d-as-linear).
# ----------------------------------------------------------------------------- #


class MiniMaxM3VisionPatchEmbed(nn.Module):
    """Conv3d patch embed realized as a linear ``[in_dim → hidden]``, ``in_dim = channels ·
    temporal_patch · patch · patch`` (== ``3·2·14·14 = 1176``). The loader reshapes the on-disk
    ``[hidden,3,2,14,14]`` conv weight to ``[hidden, in_dim]`` (same ``[channel,temporal,h,w]`` flatten
    order as the processor's patch vector), so ``pixel_values[N,in_dim] @ W.T → [N,hidden]``. No bias
    (the conv ships none)."""

    def __init__(self, cfg: MiniMaxVisionConfig) -> None:
        super().__init__()
        in_dim = cfg.num_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        self.proj = nn.Linear(in_dim, cfg.hidden_size, bias=False)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        return self.proj(pixel_values)


# ----------------------------------------------------------------------------- #
# CLIP encoder layer (pre-norm, biased q/k/v/out, GELU MLP) + 3-D RoPE.
# ----------------------------------------------------------------------------- #


class MiniMaxM3VisionAttention(nn.Module):
    """Bidirectional multi-head self-attention (16 heads, head_dim 80) with biased q/k/v/out
    projections (stock CLIP) + 3-D vision RoPE on q/k. Attention is **full within an image** (no causal
    mask); multiple images are forwarded one at a time (each attends only over its own patches)."""

    def __init__(self, cfg: MiniMaxVisionConfig) -> None:
        super().__init__()
        self.nh = cfg.num_attention_heads
        self.hd = cfg.hidden_size // cfg.num_attention_heads
        self.scale = self.hd ** -0.5
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, *, use_fast: bool = True) -> mx.array:
        # x [N, hidden] (one image's patch tokens). RoPE is per-patch, shared across heads.
        n, _ = x.shape
        q = self.q_proj(x).reshape(n, self.nh, self.hd)
        k = self.k_proj(x).reshape(n, self.nh, self.hd)
        v = self.v_proj(x).reshape(n, self.nh, self.hd)
        q = apply_rope_vision(q, cos, sin)
        k = apply_rope_vision(k, cos, sin)
        q = mx.transpose(q, (1, 0, 2))[None]                 # [1, H, N, hd]
        k = mx.transpose(k, (1, 0, 2))[None]
        v = mx.transpose(v, (1, 0, 2))[None]
        if use_fast:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        else:
            scores = (q @ mx.swapaxes(k, -1, -2)) * self.scale
            wts = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
            out = wts @ v
        out = mx.transpose(out[0], (1, 0, 2)).reshape(n, self.nh * self.hd)
        return self.out_proj(out)


class MiniMaxM3VisionMLP(nn.Module):
    """``fc2( gelu( fc1(x) ) )`` (width ``intermediate_size``, biased)."""

    def __init__(self, cfg: MiniMaxVisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(gelu(self.fc1(x)))


class MiniMaxM3VisionLayer(nn.Module):
    """Pre-norm CLIP encoder layer: ``x + attn(ln1(x))`` then ``x + mlp(ln2(x))`` (LayerNorm, biased)."""

    def __init__(self, cfg: MiniMaxVisionConfig) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.self_attn = MiniMaxM3VisionAttention(cfg)
        self.mlp = MiniMaxM3VisionMLP(cfg)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, *, use_fast: bool = True) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), cos, sin, use_fast=use_fast)
        x = x + self.mlp(self.layer_norm2(x))
        return x


# ----------------------------------------------------------------------------- #
# Projector + patch merge (order forced by on-disk input dims: project then merge).
# ----------------------------------------------------------------------------- #


class MiniMaxM3VisionProjector(nn.Module):
    """``multi_modal_projector``: per-patch ``linear_2( gelu( linear_1(x) ) )``, ``hidden(1280) →
    projection(6144) → 6144`` (both biased). Runs BEFORE the patch merge (its ``linear_1`` input is the
    ViT hidden 1280, fixing the order)."""

    def __init__(self, cfg: MiniMaxVisionConfig) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(cfg.hidden_size, cfg.projection_dim, bias=True)
        self.linear_2 = nn.Linear(cfg.projection_dim, cfg.projection_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(gelu(self.linear_1(x)))


class MiniMaxM3VisionPatchMerge(nn.Module):
    """``patch_merge_mlp``: concatenate each consecutive ``merge**2`` projected tokens (one 2×2 spatial
    block — the processor already orders patches in merge-blocks) → ``linear_2( gelu( linear_1(cat) ) )``,
    ``merge**2·6144 (24576) → 6144 → 6144`` (both biased). Reduces ``N`` tokens to ``N/merge**2``."""

    def __init__(self, cfg: MiniMaxVisionConfig) -> None:
        super().__init__()
        self.merge = cfg.spatial_merge_size
        in_dim = cfg.projection_dim * self.merge * self.merge
        self.linear_1 = nn.Linear(in_dim, cfg.projection_dim, bias=True)
        self.linear_2 = nn.Linear(cfg.projection_dim, cfg.projection_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        n, d = x.shape
        g = self.merge * self.merge
        if n % g != 0:
            raise ValueError(f"patch-merge: {n} tokens not divisible by merge**2={g} (rule 6)")
        x = x.reshape(n // g, g * d)
        return self.linear_2(gelu(self.linear_1(x)))


# ----------------------------------------------------------------------------- #
# Full vision tower.
# ----------------------------------------------------------------------------- #


class MiniMaxM3VisionModel(nn.Module):
    """The full M3 vision tower: patch embed → pre_layrnorm → 32 CLIP encoder layers (3-D RoPE) →
    multi_modal_projector → patch_merge → LLM tokens ``[N/merge**2, projection_dim]``.

    Forward takes ``pixel_values`` ``[N, in_dim]`` (the processor's flattened patches) and ``grid_thw``
    (one ``(t,h,w)`` per image, ``sum(t·h·w) == N``); each image's patches attend only within that
    image (bounded per-image loop, rule 3 — an IO/segmentation boundary, not a hot loop)."""

    def __init__(self, cfg: MiniMaxVisionConfig, *, rope_section: tuple[int, int, int] | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.hd = cfg.hidden_size // cfg.num_attention_heads
        self.merge = cfg.spatial_merge_size
        self.rope_section = rope_section or default_rope_section(self.hd)
        self.patch_embed = MiniMaxM3VisionPatchEmbed(cfg)
        self.pre_layrnorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layers = [MiniMaxM3VisionLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.projector = MiniMaxM3VisionProjector(cfg)
        self.patch_merge = MiniMaxM3VisionPatchMerge(cfg)

    def encode(self, pixel_values: mx.array, grid_thw: list[tuple[int, int, int]], *,
               use_fast: bool = True) -> mx.array:
        """ViT encoder over one image's patches → ``[N, hidden]`` (pre projector/merge). One image at a
        time so attention stays within the image; concatenated back along ``N``."""
        h = self.pre_layrnorm(self.patch_embed(pixel_values))
        outs = []
        off = 0
        for (t, hh, ww) in grid_thw:
            n = int(t) * int(hh) * int(ww)
            xi = h[off:off + n]
            pos = vision_position_ids([(t, hh, ww)], self.merge)
            cos, sin = vision_rope_3d(pos, self.hd, self.cfg.rope_theta, self.rope_section)
            for layer in self.layers:
                xi = layer(xi, cos, sin, use_fast=use_fast)
            outs.append(xi)
            off += n
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)

    def __call__(self, pixel_values: mx.array, grid_thw: list[tuple[int, int, int]], *,
                 use_fast: bool = True) -> mx.array:
        """Full tower → merged LLM tokens ``[N/merge**2, projection_dim]`` (the embeddings spliced at
        the ``image_token_index`` placeholders — the splice is the V2 milestone)."""
        feats = self.encode(pixel_values, grid_thw, use_fast=use_fast)   # [N, hidden]
        projected = self.projector(feats)                                # [N, projection_dim]
        return self.patch_merge(projected)                               # [N/merge**2, projection_dim]
