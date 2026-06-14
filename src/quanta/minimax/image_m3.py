"""MiniMax-M3-VL **image processor** — native numpy, self-contained (no torch/torchvision/PIL).

The vision-tower front end: an RGB image → the flattened patch tensor ``pixel_values`` ``[N, 1176]``
(``1176 = channels·temporal·patch·patch = 3·2·14·14``) + the ``grid_thw`` ``(t, h, w)`` the
:mod:`quanta.minimax.model_vision_m3` tower consumes. It reproduces the shipped reference
``~/models/MiniMax-M3/image_processor.py`` (``MiniMaxM3VLImageProcessor``, a Qwen2-VL-style fast
processor) **without** its ``torch``/``torchvision`` deps so it runs on the inference path (rule 5).

Pipeline (verbatim from the shipped ``_preprocess`` / ``smart_resize``):

1. **smart_resize** — round H,W to a multiple of ``factor = patch·merge = 28`` and clamp the patch
   count to ``[min_pixels, max_pixels]`` (``4·28·28`` … ``672·672``). Pure integer geometry; pinned
   bit-exact in the gate.
2. **resize** to the smart-resize target (bicubic). On a **factor-aligned, in-bounds** image the
   target equals the input ⇒ resize is the identity (no interpolation) — the path the parity gates and
   the V3 e2e use, so it is exactly pinnable. For an off-grid image a separable cubic-convolution
   resampler runs (Keys ``a=-0.75``, half-pixel centers, antialias on downscale): geometry-exact, but
   the interpolation is **best-effort** — there is no ``torchvision`` in this env to pin it against
   (documented; it never enters the gated/e2e path).
3. **rescale + CLIP-normalize** — ``(x/255 − mean)/std`` per channel (CLIP mean/std). Exact.
4. **temporal pad** — a single image is one frame; ``temporal_patch_size = 2`` ⇒ the frame is
   duplicated to 2 (so ``grid_t = 1``). Exact.
5. **patchify** — the shipped view + ``permute(0,1,4,7,5,8,3,2,6,9)`` + reshape, emitting tokens in
   **merge-block order** (consecutive ``merge²`` patches = one 2×2 spatial block — what
   :class:`quanta.minimax.model_vision_m3.MiniMaxM3VisionPatchMerge` later concatenates and what
   :func:`quanta.minimax.model_vision_m3.vision_position_ids` mirrors) and each patch flattened in
   ``[channel, temporal, patch_h, patch_w]`` order (== the conv weight ``[1280,3,2,14,14]`` reshaped
   ``[1280,1176]``, so the patch-embed linear applies directly). Exact vs the numpy-fp64 oracle.

One image ``grid=(t,h,w)`` → ``t·h·w`` patches → (after the tower's project+merge) ``(t·h·w)//merge²``
LLM tokens == the processor's ``num_tokens = grid.prod() // merge²`` placeholder count at
``image_token_index`` 200025 (the splice is the V3 milestone). Pure numpy throughout; the only loops
are the bounded kernel-tap loop in the resampler and the per-image grouping loop (coarse, non-hot —
rule 3 allows IO/segmentation loops), and they never touch the decode path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# CLIP normalization (shipped preprocessor_config.json / image_processor.py defaults).
IMAGE_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
RESCALE_FACTOR = 1.0 / 255.0
MAX_PIXELS = 451584          # 672*672
MIN_PIXELS = 4 * 28 * 28     # 3136
MAX_RATIO = 200


# ----------------------------------------------------------------------------- #
# smart_resize geometry (verbatim from the shipped image_processor.smart_resize).
# ----------------------------------------------------------------------------- #


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int = 28,
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """Round (H,W) to multiples of ``factor`` with the patch count clamped to
    ``[min_pixels, max_pixels]`` — a verbatim copy of the shipped ``smart_resize`` (pure ``math``, no
    torch), so the gate can pin it bit-exact. Refuses an extreme aspect ratio (rule 6)."""
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


# ----------------------------------------------------------------------------- #
# Separable bicubic resampler (best-effort; identity on a same-size target).
# ----------------------------------------------------------------------------- #


def _cubic_kernel(t: np.ndarray, a: float = -0.75) -> np.ndarray:
    """Keys cubic-convolution kernel (``a=-0.75``, the torchvision/PIL default), evaluated on |t|."""
    x = np.abs(t)
    x2, x3 = x * x, x * x * x
    inner = (a + 2.0) * x3 - (a + 3.0) * x2 + 1.0                  # |x| <= 1
    outer = a * x3 - 5.0 * a * x2 + 8.0 * a * x - 4.0 * a          # 1 < |x| < 2
    return np.where(x <= 1.0, inner, np.where(x < 2.0, outer, 0.0))


def _axis_weights(dst: int, src: int, a: float = -0.75, antialias: bool = True) -> np.ndarray:
    """Dense ``[dst, src]`` resample weight matrix for one axis (half-pixel centers, normalized rows).

    For ``dst == src`` returns the identity exactly (no interpolation). The kernel-tap loop is bounded
    (≤ ~``ceil(support)*2`` columns) and runs only at preprocess time — not a compute hot path."""
    if dst == src:
        return np.eye(src, dtype=np.float64)
    scale = src / dst
    downs = antialias and scale > 1.0
    support = 2.0 * (scale if downs else 1.0)
    inv = (1.0 / scale) if downs else 1.0
    centers = (np.arange(dst) + 0.5) * scale - 0.5                 # [dst]
    left = np.floor(centers - support).astype(int)
    width = int(math.ceil(support)) * 2 + 2
    cols = left[:, None] + np.arange(width)[None, :]              # [dst, width] src indices
    w = _cubic_kernel((centers[:, None] - cols) * inv, a)         # [dst, width]
    w = w * ((cols >= 0) & (cols < src))                         # drop out-of-range taps
    cols_cl = np.clip(cols, 0, src - 1)
    mat = np.zeros((dst, src), dtype=np.float64)
    for j in range(width):                                        # bounded (~10) preprocess loop
        np.add.at(mat, (np.arange(dst), cols_cl[:, j]), w[:, j])
    mat /= mat.sum(axis=1, keepdims=True)
    return mat


def resize_bicubic(img_hwc: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize an ``[H,W,C]`` float image to ``[out_h, out_w, C]`` (separable bicubic; identity if the
    size already matches). fp64 internally."""
    h, w, _ = img_hwc.shape
    if (h, w) == (out_h, out_w):
        return img_hwc.astype(np.float64)
    wh = _axis_weights(out_h, h)                                  # [out_h, H]
    ww = _axis_weights(out_w, w)                                  # [out_w, W]
    x = img_hwc.astype(np.float64)
    x = np.tensordot(wh, x, axes=([1], [0]))                      # [out_h, W, C]
    x = np.tensordot(x, ww, axes=([1], [1]))                      # [out_h, C, out_w]
    return np.transpose(x, (0, 2, 1))                             # [out_h, out_w, C]


# ----------------------------------------------------------------------------- #
# Patchify (the shipped view+permute, batch-stripped) — pure index/reshape.
# ----------------------------------------------------------------------------- #


def patchify(frames_fchw: np.ndarray, grid_t: int, grid_h: int, grid_w: int,
             patch: int, merge: int, temporal: int) -> np.ndarray:
    """``[frames, C, H, W]`` (frames == ``grid_t·temporal``) → ``[grid_t·grid_h·grid_w, C·temporal·patch²]``.

    Reproduces the shipped ``_preprocess`` view + ``permute(0,1,4,7,5,8,3,2,6,9)`` (batch dim stripped):
    token axis ordered ``(grid_t, gh//m, gw//m, m_h, m_w)`` (merge-block order), patch axis ordered
    ``(channel, temporal, patch_h, patch_w)``."""
    c = frames_fchw.shape[1]
    x = frames_fchw.reshape(grid_t, temporal, c,
                            grid_h // merge, merge, patch,
                            grid_w // merge, merge, patch)
    # shipped permute (0,1,4,7,5,8,3,2,6,9) without the batch axis (subtract 1 from each):
    #   token: grid_t(0) gh//m(3) gw//m(6) m_h(4) m_w(7) ; patch: channel(2) temporal(1) p_h(5) p_w(8)
    x = np.transpose(x, (0, 3, 6, 4, 7, 2, 1, 5, 8))
    return x.reshape(grid_t * grid_h * grid_w, c * temporal * patch * patch)


# ----------------------------------------------------------------------------- #
# The processor.
# ----------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ImageProcessorOutput:
    """``pixel_values`` ``[N, 1176]`` (float32) + ``grid_thw`` (one ``(t,h,w)`` per image)."""

    pixel_values: np.ndarray
    grid_thw: list[tuple[int, int, int]]

    def num_tokens(self, merge: int = MERGE_SIZE) -> list[int]:
        """Placeholder count per image == ``grid.prod() // merge²`` (the shipped processor's rule)."""
        return [(t * h * w) // (merge * merge) for (t, h, w) in self.grid_thw]


class MiniMaxM3ImageProcessor:
    """Native MiniMax-M3-VL image processor (numpy). ``preprocess`` takes a single RGB image or a list
    of them and returns concatenated ``pixel_values`` + per-image ``grid_thw``."""

    def __init__(self, *, patch_size: int = PATCH_SIZE, temporal_patch_size: int = TEMPORAL_PATCH_SIZE,
                 merge_size: int = MERGE_SIZE, max_pixels: int = MAX_PIXELS,
                 min_pixels: int = MIN_PIXELS,
                 image_mean: tuple[float, float, float] = IMAGE_MEAN,
                 image_std: tuple[float, float, float] = IMAGE_STD) -> None:
        self.patch = patch_size
        self.temporal = temporal_patch_size
        self.merge = merge_size
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.mean = np.asarray(image_mean, dtype=np.float64)
        self.std = np.asarray(image_std, dtype=np.float64)
        self.factor = patch_size * merge_size

    def _one(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
        """One RGB image ``[H,W,3]`` (uint8 or float 0–255) → ``([t·h·w, 1176] float32, (t,h,w))``."""
        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"expected an [H,W,3] RGB image, got shape {img.shape} (rule 6)")
        h0, w0, _ = img.shape
        rh, rw = smart_resize(h0, w0, self.factor, self.min_pixels, self.max_pixels)
        x = resize_bicubic(img.astype(np.float64), rh, rw)        # [rh, rw, 3] (identity if aligned)
        x = (x * RESCALE_FACTOR - self.mean) / self.std          # rescale + CLIP-normalize
        x = np.transpose(x, (2, 0, 1))                           # [C, H, W]
        # temporal pad: one image -> duplicate to `temporal` frames (grid_t = 1).
        frames = np.repeat(x[None], self.temporal, axis=0)       # [temporal, C, H, W]
        grid_t = frames.shape[0] // self.temporal                # == 1
        grid_h, grid_w = rh // self.patch, rw // self.patch
        flat = patchify(frames, grid_t, grid_h, grid_w, self.patch, self.merge, self.temporal)
        return flat.astype(np.float32), (grid_t, grid_h, grid_w)

    def preprocess(self, images: np.ndarray | list[np.ndarray]) -> ImageProcessorOutput:
        """Preprocess one image or a list. ``pixel_values`` are concatenated along ``N`` in image
        order; ``grid_thw`` keeps one ``(t,h,w)`` per image (the splice consumes them in order)."""
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
        flats, grids = [], []
        for im in images:                                        # coarse per-image loop (rule 3 ok)
            f, g = self._one(im)
            flats.append(f)
            grids.append(g)
        return ImageProcessorOutput(pixel_values=np.concatenate(flats, axis=0), grid_thw=grids)
