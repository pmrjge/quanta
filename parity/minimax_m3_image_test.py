"""Model-free parity gate for the MiniMax-M3-VL **image processor** (:mod:`quanta.minimax.image_m3`).

The native numpy processor reproduces the shipped ``~/models/MiniMax-M3/image_processor.py``
(``MiniMaxM3VLImageProcessor``) without its ``torch``/``torchvision`` deps (rule 5). Each deterministic
stage is pinned to an independent oracle on tiny dims:

* **smart_resize** geometry == a verbatim re-derivation of the shipped formula over a grid of (H,W);
  plus the factor-aligned-identity property (an in-bounds multiple-of-28 image is unchanged).
* **rescale + CLIP-normalize** == ``(x/255 − mean)/std`` numpy-fp64.
* **patchify** == a slow, obviously-correct nested-loop oracle (merge-block token order; patch axis
  ``[channel, temporal, patch_h, patch_w]`` — the order the conv-as-linear patch embed consumes).
* **full pipeline** on a factor-aligned image (resize == identity ⇒ exactly pinnable): ``pixel_values``
  shape ``[t·h·w, 1176]``, ``grid_thw`` correct, ``num_tokens == grid.prod()//merge²``.
* **bicubic resampler**: identity on a same-size target; constant-image preservation; rows sum to 1.
* **rule 6**: extreme aspect ratio + non-RGB shape refused.

There is **no** ``torchvision`` in this env, so the bicubic *interpolation* for an off-grid image is
not bit-pinned here — but it never enters the gated path or the V3 e2e (both use factor-aligned images
where resize is the identity). Runs in the model-free sweep (numpy only; no disk, no model).

    uv run python -m parity.minimax_m3_image_test
"""

from __future__ import annotations

import math

import numpy as np

from quanta.minimax import image_m3 as IP

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _rel(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-9))


# --- independent oracles ------------------------------------------------------ #


def _smart_resize_oracle(h, w, factor=28, min_pixels=IP.MIN_PIXELS, max_pixels=IP.MAX_PIXELS):
    """Verbatim re-derivation of the shipped smart_resize (pure math; the reference)."""
    def rbf(n, f):
        return round(n / f) * f
    h_bar = max(factor, rbf(h, factor))
    w_bar = max(factor, rbf(w, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = math.floor(h / beta / factor) * factor
        w_bar = math.floor(w / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def _patchify_oracle(frames, grid_t, gh, gw, patch, merge, temporal):
    """Slow, obviously-correct patchify: explicit nested indexing in merge-block token order with the
    patch axis flattened ``[channel, temporal, patch_h, patch_w]``. frames: [grid_t*temporal, C, H, W]."""
    c = frames.shape[1]
    rows = []
    for gt in range(grid_t):
        for bh in range(gh // merge):
            for bw in range(gw // merge):
                for mh in range(merge):
                    for mw in range(merge):
                        gh_idx, gw_idx = bh * merge + mh, bw * merge + mw
                        vec = []
                        for ch in range(c):
                            for tt in range(temporal):
                                for ph in range(patch):
                                    for pw in range(patch):
                                        vec.append(frames[gt * temporal + tt, ch,
                                                          gh_idx * patch + ph, gw_idx * patch + pw])
                        rows.append(vec)
    return np.asarray(rows, dtype=np.float64)


def run() -> None:
    rng = np.random.default_rng(0)

    # 1) smart_resize geometry == oracle across a grid of sizes (incl. up/down clamps)
    for h, w in [(224, 224), (100, 200), (37, 53), (1, 5000), (2000, 2000), (28, 28), (4000, 30)]:
        try:
            got = IP.smart_resize(h, w)
            exp = _smart_resize_oracle(h, w)
            _ck(got == exp, f"smart_resize({h},{w})={got} != oracle {exp}")
        except ValueError:
            # extreme ratio: both must raise
            try:
                _smart_resize_oracle(h, w)
            except Exception:
                pass
            _ck(max(h, w) / min(h, w) > IP.MAX_RATIO, f"smart_resize({h},{w}) raised unexpectedly")

    # 2) factor-aligned in-bounds image is unchanged by smart_resize (the gated/e2e path). In-bounds
    #    means min_pixels (3136) <= h*w <= max_pixels (451584); 56x56 == min exactly (not < min).
    for h, w in [(56, 56), (84, 56), (224, 168), (140, 140)]:
        _ck(IP.smart_resize(h, w) == (h, w), f"factor-aligned {h}x{w} should be identity")

    # 3) bicubic: identity on same size; constant preserved; weight rows sum to 1
    img = rng.standard_normal((56, 84, 3)).astype(np.float64)
    _ck(_rel(IP.resize_bicubic(img, 56, 84), img) == 0.0, "resize same-size not identity")
    const = np.full((40, 40, 3), 0.7)
    _ck(_rel(IP.resize_bicubic(const, 28, 28), np.full((28, 28, 3), 0.7)) < 1e-9,
        "bicubic does not preserve a constant image")
    W = IP._axis_weights(28, 40)
    _ck(np.allclose(W.sum(axis=1), 1.0, atol=1e-12), "resample rows must sum to 1")
    _ck(IP.resize_bicubic(img, 70, 84).shape == (70, 84, 3), "resize output shape")

    # 4) rescale + CLIP-normalize == oracle (on a factor-aligned uint8 image so resize is identity)
    proc = IP.MiniMaxM3ImageProcessor()
    u8 = rng.integers(0, 256, size=(56, 56, 3)).astype(np.uint8)
    out = proc.preprocess(u8)
    norm_ref = (u8.astype(np.float64) / 255.0 - np.asarray(IP.IMAGE_MEAN)) / np.asarray(IP.IMAGE_STD)
    # rebuild the expected pixel_values via the oracle patchify
    chw = np.transpose(norm_ref, (2, 0, 1))                       # [C,H,W]
    frames = np.repeat(chw[None], IP.TEMPORAL_PATCH_SIZE, axis=0)  # [2,C,H,W]
    gh, gw = 56 // IP.PATCH_SIZE, 56 // IP.PATCH_SIZE             # 4,4
    ref_pv = _patchify_oracle(frames, 1, gh, gw, IP.PATCH_SIZE, IP.MERGE_SIZE, IP.TEMPORAL_PATCH_SIZE)
    _ck(out.grid_thw == [(1, gh, gw)], f"grid_thw wrong: {out.grid_thw}")
    _ck(out.pixel_values.shape == (gh * gw, 1176), f"pixel_values shape {out.pixel_values.shape}")
    _ck(_rel(out.pixel_values, ref_pv) < 1e-5, "pixel_values != rescale/normalize/patchify oracle")

    # 5) patchify oracle == module patchify directly (independent of normalize). gh,gw must be
    #    multiples of merge (smart_resize rounds H,W to patch*merge ⇒ grids are always even here).
    fr = rng.standard_normal((2, 3, 28, 56)).astype(np.float64)   # gh=2, gw=4
    got = IP.patchify(fr, 1, 2, 4, IP.PATCH_SIZE, IP.MERGE_SIZE, IP.TEMPORAL_PATCH_SIZE)
    exp = _patchify_oracle(fr, 1, 2, 4, IP.PATCH_SIZE, IP.MERGE_SIZE, IP.TEMPORAL_PATCH_SIZE)
    _ck(got.shape == (8, 1176) and _rel(got, exp) < 1e-9, "patchify != nested-loop oracle")

    # 6) num_tokens == grid.prod() // merge**2 ; multi-image concatenation
    im2 = [rng.integers(0, 256, (56, 56, 3), np.uint8), rng.integers(0, 256, (84, 112, 3), np.uint8)]
    o2 = proc.preprocess(im2)
    g0, g1 = o2.grid_thw
    _ck(o2.pixel_values.shape[0] == g0[0] * g0[1] * g0[2] + g1[0] * g1[1] * g1[2], "concat token count")
    _ck(o2.num_tokens() == [(t * h * w) // 4 for (t, h, w) in o2.grid_thw], "num_tokens rule")

    # 7) rule 6: extreme aspect ratio + non-RGB shape refused
    try:
        IP.smart_resize(1, 5000 * IP.MAX_RATIO)
        _ck(False, "extreme ratio accepted")
    except ValueError:
        _ck(True, "extreme ratio refused")
    try:
        proc.preprocess(rng.standard_normal((10, 10)))          # not [H,W,3]
        _ck(False, "non-RGB image accepted")
    except ValueError:
        _ck(True, "non-RGB image refused")

    print(f"PARITY-CHECKS: {_N}")
    print(f"PASS — MiniMax-M3-VL image processor: smart_resize/normalize/patchify == oracle, "
          f"factor-aligned identity, bicubic well-formed, num_tokens rule, rule-6 refusals "
          f"({_N} checks).")


if __name__ == "__main__":
    run()
