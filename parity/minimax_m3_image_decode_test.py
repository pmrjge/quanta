"""Model-free V3b-prep gate: the offline image decoder (:mod:`parity._image_decode`) round-trips.

The V3b real-weight arbiter turns a real image *file* into the ``[H,W,3]`` uint8 RGB array that the
native :class:`quanta.minimax.image_m3.MiniMaxM3ImageProcessor` consumes. That decode is the one new
piece of code on the arbiter's input path, so it gets its own gate:

* **Lossless round-trip.** A synthetic uint8 RGB array → PNG (in-memory) → ``decode_image_rgb_from_bytes``
  reproduces it **bit-exact** (PNG is lossless), as does the file path via ``decode_image_rgb``.
* **Mode coercion (rule 6).** Grayscale (``L``) and alpha (``RGBA``) inputs decode to a 3-channel RGB
  array (``[H,W,3]`` uint8) — grayscale broadcasts to R==G==B; RGBA drops alpha onto the RGB it carries.
* **Processor integration.** The decoded array flows through ``MiniMaxM3ImageProcessor.preprocess`` on a
  factor-aligned (28-multiple) size — where the bicubic resize is the identity, so the path is exact —
  yielding the expected ``grid_thw`` + ``pixel_values [N,1176]``.

Imports PIL/pillow (the ``reference`` extra), so on a base-deps-only env this gate is **SKIPPED**, not
failed (``optional_deps`` maps ``pillow`` → ``PIL``).

    uv run --extra reference python -m parity.minimax_m3_image_decode_test
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image  # noqa: PLC0415 — reference extra; a base-deps env SKIPs this gate

from parity._image_decode import decode_image_rgb, decode_image_rgb_from_bytes
from quanta.minimax.image_m3 import MiniMaxM3ImageProcessor

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _png_bytes(arr: np.ndarray, mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def run() -> None:
    rng = np.random.default_rng(0)

    # --- (1) lossless RGB round-trip (bytes) ------------------------------------------------------
    h, w = 28, 56
    rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    dec = decode_image_rgb_from_bytes(_png_bytes(rgb))
    _ck(dec.shape == (h, w, 3), f"decoded shape {dec.shape} != {(h, w, 3)}")
    _ck(dec.dtype == np.uint8, f"decoded dtype {dec.dtype} != uint8")
    _ck(np.array_equal(dec, rgb), "PNG RGB round-trip not bit-exact")
    _ck(dec.flags["C_CONTIGUOUS"], "decoded array must be C-contiguous")

    # --- (2) file-path variant matches the bytes variant -----------------------------------------
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "img.png"
        Image.fromarray(rgb, mode="RGB").save(p)
        dec_file = decode_image_rgb(p)
        _ck(np.array_equal(dec_file, rgb), "file-path decode != in-memory decode")
        try:
            decode_image_rgb(Path(d) / "missing.png")
            _ck(False, "missing file accepted")
        except FileNotFoundError:
            _ck(True, "missing file refused (rule 6)")

    # --- (3) mode coercion: grayscale + RGBA → [H,W,3] RGB ---------------------------------------
    gray = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    dg = decode_image_rgb_from_bytes(_png_bytes(gray, mode="L"))
    _ck(dg.shape == (h, w, 3), f"grayscale decode shape {dg.shape}")
    _ck(np.array_equal(dg[..., 0], dg[..., 1]) and np.array_equal(dg[..., 1], dg[..., 2]),
        "grayscale must broadcast to R==G==B")
    _ck(np.array_equal(dg[..., 0], gray), "grayscale luminance not preserved on the R channel")
    rgba = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], axis=-1)
    da = decode_image_rgb_from_bytes(_png_bytes(rgba, mode="RGBA"))
    _ck(da.shape == (h, w, 3), f"RGBA decode shape {da.shape}")
    _ck(np.array_equal(da, rgb), "opaque RGBA must drop alpha onto the carried RGB")

    # --- (4) decoded array flows through the native processor (factor-aligned ⇒ resize identity) --
    aligned = rng.integers(0, 256, size=(56, 56, 3), dtype=np.uint8)   # 56 = 2*28, in-bounds
    out = MiniMaxM3ImageProcessor().preprocess(decode_image_rgb_from_bytes(_png_bytes(aligned)))
    _ck(out.grid_thw == [(1, 4, 4)], f"processor grid_thw {out.grid_thw} (want [(1,4,4)] for 56x56)")
    _ck(out.pixel_values.shape == (16, 1176), f"pixel_values shape {out.pixel_values.shape}")
    _ck(out.num_tokens() == [4], f"num_tokens {out.num_tokens()} (want [4])")

    print(f"PARITY-CHECKS: {_N}")
    print(f"PASS — MiniMax-M3-VL V3b-prep image decoder: lossless PNG round-trip (bytes + file), "
          f"grayscale/RGBA → [H,W,3] RGB, decoded array flows through the native processor "
          f"({_N} checks).")


if __name__ == "__main__":
    run()
