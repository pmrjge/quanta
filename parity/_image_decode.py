"""Offline image decoder (file/bytes → ``[H,W,3]`` uint8 RGB numpy) for the vision-tower e2e arbiter.

The native :mod:`quanta.minimax.image_m3` processor takes an already-decoded ``[H,W,3]`` RGB array
(it is the runtime front-end and stays torch/PIL-free, rule 5). Turning an actual JPEG/PNG *file* into
that array needs a codec, which only the offline parity/arbiter path requires — so this lives under
``parity/`` (never imported by ``src/``) and imports **PIL/pillow lazily**, under the ``reference``
extra. It is NOT on the runtime hot path; the eventual serving image-input path (the oMLX shim) decides
its own decode strategy separately.

``decode_image_rgb`` / ``decode_image_rgb_from_bytes`` return a C-contiguous ``uint8`` ``[H,W,3]``
array (alpha/grayscale/palette images are converted to 3-channel RGB), the exact shape
:meth:`quanta.minimax.image_m3.MiniMaxM3ImageProcessor.preprocess` consumes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _to_rgb_array(img) -> np.ndarray:
    """A PIL ``Image`` → ``[H,W,3]`` uint8 RGB numpy (convert any mode to RGB, drop alpha/palette)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"decoded image is not [H,W,3] RGB (got shape {arr.shape}) — rule 6")
    return np.ascontiguousarray(arr)


def decode_image_rgb(path: str | Path) -> np.ndarray:
    """Decode an image file at ``path`` to a ``[H,W,3]`` uint8 RGB array (PIL, lazily imported)."""
    from PIL import Image  # noqa: PLC0415 — offline-only (reference extra); never on the hot path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {p}")
    with Image.open(p) as img:
        img.load()
        return _to_rgb_array(img)


def decode_image_rgb_from_bytes(data: bytes) -> np.ndarray:
    """Decode encoded image ``bytes`` (PNG/JPEG/…) to a ``[H,W,3]`` uint8 RGB array (PIL, lazy)."""
    import io  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415 — offline-only (reference extra)

    with Image.open(io.BytesIO(data)) as img:
        img.load()
        return _to_rgb_array(img)
