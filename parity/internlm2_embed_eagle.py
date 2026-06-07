"""Embed the trained InternLM2.5 EAGLE-3 drafter into the int8-g64 artifact as a portable sidecar.

One-shot bake-time op (CLAUDE.md artifact rule: an additive ``eagle/`` subdir with **relative refs
only**, ``manifest.json`` untouched) so the oMLX shim
(:meth:`quanta.shim.omlx.QuantaOmlxEngine._ensure_eagle`) can auto-load the drafter via
:func:`quanta.eagle.artifact.load_eagle` and serve EAGLE-3 spec-decode — spec output bit-identical to
plain greedy (lossless), 1.42× @ k=2 at the int4-PTQ serving operating point. Re-serializes the standalone
``drafter_int8g64_refined2.safetensors`` (the finalized 0.46-holdout drafter) into ``<art>/eagle/``;
refuses to overwrite an existing ``eagle/`` (remove it first to re-embed). Solo, ~seconds (loads the ~1 GB
bf16 drafter, validates it against the declared config, re-saves canonically).

    uv run python -m parity.internlm2_embed_eagle [art_dir] [drafter.safetensors]
"""

from __future__ import annotations

import sys
from pathlib import Path

from quanta.eagle.artifact import embed_eagle
from quanta.internlm2.eagle import DEFAULT_CAPTURE_LAYERS, INTERNLM2_DRAFTER_CFG

ART = "/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64"
DRAFTER = "/Users/pmrj/models/internlm2_eagle/drafter_int8g64_refined2.safetensors"


def run(art: str = ART, drafter: str = DRAFTER) -> None:
    out = embed_eagle(
        art, drafter,
        capture_layers=DEFAULT_CAPTURE_LAYERS,
        drafter_cfg=INTERNLM2_DRAFTER_CFG,
        training_meta={
            "target_model": "internlm2_5-7b-chat-1m",
            "source_drafter": Path(drafter).name,
            "holdout_accept": 0.46,
            "serving": "int4-PTQ drafter + head_bits=4 -> 1.42x lossless @ k=2",
            "track": "project_internlm2_eagle (M0-M3, commit ec0f6f3)",
        },
    )
    print(f"embedded EAGLE-3 drafter sidecar -> {out}")
    print(f"  capture_layers={DEFAULT_CAPTURE_LAYERS}")
    print(f"  drafter_cfg={INTERNLM2_DRAFTER_CFG}")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else ART
    d = sys.argv[2] if len(sys.argv) > 2 else DRAFTER
    run(a, d)
