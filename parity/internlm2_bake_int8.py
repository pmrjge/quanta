"""Full int8 g64 bake of InternLM2.5-7B-Chat-1M → the resident serving artifact.

int8 attention + int8 FFN, g64, bf16 embed/norms/output. A 7B fits comfortably resident at int8
(~7 GB) on the 512 GB box, and int8 affine RTN is ~lossless — so int8-everywhere is the right
serving policy here (no int4 byte-budget pressure like the giant MoE targets). The runtime reads
the per-weight width back from the manifest, so no runtime flag needs to change.

    uv run --with numpy python -m parity.internlm2_bake_int8
"""

from __future__ import annotations

import time
from pathlib import Path

from quanta.internlm2.bake import bake_internlm2

SOURCE = "/Users/pmrj/models/internlm2_5-7b-chat-1m"
OUT = "/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64"


def run() -> None:
    t0 = time.time()
    summary = bake_internlm2(SOURCE, OUT, group_size=64, attn_bits=8, mlp_bits=8)
    dt = time.time() - t0
    gb = summary["bytes"] / 1e9
    print(f"baked -> {OUT}")
    print(f"  layers={summary['layers']}  counts={summary['counts']}  "
          f"size={gb:.2f} GB  in {dt:.1f}s")
    print(f"  sidecars={summary['sidecars']}")
    # sanity: 32 layers × (4 attn + 3 mlp) quantized; dense = embed + final_norm + output + 64 norms
    exp_attn, exp_mlp = 32 * 4, 32 * 3
    ok = (summary["counts"]["attn_quant"] == exp_attn
          and summary["counts"]["mlp_quant"] == exp_mlp)
    print(f"  quant counts as expected (attn={exp_attn}, mlp={exp_mlp}): {ok}")
    assert ok and Path(OUT, "manifest.json").exists()
    print("\nPASS")


if __name__ == "__main__":
    run()
