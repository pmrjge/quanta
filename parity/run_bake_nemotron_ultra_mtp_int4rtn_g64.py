"""Nemotron-3-Ultra MTP draft-head RTN sidecar bake → ...-quanta_int4rtn_g64_mtp.

The MTP-M1 driver: bakes the native-MTP self-speculative draft head (#40) as a self-contained
**sidecar** of the shipped int4-RTN backbone artifact (`...-quanta_int4rtn_g64`). Same per-tensor
policy as the backbone — int4-RTN routed experts (512, relu² latent-MoE), int8 affine dense
(`eh_proj`, attn q/k/v/o, latent fc1/fc2, shared expert), bf16 fusion/sub-block/final norms +
router gate/bias — written as its **own** bundle so the immutable backbone artifact is untouched
(the MTP-M2 loader pairs the two). The MTP head's group size (g64) matches the backbone arm.

RTN is data-free (no calibration) and streamed one expert resident at a time (rule 8): the ~21.5 GiB
expert stack is never materialized, so this is a small, fast bake (~minutes), not the 1023 GiB whole
model. The recon gate `parity/nemotron_ultra_mtp_bake_parity.py` then verifies the sidecar
dequantizes faithfully (baked-head forward ≈ bf16-head forward). **Run solo (OOM hazard — one model
resident at a time).**

    uv run python -m parity.run_bake_nemotron_ultra_mtp_int4rtn_g64
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.nemotron.bake import bake_nemotron_mtp

ULTRA = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
OUT = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-quanta_int4rtn_g64_mtp"
GROUP_SIZE = 64  # same expert footprint as the backbone int4-RTN arm


def run() -> None:
    mx.set_cache_limit(32 * 1024**3)  # cap the MLX buffer cache so the resident set can't balloon
    print(f"baking MTP draft head: int4-RTN g{GROUP_SIZE} experts, int8 dense, bf16 core -> {OUT}",
          flush=True)
    t0 = time.perf_counter()
    stats = bake_nemotron_mtp(ULTRA, OUT, group_size=GROUP_SIZE, expert_method="rtn",
                              scale_dtype=mx.bfloat16)
    print(f"NEMOTRON-ULTRA MTP RTN BAKE DONE in {(time.perf_counter() - t0) / 60:.2f} min\n{stats}",
          flush=True)


if __name__ == "__main__":
    run()
