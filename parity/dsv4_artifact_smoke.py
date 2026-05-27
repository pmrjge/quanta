"""DSV4 baked-artifact smoke: load 1 layer, print bytes/shapes, no forward.

Diagnostic for the int4-g64 bake: confirms the artifact's manifest + key map + dequant paths
work end-to-end on a single layer (rule-8 streaming load), without the full 169 GB resident
build. Useful when the full ppl gate hangs/exits silently — this isolates artifact-reader
failures from runtime/forward failures.

    uv run python -u -m parity.dsv4_artifact_smoke
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.dsv4.artifact import DSV4Artifact
from quanta.dsv4.model import load_block_params

ART = "/Users/pmrj/models/DeepSeek-V4-Flash-quanta_int4g64"


def run() -> None:
    print(f"[smoke] opening artifact at {ART}", flush=True)
    t0 = time.perf_counter()
    art = DSV4Artifact(ART)
    cfg = art.cfg
    print(f"[smoke] cfg loaded in {(time.perf_counter() - t0) * 1000:.1f}ms  "
          f"layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
          f"head_dim={cfg.head_dim} vocab={cfg.vocab_size}", flush=True)

    t1 = time.perf_counter()
    p = load_block_params(art, cfg, 0)
    arrays = []
    for v in p.values():
        if isinstance(v, dict):
            arrays.extend(a for a in v.values() if isinstance(a, mx.array))
            for sub in v.values():
                if isinstance(sub, dict):
                    arrays.extend(a for a in sub.values() if isinstance(a, mx.array))
        elif isinstance(v, mx.array):
            arrays.append(v)
    mx.eval(arrays)
    art.release()
    mx.clear_cache()
    n_arr = len(arrays)
    total_bytes = sum(a.size * a.dtype.size for a in arrays)
    print(f"[smoke] layer 0 materialized in {(time.perf_counter() - t1):.2f}s  "
          f"arrays={n_arr}  ~{total_bytes / 1024**3:.2f} GiB", flush=True)

    # Sample one expert weight + one attention projection to confirm dequant paths fired.
    if "experts" in p and "w1" in p["experts"]:
        w1 = p["experts"]["w1"]
        print(f"[smoke] experts.w1 shape={tuple(w1.shape)} dtype={w1.dtype}", flush=True)
    if "attn" in p and "wq_a" in p["attn"]:
        wq = p["attn"]["wq_a"]
        print(f"[smoke] attn.wq_a shape={tuple(wq.shape)} dtype={wq.dtype}", flush=True)
    print("[smoke] PASS (artifact + per-layer load OK)", flush=True)


if __name__ == "__main__":
    run()
