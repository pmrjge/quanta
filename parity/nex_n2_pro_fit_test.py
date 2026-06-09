"""Nex-N2-Pro enablement gate (N0): config parses + correct ChatML eos + the loader key contract
holds + the int4/int6 quant mix is RAM-resident under the 490.4 GiB ceiling — before any bake.

Nex-N2-Pro is the post-trained **Qwen3.5-397B-A17B** (``qwen3_5_moe``), so the existing ``qwen35``
runtime/bake target it directly. This gate proves, against the REAL 122-shard checkpoint (``config`` +
``model.safetensors.index.json`` + the per-shard safetensors HEADERS only — **no tensor is
materialized**, sub-second):

* :meth:`Qwen35Config.from_pretrained` parses Nex (60L hybrid: 45 linear + 15 full, 512e top-10) and
  derives the CORRECT chat stop set ``{<|im_end|>=248046, <|endoftext|>=248044}`` despite Nex shipping
  **no** ``generation_config.json`` — its ``config.json`` lists only ``eos=248044`` (``<|endoftext|>``,
  a doc separator that never ends a turn; serving it alone is a rule-6 bug);
* the bake's quant policy classifies EVERY source tensor (rule 6 — no unmapped key) AND the expected
  keymap EXACTLY matches Nex's on-disk index (the loader's key contract holds at 397B scale — no drift
  vs the 35B sibling it was verified on);
* the milestone: the **int4-g64 AND int6-g64** expert mixes (int8 dense + bf16 control) are both
  resident under 490.4 GiB; a header-vs-file cross-check validates the byte accounting.

Real-weight-PATH gate (reads ``~/models/Nex-N2-Pro``), so it is EXCLUDED from the model-free sweep and
run SOLO / by hand — like ``parity/nemotron_ultra_fit_test.py``. It loads no weights (headers only).

    uv run python -m parity.nex_n2_pro_fit_test
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.quant_policy import VISION_PREFIX, coverage, dtype_bytes, project_resident

NEX = "/Users/pmrj/models/Nex-N2-Pro"
CEILING_GIB = 490.4  # M3 Ultra recommended max working set (CLAUDE.md)


def _read_header(path: Path) -> dict:
    """The safetensors header (tensor → {dtype, shape, data_offsets}) — first 8 bytes are the u64-LE
    header length, then that many bytes of JSON. Reads ~one block per shard; materializes no tensor."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def _tensor_sizes(model_dir: Path, wm: dict) -> dict[str, tuple[str, int]]:
    """``{key: (dtype, numel)}`` from each shard's header, read once per shard (no tensor loaded)."""
    by_shard: dict[str, list[str]] = {}
    for k, shard in wm.items():
        by_shard.setdefault(shard, []).append(k)
    sizes: dict[str, tuple[str, int]] = {}
    for shard, keys in by_shard.items():
        hdr = _read_header(model_dir / shard)
        for k in keys:
            shape = hdr[k]["shape"]
            sizes[k] = (hdr[k]["dtype"], math.prod(shape) if shape else 1)
    return sizes


def run() -> None:
    d = Path(NEX)
    cfg = Qwen35Config.from_pretrained(d)

    # 1. config + eos correctness (the must-do #1 fix, verified on the real checkpoint)
    dims_ok = (cfg.num_hidden_layers == 60 and cfg.num_experts == 512
               and cfg.num_experts_per_tok == 10 and cfg.head_dim == 256
               and cfg.partial_rotary_factor == 0.25 and cfg.mrope_section == (11, 11, 10)
               and cfg.has_shared_expert and cfg.vocab_size == 248320)
    full = sum(1 for i in range(cfg.num_hidden_layers) if cfg.is_full_attention(i))
    raw_tc = json.loads((d / "config.json").read_text()).get("text_config", {})
    declared_mtp = int(raw_tc.get("mtp_num_hidden_layers", 0))
    assert dims_ok, "Nex key dims mismatch"
    assert full == 15, f"expected 15 full-attention layers, got {full}"
    assert cfg.eos_token_ids == (248046, 248044), f"wrong chat stop set {cfg.eos_token_ids}"
    assert cfg.eos_token_id == 248046, "primary eos must be <|im_end|> (the turn-ender)"
    # Nex declares mtp_num_hidden_layers=1 but ships NO mtp.* weights → from_pretrained refines to 0,
    # so native-MTP spec-decode is unavailable for Nex (a drafter would have to come from elsewhere).
    assert declared_mtp == 1 and cfg.num_mtp_modules == 0, \
        f"MTP presence-refine wrong: declared {declared_mtp}, effective {cfg.num_mtp_modules}"

    # 2. quant-policy coverage vs the REAL index (rule 6 + the loader contract @ 397B)
    wm = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    cov = coverage(list(wm), cfg)
    assert not cov["missing"], f"policy expects keys absent on disk (loader drift): {cov['missing'][:6]}"
    assert not cov["extra"], f"unclassified on-disk keys (rule-6 violation): {cov['extra'][:6]}"
    from collections import Counter
    by_scheme = Counter(cov["keymap"].values())

    # 3. resident projection from REAL header shapes: int4 AND int6 mixes under the ceiling
    sizes = _tensor_sizes(d, wm)
    p4 = project_resident(sizes, cov["keymap"], expert_bits=4)
    p6 = project_resident(sizes, cov["keymap"], expert_bits=6)
    fits4, fits6 = p4["mix_gib"] <= CEILING_GIB, p6["mix_gib"] <= CEILING_GIB

    # 3b. header accounting cross-check: the per-tensor header sizes sum to ~the on-disk file bytes
    #     (only the small per-shard JSON header is unaccounted) — proves the shapes are read right.
    hdr_total = sum(numel * dtype_bytes(dt) for dt, numel in sizes.values()) / 2**30
    on_disk = sum(p.stat().st_size for p in d.glob("model-*.safetensors")) / 2**30
    acct_ok = abs(hdr_total - on_disk) / on_disk < 0.01
    vision_gib = sum(numel * dtype_bytes(dt) for k, (dt, numel) in sizes.items()
                     if k.startswith(VISION_PREFIX)) / 2**30

    print("\n=== Nex-N2-Pro N0 (config + eos + fit) ===")
    print(f"layers (linear/full = total)  : {cfg.num_hidden_layers - full}/{full} = "
          f"{cfg.num_hidden_layers}  | experts {cfg.num_experts} top-{cfg.num_experts_per_tok} "
          f"+ shared")
    print(f"MTP head                      : declared {declared_mtp}, weights ABSENT -> effective "
          f"{cfg.num_mtp_modules} (native-MTP spec-decode unavailable for Nex)")
    print(f"chat stop set                 : {cfg.eos_token_ids}  (im_end 248046, endoftext 248044)")
    print(f"quant policy coverage         : {len(cov['keymap'])} text tensors -> {dict(by_scheme)}"
          f"  (+{len(cov['vision'])} vision, excluded)")
    print(f"source (text bf16)            : {p4['bf16_gib']:.1f} GiB  | +vision {vision_gib:.1f} GiB "
          f"| on-disk {on_disk:.1f} GiB over 122 shards")
    print(f"resident int4-g64 mix         : {p4['mix_gib']:.1f} GiB  "
          f"(experts {p4['gib']['expert_int4']:.1f} + int8 {p4['gib']['int8']:.1f} + dense "
          f"{p4['gib']['dense']:.1f})  headroom {CEILING_GIB - p4['mix_gib']:.1f}")
    print(f"resident int6-g64 mix         : {p6['mix_gib']:.1f} GiB  "
          f"(experts {p6['gib']['expert_int4']:.1f})  headroom {CEILING_GIB - p6['mix_gib']:.1f}")
    print(f"fits <= {CEILING_GIB} GiB          : int4 {fits4} / int6 {fits6}")

    assert acct_ok, f"header bytes {hdr_total:.1f} GiB drifted from on-disk {on_disk:.1f} GiB (>1%)"
    assert fits4, f"int4 mix {p4['mix_gib']:.1f} GiB exceeds {CEILING_GIB} GiB"
    assert fits6, f"int6 mix {p6['mix_gib']:.1f} GiB exceeds {CEILING_GIB} GiB"
    assert p4["mix_gib"] < p6["mix_gib"] < p4["bf16_gib"], "mix ordering wrong (int4<int6<bf16)"

    print("PARITY-CHECKS: 11")
    print(f"PASS — Nex parses, eos {cfg.eos_token_ids}, policy covers all {len(cov['keymap'])} "
          f"tensors, resident int4 {p4['mix_gib']:.0f} / int6 {p6['mix_gib']:.0f} GiB < {CEILING_GIB}.")


if __name__ == "__main__":
    run()
