"""MiniMax-M3-VL enablement gate (M0): config parses (nested text+vision) + correct eos + MTP
presence-refine + the quant policy covers EVERY source tensor (rule 6) + the int6/int4 quant mix is
RAM-resident under the 490.4 GiB ceiling — before any bake.

MiniMax-M3-VL (``minimax_m3_vl``) is a DIFFERENT architecture from the in-tree M2.7 module: 60L
(3 dense + 57 MoE), GQA 64q/4kv, 128 experts top-4 + 1 shared, sigmoid noaux_tc routing, clamped
SwiGLU-OAI, a native TRAINED block-sparse attention indexer, and a CLIP-ViT vision tower (full-VL
build). This gate proves, against the REAL 59-shard checkpoint (``config`` + the index + the
per-shard safetensors HEADERS only — **no tensor is materialized**, sub-second):

* :meth:`MiniMaxM3Config.from_pretrained` parses the nested config, derives eos ``(200020,)``, and
  refines ``num_mtp_modules 7 -> 0`` (M3 declares 7 MTP modules but ships ZERO ``mtp.*`` weights —
  exactly the Nex case ⇒ native-MTP spec-decode is N/A);
* the quant policy classifies EVERY source tensor — text keys EXACTLY match the on-disk index
  (rule 6, the loader key contract at 397B scale) and the whole vision tower is covered (dense);
* the milestone: the **int4-g64** expert mix (int8 dense/attn/shared + bf16 control/indexer/vision)
  is resident under 490.4 GiB (int6 projected too, the retired arm); a header-vs-file cross-check
  validates the byte accounting.

Real-weight-PATH gate (reads ``~/models/MiniMax-M3``) ⇒ EXCLUDED from the model-free sweep, run
SOLO / by hand — like ``parity/nex_n2_pro_fit_test.py``. It loads no weights (headers only).

    uv run python -m parity.minimax_m3_fit_test
"""

from __future__ import annotations

import json
import math
import struct
from collections import Counter
from pathlib import Path

from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.quant_policy_m3 import (
    coverage,
    dtype_bytes,
    is_vision,
    project_resident,
)

MINIMAX = "/Users/pmrj/models/MiniMax-M3"
CEILING_GIB = 490.4  # M3 Ultra recommended max working set (CLAUDE.md)


def _read_header(path: Path) -> dict:
    """The safetensors header (tensor → {dtype, shape, data_offsets}); reads ~one block per shard,
    materializes no tensor."""
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
    d = Path(MINIMAX)
    cfg = MiniMaxM3Config.from_pretrained(d)

    # 1. config correctness on the real checkpoint
    dims_ok = (cfg.num_hidden_layers == 60 and cfg.num_local_experts == 128
               and cfg.num_experts_per_tok == 4 and cfg.head_dim == 128
               and cfg.num_attention_heads == 64 and cfg.num_key_value_heads == 4
               and cfg.partial_rotary_factor == 0.5 and cfg.has_shared_expert
               and cfg.use_gemma_norm and cfg.hidden_act == "swigluoai"
               and cfg.vocab_size == 200064)
    n_dense = sum(1 for i in range(cfg.num_hidden_layers) if cfg.is_dense_layer(i))
    n_moe = sum(1 for i in range(cfg.num_hidden_layers) if cfg.is_moe_layer(i))
    n_sparse = sum(1 for i in range(cfg.num_hidden_layers) if cfg.is_sparse_attention_layer(i))
    assert dims_ok, "M3 key dims mismatch"
    assert (n_dense, n_moe) == (3, 57), f"layer split wrong: {n_dense} dense / {n_moe} moe"
    assert n_sparse == 57, f"expected 57 sparse-attention layers, got {n_sparse}"
    assert cfg.eos_token_ids == (200020,), f"wrong eos stop set {cfg.eos_token_ids}"
    assert cfg.num_mtp_modules_declared == 7 and cfg.num_mtp_modules == 0, \
        f"MTP presence-refine wrong: declared {cfg.num_mtp_modules_declared}, eff {cfg.num_mtp_modules}"
    assert cfg.max_position_embeddings == 1048576, "M3 declares the 1M native window"
    assert cfg.vision is not None and cfg.vision.num_hidden_layers == 32, "vision tower must parse"

    # 2. quant-policy coverage vs the REAL index (rule 6 — text exact + vision covered)
    wm = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    cov = coverage(list(wm), cfg)
    assert not cov["missing"], f"policy expects keys absent on disk (drift): {cov['missing'][:6]}"
    assert not cov["extra"], f"unclassified on-disk keys (rule-6 violation): {cov['extra'][:6]}"
    assert len(cov["keymap"]) == len(wm), \
        f"keymap {len(cov['keymap'])} != index {len(wm)} (every tensor must be classified)"
    by_scheme = Counter(cov["keymap"].values())

    # 3. resident projection from REAL header shapes: int4 (ship) AND int6 (retired) under ceiling
    sizes = _tensor_sizes(d, wm)
    p6 = project_resident(sizes, cov["keymap"], expert_bits=6)
    p4 = project_resident(sizes, cov["keymap"], expert_bits=4)
    fits6, fits4 = p6["mix_gib"] <= CEILING_GIB, p4["mix_gib"] <= CEILING_GIB

    # 3b. header accounting cross-check: per-tensor header sizes sum to ~the on-disk file bytes.
    hdr_total = sum(numel * dtype_bytes(dt) for dt, numel in sizes.values()) / 2**30
    on_disk = sum(p.stat().st_size for p in d.glob("model-*.safetensors")) / 2**30
    acct_ok = abs(hdr_total - on_disk) / on_disk < 0.01
    vision_gib = sum(numel * dtype_bytes(dt) for k, (dt, numel) in sizes.items()
                     if is_vision(k)) / 2**30

    print("\n=== MiniMax-M3-VL M0 (config + eos + MTP refine + fit) ===")
    print(f"layers (dense/moe = total)    : {n_dense}/{n_moe} = {cfg.num_hidden_layers}  | "
          f"experts {cfg.num_local_experts} top-{cfg.num_experts_per_tok} + {cfg.n_shared_experts} "
          f"shared  | sparse-attn layers {n_sparse}")
    print(f"attention                     : GQA {cfg.num_attention_heads}q/{cfg.num_key_value_heads}"
          f"kv head_dim {cfg.head_dim} partial-RoPE {cfg.partial_rotary_factor} theta {cfg.rope_theta:g}"
          f"  gemma_norm {cfg.use_gemma_norm}  act {cfg.hidden_act}")
    print(f"MTP head                      : declared {cfg.num_mtp_modules_declared}, weights ABSENT "
          f"-> effective {cfg.num_mtp_modules} (native-MTP spec-decode N/A)")
    print(f"eos stop set / bos            : {cfg.eos_token_ids} / {cfg.bos_token_id}  | "
          f"native ctx {cfg.max_position_embeddings}")
    print(f"vision tower                  : ViT {cfg.vision.num_hidden_layers}L hidden "
          f"{cfg.vision.hidden_size} patch {cfg.vision.patch_size}  ({vision_gib:.1f} GiB bf16)")
    print(f"quant policy coverage         : {len(cov['keymap'])} tensors -> {dict(by_scheme)}  "
          f"(incl. {len(cov['vision'])} vision dense)")
    print(f"source (mapped bf16)          : {p6['bf16_gib']:.1f} GiB  | on-disk {on_disk:.1f} GiB "
          f"over 59 shards")
    print(f"resident int4-g64 mix (SHIP)  : {p4['mix_gib']:.1f} GiB  (experts "
          f"{p4['gib']['expert_int']:.1f} + int8 {p4['gib']['int8']:.1f} + dense "
          f"{p4['gib']['dense']:.1f})  headroom {CEILING_GIB - p4['mix_gib']:.1f}")
    print(f"resident int6-g64 mix (retired): {p6['mix_gib']:.1f} GiB  (experts "
          f"{p6['gib']['expert_int']:.1f})  headroom {CEILING_GIB - p6['mix_gib']:.1f}")
    print(f"fits <= {CEILING_GIB} GiB          : int4 {fits4} / int6 {fits6}")

    assert acct_ok, f"header bytes {hdr_total:.1f} GiB drifted from on-disk {on_disk:.1f} GiB (>1%)"
    assert fits6, f"int6 mix {p6['mix_gib']:.1f} GiB exceeds {CEILING_GIB} GiB"
    assert fits4, f"int4 mix {p4['mix_gib']:.1f} GiB exceeds {CEILING_GIB} GiB"
    assert p4["mix_gib"] < p6["mix_gib"] < p6["bf16_gib"], "mix ordering wrong (int4<int6<bf16)"

    print("PARITY-CHECKS: 13")
    print(f"PASS — M3-VL parses, eos {cfg.eos_token_ids}, MTP {cfg.num_mtp_modules}, policy covers "
          f"all {len(cov['keymap'])} tensors, resident int4 {p4['mix_gib']:.0f} GiB < {CEILING_GIB}.")


if __name__ == "__main__":
    run()
