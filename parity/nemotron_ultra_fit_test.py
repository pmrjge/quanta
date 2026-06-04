"""Nemotron-3-Ultra-550B enablement gate (U0): config parses (new schema) + the quant mix fits.

The 550B-Ultra checkpoint ships the *newer* Nemotron-H config schema — an explicit
``layers_block_type`` list instead of the compact ``hybrid_override_pattern`` letter string the
already-baked 120B-Super sibling uses, and it omits ``num_hidden_layers`` entirely. This gate proves
:meth:`NemotronHConfig.from_pretrained` now normalises BOTH schemas to the same ``(pattern,
n_layers)`` the runtime consumes, that the derived layer split reproduces the explicit list
bit-for-bit, that the int4/int8/bf16 quant policy classifies EVERY Ultra tensor (rule #6 — no silent
default), and — the milestone number — that the int4-GPTQ-experts + int8-dense + bf16-core mix is
**RAM-resident under the 490.4 GiB ceiling** before any multi-hour bake is committed. Expected split:
48 mamba / 48 moe / 12 attention (= 108 layers); hidden 8192, 512 experts top-22, latent 2048.

No weights are loaded (config + weight-map index + analytic sizing only) — sub-second. The Super
scaffold (:mod:`parity.nemotron_scaffold_test`) covers the old schema unchanged; this also re-parses
Super to prove the shared adapter stays backward-compatible.

    uv run python -m parity.nemotron_ultra_fit_test
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.quant_policy import bake_plan, estimate_storage

ULTRA = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
SUPER = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
CEILING_GIB = 490.4  # M3 Ultra recommended max working set (CLAUDE.md)


def run() -> None:
    # 1. Ultra parses through the new adapter (KeyErrored on num_hidden_layers/pattern before U0)
    cfg = NemotronHConfig.from_pretrained(ULTRA)
    raw = json.loads((Path(ULTRA) / "config.json").read_text())

    # 2. parser faithfulness: the derived split reproduces the explicit list bit-for-bit, and Ultra
    #    genuinely lacks the old-schema keys (so the NEW branch is what's exercised)
    explicit = list(raw["layers_block_type"])
    derived = cfg.layers_block_type
    assert derived == explicit, "derived layer split != config.json layers_block_type"
    assert cfg.num_hidden_layers == len(explicit), "num_hidden_layers != len(layers_block_type)"
    assert "hybrid_override_pattern" not in raw and "num_hidden_layers" not in raw, \
        "Ultra config carries old-schema keys — the new branch isn't being exercised"
    split = (cfg.count("mamba"), cfg.count("moe"), cfg.count("attention"))
    assert split == (explicit.count("mamba"), explicit.count("moe"), explicit.count("attention"))
    assert sum(split) == cfg.num_hidden_layers

    # 3. key dims at the new 2x-wide scale (single scalar keys — read straight from config.json)
    dims_ok = (cfg.hidden_size == 8192 and cfg.n_routed_experts == 512
               and cfg.num_experts_per_tok == 22 and cfg.moe_latent_size == 2048
               and cfg.moe_intermediate_size == 5120 and cfg.num_attention_heads == 64
               and cfg.num_key_value_heads == 2 and cfg.mamba_num_heads == 256
               and cfg.max_position_embeddings == 262144 and cfg.vocab_size == 131072)
    assert dims_ok, "Ultra key dims mismatch"

    # 4. quant policy: 100% coverage over the REAL Ultra weight map (rule #6 — no unmapped tensor)
    wm = json.loads((Path(ULTRA) / "model.safetensors.index.json").read_text())["weight_map"]
    plan = bake_plan(wm.keys())  # raises ValueError on any tensor with no policy
    by_scheme = Counter(s.kind for s in plan.values())

    # 5. THE milestone: the int4-GPTQ + int8 + bf16 mix is resident under the 490.4 GiB ceiling
    est = estimate_storage(cfg)
    mix, bf16 = est["total_gib_mix"], est["total_gib_bf16"]
    fits = mix <= CEILING_GIB

    # 6. backward-compat: the Super (old-schema) config still parses through the shared adapter
    sup = NemotronHConfig.from_pretrained(SUPER)
    super_ok = (sup.num_hidden_layers == 88 and sup.count("attention") == 8
                and sup.hidden_size == 4096)

    print("\n=== Nemotron-3-Ultra-550B U0 (config + fit) ===")
    print(f"layers mamba/moe/attn = total : {split} = {sum(split)}")
    print(f"derived split == explicit     : {derived == explicit}")
    print(f"key dims (8192/512e/top-22)    : {dims_ok}")
    print(f"quant policy coverage         : {len(plan)} tensors -> {dict(by_scheme)}")
    print(f"resident mix                  : {mix:.1f} GiB  (bf16 {bf16:.1f} GiB, {bf16 / mix:.1f}x)")
    print(f"  by scheme (GiB)             : {dict((k, round(v, 1)) for k, v in est['gib'].items())}")
    print(f"fits <= {CEILING_GIB} GiB           : {fits}  (headroom {CEILING_GIB - mix:.1f} GiB)")
    print(f"Super old-schema parses       : {super_ok}")
    print("  note: backbone-only; +~2.8B MTP head ~1-2 GiB more (still well under).")

    assert fits, f"Ultra int4 mix {mix:.1f} GiB exceeds the {CEILING_GIB} GiB ceiling"
    assert mix < bf16 and super_ok
    print(f"PASS — Ultra parses (new schema), policy covers all {len(plan)} tensors, "
          f"resident {mix:.1f} GiB under {CEILING_GIB} GiB.")


if __name__ == "__main__":
    run()
