"""Validate the Nemotron-H scaffold: config, tokenizer (encode), and quant policy.

Checks the hybrid layer split (40 mamba / 40 moe / 8 attention), key dims, the HF-tokenizer
encode round-trip + two-eos stop set {2, 11}, a faithful chat-template render, and that the
quant policy classifies EVERY tensor in the real weight map (no silent default) — plus the
analytic size of the int4/int8/bf16 mix vs bf16.

    uv run --with tokenizers python -m parity.nemotron_scaffold_test
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.quant_policy import bake_plan, classify, estimate_storage

MODEL = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"


def run() -> None:
    cfg = NemotronHConfig.from_pretrained(MODEL)
    layers_ok = (cfg.num_hidden_layers == 88 and cfg.count("mamba") == 40
                 and cfg.count("moe") == 40 and cfg.count("attention") == 8)
    dims_ok = (cfg.hidden_size == 4096 and cfg.n_routed_experts == 512
               and cfg.num_experts_per_tok == 22 and cfg.mamba_d_inner == 8192
               and cfg.moe_latent_size == 1024 and cfg.mamba_in_proj_dim == 18560)

    # quant policy: 100% coverage over the real tensor list (rule #6: no unmapped tensor)
    wm = json.loads((Path(MODEL) / "model.safetensors.index.json").read_text())["weight_map"]
    plan = bake_plan(wm.keys())  # raises if any tensor is unmapped
    by_scheme = Counter(s.kind for s in plan.values())
    spot_ok = (
        classify("backbone.layers.1.mixer.experts.0.up_proj.weight").kind == "int4_gptq"
        and classify("backbone.layers.1.mixer.shared_experts.down_proj.weight").kind == "int8_affine"
        and classify("backbone.layers.0.mixer.in_proj.weight").kind == "int8_affine"
        and classify("backbone.layers.7.mixer.q_proj.weight").kind == "int8_affine"
        and classify("backbone.layers.1.mixer.fc1_latent_proj.weight").kind == "int8_affine"
        and classify("backbone.layers.0.mixer.A_log").kind == "bf16"
        and classify("backbone.layers.0.mixer.conv1d.weight").kind == "bf16"
        and classify("backbone.layers.1.mixer.gate.weight").kind == "bf16"
        and classify("backbone.layers.0.norm.weight").kind == "bf16"
        and classify("lm_head.weight").kind == "bf16"
    )
    est = estimate_storage(cfg)
    size_ok = est["total_gib_mix"] < est["total_gib_bf16"] and est["total_gib_mix"] < 100

    # tokenizer (encode) — imported here so the config/quant checks don't require `tokenizers`
    from quanta.nemotron.tokenizer import NemotronTokenizer
    tok = NemotronTokenizer(MODEL)
    prose = "The hybrid model interleaves Mamba-2 and attention layers."
    ids = tok.encode(prose)
    rt = tok.decode(ids)
    tok_ok = (tok.bos_id == 1 and tok.eos_id == 11 and tok.stop_ids == {2, 11}
              and len(ids) > 0 and rt.strip() == prose.strip())
    chat = tok.apply_chat_template([{"role": "user", "content": "hi"}])
    chat_ids = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=True)
    chat_ok = ("<|im_start|>" in chat and chat.rstrip().endswith("<think>")
               and 11 in chat_ids and 1 not in chat_ids)

    print("\n=== Nemotron-H scaffold ===")
    print(f"layers 40 mamba / 40 moe / 8 attn (=88) : {layers_ok}")
    print(f"dims (hidden/experts/d_inner/latent)    : {dims_ok}")
    print(f"quant policy coverage                   : {len(plan)} tensors -> {dict(by_scheme)}")
    print(f"quant spot-checks                       : {spot_ok}")
    print(f"size mix vs bf16 (backbone)             : {est['total_gib_mix']:.1f} GiB vs "
          f"{est['total_gib_bf16']:.1f} GiB  {dict((k, round(v, 1)) for k, v in est['gib'].items())}")
    print(f"tokenizer encode + stop set             : {tok_ok}  "
          f"bos={tok.bos_id} eos={tok.eos_id} stop={sorted(tok.stop_ids)}")
    print(f"chat template + tokenized specials      : {chat_ok}")
    if not tok_ok:
        print(f"  prose={prose!r} roundtrip={rt!r}")
    if not chat_ok:
        print(f"  chat={chat!r}")
    assert all([layers_ok, dims_ok, spot_ok, size_ok, tok_ok, chat_ok])
    print("Nemotron-H scaffold OK (config + encode + quant policy)")


if __name__ == "__main__":
    run()
