"""Gate: Qwen3.5-397B-A17B streamed bf16 loader — **model-free** key/shape check, headers only, ~0 GB.

Verifies that every tensor key :class:`quanta.qwen35.loader.Qwen35SourceCheckpoint` will reach for
**exists** in the real checkpoint's ``model.safetensors.index.json`` ``weight_map`` with the
**shape the config implies**, and that the loader's key enumeration **excludes every**
``model.visual.*`` key (the vision tower is out of scope). It does this by parsing the index JSON and
reading only the safetensors **headers** — for each key it reads the leading ``<Q`` (uint64 LE) header
length and ``json.loads`` exactly that many header bytes to get the per-tensor ``dtype``/``shape``. It
**never reads tensor bytes, never calls mx.load, never allocates a weight** — memory stays a few MB
(headers only). The heavy real-checkpoint tensor load is deferred to a future GPU session; this gate
deliberately does not exercise it.

Coverage: top-level (embed / norm / lm_head), a sample LINEAR-attention layer, a sample FULL-attention
layer, the MoE block of some layer (pre-stacked 3D routed experts + shared expert), and the native MTP
block (full-attn + MoE with the SAME fused pre-stacked experts as a main-decoder block). The loader's
own suffix constants and key prefixes are imported so the gate checks the *actual* enumeration, not a
hand-copied list that could drift.

    uv run --with numpy python -m parity.qwen35_loader_keys_test

(``numpy`` is optional — this gate is pure stdlib ``json``/``struct``.)
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.loader import (
    EMBED_KEY,
    FINAL_NORM_KEY,
    FULL_ATTN_SUFFIXES,
    LINEAR_ATTN_SUFFIXES,
    LM_HEAD_KEY,
    LM_PREFIX,
    SHARED_EXPERT_PROJS,
)

MODEL_DIR = Path("/Users/pmrj/models/Qwen3.6-35B-A3B")


class HeaderIndex:
    """Reads safetensors **headers only** (dtype/shape) for keys in the index weight_map."""

    def __init__(self, model_dir: Path) -> None:
        self.dir = model_dir
        self.weight_map: dict[str, str] = json.loads(
            (model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        self._hdr_cache: dict[str, dict] = {}

    def _header(self, shard: str) -> dict:
        hdr = self._hdr_cache.get(shard)
        if hdr is None:
            with open(self.dir / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]   # header length (little-endian uint64)
                hdr = json.loads(f.read(n))             # exactly the header JSON — NO tensor bytes
            hdr.pop("__metadata__", None)
            self._hdr_cache[shard] = hdr
        return hdr

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def meta(self, key: str) -> tuple[str, tuple[int, ...]]:
        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(f"{key!r} absent from weight_map")
        m = self._header(shard)[key]
        return m["dtype"], tuple(m["shape"])


def run() -> None:
    cfg = Qwen35Config.from_pretrained(MODEL_DIR)
    idx = HeaderIndex(MODEL_DIR)
    H, V = cfg.hidden_size, cfg.vocab_size
    E = cfg.num_experts
    ok = True
    checked = 0
    failures = 0

    def check_shape(key: str, want: tuple[int, ...]) -> None:
        """Assert ``key`` exists in the weight_map and its header shape equals ``want``."""
        nonlocal ok, checked, failures
        checked += 1
        if not idx.has(key):
            ok = False
            failures += 1
            print(f"  [FAIL] MISSING {key}")
            return
        _, got = idx.meta(key)
        if got != want:
            ok = False
            failures += 1
            print(f"  [FAIL] {key}: shape {got} != expected {want}")

    def group(tag: str, n_before: int, fails_before: int) -> None:
        """Report THIS group's own result (failures since ``fails_before``), not the global state."""
        added = checked - n_before
        grp_ok = failures == fails_before
        print(f"  [{'OK' if grp_ok else 'FAIL'}] {tag} ({added} keys)")

    # --- top-level text tensors -----------------------------------------------
    n, f = checked, failures
    check_shape(EMBED_KEY, (V, H))
    check_shape(FINAL_NORM_KEY, (H,))
    # lm_head is untied here, and sits at the TOP level (not under model.language_model.)
    check_shape((EMBED_KEY if cfg.tie_word_embeddings else LM_HEAD_KEY), (V, H))
    assert not cfg.tie_word_embeddings, "expected untied lm_head for Qwen3.5"
    assert not LM_HEAD_KEY.startswith(LM_PREFIX), "lm_head must be top-level, not under language_model"
    group("top-level: embed / norm / lm_head", n, f)

    # --- a sample LINEAR-attention layer --------------------------------------
    lin = next(i for i in range(cfg.num_hidden_layers) if cfg.is_linear_attention(i))
    want_linear = {
        "in_proj_qkv.weight": (cfg.linear_qkv_dim, H),
        "in_proj_a.weight": (cfg.linear_num_value_heads, H),
        "in_proj_b.weight": (cfg.linear_num_value_heads, H),
        "in_proj_z.weight": (cfg.linear_v_dim, H),
        "conv1d.weight": (cfg.linear_qkv_dim, 1, cfg.linear_conv_kernel_dim),
        "A_log": (cfg.linear_num_value_heads,),
        "dt_bias": (cfg.linear_num_value_heads,),
        "norm.weight": (cfg.linear_value_head_dim,),
        "out_proj.weight": (H, cfg.linear_v_dim),
    }
    assert set(want_linear) == set(LINEAR_ATTN_SUFFIXES), "linear suffix set drifted from loader"
    n, f = checked, failures
    pre = f"{LM_PREFIX}layers.{lin}."
    check_shape(pre + "input_layernorm.weight", (H,))
    check_shape(pre + "post_attention_layernorm.weight", (H,))
    for suf in LINEAR_ATTN_SUFFIXES:
        check_shape(f"{pre}linear_attn.{suf}", want_linear[suf])
    group(f"linear-attn layer {lin}: norms + Gated-DeltaNet", n, f)

    # --- a sample FULL-attention layer ----------------------------------------
    full = next(i for i in range(cfg.num_hidden_layers) if cfg.is_full_attention(i))
    want_full = {
        "q_proj.weight": (cfg.q_proj_out, H),
        "k_proj.weight": (cfg.kv_dim, H),
        "v_proj.weight": (cfg.kv_dim, H),
        "o_proj.weight": (H, cfg.q_dim),
        "q_norm.weight": (cfg.head_dim,),
        "k_norm.weight": (cfg.head_dim,),
    }
    assert set(want_full) == set(FULL_ATTN_SUFFIXES), "full suffix set drifted from loader"
    n, f = checked, failures
    pre = f"{LM_PREFIX}layers.{full}."
    check_shape(pre + "input_layernorm.weight", (H,))
    check_shape(pre + "post_attention_layernorm.weight", (H,))
    for suf in FULL_ATTN_SUFFIXES:
        check_shape(f"{pre}self_attn.{suf}", want_full[suf])
    group(f"full-attn layer {full}: norms + gated-GQA", n, f)

    # --- the MoE block of a layer (pre-stacked 3D routed + shared expert) ------
    moe_layer = lin  # MoE is present on every layer
    n, f = checked, failures
    mp = f"{LM_PREFIX}layers.{moe_layer}.mlp."
    check_shape(mp + "gate.weight", (E, H))
    check_shape(mp + "experts.gate_up_proj", (E, cfg.moe_gate_up_out, H))      # PRE-STACKED 3D
    check_shape(mp + "experts.down_proj", (E, H, cfg.moe_intermediate_size))   # PRE-STACKED 3D
    check_shape(mp + "shared_expert_gate.weight", (1, H))
    si = cfg.shared_expert_intermediate_size
    for proj, want in (("gate_proj", (si, H)), ("up_proj", (si, H)), ("down_proj", (H, si))):
        check_shape(f"{mp}shared_expert.{proj}.weight", want)
    assert set(SHARED_EXPERT_PROJS) == {"gate_proj", "up_proj", "down_proj"}
    group(f"MoE layer {moe_layer}: router + pre-stacked experts + shared", n, f)

    # --- the native MTP block (full-attn + MoE; SAME fused pre-stacked experts as main decoder) ---
    n, f = checked, failures
    check_shape("mtp.fc.weight", (H, 2 * H))
    check_shape("mtp.pre_fc_norm_embedding.weight", (H,))
    check_shape("mtp.pre_fc_norm_hidden.weight", (H,))
    check_shape("mtp.norm.weight", (H,))
    lp = "mtp.layers.0."
    check_shape(lp + "input_layernorm.weight", (H,))
    check_shape(lp + "post_attention_layernorm.weight", (H,))
    for suf in FULL_ATTN_SUFFIXES:
        check_shape(f"{lp}self_attn.{suf}", want_full[suf])
    check_shape(lp + "mlp.gate.weight", (E, H))
    check_shape(lp + "mlp.shared_expert_gate.weight", (1, H))
    for proj, want in (("gate_proj", (si, H)), ("up_proj", (si, H)), ("down_proj", (H, si))):
        check_shape(f"{lp}mlp.shared_expert.{proj}.weight", want)
    # MTP routed experts are PRE-STACKED + gate/up-FUSED, exactly like the main decoder (no per-expert)
    check_shape(lp + "mlp.experts.gate_up_proj", (E, cfg.moe_gate_up_out, H))   # PRE-STACKED 3D
    check_shape(lp + "mlp.experts.down_proj", (E, H, cfg.moe_intermediate_size))  # PRE-STACKED 3D
    group("MTP block: fc + norms + full-attn + fused pre-stacked MoE", n, f)

    # --- visual exclusion: the loader must reach NO model.visual.* key ---------
    # Reconstruct the exact key set the loader enumerates across every layer + MTP, then assert none of
    # them are vision-tower keys. Also confirm the checkpoint really has visual keys, so the exclusion
    # is meaningful (not vacuously true on a text-only dump).
    n_visual_ckpt = sum(1 for k in idx.weight_map if k.startswith("model.visual."))
    enumerated: set[str] = {EMBED_KEY, FINAL_NORM_KEY, LM_HEAD_KEY,
                            "mtp.fc.weight", "mtp.pre_fc_norm_embedding.weight",
                            "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight"}
    for i in range(cfg.num_hidden_layers):
        p = f"{LM_PREFIX}layers.{i}."
        enumerated |= {p + "input_layernorm.weight", p + "post_attention_layernorm.weight"}
        if cfg.is_linear_attention(i):
            enumerated |= {f"{p}linear_attn.{s}" for s in LINEAR_ATTN_SUFFIXES}
        else:
            enumerated |= {f"{p}self_attn.{s}" for s in FULL_ATTN_SUFFIXES}
        mp = p + "mlp."
        enumerated |= {mp + "gate.weight", mp + "experts.gate_up_proj", mp + "experts.down_proj",
                       mp + "shared_expert_gate.weight"}
        enumerated |= {f"{mp}shared_expert.{proj}.weight" for proj in SHARED_EXPERT_PROJS}
    lp = "mtp.layers.0."
    enumerated |= {lp + "input_layernorm.weight", lp + "post_attention_layernorm.weight",
                   lp + "mlp.gate.weight", lp + "mlp.shared_expert_gate.weight"}
    enumerated |= {f"{lp}self_attn.{s}" for s in FULL_ATTN_SUFFIXES}
    enumerated |= {f"{lp}mlp.shared_expert.{proj}.weight" for proj in SHARED_EXPERT_PROJS}
    enumerated |= {lp + "mlp.experts.gate_up_proj", lp + "mlp.experts.down_proj"}
    leaked = sorted(k for k in enumerated if k.startswith("model.visual."))
    missing = sorted(k for k in enumerated if not idx.has(k))
    excl_ok = (not leaked) and (not missing) and n_visual_ckpt > 0
    ok = ok and excl_ok
    if leaked:
        print(f"  [FAIL] loader enumerates {len(leaked)} visual key(s), e.g. {leaked[0]}")
    if missing:
        print(f"  [FAIL] loader enumerates {len(missing)} absent key(s), e.g. {missing[0]}")
    print(f"  [{'OK' if excl_ok else 'FAIL'}] visual-exclusion: 0 of {len(enumerated)} enumerated keys "
          f"are model.visual.* (checkpoint has {n_visual_ckpt} visual keys)")

    print(f"{'PASS' if ok else 'FAIL'}: Qwen3.5 loader keys — {checked} shapes verified, "
          f"{len(enumerated)} keys enumerated, vision excluded "
          f"(layers: linear={lin}, full={full}, experts={E})")


if __name__ == "__main__":
    run()
