"""Key→quant-scheme map + resident-storage projection for a Qwen3.5 (``qwen3_5_moe``) checkpoint.

The bake (:mod:`quanta.qwen35.bake`) classifies tensors per loader sub-dict; this module expresses the
**same** policy over the on-disk safetensors INDEX KEYS, so the enablement (N0) gate can prove rule-6
coverage (every source tensor maps to a scheme — no silent default) and project the int4/int6 resident
footprint **before** any multi-hour bake. The per-suffix int8/bf16 partition and the loader's key
enumeration are IMPORTED from the bake/loader, so this can never drift from what the bake writes.

Schemes:

* ``expert_int4``  — the pre-stacked routed-expert stacks (``mlp.experts.gate_up_proj`` / ``down_proj``)
  → int4 affine g64 (the dominant footprint; ``expert_bits`` lets the projection model int6 too).
* ``int8``         — matmul projections (gated-GQA q/k/v/o, Gated-DeltaNet in/out projections) + the
  shared expert → int8 affine.
* ``dense``        — bf16/f32 control kept verbatim: every RMSNorm, the SSM control (``A_log`` /
  ``dt_bias`` / ``conv1d`` / DeltaNet ``norm``), the router ``gate``, the ``shared_expert_gate``
  sigmoid, the ``embed_tokens`` / ``lm_head`` token tables, and the MTP ``fc`` fusion + pre-norms.

``model.visual.*`` (the ViT) is **not** baked by the language-model-only loader and is excluded here.
"""

from __future__ import annotations

from quanta.qwen35.bake import _FULL_BF16, _FULL_INT8, _LINEAR_BF16, _LINEAR_INT8
from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.loader import (
    EMBED_KEY,
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    LM_PREFIX,
    SHARED_EXPERT_PROJS,
)

VISION_PREFIX = "model.visual."
DENSE, INT8, EXPERT_INT4 = "dense", "int8", "expert_int4"

# safetensors dtype string → bytes/element (only the dtypes a Qwen3.5 bf16 checkpoint uses).
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I16": 2, "I32": 4}


def dtype_bytes(dtype: str) -> int:
    """Bytes per element for a safetensors dtype string. Fail loud on an unknown dtype (rule 6 — a
    silent 0 would under-count the footprint)."""
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unknown safetensors dtype {dtype!r} (cannot size tensor)")
    return _DTYPE_BYTES[dtype]


def affine_bpp(bits: int, group_size: int, scale_bytes: int = 2) -> float:
    """Effective bits-per-param of an affine-packed weight: ``bits`` for the codes + one ``scale`` and
    one ``bias`` (``scale_bytes`` each) per ``group_size`` elements. int4 g64 bf16-scale = 4.5 bpp;
    int6 = 6.5; int8 = 8.5."""
    return bits + (2 * scale_bytes * 8) / group_size


def _attn_keys(prefix: str, int8_suffixes: tuple[str, ...],
               bf16_suffixes: tuple[str, ...]) -> dict[str, str]:
    km = {prefix + s: INT8 for s in int8_suffixes}
    km.update({prefix + s: DENSE for s in bf16_suffixes})
    return km


def _moe_keys(prefix: str) -> dict[str, str]:
    """``{key: scheme}`` for one MoE block at ``prefix`` (e.g. ``...layers.3.mlp.``): router + shared
    gate bf16, shared expert int8, the two pre-stacked routed-expert stacks int4."""
    km = {prefix + "gate.weight": DENSE, prefix + "shared_expert_gate.weight": DENSE}
    km.update({prefix + f"shared_expert.{proj}.weight": INT8 for proj in SHARED_EXPERT_PROJS})
    km[prefix + "experts.gate_up_proj"] = EXPERT_INT4
    km[prefix + "experts.down_proj"] = EXPERT_INT4
    return km


def _mtp_source_keys(cfg: Qwen35Config) -> dict[str, str]:
    """Source keys of the native MTP block (``mtp.fc`` + pre-norms + ``mtp.layers.0.*`` full-attn +
    MoE), mirroring :meth:`quanta.qwen35.loader.Qwen35SourceCheckpoint.mtp` (single head, j=0)."""
    if cfg.num_mtp_modules <= 0:
        return {}
    km = {
        "mtp.fc.weight": DENSE,
        "mtp.pre_fc_norm_embedding.weight": DENSE,
        "mtp.pre_fc_norm_hidden.weight": DENSE,
        "mtp.norm.weight": DENSE,
    }
    lp = "mtp.layers.0."
    km[lp + "input_layernorm.weight"] = DENSE
    km[lp + "post_attention_layernorm.weight"] = DENSE
    km.update(_attn_keys(lp + "self_attn.", _FULL_INT8, _FULL_BF16))
    km.update(_moe_keys(lp + "mlp."))
    return km


def expected_keymap(cfg: Qwen35Config) -> dict[str, str]:
    """The complete ``{source_key: scheme}`` map the bake writes, reconstructed from the loader's key
    enumeration + the bake's int8/bf16 partition + the config's per-layer schedule. The N0 gate asserts
    this equals the (non-vision) keys in the real ``model.safetensors.index.json`` — proving rule-6
    coverage AND that Nex-N2-Pro's key contract matches the loader (no model-specific drift)."""
    km: dict[str, str] = {EMBED_KEY: DENSE, FINAL_NORM_KEY: DENSE}
    if not cfg.tie_word_embeddings:
        km[LM_HEAD_KEY] = DENSE
    for i in range(cfg.num_hidden_layers):
        lp = f"{LM_PREFIX}layers.{i}."
        km[lp + "input_layernorm.weight"] = DENSE
        km[lp + "post_attention_layernorm.weight"] = DENSE
        if cfg.is_linear_attention(i):
            km.update(_attn_keys(lp + "linear_attn.", _LINEAR_INT8, _LINEAR_BF16))
        else:
            km.update(_attn_keys(lp + "self_attn.", _FULL_INT8, _FULL_BF16))
        km.update(_moe_keys(lp + "mlp."))
    km.update(_mtp_source_keys(cfg))
    return km


def coverage(index_keys: list[str], cfg: Qwen35Config) -> dict:
    """Partition ``index_keys`` into ``text`` vs ``vision`` and check the text keys EXACTLY equal
    :func:`expected_keymap` (rule 6: no source tensor without a policy, no policy for a key that does
    not exist). Returns ``{keymap, vision, missing, extra}``; ``missing``/``extra`` empty ⇔ covered."""
    text = [k for k in index_keys if not k.startswith(VISION_PREFIX)]
    vision = [k for k in index_keys if k.startswith(VISION_PREFIX)]
    km = expected_keymap(cfg)
    expected, actual = set(km), set(text)
    return {
        "keymap": km,
        "vision": vision,
        "missing": sorted(expected - actual),   # expected by the bake but absent on disk
        "extra": sorted(actual - expected),     # on disk but unclassified (rule-6 violation)
    }


def project_resident(tensor_sizes: dict[str, tuple[str, int]], keymap: dict[str, str], *,
                     expert_bits: int, group_size: int = 64, scale_bytes: int = 2) -> dict:
    """Project the RAM-resident footprint of the baked mix from real tensor sizes.

    ``tensor_sizes``: ``{key: (dtype_str, numel)}`` (read from the safetensors headers — exact shapes,
    no tensor materialized). ``keymap``: ``{key: scheme}``. Experts → ``expert_bits`` affine, int8 →
    int8 affine, dense → kept verbatim in the source dtype. Returns per-scheme + total GiB for the mix
    and the all-bf16 baseline (the source size of the mapped tensors)."""
    e_bpp, i_bpp = affine_bpp(expert_bits, group_size, scale_bytes), affine_bpp(8, group_size,
                                                                                scale_bytes)
    by = {DENSE: 0.0, INT8: 0.0, EXPERT_INT4: 0.0}
    bf16 = 0.0
    for key, (dtype, numel) in tensor_sizes.items():
        scheme = keymap.get(key)
        if scheme is None:  # vision / unmapped — coverage() owns the rule-6 assertion; skip here
            continue
        src = numel * dtype_bytes(dtype)
        bf16 += src
        if scheme == EXPERT_INT4:
            by[scheme] += numel * e_bpp / 8
        elif scheme == INT8:
            by[scheme] += numel * i_bpp / 8
        else:
            by[scheme] += src  # dense kept verbatim
    g = float(2**30)
    return {
        "gib": {k: v / g for k, v in by.items()},
        "mix_gib": sum(by.values()) / g,
        "bf16_gib": bf16 / g,
        "expert_bits": expert_bits,
    }
