"""Key→quant-scheme map + resident-storage projection for a MiniMax-M3-VL checkpoint.

Expresses the bake's quantization policy over the on-disk safetensors INDEX KEYS so the M0
enablement gate can prove rule-6 coverage (every source tensor maps to a scheme — no silent
default) and project the int6/int4 resident footprint **before** any multi-hour bake.

This module is **self-contained** for M0: the key schema is enumerated here directly (M3 ships
per-expert tensors, not pre-stacked stacks, so there is no loader/bake to import a suffix
partition from yet). Once :mod:`quanta.minimax.loader_m3`/``bake_m3`` exist (M1/M2) the partition
should be imported from them so this can never drift — for now the M0 fit-test gates this map
against the REAL index, which is the equivalent guarantee.

Schemes:

* ``expert_int``  — the routed-expert FFN weights (``block_sparse_moe.experts.<e>.{w1,w2,w3}``)
  → affine ``expert_bits`` g64 (the dominant footprint; the user ships **int6**, ``expert_bits=6``).
* ``int8``        — matmul projections (GQA q/k/v/o), the dense-FFN ``mlp.{gate,up,down}_proj``
  (layers 0–2), and the shared expert (``block_sparse_moe.shared_experts.*``) → int8 affine.
* ``dense``       — kept verbatim (bf16/f32): every RMSNorm (incl. per-head q/k norm + the
  Gemma ``(1+w)`` norms), the router ``gate`` + ``e_score_correction_bias`` (fp32), the **trained
  sparse-attention indexer** (``index_{q,k}_proj`` + ``index_{q,k}_norm`` — kept full precision to
  protect block selection), and the ``embed_tokens`` / ``lm_head`` token tables.

**Vision** (full-VL build): the whole ViT (``vision_tower.*``), ``multi_modal_projector.*`` and
``patch_merge_mlp.*`` are classified ``dense`` (bf16 verbatim) — small and precision-sensitive;
a vision-specific quant scheme can be revisited once the VL track is gated.
"""

from __future__ import annotations

DENSE, INT8, EXPERT_INT = "dense", "int8", "expert_int"

LM_PREFIX = "language_model.model."
EMBED_KEY = LM_PREFIX + "embed_tokens.weight"
FINAL_NORM_KEY = LM_PREFIX + "norm.weight"
LM_HEAD_KEY = "language_model.lm_head.weight"
VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")

# safetensors dtype string → bytes/element.
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I16": 2, "I32": 4}


def dtype_bytes(dtype: str) -> int:
    """Bytes per element for a safetensors dtype string. Fail loud on an unknown dtype (rule 6)."""
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unknown safetensors dtype {dtype!r} (cannot size tensor)")
    return _DTYPE_BYTES[dtype]


def affine_bpp(bits: int, group_size: int, scale_bytes: int = 2) -> float:
    """Effective bits-per-param of an affine-packed weight: ``bits`` for the codes + one ``scale``
    and one ``bias`` (``scale_bytes`` each) per ``group_size`` elements. int4 g64 = 4.5 bpp; int6 =
    6.5; int8 = 8.5."""
    return bits + (2 * scale_bytes * 8) / group_size


def is_vision(key: str) -> bool:
    return key.startswith(VISION_PREFIXES)


def expected_keymap(cfg) -> dict[str, str]:
    """The complete ``{text_source_key: scheme}`` map the bake writes, enumerated from the config's
    per-layer schedule. The M0 gate asserts this EXACTLY equals the non-vision keys in the real
    ``model.safetensors.index.json`` (rule-6 coverage at 397B-class scale)."""
    km: dict[str, str] = {EMBED_KEY: DENSE, FINAL_NORM_KEY: DENSE}
    if not cfg.tie_word_embeddings:
        km[LM_HEAD_KEY] = DENSE
    for i in range(cfg.num_hidden_layers):
        lp = f"{LM_PREFIX}layers.{i}."
        km[lp + "input_layernorm.weight"] = DENSE
        km[lp + "post_attention_layernorm.weight"] = DENSE
        # GQA attention (every layer)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            km[lp + f"self_attn.{proj}.weight"] = INT8
        km[lp + "self_attn.q_norm.weight"] = DENSE
        km[lp + "self_attn.k_norm.weight"] = DENSE
        # trained block-sparse indexer (sparse layers only) — kept bf16
        if cfg.is_sparse_attention_layer(i):
            for t in ("index_q_proj.weight", "index_k_proj.weight",
                      "index_q_norm.weight", "index_k_norm.weight"):
                km[lp + f"self_attn.{t}"] = DENSE
        # FFN: MoE (3–59) vs dense (0–2)
        if cfg.is_moe_layer(i):
            mp = lp + "block_sparse_moe."
            km[mp + "gate.weight"] = DENSE
            km[mp + "e_score_correction_bias"] = DENSE
            for proj in ("gate_proj", "up_proj", "down_proj"):
                km[mp + f"shared_experts.{proj}.weight"] = INT8
            for e in range(cfg.num_local_experts):
                for w in ("w1", "w2", "w3"):
                    km[mp + f"experts.{e}.{w}.weight"] = EXPERT_INT
        else:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                km[lp + f"mlp.{proj}.weight"] = INT8
    return km


def coverage(index_keys, cfg) -> dict:
    """Classify every on-disk key. ``vision_*`` → ``dense``; text keys must EXACTLY equal
    :func:`expected_keymap` (rule 6: no source tensor without a policy, no policy for an absent
    key). Returns ``{keymap, vision, missing, extra}``; ``missing``/``extra`` empty ⇔ covered."""
    km = expected_keymap(cfg)
    classified: dict[str, str] = {}
    vision: list[str] = []
    extra: list[str] = []
    missing = set(km)
    for k in index_keys:
        if is_vision(k):
            classified[k] = DENSE
            vision.append(k)
        elif k in km:
            classified[k] = km[k]
            missing.discard(k)
        else:
            extra.append(k)
    return {
        "keymap": classified,        # every recognized on-disk key → scheme (text + vision)
        "vision": vision,
        "missing": sorted(missing),  # expected by the bake but absent on disk
        "extra": sorted(extra),      # on disk but unclassified (rule-6 violation)
    }


def project_resident(tensor_sizes: dict[str, tuple[str, int]], keymap: dict[str, str], *,
                     expert_bits: int, group_size: int = 64, scale_bytes: int = 2) -> dict:
    """Project the RAM-resident footprint of the baked mix from real tensor sizes.

    ``tensor_sizes``: ``{key: (dtype_str, numel)}`` (read from the safetensors headers — exact
    shapes, no tensor materialized). ``keymap``: ``{key: scheme}`` from :func:`coverage`. Experts →
    ``expert_bits`` affine, int8 → int8 affine, dense → kept verbatim in the source dtype. Returns
    per-scheme + total GiB for the mix and the all-bf16 baseline of the mapped tensors."""
    e_bpp = affine_bpp(expert_bits, group_size, scale_bytes)
    i_bpp = affine_bpp(8, group_size, scale_bytes)
    by = {DENSE: 0.0, INT8: 0.0, EXPERT_INT: 0.0}
    src_total = 0.0
    for key, (dtype, numel) in tensor_sizes.items():
        scheme = keymap.get(key)
        if scheme is None:  # unclassified — coverage() owns the rule-6 assertion; skip here
            continue
        src = numel * dtype_bytes(dtype)
        src_total += src
        if scheme == EXPERT_INT:
            by[scheme] += numel * e_bpp / 8
        elif scheme == INT8:
            by[scheme] += numel * i_bpp / 8
        else:
            by[scheme] += src  # dense kept verbatim
    g = float(2**30)
    return {
        "gib": {k: v / g for k, v in by.items()},
        "mix_gib": sum(by.values()) / g,
        "bf16_gib": src_total / g,
        "expert_bits": expert_bits,
    }
