"""Per-tensor quantization policy for the Nemotron-H bake (the int4/int8/bf16 mix).

For **Super-120B** the bf16 model fits (~247 GiB < 490), so the mix is purely a decode-bandwidth
play; for **Ultra-550B** the bf16 model is ~1023 GiB and does NOT fit, so the int4 mix (~306 GiB
resident) is what lets it serve at all. Both ship at **group-size 64**:

  * **routed relu^2 experts** (the sparse top-k bulk) -> ``int4`` g64 (RTN — AWQ regressed e2e on
    the relu^2 down-proj, #38). The bf16 source gives the headroom Kimi's int4 source never had;
  * **dense always-on** (mamba in/out-proj, attention q/k/v/o, shared experts, MoE latent proj,
    MTP eh_proj) -> ``int8`` affine g64 — this path sets the decode floor here (inverted vs Kimi);
  * **SSM core** (``A_log``/``D``/``dt_bias``/``conv1d``), all norms, router gate + correction
    bias, embeddings, and lm_head -> ``bf16`` (recurrence stability + logit sensitivity; small
    fraction of bytes).

:func:`classify` fails loud on any unmapped tensor (rule #6 — refuse to bake a tensor with no
policy). :func:`estimate_storage` sizes the mix analytically from the config (backbone only).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import NemotronHConfig


@dataclass(frozen=True)
class QScheme:
    kind: str  # "int4_gptq" | "int8_affine" | "bf16"
    bits: int
    group_size: int  # 0 for bf16

    @property
    def bytes_per_param(self) -> float:
        if self.kind == "bf16":
            return 2.0
        return (self.bits + 32 / self.group_size) / 8  # affine: + per-group scale/zero overhead


INT4_GPTQ = QScheme("int4_gptq", 4, 64)   # both Nemotron bakes ship g64 (…-quanta_int4{rtn,awq}_g64)
INT8 = QScheme("int8_affine", 8, 64)      # dense shares the experts' group size (bake.py: one `gs`)
BF16 = QScheme("bf16", 16, 0)

_BYTES = {INT4_GPTQ.kind: INT4_GPTQ.bytes_per_param, INT8.kind: INT8.bytes_per_param,
          BF16.kind: BF16.bytes_per_param}


def classify(name: str) -> QScheme:
    """Map a tensor name to its quant scheme. Works for both backbone.* and mtp.* tensors."""
    n = name
    # routed relu^2 experts (up/down) — the int4-GPTQ bulk
    if ".experts." in n and ("up_proj" in n or "down_proj" in n) and "shared" not in n:
        return INT4_GPTQ
    # shared expert — always-on dense
    if "shared_experts" in n:
        return INT8
    # mamba in/out projections
    if n.endswith("in_proj.weight") or n.endswith("out_proj.weight"):
        return INT8
    # attention projections
    if any(n.endswith(p + ".weight") for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return INT8
    # MoE latent projections (fc1/fc2) + MTP embed-hidden fusion
    if "latent_proj" in n or n.endswith("eh_proj.weight"):
        return INT8
    # SSM core — never quantize (recurrence stability; cache state is fp32 upstream)
    if n.endswith(".A_log") or n.endswith(".D") or n.endswith(".dt_bias") or ".conv1d." in n:
        return BF16
    # router (gate weight + sigmoid correction bias)
    if n.endswith("gate.weight") or "e_score_correction_bias" in n:
        return BF16
    # every norm (per-layer, gated mamba norm, final norm, MTP enorm/hnorm/final_layernorm)
    if n.endswith("norm.weight") or n.endswith("norm_f.weight"):
        return BF16
    # token table + output head (logit-sensitive; small fraction of bytes)
    if n.endswith("embeddings.weight") or n.endswith("lm_head.weight"):
        return BF16
    raise ValueError(f"no quant policy for tensor: {name!r}")


def bake_plan(tensor_names) -> dict[str, QScheme]:
    """Classify every tensor; raises if any is unmapped (full-coverage guarantee, rule #6)."""
    return {name: classify(name) for name in tensor_names}


def estimate_storage(cfg: NemotronHConfig) -> dict:
    """Analytic storage of the shipped int4-RTN/AWQ g64 mix vs bf16 (backbone only; ignores the
    ~2.8B MTP sidecar).

    Faithful to the as-baked artifact (rule #6 — the fit gate must match what ships): each affine
    group stores a bf16 scale + bf16 zero (= 32 bits/group, the ``32/group_size`` term in
    :attr:`QScheme.bytes_per_param`), and every routed expert additionally stores a per-input-channel
    bf16 ``awq_scale`` vector (``ones`` under RTN, real scales under AWQ — same bytes either way). At
    g64 this projects ~305.9 GiB, matching the 305.97 GiB on-disk Ultra backbone
    (``…-quanta_int4rtn_g64``); the prior g128 sizing under-counted by ~16 GiB (the U0 fit-test
    cross-checks the projection against the real artifact)."""
    h = cfg.hidden_size
    counts = {INT4_GPTQ.kind: 0, INT8.kind: 0, BF16.kind: 0}

    def add(scheme: QScheme, params: int) -> None:
        counts[scheme.kind] += params

    for kind in cfg.layers_block_type:
        add(BF16, h)  # per-layer input norm
        if kind == "mamba":
            add(INT8, h * cfg.mamba_in_proj_dim + cfg.mamba_d_inner * h)  # in + out proj
            add(BF16, cfg.mamba_conv_dim * cfg.conv_kernel + cfg.mamba_conv_dim
                + 3 * cfg.mamba_num_heads + cfg.mamba_d_inner)  # conv w+b, A_log, D, dt_bias, gated norm
        elif kind == "attention":
            add(INT8, 2 * h * cfg.attn_q_dim + 2 * h * cfg.attn_kv_dim)  # q,o + k,v
        elif kind == "moe":
            lat, inter = cfg.moe_latent_size, cfg.moe_intermediate_size
            shared = cfg.moe_shared_expert_intermediate_size
            add(INT4_GPTQ, cfg.n_routed_experts * (lat * inter + inter * lat))  # up + down
            add(INT8, 2 * h * shared)  # shared up + down
            add(INT8, 2 * h * lat)  # fc1 + fc2 latent proj
            add(BF16, h * cfg.n_routed_experts + cfg.n_routed_experts)  # gate + correction bias
    add(BF16, 2 * cfg.vocab_size * h + h)  # embeddings + lm_head + final norm

    gib = {k: counts[k] * _BYTES[k] / 2**30 for k in counts}
    # per-routed-expert awq_scale vector (bf16, one per input channel: up<-latent, down<-intermediate);
    # stored for EVERY expert — the runtime always divides by a scale (`ones` in the RTN bake, #38)
    scale_overhead_gib = (cfg.count("moe") * cfg.n_routed_experts
                          * (cfg.moe_latent_size + cfg.moe_intermediate_size) * 2) / 2**30
    total_params = sum(counts.values())
    return {
        "params": counts,
        "gib": gib,
        "scale_overhead_gib": scale_overhead_gib,
        "total_params": total_params,
        "total_gib_mix": sum(gib.values()) + scale_overhead_gib,
        "total_gib_bf16": total_params * 2 / 2**30,
    }
