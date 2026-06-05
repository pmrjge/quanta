"""Nemotron-H bake orchestration → a self-contained int4(AWQ)/int8/bf16 artifact.

Streamed, one layer resident at a time (rule-8). Per-tensor scheme comes from
:func:`quanta.nemotron.quant_policy.classify` (fail-loud coverage, rule-6):

* routed relu^2 experts (up/down) → **int4 AWQ** — the bf16 source has the sub-grid headroom
  Kimi's int4 source never had (settled). AWQ stores, per expert, the affine codes of
  ``W·diag(s)`` plus the per-input-channel scale ``s``; the runtime applies ``x·diag(1/s)``
  folded into the gather (``expert_method="rtn"`` stores ``s=1`` → plain int4, one runtime path);
* dense always-on (mamba in/out-proj, attention q/k/v/o, latent fc1/fc2, shared expert) → int8
  affine — the decode floor here (inverted vs Kimi);
* SSM core + every norm + router + embeddings/head → bf16.

Two streamed passes mirror the Kimi bake: (1) capture per-MoE-layer latent + routing for AWQ;
(2) write the artifact. :func:`bake_nemotron` bakes the **backbone** + embeddings/head/norm_f;
the native MTP draft head (``mtp.*``, the #40 speculative path) is baked separately by
:func:`bake_nemotron_mtp` into a **sidecar** artifact under the same per-tensor policy, so the
immutable backbone bundle is never touched. Runnable on a slice (``n_layers``, ``expert_subset``)
for bounded validation; the full call is the real bake.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import mlx.core as mx

from quanta.bake.artifact import ArtifactWriter
from quanta.bake.awq import awq_quantize
from quanta.bake.calibrate import expert_rows
from quanta.bake.quant import quantize_affine
from quanta.nemotron.calibrate import capture_calibration
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.loader import NemotronSourceCheckpoint
from quanta.nemotron.moe import relu2
from quanta.nemotron.quant_policy import classify

EMBED, NORMF, HEAD = "backbone.embeddings.weight", "backbone.norm_f.weight", "lm_head.weight"
_EXPERT_BITS = 4


def _write_int8(writer: ArtifactWriter, key: str, w: mx.array, gs: int,
                scale_dtype: mx.Dtype | None) -> None:
    writer.add_quantized(key, *quantize_affine(w, 8, gs, scale_dtype=scale_dtype), 8, gs)


def _write_by_policy(writer: ArtifactWriter, name: str, arr: mx.array, gs: int,
                     scale_dtype: mx.Dtype | None) -> None:
    """Dispatch one named tensor by its quant_policy scheme (fail-loud on unmapped, rule-6).
    Experts are excluded here — they go through the dedicated AWQ pass."""
    kind = classify(name).kind
    if kind == "bf16":
        writer.add_dense(name, arr)
    elif kind == "int8_affine":
        _write_int8(writer, name[: -len(".weight")], arr, gs, scale_dtype)
    else:
        raise ValueError(f"{name}: scheme {kind!r} not handled in the dense pass")


def _bake_expert(writer: ArtifactWriter, base: str, w_up: mx.array, w_down: mx.array,
                 latent: mx.array | None, idx: mx.array | None, gs: int, method: str,
                 scale_dtype: mx.Dtype | None = None) -> int:
    """AWQ (or RTN) int4 the two relu^2 expert matrices. Returns rows used (0 = cold/RTN).

    ``up`` calibrates on the expert's routed latent rows; ``down`` on ``relu2(up·latent)`` of
    those rows. Cold experts (no routed rows) and ``method='rtn'`` use plain int4 with ``s=1``
    so the runtime always divides by a stored scale (one path). NB: AWQ misfires on the relu^2
    down-proj (degenerate per-channel scales → +75% e2e ppl); plain int4 (``method='rtn'``) is
    lossless e2e (+0.1%) at the same 4-bit footprint — prefer it for Nemotron (see #38)."""
    xe = expert_rows(latent, idx, _expert_id(base)) if (method == "awq" and latent is not None) else None
    if xe is not None and xe.shape[0] > 0:
        s_up, p, sc, b = awq_quantize(w_up, xe, _EXPERT_BITS, gs)
        writer.add_awq_quantized(f"{base}.up_proj", p, sc, b, s_up.astype(mx.bfloat16), _EXPERT_BITS, gs)
        up_out = relu2(xe.astype(mx.float32) @ w_up.astype(mx.float32).T)  # [n, inter] down-proj input
        s_dn, p, sc, b = awq_quantize(w_down, up_out, _EXPERT_BITS, gs)
        writer.add_awq_quantized(f"{base}.down_proj", p, sc, b, s_dn.astype(mx.bfloat16), _EXPERT_BITS, gs)
        return int(xe.shape[0])
    for proj, w in (("up_proj", w_up), ("down_proj", w_down)):  # RTN / cold → s=1 identity (plain int4)
        p, sc, b = quantize_affine(w, _EXPERT_BITS, gs, scale_dtype=scale_dtype)
        ones = mx.ones((w.shape[1],), dtype=mx.bfloat16)
        writer.add_awq_quantized(f"{base}.{proj}", p, sc, b, ones, _EXPERT_BITS, gs)
    return 0


def _expert_id(base: str) -> int:
    return int(base.rsplit(".", 1)[1])


def bake_nemotron(
    source: str | Path,
    out_dir: str | Path,
    calib_ids: mx.array,
    *,
    n_layers: int | None = None,
    expert_subset: Iterable[int] | None = None,
    include_head: bool = True,
    group_size: int = 128,
    expert_method: str = "awq",
    scale_dtype: mx.Dtype | None = None,
) -> dict:
    assert expert_method in ("awq", "rtn"), f"expert_method must be 'awq'|'rtn', got {expert_method!r}"
    cfg = NemotronHConfig.from_pretrained(source)
    ck = NemotronSourceCheckpoint(source)
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    experts = list(range(cfg.n_routed_experts)) if expert_subset is None else list(expert_subset)

    caps = capture_calibration(ck, cfg, calib_ids, n_layers=n) if expert_method == "awq" else {}

    writer = ArtifactWriter(out_dir, Path(source) / "config.json")
    if include_head:
        writer.add_dense(EMBED, ck.read(EMBED))  # bf16 (logit-sensitive; policy)
        writer.add_dense(NORMF, ck.read(NORMF))
        if not cfg.tie_word_embeddings:
            writer.add_dense(HEAD, ck.read(HEAD))
        ck.release()

    warm = 0
    for i in range(n):
        kind = cfg.layer_kind(i)
        norm_name = ck.norm_key(i)
        if kind == "mamba":
            t = ck.mamba_tensors(i)
        elif kind == "attention":
            t = ck.attention_tensors(i)
        else:
            t = ck.moe_nonexpert_tensors(i)
        for suf, arr in t.items():  # in/out-proj int8; SSM core, conv, norms bf16; latent fc1/fc2 int8
            _write_by_policy(writer, norm_name if suf == "layer_norm" else ck.mixer_key(i, suf),
                             arr, group_size, scale_dtype)

        if kind == "moe":
            es = ck.expert_stacks(i, cfg.n_routed_experts)
            up_st, down_st = es["up"], es["down"]
            latent, idx = caps.get(i, (None, None))
            for e in experts:
                base = f"backbone.layers.{i}.mixer.experts.{e}"
                warm += int(_bake_expert(writer, base, up_st[e], down_st[e], latent, idx,
                                         group_size, expert_method, scale_dtype) > 0)
            del es, up_st, down_st
        ck.release()
        mx.clear_cache()

    n_moe = sum(cfg.layer_kind(i) == "moe" for i in range(n))
    policy = {"experts": f"int4 {expert_method} g{group_size}", "dense": f"int8 g{group_size}",
              "ssm_norms_router_head": "bf16", "scales": "bf16" if scale_dtype == mx.bfloat16 else "fp32"}
    writer.finalize(policy)
    return {"layers": n, "moe_layers": n_moe, "experts_per_layer": len(experts),
            "warm_experts": warm, "expert_method": expert_method}


def bake_nemotron_mtp(
    source: str | Path,
    out_dir: str | Path,
    *,
    group_size: int = 128,
    expert_method: str = "rtn",
    scale_dtype: mx.Dtype | None = None,
) -> dict:
    """Bake the native MTP draft head (``mtp.layers.0/1.*``) into a self-contained **sidecar**.

    The companion to the backbone artifact for native-MTP self-speculative decode (#40), written as
    its own bundle (sharded safetensors + index + config + manifest) so the immutable backbone
    artifact is never mutated — the MTP-M2 loader reads the backbone from one dir and this head from
    the sidecar. Same per-tensor policy as the backbone (:func:`quant_policy.classify` already covers
    every ``mtp.*`` tensor): the 512-expert relu^2 latent-MoE (``mtp.layers.1.mixer.experts.*``) →
    int4; the dense always-on projections (``eh_proj``, attn q/k/v/o, latent fc1/fc2, shared expert)
    → int8 affine; the fusion/sub-block/final norms + router gate/bias → bf16. Keys match the
    backbone scheme (``mtp.layers.{0,1}.mixer.*``) so a ``build_resident_mtp`` mirrors
    :func:`quanta.nemotron.runtime.build_resident_block`.

    RTN only: the head follows the backbone's ship decision (finding #38 — AWQ regresses the relu^2
    down-proj e2e; RTN is ~lossless at the same 4-bit footprint), and MTP latent calibration capture
    is not built, so ``expert_method`` must be ``"rtn"`` (fail loud, rule 6). Streamed one expert
    resident at a time (rule 8 / rule 3 IO loop) — no 21.5 GiB stack is ever materialized. Run solo.
    """
    assert expert_method == "rtn", (
        f"MTP sidecar bakes RTN experts only (finding #38; no MTP calib capture), got {expert_method!r}")
    cfg = NemotronHConfig.from_pretrained(source)
    ck = NemotronSourceCheckpoint(source)
    n_experts = cfg.n_routed_experts
    mpre = "mtp.layers.1.mixer."

    source_mtp = {k for k in ck.weight_map if k.startswith("mtp.")}
    assert source_mtp, f"no mtp.* tensors in {source} — not a native-MTP checkpoint"
    # nonexpert = fusion + attn sub-block + moe-nonexpert + norms; ".experts." excludes the routed
    # experts but keeps shared_experts ("_experts." not ".experts.") → dense int8 by policy.
    nonexpert = sorted(k for k in source_mtp if ".experts." not in k)

    writer = ArtifactWriter(out_dir, Path(source) / "config.json")
    consumed: set[str] = set()
    for key in nonexpert:  # int8 (eh_proj/qkvo/fc1/fc2/shared) or bf16 (norms/gate/bias) by policy
        _write_by_policy(writer, key, ck.read(key), group_size, scale_dtype)
        consumed.add(key)
    ck.release()

    warm = 0
    for e in range(n_experts):  # routed relu^2 experts → int4 RTN, one expert resident (rule 8)
        base = f"{mpre}experts.{e}"
        up, down = ck.read(f"{base}.up_proj.weight"), ck.read(f"{base}.down_proj.weight")
        warm += int(_bake_expert(writer, base, up, down, None, None,
                                 group_size, expert_method, scale_dtype) > 0)
        consumed |= {f"{base}.up_proj.weight", f"{base}.down_proj.weight"}
        if e % 32 == 31:
            ck.release()
            mx.clear_cache()
    ck.release()

    orphans = source_mtp - consumed  # rule 6: every source mtp.* tensor has a runtime consumer
    assert not orphans, f"rule 6: {len(orphans)} mtp tensors unconsumed, e.g. {sorted(orphans)[:5]}"

    policy = {"head": "mtp", "experts": f"int4 {expert_method} g{group_size}",
              "dense": f"int8 g{group_size}", "norms_router": "bf16",
              "scales": "bf16" if scale_dtype == mx.bfloat16 else "fp32"}
    writer.finalize(policy)
    return {"head": "mtp", "n_experts": n_experts, "warm_experts": warm,
            "expert_method": expert_method, "tensors": len(consumed)}
