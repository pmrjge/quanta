"""Nemotron-H model assembly: embed → [M/E/* blocks] → norm_f → lm_head, MLX-native.

Each layer is a single **pre-norm mixer with a residual** — ``x + mixer(norm(x))`` — with one
norm per layer (the Mamba mixer carries its own internal gated RMSNorm, separate from this
per-layer norm). The block kind (``mamba`` / ``attention`` / ``moe``) follows the config's
``hybrid_override_pattern``. The model threads per-layer state for incremental decode: a growing
KV cache on attention layers, the ``(ssm_state, conv_state)`` recurrence on mamba layers, and
nothing on the stateless MoE layers.

Weights are random-init here — enough to run and to gate the prefill==decode equivalence.
Checkpoint loading (the streamed loader → these modules) is a separate step.
"""

from __future__ import annotations

import mlx.nn as nn

from quanta.nemotron.attention import NemotronAttention
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.loader import NemotronSourceCheckpoint
from quanta.nemotron.mamba_mixer import MambaMixer
from quanta.nemotron.moe import NemotronLatentMoE

_MIXER = {"mamba": MambaMixer, "attention": NemotronAttention, "moe": NemotronLatentMoE}


def load_block(block: "NemotronBlock", ck: NemotronSourceCheckpoint, cfg: NemotronHConfig, i: int) -> None:
    """Load one layer's real bf16 weights from the source checkpoint into a built block.

    Direct attribute assignment (the modules mix ``nn.Linear``/``nn.RMSNorm`` with raw-array
    params); ``ck.read`` already materializes each tensor, so the shard mmaps can be released
    after. The per-layer pre-norm is ``layer_norm``; the mamba mixer's own gated norm is
    ``norm.weight``.
    """
    m = block.mixer
    if block.kind == "mamba":
        t = ck.mamba_tensors(i)
        m.in_proj.weight, m.out_proj.weight = t["in_proj.weight"], t["out_proj.weight"]
        m.norm.weight = t["norm.weight"]
        m.conv_weight, m.conv_bias = t["conv1d.weight"], t["conv1d.bias"]
        m.A_log, m.D, m.dt_bias = t["A_log"], t["D"], t["dt_bias"]
    elif block.kind == "attention":
        t = ck.attention_tensors(i)
        m.q_proj.weight, m.k_proj.weight = t["q_proj.weight"], t["k_proj.weight"]
        m.v_proj.weight, m.o_proj.weight = t["v_proj.weight"], t["o_proj.weight"]
    else:  # moe
        t = ck.moe_nonexpert_tensors(i)
        m.gate_weight = t["gate.weight"]
        m.e_score_correction_bias = t["gate.e_score_correction_bias"]
        m.fc1_latent_proj.weight = t["fc1_latent_proj.weight"]
        m.fc2_latent_proj.weight = t["fc2_latent_proj.weight"]
        m.shared_up.weight = t["shared_experts.up_proj.weight"]
        m.shared_down.weight = t["shared_experts.down_proj.weight"]
        es = ck.expert_stacks(i, cfg.n_routed_experts)
        m.set_experts(es["up"], es["down"])
    block.norm.weight = t["layer_norm"]


class NemotronBlock(nn.Module):
    """One Nemotron-H layer: ``x + mixer(norm(x))``. Returns ``(out, ssm_state, conv_state)`` —
    the mamba recurrence state passes through functionally; attention mutates its KV cache in
    place; for non-mamba layers the state passthroughs are ``None``."""

    def __init__(self, cfg: NemotronHConfig, kind: str) -> None:
        super().__init__()
        if kind not in _MIXER:
            raise ValueError(f"unknown block kind {kind!r}")
        self.kind = kind
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.mixer = _MIXER[kind](cfg)

    def __call__(self, x, *, cache=None, ssm_state=None, conv_state=None, use_fast=True,
                 topk_override: int | None = None, chunked_cont: bool = False):
        """``topk_override`` (moe layers only): if set, route through that many experts instead of
        cfg.num_experts_per_tok — used by the MTP draft head's moe sub-block for a lighter drafter.
        ``chunked_cont`` (mamba layers only): when prefilling a suffix on top of a restored recurrent
        state, use the chunked-SSD continuation (#152 paged) instead of the per-token steps."""
        h = self.norm(x)
        if self.kind == "mamba":
            y, ssm_state, conv_state = self.mixer(h, state=ssm_state, conv_state=conv_state,
                                                  chunked_cont=chunked_cont)
        elif self.kind == "attention":
            y = self.mixer(h, cache=cache, use_fast=use_fast)
        else:  # moe (stateless)
            y = self.mixer(h, topk_override=topk_override) if topk_override is not None else self.mixer(h)
        return x + y, ssm_state, conv_state


class NemotronModel(nn.Module):
    def __init__(self, cfg: NemotronHConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.kinds = cfg.layers_block_type
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [NemotronBlock(cfg, k) for k in self.kinds]
        self.norm_f = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def __call__(self, token_ids, *, caches=None, ssm=None, conv=None, use_fast=True,
                 topk_override: int | None = None):
        """Logits for ``token_ids`` ``[t]`` → ``[1, t, vocab]``, plus the updated mamba state
        lists ``(ssm, conv)``. ``caches`` (per-attention-layer KV caches) grow in place. With all
        of ``caches``/``ssm``/``conv`` ``None`` this is a fresh prefill; pass the prior state to
        continue (one token at a time for decode). Mamba ``conv_state=None`` ⇒ chunked prefill;
        a real (zero-initialised) ``conv_state`` ⇒ the O(1) step recurrence.

        ``topk_override`` (moe layers only): route through that many experts instead of
        cfg.num_experts_per_tok — used by the MTP draft head's sub-blocks for a lighter drafter."""
        h = self.embed_tokens(token_ids)[None]  # [1, t, hidden]
        n = len(self.layers)
        caches = caches if caches is not None else [None] * n
        ssm = ssm if ssm is not None else [None] * n
        conv = conv if conv is not None else [None] * n
        for i, blk in enumerate(self.layers):
            h, ssm[i], conv[i] = blk(h, cache=caches[i], ssm_state=ssm[i], conv_state=conv[i],
                                     use_fast=use_fast, topk_override=topk_override)
        return self.lm_head(self.norm_f(h)), ssm, conv
