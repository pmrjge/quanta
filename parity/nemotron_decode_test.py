"""Prefill == stepwise-decode for the assembled Nemotron-H model (state threading across M/*/E).

A tiny random-init NemotronModel with a mixed ``M*EM`` layer pattern (real config dims via
dataclasses.replace, so it's small + safe — no checkpoint). The chunked-prefill forward over the
whole sequence must equal feeding the same tokens one at a time through the decode path, which
exercises all three state kinds together: growing KV caches (attention), the (ssm, conv)
recurrence (mamba), and stateless MoE. A logic bug in the threading would be O(1), not sub-1%.

    uv run python -m parity.nemotron_decode_test
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx

from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.generate import decode_state, generate
from quanta.nemotron.mamba_mixer import MambaMixer
from quanta.nemotron.model import NemotronModel
from quanta.nemotron.moe import NemotronLatentMoE

MODEL = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"


def _rel(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)) / (mx.max(mx.abs(b)) + 1e-6))


def _small_cfg() -> NemotronHConfig:
    return replace(
        NemotronHConfig.from_pretrained(MODEL),
        vocab_size=128, hidden_size=64, num_hidden_layers=4, hybrid_override_pattern="M*EM",
        num_attention_heads=8, num_key_value_heads=2, head_dim=8,
        mamba_num_heads=8, mamba_head_dim=16, mamba_n_groups=2, ssm_state_size=16,
        conv_kernel=4, expand=2, chunk_size=8,
        n_routed_experts=8, num_experts_per_tok=3, moe_intermediate_size=24,
        moe_latent_size=16, moe_shared_expert_intermediate_size=20,
    )


def _randomize(model: NemotronModel) -> None:
    """Mamba SSM params + MoE expert stacks default to zeros/ones — give them real dynamics."""
    cfg = model.cfg
    for blk in model.layers:
        m = blk.mixer
        if isinstance(m, MambaMixer):
            m.conv_weight = mx.random.normal(m.conv_weight.shape) * 0.2
            m.A_log = mx.random.normal((cfg.mamba_num_heads,)) * 0.5
            m.dt_bias = mx.random.normal((cfg.mamba_num_heads,)) * 0.1
            m.D = mx.random.normal((cfg.mamba_num_heads,))
        elif isinstance(m, NemotronLatentMoE):
            e, lat, inter = cfg.n_routed_experts, cfg.moe_latent_size, cfg.moe_intermediate_size
            m.gate_weight = mx.random.normal((e, cfg.hidden_size))
            m.e_score_correction_bias = mx.random.normal((e,))
            m.up_stack = mx.random.normal((e, inter, lat)) * 0.1
            m.down_stack = mx.random.normal((e, lat, inter)) * 0.1


def run() -> None:
    mx.random.seed(0)
    cfg = _small_cfg()
    model = NemotronModel(cfg)
    _randomize(model)
    length = 12
    ids = mx.random.randint(0, cfg.vocab_size, (length,))

    logits_pf, _, _ = model(ids)  # chunked prefill over the whole sequence

    caches, ssm, conv = decode_state(model)  # zero conv-state → step path from token 0
    dec = []
    for t in range(length):
        lg, ssm, conv = model(ids[t : t + 1], caches=caches, ssm=ssm, conv=conv)
        dec.append(lg)
    logits_dec = mx.concatenate(dec, axis=1)
    rel = _rel(logits_dec, logits_pf)
    decode_ok = rel < 1e-2

    gen = generate(model, [1, 2, 3, 4], max_new_tokens=5, temperature=0.0)
    gen_ok = len(gen) == 5 and all(0 <= t < cfg.vocab_size for t in gen)

    print("\n=== Nemotron-H decode loop (M/*/E state threading) ===")
    print(f"prefill == stepwise decode           : {decode_ok}  rel={rel:.2e}")
    print(f"generate loop (length/in-vocab)      : {gen_ok}  tokens={gen}")
    assert all([decode_ok, gen_ok])
    print("Nemotron-H decode OK (prefill==decode across mamba/attn/moe; generate runs)")


if __name__ == "__main__":
    run()
