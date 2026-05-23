"""Self-consistency parity for the Nemotron-H layer modules + loader wiring.

Real config dims, random-init weights (the prefill==decode equivalences are algebraic, so
random weights suffice), tiny sequence — safe alongside the bake (no checkpoint weights
loaded; the loader section only reads tensor *shapes* lazily + one tiny tensor).

  * MambaMixer:  chunked prefill == token-by-token decode (SSD + conv state)
  * NemotronAttention: naive == fast (rope+sdpa), and prefill == incremental decode (KV cache)
  * loader: resolves backbone.* names; tensor shapes match the config dataclass

    uv run python -m parity.nemotron_layers_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.nemotron.attention import KVCache, NemotronAttention
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.loader import NemotronSourceCheckpoint
from quanta.nemotron.mamba_mixer import MambaMixer

MODEL = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"


def _rel(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)) / (mx.max(mx.abs(b)) + 1e-6))


def run() -> None:
    mx.random.seed(0)
    cfg = NemotronHConfig.from_pretrained(MODEL)
    length = 16

    # --- Mamba mixer: prefill == decode -------------------------------------
    mix = MambaMixer(cfg)
    mix.conv_weight = mx.random.normal(mix.conv_weight.shape) * 0.2
    mix.A_log = mx.random.normal((cfg.mamba_num_heads,)) * 0.5
    mix.dt_bias = mx.random.normal((cfg.mamba_num_heads,)) * 0.1
    mix.D = mx.random.normal((cfg.mamba_num_heads,))
    x = mx.random.normal((1, length, cfg.hidden_size)) * 0.5
    y_pf, _, _ = mix(x)
    state, cstate = None, mx.zeros((1, cfg.conv_kernel - 1, cfg.mamba_conv_dim))
    ys = []
    for t in range(length):
        y_t, state, cstate = mix(x[:, t : t + 1], state=state, conv_state=cstate)
        ys.append(y_t)
    mamba_ok = _rel(mx.concatenate(ys, axis=1), y_pf) < 3e-3

    # --- GQA attention: naive == fast, and prefill == incremental decode ----
    attn = NemotronAttention(cfg)
    xa = mx.random.normal((1, length, cfg.hidden_size)) * 0.5
    o_fast = attn(xa, offset=0, use_fast=True)
    o_naive = attn(xa, offset=0, use_fast=False)
    nf_ok = _rel(o_fast, o_naive) < 2e-3
    cache = KVCache()
    od = [attn(xa[:, t : t + 1], cache=cache, use_fast=True) for t in range(length)]
    dec_ok = _rel(mx.concatenate(od, axis=1), o_fast) < 2e-3

    # --- loader: name resolution + shapes match the config ------------------
    ck = NemotronSourceCheckpoint(MODEL)
    i_m = cfg.layers_block_type.index("mamba")
    i_e = cfg.layers_block_type.index("moe")
    i_a = cfg.layers_block_type.index("attention")
    sh_in = ck.shape(ck.mixer_key(i_m, "in_proj.weight"))
    sh_conv = ck.shape(ck.mixer_key(i_m, "conv1d.weight"))
    sh_up = ck.shape(ck.expert_key(i_e, 0, "up_proj"))
    sh_q = ck.shape(ck.mixer_key(i_a, "q_proj.weight"))
    a_log = ck.read(ck.mixer_key(i_m, "A_log"))  # tiny real read
    loader_ok = (
        sh_in == (cfg.mamba_in_proj_dim, cfg.hidden_size)
        and sh_conv == (cfg.mamba_conv_dim, 1, cfg.conv_kernel)
        and sh_up == (cfg.moe_intermediate_size, cfg.moe_latent_size)
        and sh_q == (cfg.attn_q_dim, cfg.hidden_size)
        and a_log.shape == (cfg.mamba_num_heads,)
    )

    print("\n=== Nemotron-H layers (self-consistency) ===")
    print(f"Mamba mixer prefill == decode        : {mamba_ok}  rel={_rel(mx.concatenate(ys, axis=1), y_pf):.2e}")
    print(f"GQA attention naive == fast          : {nf_ok}  rel={_rel(o_fast, o_naive):.2e}")
    print(f"GQA attention prefill == decode      : {dec_ok}  rel={_rel(mx.concatenate(od, axis=1), o_fast):.2e}")
    print(f"loader names/shapes match config     : {loader_ok}")
    print(f"  in_proj{sh_in} conv{sh_conv} expert_up{sh_up} q_proj{sh_q}")
    assert all([mamba_ok, nf_ok, dec_ok, loader_ok])
    print("Nemotron-H layers OK (mixer + GQA self-consistent; loader wired)")


if __name__ == "__main__":
    run()
