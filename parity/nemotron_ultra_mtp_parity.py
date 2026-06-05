"""Nemotron-Ultra U4 / MTP-M0 — native MTP draft-head numeric parity (bf16, source weights).

The first milestone of the **native MTP self-speculative decode** stream (task #40). Builds the
runtime's :class:`quanta.nemotron.mtp.NemotronMTPModule` — one MTP head: ``fuse(embed, prev_hidden) ->
attn sub-block -> moe sub-block -> final_layernorm -> shared head`` — fills it from the Ultra BF16
source's ``mtp.layers.0/1.*`` tensors (1040 of them: the ``mtp.layers.0`` attention sub-block with the
``enorm``/``hnorm``/``eh_proj`` fusion, and the ``mtp.layers.1`` 512-expert relu^2 latent-MoE sub-block
+ ``final_layernorm``), and diffs its forward against an **independent inline reference**.

What this gates (the genuinely NEW surface vs U1): the fusion concat order
``eh_proj(concat([enorm(embed), hnorm(prev_hidden)]))``, which per-sub-block norm feeds which mixer,
the ``x + mixer(norm(x))`` residual wiring, the sub-block **kinds** (attn then moe), and the
``final_layernorm -> head`` readout — at full Ultra scale (hidden 8192, 512 experts top-22, latent
2048). The reference recomputes the fusion / pre-norms / residuals / readout with raw ``mx`` ops and
runs the two sub-blocks through **standalone** :class:`NemotronAttention` / :class:`NemotronLatentMoE`
modules whose internals are independently U1-gated vs transformers
(``parity/nemotron_ultra_layer_parity.py``: attn Δ 4.5e-06, moe Δ 7e-04). Chaining "M0: head ==
inline-assembly" with "U1: mixers == transformers" is a transitive reference for the whole head.

What it does NOT gate (deferred, by design): the head's *functional* quality — whether it actually
predicts token ``p+2`` well, i.e. a high accept rate — is the separate MTP-M2 real-decode gate.
Losslessness holds for ANY head quality (the main model verifies every draft, CLAUDE.md rule 4), so
the structural (M0) vs functional (M2) split is clean; a mis-wired head would still decode greedy-exact,
just with accept rate ~= chance. M0 is parity-first: it must be green BEFORE the head is quantized/baked
(MTP-M1) and wired into the resident spec loop (MTP-M2).

Layer-streamed (rule 8): one MTP head resident — the 512-expert bf16 stack (~21.5 GiB) is the peak
(the documented per-moe-layer exception, same as U1). Run solo.

    uv run python -m parity.nemotron_ultra_mtp_parity
"""

from __future__ import annotations

import mlx.core as mx

from quanta.nemotron.attention import NemotronAttention
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.loader import NemotronSourceCheckpoint
from quanta.nemotron.moe import NemotronLatentMoE
from quanta.nemotron.mtp import NemotronMTPModule

ULTRA = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
T = 8          # draft window: T=1 is the real single-token draft; T>1 also exercises causal attn + multi-token route
HEADV = 2048   # small stand-in LM head [HEADV, hidden] — the readout *orientation* is under test, not the vocab


def _rel(a: mx.array, b: mx.array) -> float:
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float(mx.max(mx.abs(a - b)) / (mx.max(mx.abs(b)) + 1e-6))


# Map the runtime's MTP-module slots <- source ``mtp.*`` tensor keys (non-expert; experts streamed below).
_L0, _L1 = "mtp.layers.0.", "mtp.layers.1."
_KEYMAP: dict[str, str] = {
    "enorm": _L0 + "enorm.weight",
    "hnorm": _L0 + "hnorm.weight",
    "eh_proj": _L0 + "eh_proj.weight",
    "a_norm": _L0 + "norm.weight",                 # attn sub-block pre-norm
    "q": _L0 + "mixer.q_proj.weight",
    "k": _L0 + "mixer.k_proj.weight",
    "v": _L0 + "mixer.v_proj.weight",
    "o": _L0 + "mixer.o_proj.weight",
    "m_norm": _L1 + "norm.weight",                 # moe sub-block pre-norm
    "gate": _L1 + "mixer.gate.weight",
    "gate_bias": _L1 + "mixer.gate.e_score_correction_bias",
    "fc1": _L1 + "mixer.fc1_latent_proj.weight",
    "fc2": _L1 + "mixer.fc2_latent_proj.weight",
    "s_up": _L1 + "mixer.shared_experts.up_proj.weight",
    "s_down": _L1 + "mixer.shared_experts.down_proj.weight",
    "final": _L1 + "final_layernorm.weight",
}


def _mtp_tensors(ck: NemotronSourceCheckpoint, n_experts: int) -> tuple[dict[str, mx.array], set[str]]:
    """Read the source ``mtp.layers.0/1.*`` tensors (one head) into a slot dict; stream the 512
    routed experts into ``[E, out, in]`` bf16 stacks with periodic shard release (rule 8 / rule 3 IO
    loop). Also returns the set of source keys consumed (for the rule-6 coverage check)."""
    t: dict[str, mx.array] = {name: ck.read(key) for name, key in _KEYMAP.items()}
    consumed: set[str] = set(_KEYMAP.values())

    up0 = ck.read(_L1 + "mixer.experts.0.up_proj.weight")
    down0 = ck.read(_L1 + "mixer.experts.0.down_proj.weight")
    consumed |= {_L1 + "mixer.experts.0.up_proj.weight", _L1 + "mixer.experts.0.down_proj.weight"}
    up = mx.zeros((n_experts, *up0.shape), up0.dtype)
    down = mx.zeros((n_experts, *down0.shape), down0.dtype)
    up[0], down[0] = up0, down0
    for e in range(1, n_experts):
        uk = _L1 + f"mixer.experts.{e}.up_proj.weight"
        dk = _L1 + f"mixer.experts.{e}.down_proj.weight"
        up[e], down[e] = ck.read(uk), ck.read(dk)
        consumed |= {uk, dk}
        if e % 32 == 31:
            mx.eval(up, down)
            ck.release()
    mx.eval(list(t.values()), up, down)
    ck.release()
    t["up"], t["down"] = up, down
    return t, consumed


def _fill_module(mtp: NemotronMTPModule, t: dict[str, mx.array]) -> None:
    """Assign the source ``mtp.*`` tensors into the runtime module's slots (the future MTP-M2 loader)."""
    mtp.enorm.weight, mtp.hnorm.weight, mtp.eh_proj.weight = t["enorm"], t["hnorm"], t["eh_proj"]
    a = mtp.attn_block
    a.norm.weight = t["a_norm"]
    a.mixer.q_proj.weight, a.mixer.k_proj.weight = t["q"], t["k"]
    a.mixer.v_proj.weight, a.mixer.o_proj.weight = t["v"], t["o"]
    m = mtp.moe_block
    m.norm.weight = t["m_norm"]
    m.mixer.gate_weight = t["gate"]
    m.mixer.e_score_correction_bias = t["gate_bias"]
    m.mixer.fc1_latent_proj.weight, m.mixer.fc2_latent_proj.weight = t["fc1"], t["fc2"]
    m.mixer.shared_up.weight, m.mixer.shared_down.weight = t["s_up"], t["s_down"]
    m.mixer.set_experts(t["up"], t["down"])
    mtp.final_layernorm.weight = t["final"]


def _ref_forward(t: dict[str, mx.array], cfg: NemotronHConfig, prev_hidden: mx.array,
                 token_emb: mx.array, head: mx.array) -> tuple[mx.array, mx.array]:
    """Independent inline MTP head: fuse -> attn sub-block -> moe sub-block -> final-norm -> head.

    Raw ``mx`` ops for the fusion / pre-norms / residuals / readout (the new surface); standalone
    U1-gated mixers for the two sub-block forwards. Returns ``(logits, new_hidden)`` where
    ``new_hidden`` is the post-moe-block residual (the chained-draft feature)."""
    eps = cfg.norm_eps
    # fusion: eh_proj(concat([enorm(embed), hnorm(prev_hidden)])) — DeepSeek-V3 / Nemotron-H MTP order
    e = mx.fast.rms_norm(token_emb, t["enorm"], eps)
    h = mx.fast.rms_norm(prev_hidden, t["hnorm"], eps)
    x = mx.concatenate([e, h], axis=-1) @ t["eh_proj"].T

    # attn sub-block: x + attn(norm(x))
    attn = NemotronAttention(cfg)
    attn.q_proj.weight, attn.k_proj.weight = t["q"], t["k"]
    attn.v_proj.weight, attn.o_proj.weight = t["v"], t["o"]
    x = x + attn(mx.fast.rms_norm(x, t["a_norm"], eps), cache=None, use_fast=False)

    # moe sub-block: x + moe(norm(x))
    moe = NemotronLatentMoE(cfg)
    moe.gate_weight, moe.e_score_correction_bias = t["gate"], t["gate_bias"]
    moe.fc1_latent_proj.weight, moe.fc2_latent_proj.weight = t["fc1"], t["fc2"]
    moe.shared_up.weight, moe.shared_down.weight = t["s_up"], t["s_down"]
    moe.set_experts(t["up"], t["down"])
    x = x + moe(mx.fast.rms_norm(x, t["m_norm"], eps))

    logits = mx.fast.rms_norm(x, t["final"], eps) @ head.T
    return logits, x


def run() -> None:
    mx.random.seed(0)
    cfg = NemotronHConfig.from_pretrained(ULTRA)
    ck = NemotronSourceCheckpoint(ULTRA)
    t, consumed = _mtp_tensors(ck, cfg.n_routed_experts)

    # rule 6: every source mtp.* tensor must have a consumer (no orphan baked into the head later).
    source_mtp = {k for k in ck.weight_map if k.startswith("mtp.")}
    orphans = source_mtp - consumed
    assert not orphans, f"rule 6: {len(orphans)} source mtp tensors unconsumed, e.g. {sorted(orphans)[:5]}"

    mtp = NemotronMTPModule(cfg)
    _fill_module(mtp, t)

    # guard against a two-random-modules false PASS: weights genuinely loaded + correct sub-block kinds.
    assert mx.array_equal(mtp.eh_proj.weight, t["eh_proj"]), "eh_proj not loaded"
    assert mx.array_equal(mtp.moe_block.mixer.up_stack, t["up"]), "experts not loaded"
    assert mtp.attn_block.kind == "attention" and mtp.moe_block.kind == "moe", "sub-block kinds wrong"

    prev_hidden = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.bfloat16)
    token_emb = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.bfloat16)
    head = mx.random.normal((HEADV, cfg.hidden_size)).astype(mx.bfloat16)

    m_logits, m_hidden = mtp(prev_hidden, token_emb, head, use_fast=False, return_hidden=True)
    r_logits, r_hidden = _ref_forward(t, cfg, prev_hidden, token_emb, head)
    mx.eval(m_logits, m_hidden, r_logits, r_hidden)

    d_logits, d_hidden = _rel(m_logits, r_logits), _rel(m_hidden, r_hidden)
    ok = d_logits < 1e-2 and d_hidden < 1e-2

    print("\n=== Nemotron-Ultra MTP-M0 (native MTP draft-head parity vs inline reference) ===")
    print(f"head: 1 module (mtp.layers.0 attn + mtp.layers.1 {cfg.n_routed_experts}e top-"
          f"{cfg.num_experts_per_tok} latent-moe), hidden {cfg.hidden_size}, T {T}")
    print(f"source mtp tensors covered : {len(consumed)}/{len(source_mtp)} (rule 6)")
    print(f"logits      Δ {d_logits:.2e}")
    print(f"new_hidden  Δ {d_hidden:.2e}   (the chained-draft feature: pre-final-norm residual)")
    print("PASS" if ok else "FAIL (MTP head != inline reference)")
    assert ok, f"MTP head parity failed: logits Δ {d_logits:.2e}, new_hidden Δ {d_hidden:.2e}"


if __name__ == "__main__":
    run()
