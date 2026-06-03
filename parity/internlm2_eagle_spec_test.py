"""Model-free lossless gate for InternLM2.5's EAGLE-3 spec-decode adapter (M1).

M0 (``parity/internlm2_eagle_capture_test.py``) proved the two substrate primitives on the bf16
reference (capture + lossless truncate). M1 wires the per-model adapter
:func:`quanta.internlm2.eagle.spec_generate` around the generic
:func:`quanta.eagle.spec_core.spec_generate`, extends capture to the **packed** runtime, and adds the
:meth:`~quanta.internlm2.runtime.InternLM2ResidentModel.embed_head` accessor. This gate pins the EAGLE
guarantee — **output bit-identical to plain greedy regardless of drafter quality** — on a tiny
random-init model (NO checkpoint, NO GPU), with an **untrained** (random) drafter:

A. **bf16 spec == greedy** — drives the adapter over the bf16 reference forward (capture + cache +
   truncate + offset all wired) and asserts the emitted token stream equals plain greedy decode of
   the same model, token-for-token. Accept rate is ~0 (random drafter) but the output is identical.
B. **packed capture property** — the new ``_PackedModel`` capture insertion: ``capture_layers`` does
   NOT perturb the packed logits (max|Δ|=0), each cap is ``[T, H]``, and the last cap reconstructs the
   logits through the packed final norm + head (so it is the true pre-norm residual, not off-by-one).
C. **packed spec == greedy** — the full packed forward (``mx.quantized_matmul``) through the spec
   loop: spec-decode over the packed model equals greedy decode of the *same packed model*, proving
   the packed capture + cache + truncate path is lossless end-to-end. (Quantization changes which
   token is greedy vs bf16 — that is expected; the EAGLE guarantee is spec==greedy *within one model*.)

    uv run python -m parity.internlm2_eagle_spec_test
"""

from __future__ import annotations

from dataclasses import asdict

import mlx.core as mx

from quanta.eagle.artifact import DrafterConfig
from quanta.eagle.drafter import EagleDrafter
from quanta.internlm2.config import InternLM2Config
from quanta.internlm2.eagle import spec_generate as il2_spec_generate
from quanta.internlm2.model import InternLM2Model
from quanta.internlm2.runtime import InternLM2ResidentModel, _PackedLayer, _PackedModel, _rmsnorm

_TINY_LAYERS: tuple[int, int, int] = (0, 1, 2)   # low / mid / high of the 4-layer toy model
_PROMPT: list[int] = [3, 9, 1, 40, 22]
_MAX_NEW = 12
_K = 4


def _tiny_cfg() -> InternLM2Config:
    """A 4-layer toy InternLM2.5 (random init) — same field set as the M0 capture gate."""
    return InternLM2Config(
        vocab_size=64, hidden_size=32, num_hidden_layers=4, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, attention_bias=False,
        rope_theta=1.0e4, rope_scaling_type="dynamic", rope_scaling_factor=2.5,
        max_position_embeddings=4096, hidden_act="silu", norm_eps=1e-5, tie_word_embeddings=False,
        eos_token_id=2, eos_token_ids=(2,), pad_token_id=2, bos_token_id=1, add_bos_token=True,
    )


def _build_bf16(cfg: InternLM2Config) -> InternLM2Model:
    model = InternLM2Model(cfg)
    mx.eval(model.parameters())
    return model


def _quantize_layer(layer, *, bits: int = 4, gs: int = 32) -> _PackedLayer:
    """Affine-quantize one bf16 decoder layer's 7 projections into a ``_PackedLayer`` (group ``gs``).

    The packed runtime's own ``_qmm`` consumes exactly ``mx.quantize``'s ``(packed, scale, biases)``
    triplet, so this gives a runnable ``_PackedModel`` without a baked artifact. The toy dims
    (in ∈ {32, 64, 128}) are all divisible by gs=32; bits=4."""
    def q(w):
        return mx.quantize(w, group_size=gs, bits=bits)   # (packed, scale, biases)
    qp, qs, qb = q(layer.attention.wq.weight)
    kp, ks, kb = q(layer.attention.wk.weight)
    vp, vs, vb = q(layer.attention.wv.weight)
    op, osc, ob = q(layer.attention.wo.weight)
    w1p, w1s, w1b = q(layer.feed_forward.w1.weight)
    w3p, w3s, w3b = q(layer.feed_forward.w3.weight)
    w2p, w2s, w2b = q(layer.feed_forward.w2.weight)
    return _PackedLayer(
        attn_norm=layer.attention_norm.weight, ffn_norm=layer.ffn_norm.weight,
        q_packed=qp, q_scale=qs, q_wbias=qb,
        k_packed=kp, k_scale=ks, k_wbias=kb,
        v_packed=vp, v_scale=vs, v_wbias=vb,
        o_packed=op, o_scale=osc, o_wbias=ob,
        w1_packed=w1p, w1_scale=w1s, w1_wbias=w1b,
        w3_packed=w3p, w3_scale=w3s, w3_wbias=w3b,
        w2_packed=w2p, w2_scale=w2s, w2_wbias=w2b,
        attn_bits=bits, attn_gs=gs, mlp_bits=bits, mlp_gs=gs,
    )


def _build_packed(model: InternLM2Model, cfg: InternLM2Config) -> _PackedModel:
    """A runnable ``_PackedModel`` from a tiny bf16 model — bypasses the artifact loader (``object.__new__``)
    and fills the same fields ``_PackedModel.__init__`` would, so ``__call__`` runs unchanged."""
    pm = object.__new__(_PackedModel)
    pm.cfg = cfg
    pm.n_layers = cfg.num_hidden_layers
    pm.embed = model.tok_embeddings.weight
    pm.final_norm = model.norm.weight
    pm.lm_head = None if cfg.tie_word_embeddings else model.output.weight
    pm.layers = [_quantize_layer(layer) for layer in model.layers]
    return pm


def _resident(inner, cfg: InternLM2Config) -> InternLM2ResidentModel:
    """Wrap a tiny inner forward (bf16 ``InternLM2Model`` or ``_PackedModel``) in the production
    resident wrapper without its heavy artifact ``__init__`` — sets only the attributes the EAGLE
    adapter touches (``_model`` / ``cfg`` / KV-cache flags), so ``new_cache`` / ``__call__`` /
    ``embed_head`` behave exactly as in production."""
    rm = object.__new__(InternLM2ResidentModel)
    rm._model = inner
    rm.cfg = cfg
    rm.quantized_kv = False           # bf16 KV — tightest match vs the greedy reference
    rm.kv_group_size = 64
    rm.kv_bits = 8
    rm.packed = isinstance(inner, _PackedModel)
    return rm


def _tiny_drafter(cfg: InternLM2Config) -> EagleDrafter:
    """An untrained (random-init) drafter sized to the toy model: hidden = H, 3 feature layers."""
    dcfg = DrafterConfig(
        hidden=cfg.hidden_size, n_heads=4, head_dim=8, intermediate=2 * cfg.hidden_size,
        eps=cfg.norm_eps, rope_base=1.0e4, n_feature_layers=len(_TINY_LAYERS), layerscale_init=1e-4,
    )
    drafter = EagleDrafter(**asdict(dcfg))
    mx.eval(drafter.parameters())
    return drafter


def _greedy(model: InternLM2ResidentModel, prompt: list[int], max_new: int,
            eos_id: int | None = None) -> list[int]:
    """Plain greedy decode through the resident wrapper — the lossless reference for spec-decode."""
    cache = model.new_cache()
    logits = model(mx.array([list(prompt)]), caches=cache)        # prefill [1, T, V]
    cur = int(mx.argmax(logits[0, -1]).item())
    out = [cur]
    while len(out) < max_new:
        if eos_id is not None and cur == eos_id:
            break
        logits = model(mx.array([[cur]]), caches=cache)           # decode at cache.offset
        cur = int(mx.argmax(logits[0, -1]).item())
        out.append(cur)
    return out[:max_new]


def test_spec_eq_greedy_bf16() -> None:
    cfg = _tiny_cfg()
    rm = _resident(_build_bf16(cfg), cfg)
    drafter = _tiny_drafter(cfg)
    embed, head = rm.embed_head()
    spec_out, stats = il2_spec_generate(rm, drafter, embed, head, _PROMPT,
                                        max_new=_MAX_NEW, k=_K, layers=_TINY_LAYERS, eos_id=None)
    greedy_out = _greedy(rm, _PROMPT, _MAX_NEW)
    assert spec_out == greedy_out, f"bf16 spec != greedy:\n  spec  ={spec_out}\n  greedy={greedy_out}"
    print(f"A bf16   spec==greedy: {len(spec_out)} toks, mean_accept={stats['mean_accept']:.2f} "
          f"(untrained drafter)  ok")


def test_packed_capture_property() -> None:
    cfg = _tiny_cfg()
    pm = _build_packed(_build_bf16(cfg), cfg)
    ids = mx.array([[3, 9, 1, 40, 22, 5, 18]])
    t = ids.shape[1]
    layers = tuple(range(cfg.num_hidden_layers))

    logits_plain = pm(ids)
    logits_cap, caps = pm(ids, capture_layers=layers)

    d_logits = float(mx.max(mx.abs(logits_plain - logits_cap)))
    assert d_logits == 0.0, f"packed capture changed logits: max|Δ|={d_logits}"
    for layer in layers:
        assert caps[layer].shape == (t, cfg.hidden_size), f"caps[{layer}] shape {caps[layer].shape}"

    last = cfg.num_hidden_layers - 1
    recon_h = _rmsnorm(caps[last][None], pm.final_norm, cfg.norm_eps)        # [1, T, H]
    head = pm.embed if cfg.tie_word_embeddings else pm.lm_head
    recon = recon_h @ head.T
    d_recon = float(mx.max(mx.abs(recon - logits_cap)))
    assert d_recon < 1e-4, f"packed last cap is not the pre-norm residual: max|Δ|={d_recon}"
    print(f"B packed capture: logits Δ={d_logits}, caps {len(caps)}×[T={t},H={cfg.hidden_size}], "
          f"recon Δ={d_recon:.2e}  ok")


def test_spec_eq_greedy_packed() -> None:
    cfg = _tiny_cfg()
    rm = _resident(_build_packed(_build_bf16(cfg), cfg), cfg)
    drafter = _tiny_drafter(cfg)
    embed, head = rm.embed_head()
    spec_out, stats = il2_spec_generate(rm, drafter, embed, head, _PROMPT,
                                        max_new=_MAX_NEW, k=_K, layers=_TINY_LAYERS, eos_id=None)
    greedy_out = _greedy(rm, _PROMPT, _MAX_NEW)
    assert spec_out == greedy_out, f"packed spec != greedy:\n  spec  ={spec_out}\n  greedy={greedy_out}"
    print(f"C packed spec==greedy: {len(spec_out)} toks, mean_accept={stats['mean_accept']:.2f}  ok")


def run() -> None:
    test_spec_eq_greedy_bf16()
    test_packed_capture_property()
    test_spec_eq_greedy_packed()
    print("PASS — InternLM2.5 EAGLE-3 adapter lossless (bf16 + packed spec==greedy, packed capture)")


if __name__ == "__main__":
    run()
