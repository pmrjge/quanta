"""Model-free paged-KV gate for InternLM2.5-7B-Chat-1M (#176) — the clean dense-GQA case.

InternLM2.5 has NO recurrent/derived state, so paging is the textbook k/v-pair scenario: every layer's
:class:`~quanta.internlm2.attention.KVCache` is paged through the shared
:class:`~quanta.paged.PagedKVCacheManager`, prefix blocks dedup across requests, and there is **nothing**
to content-address at a block boundary. This gate proves, on a tiny random-init :class:`InternLM2Model`
(NO checkpoint, NO GPU — a few KB of tensors), that:

A. **core** — a paged forward that re-references a resident prefix's blocks and prefills only the uncached
   suffix is **bit-identical** to a one-shot discrete prefill of the whole prompt (the
   ``cache_quant`` orthogonal-axes foundation: int8 quant groups on ``head_dim`` vs blocks on the seq
   axis). Checked for BOTH KV modes (int8 g32 default + bf16) and asserts ``prefill_paged`` yields **no**
   boundary payloads (``has_recurrent_state=False``).
B. **engine wiring** — a real :class:`quanta.shim.omlx._InternLM2BatchedSession` (paged_kv=True) admits a
   request from scratch (== discrete one-shot), releases it, then admits an overlapping request that
   **reuses the resident prefix blocks** (== discrete one-shot), with the manager's stats confirming the
   prefix-hit (``prefix_hit_tokens > 0``).

NTK note: ``max_position_embeddings`` is set far above the test lengths so the dynamic-NTK base stays at
``rope_theta`` (constant) — that is the regime where cross-request prefix reuse is bit-exact (the agentic
sub-256K case). Real-model teacher-forced ppl parity (paged ON == OFF) is deferred to a one-model-at-a-time
GPU session.

    uv run --with numpy python -m parity.internlm2_paged_test
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from quanta.internlm2.batched_runtime import InternLM2BatchedResidentModel
from quanta.internlm2.config import InternLM2Config
from quanta.internlm2.decode import InternLM2Cache
from quanta.internlm2.model import InternLM2Model
from quanta.paged import PagedKVCacheManager
from quanta.shim.omlx import _InternLM2BatchedSession


def _tiny_cfg() -> InternLM2Config:
    """Tiny dense InternLM2.5 config built directly (no checkpoint). ``head_dim=32`` is the smallest the
    int8 KV path accepts (mx.quantize's min group_size is 32); ``max_position_embeddings`` >> test lengths
    so the dynamic-NTK base stays constant (== rope_theta) and cross-request reuse is bit-exact."""
    return InternLM2Config(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, attention_bias=False,
        rope_theta=1.0e4, rope_scaling_type="dynamic", rope_scaling_factor=2.5,
        max_position_embeddings=4096, hidden_act="silu", norm_eps=1e-5, tie_word_embeddings=False,
        eos_token_id=2, eos_token_ids=(2,), pad_token_id=2, bos_token_id=1, add_bos_token=True,
    )


class _FakeInner:
    """Ducks :class:`~quanta.internlm2.runtime.InternLM2ResidentModel`'s surface around a bf16
    :class:`InternLM2Model` so the batched runtime + omlx session run fully model-free (no artifact)."""

    def __init__(self, model: InternLM2Model, cfg: InternLM2Config, *, quantized_kv: bool,
                 kv_group_size: int, kv_bits: int) -> None:
        self._m = model
        self.cfg = cfg
        self.quantized_kv = quantized_kv
        self.kv_group_size = kv_group_size
        self.kv_bits = kv_bits

    @property
    def num_layers(self) -> int:
        return self._m.n_layers

    def new_cache(self) -> InternLM2Cache:
        return InternLM2Cache(self.cfg, quantized=self.quantized_kv,
                              group_size=self.kv_group_size, bits=self.kv_bits)

    def __call__(self, token_ids: mx.array, *, cache: Any = None, caches: Any = None,
                 offset: int | None = None, use_fast: bool = True, last_only: bool = False) -> mx.array:
        del use_fast
        if token_ids.ndim == 1:
            token_ids = token_ids[None]
        cache_obj = cache if cache is not None else caches
        if cache_obj is None:
            cache_list: list | None = None
        elif isinstance(cache_obj, list):
            cache_list = cache_obj
        else:
            cache_list = cache_obj.as_list()
        if offset is not None:
            abs_pos = int(offset)
        elif cache_obj is not None and not isinstance(cache_obj, list):
            abs_pos = cache_obj.offset
        elif cache_list and cache_list[0] is not None:
            abs_pos = cache_list[0].offset
        else:
            abs_pos = 0
        return self._m(token_ids, caches=cache_list, use_fast=True,
                       abs_pos_start=abs_pos, last_only=last_only)


def _prompt(cfg: InternLM2Config, n: int) -> list[int]:
    return [int(x) for x in mx.random.randint(0, cfg.vocab_size, (n,)).tolist()]


def _bf16_model(cfg: InternLM2Config) -> InternLM2Model:
    """Random-init :class:`InternLM2Model` cast to **bf16** — the real runtime runs bf16 end-to-end
    (embeddings + packed matmuls are bf16) and both the discrete :class:`~quanta.internlm2.attention.KVCache`
    and the paged manager's ``gather`` return bf16. A default-init (float32) model would compare a bf16
    paged stream against a float32 discrete stream and show ~1e-3 bf16 rounding — NOT a paging error. So
    the gate must run bf16 to be a true paged==discrete test (mirrors the Nemotron/DSV4 gates' bf16 embed)."""
    from mlx.utils import tree_map

    model = InternLM2Model(cfg)
    model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
    mx.eval(model.parameters())
    return model


def _check_regime(quantized: bool, kv_bits: int) -> None:
    """Core: paged prefix-reuse + suffix prefill == discrete one-shot prefill (bit-exact last logits)."""
    mx.random.seed(0)
    cfg = _tiny_cfg()
    model = _bf16_model(cfg)
    kv_gs = 32 if quantized else 64
    inner = _FakeInner(model, cfg, quantized_kv=quantized, kv_group_size=kv_gs, kv_bits=kv_bits)
    batched = InternLM2BatchedResidentModel.from_inner(inner, max_batch=2)

    P, block = 11, 4
    prompt = _prompt(cfg, P)
    n_pref = ((P - 1) // block) * block  # block-aligned prefix the paged path will reuse

    # discrete CONTINUE-FROM-PREFIX reference: prefill the same prefix chunk, then the same suffix chunk,
    # so the SDPA shapes match the paged path exactly (-> bit-exact). A one-shot whole-prompt prefill would
    # give the suffix a different q_len and drift ~1e-3 in bf16 — real SDPA-tiling noise, not a paging bug.
    disc = inner.new_cache()
    inner(mx.array(prompt[:n_pref]), cache=disc, last_only=True)
    ref = inner(mx.array(prompt[n_pref:]), cache=disc, last_only=True)[0, -1]
    mx.eval(ref)

    spec = batched.paged_kv_spec
    mgr = PagedKVCacheManager(num_layers=spec["n_layers"], block_size=block, max_blocks=64,
                              group_size=spec["group_size"], bits=spec["bits"],
                              quantized=spec["quantized"], model_name="ilm2t")

    # req1: prefill the block-aligned prefix so full blocks get content-addressed (committed).
    seq1 = mgr.new_sequence()
    mgr.advance(seq1, prompt[:n_pref])
    st1 = batched.make_paged_state(mgr, seq1)
    batched.prefill_paged(mx.array(prompt[:n_pref]), st1, offset=0, recurrent_in=None, block_size=block)
    mgr.commit(seq1)

    # req2: full prompt, reuse the resident prefix blocks + prefill only the uncached suffix.
    seq2 = mgr.new_sequence()
    n_attn = mgr.match_prefix(seq2, prompt[:-1])
    suffix = prompt[n_attn:]
    mgr.advance(seq2, suffix)
    st2 = batched.make_paged_state(mgr, seq2)
    logits_pg, boundaries = batched.prefill_paged(mx.array(suffix), st2, offset=n_attn,
                                                  recurrent_in=None, block_size=block)
    mgr.commit(seq2)
    row = logits_pg[0, -1]
    mx.eval(row)

    d = float(mx.max(mx.abs(row - ref)))
    mode = f"int{kv_bits} g{kv_gs}" if quantized else "bf16"
    ok = (boundaries == []) and (n_attn == n_pref > 0) and (d == 0.0)
    print(f"  [{'OK' if ok else 'XX'}] {mode:<9} P={P} blk={block} n_attn={n_attn} "
          f"suffix={len(suffix)} boundaries={len(boundaries)} |Δ|={d:.2e}")
    assert boundaries == [], f"dense model must emit no boundary payloads, got {boundaries}"
    assert n_attn == n_pref > 0, f"expected prefix reuse of {n_pref} tokens, got n_attn={n_attn}"
    assert d == 0.0, f"paged != discrete ({mode}): |Δ|={d}"


def _check_engine() -> None:
    """Engine wiring: real _InternLM2BatchedSession admit (from-scratch == discrete; reuse == discrete)."""
    mx.random.seed(0)
    cfg = _tiny_cfg()
    model = _bf16_model(cfg)
    inner = _FakeInner(model, cfg, quantized_kv=True, kv_group_size=32, kv_bits=8)
    batched = InternLM2BatchedResidentModel.from_inner(inner, max_batch=2)
    sess = _InternLM2BatchedSession(runtime=batched, capacity=2, paged_kv=True,
                                    block_size=4, model_name="ilm2eng")

    P, block = 11, 4
    prompt = _prompt(cfg, P)
    n_pref = ((P - 1) // block) * block

    # two apples-to-apples references: req1 admits from scratch (one-shot, q_len=P) -> discrete one-shot;
    # req2 reuses the prefix (suffix prefill, q_len=P-n_pref) -> discrete continue-from-prefix with the
    # SAME split (identical SDPA shapes -> bit-exact, unlike a one-shot reference which drifts in bf16).
    ref_oneshot = inner(mx.array(prompt), cache=inner.new_cache(), last_only=True)[0, -1]
    disc = inner.new_cache()
    inner(mx.array(prompt[:n_pref]), cache=disc, last_only=True)
    ref_cont = inner(mx.array(prompt[n_pref:]), cache=disc, last_only=True)[0, -1]
    mx.eval(ref_oneshot, ref_cont)

    row1 = sess.admit(0, prompt)              # from scratch (no prior cache) -> one-shot q_len=P
    d1 = float(mx.max(mx.abs(row1 - ref_oneshot)))
    sess.release(0)                           # ref-- ; prefix blocks stay resident for reuse
    row2 = sess.admit(1, prompt)             # reuses req1's committed prefix blocks -> suffix prefill
    d2 = float(mx.max(mx.abs(row2 - ref_cont)))

    stats = sess.get_cache_stats()
    hit = int(stats["prefix_hit_tokens"])
    cow = int(stats["cow_copies"])
    has_rec = "recurrent" in stats
    ok = sess.prefix_cache_enabled and d1 < 5e-4 and d2 < 5e-4 and hit > 0 and not has_rec
    print(f"  [{'OK' if ok else 'XX'}] engine: req1|Δ|={d1:.2e} req2|Δ|={d2:.2e} "
          f"hit_tok={hit} cow={cow} recurrent={has_rec}")
    assert sess.prefix_cache_enabled, "paged session should report prefix_cache_enabled"
    assert d1 < 5e-4 and d2 < 5e-4, f"engine paged != discrete: req1={d1} req2={d2}"
    assert hit > 0, "expected prefix reuse across requests (prefix_hit_tokens>0)"
    assert not has_rec, "dense InternLM2.5 must not allocate a RecurrentPrefixCache"


def run() -> None:
    print("A. core: paged prefix-reuse + suffix == discrete one-shot (per KV mode)")
    _check_regime(quantized=True, kv_bits=8)
    _check_regime(quantized=False, kv_bits=8)
    print("B. engine wiring: real _InternLM2BatchedSession paged admit (from-scratch + reuse)")
    _check_engine()
    print("PASS")


if __name__ == "__main__":
    run()
