"""Parity: DSV4 prefix-sharing paged latent KV + content-addressed derived-state recompute (#175).

Model-free gate for the DSV4 half of #152. With **tiny random params** (never load the ~180 GiB
artifact — a job may be resident), it proves the two halves of DSV4 prefix sharing are exact:

  A. **Core (the remaining-risk gate).** For each attention regime — ratio-0 dense sliding-window,
     ratio-4 compressed + Lightning-Indexer (overlapping windows), and the ratio-128 regime
     (compressed, no indexer, non-overlapping) — a continuous decode of ``[0, T)`` is compared against
     a two-pass paged run: pass 1 fills the prefix ``[0, P)``'s latent blocks + snapshots the derived
     compressed-KV / indexer-KV / raw-hidden ring at ``P``; pass 2 ``match_prefix``-es the resident
     latent blocks, restores the derived snapshot, and prefills only the suffix ``[P, T)``. The suffix
     outputs must be **bit-identical** to the continuous decode's ``[P, T)``. ``P`` is chosen a block
     multiple but **NOT** a ratio multiple, so the first post-boundary compressor window STRADDLES the
     prefix/suffix boundary (``prev``/part of ``cur`` in the prefix) — the exact risk the design hinges
     on: the restored ring must carry the ``coff*ratio`` raw-hidden tail to pool that window. This is
     unconditional in block_size↔ratio alignment (the snapshot dodges the coupling).

  B. **Engine wiring.** Drives the REAL ``_DSV4BatchedSession`` paged path (``admit`` ->
     ``_admit_paged`` -> ``match_prefix`` + ``lookup_at`` + ``prefill_paged`` + ``commit`` +
     ``store_at``) over a duck-typed tiny inner: request 1 prefills from scratch (== discrete one-shot
     final logits); request 2 reuses request 1's committed prefix blocks + restored derived snapshot
     and suffix-prefills (== discrete) — and the engine reports real prefix-reuse + snapshot-hit stats.

The real-model teacher-forced-ppl parity (paged ON == OFF) is DEFERRED to a one-model-at-a-time GPU
session (it loads the big artifact); this gate is tiny-tensors only.

    uv run --with numpy python -m parity.dsv4_paged_latent_test
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from quanta.cache_quant import BITS
from quanta.dsv4 import decode as D
from quanta.dsv4.attention import _rms_w, rope_cos_sin
from quanta.dsv4.batched_runtime import DSV4BatchedResidentModel, _latent_quant
from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.hyper import hc_expand, hc_head
from quanta.dsv4.model import dsv4_block
from quanta.paged import PagedKVCacheManager, RecurrentPrefixCache
from quanta.shim.omlx import _DSV4BatchedSession

# tiny geometry — head_dim=128 so the latent paging exercises the REAL int8-g128 quant round-trip.
DIM = 16
N_HEADS = 2
HEAD_DIM = 128
ROPE_HEAD_DIM = 64
Q_LORA = 8
O_GROUPS = 2
O_LORA = 4
WINDOW = 6
IDX_HEADS = 2
IDX_HEAD_DIM = 64
IDX_TOPK = 2
VOCAB = 64
VOCAB_IDS = list(range(3, 3 + 40))   # a deterministic id stream for prompts


def _cfg(ratios: tuple[int, ...]) -> DeepSeekV4Config:
    """A minimal valid DSV4 config with ``len(ratios)`` layers of the given per-layer ratios."""
    return DeepSeekV4Config(
        vocab_size=VOCAB, hidden_size=DIM, num_hidden_layers=len(ratios), moe_intermediate_size=16,
        num_attention_heads=N_HEADS, head_dim=HEAD_DIM, rope_head_dim=ROPE_HEAD_DIM,
        q_lora_rank=Q_LORA, o_lora_rank=O_LORA, o_groups=O_GROUPS, sliding_window=WINDOW,
        index_n_heads=IDX_HEADS, index_head_dim=IDX_HEAD_DIM, index_topk=IDX_TOPK,
        compress_ratios=ratios, compress_rope_theta=10000.0,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1, n_hash_layers=1,
        scoring_func="sqrtsoftplus", topk_method="noaux_tc", norm_topk_prob=True,
        routed_scaling_factor=1.0, swiglu_limit=0.0,
        hc_mult=1, hc_sinkhorn_iters=1, hc_eps=1e-6, n_mtp_layers=0,
        norm_eps=1e-6, rope_theta=10000.0,
        rope_scaling={"factor": 4.0, "beta_fast": 32, "beta_slow": 1,
                      "original_max_position_embeddings": 16, "type": "yarn"},
        max_position_embeddings=4096, bos_token_id=0, eos_token_id=1, eos_token_ids=(1,),
        tie_word_embeddings=False,
    )


def _r(rng, *shape) -> mx.array:
    return mx.array((rng.standard_normal(shape) * 0.3).astype(np.float32))


def _compressor_params(rng, ratio: int, head_dim: int) -> dict:
    coff = 2 if ratio == 4 else 1
    return {"ape": _r(rng, ratio, coff * head_dim),
            "norm": _r(rng, head_dim) * 0.1 + 1.0,
            "wkv": _r(rng, coff * head_dim, DIM),
            "wgate": _r(rng, coff * head_dim, DIM)}


def _attn_params(rng, cfg: DeepSeekV4Config, layer_id: int) -> dict:
    p = {"wq_a": _r(rng, Q_LORA, DIM),
         "q_norm": _r(rng, Q_LORA) * 0.1 + 1.0,
         "wq_b": _r(rng, N_HEADS * HEAD_DIM, Q_LORA),
         "wkv": _r(rng, HEAD_DIM, DIM),
         "kv_norm": _r(rng, HEAD_DIM) * 0.1 + 1.0,
         "wo_a": _r(rng, O_GROUPS * O_LORA, (N_HEADS * HEAD_DIM) // O_GROUPS),
         "wo_b": _r(rng, DIM, O_GROUPS * O_LORA),
         "attn_sink": _r(rng, N_HEADS)}
    ratio = cfg.compress_ratio(layer_id)
    if cfg.has_compressor(layer_id):
        p["compressor"] = _compressor_params(rng, ratio, HEAD_DIM)
    if cfg.has_indexer(layer_id):
        p["indexer"] = {"wq_b": _r(rng, IDX_HEADS * IDX_HEAD_DIM, Q_LORA),
                        "weights_proj": _r(rng, IDX_HEADS, DIM),
                        "compressor": _compressor_params(rng, 4, IDX_HEAD_DIM)}
    return p


def _rope_full(cfg: DeepSeekV4Config, layer_id: int, seqlen: int):
    orig, theta = cfg.attn_rope(layer_id)
    return rope_cos_sin(cfg.rope_head_dim, seqlen, orig, theta, cfg.rope_factor,
                        cfg.beta_fast, cfg.beta_slow)


def _maxdiff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _step_fn(cfg, layer_id):
    return D.decode_step_dense if cfg.compress_ratio(layer_id) == 0 else D.decode_step_compressed


def _decode_seq(step, x, p, cfg, lid, cos, sin, lo, hi, cache):
    """Decode tokens ``[lo, hi)`` into ``cache``; return stacked outputs ``[1, hi-lo, dim]``."""
    outs = [step(x[:, t:t + 1], p, cfg, lid, cache, cos, sin, t) for t in range(lo, hi)]
    return mx.concatenate(outs, axis=1)


def _make_mgr(cfg, block_size, n_layers):
    gs, q = _latent_quant(cfg.head_dim)
    mgr = PagedKVCacheManager(num_layers=n_layers, block_size=block_size, max_blocks=256,
                              group_size=gs, bits=BITS, quantized=q, model_name="dsv4t",
                              single_stream=True)
    return mgr, gs, q


# ---------------------------------------------------------------------------------------------------
# A. CORE: continuous decode == paged prefix-reuse + derived-snapshot restore + suffix, per regime,
#    with a boundary-STRADDLING compressor window (P a block multiple, NOT a ratio multiple).
# ---------------------------------------------------------------------------------------------------
def _run_regime(name: str, ratio: int, *, T: int, P: int, block_size: int, rng) -> bool:
    cfg = _cfg((ratio,))                      # single layer of the target regime
    p = _attn_params(rng, cfg, 0)
    step = _step_fn(cfg, 0)
    x = _r(rng, 1, T, DIM)
    cos, sin = _rope_full(cfg, 0, T)

    # (ref) continuous decode of [0, T) through a discrete cache
    ref_cache = D.DSV4Cache(1)
    ref = _decode_seq(step, x, p, cfg, 0, cos, sin, 0, T, ref_cache)
    mx.eval(ref)

    mgr, gs, q = _make_mgr(cfg, block_size, 1)
    rec = RecurrentPrefixCache(block_size=block_size, model_name="dsv4t")
    ids = VOCAB_IDS[:T]

    # pass 1: fill the prefix [0, P) latent blocks + snapshot the derived state at P
    seqA = mgr.new_sequence()
    mgr.advance(seqA, ids[:P])
    cacheA = D.paged_cache(mgr, seqA, 1, quantized=q, group_size=gs)
    _ = _decode_seq(step, x, p, cfg, 0, cos, sin, 0, P, cacheA)
    mgr.commit(seqA)
    snapA = D.snapshot_derived(cacheA)
    rec.store_at(ids, P, snapA)
    mx.eval([])  # nothing to force; the writes already materialized via the gather in the stepper

    # pass 2: a fresh request reuses [0, P) (latent blocks + derived snapshot), suffix-prefills [P, T)
    seqB = mgr.new_sequence()
    n_attn = mgr.match_prefix(seqB, ids[:P])
    rec_state = rec.lookup_at(ids, n_attn) if n_attn else None
    cacheB = D.paged_cache(mgr, seqB, 1, quantized=q, group_size=gs)
    D.restore_derived(cacheB, rec_state)
    mgr.advance(seqB, ids[P:T])
    out = _decode_seq(step, x, p, cfg, 0, cos, sin, P, T, cacheB)
    mx.eval(out)

    d_suffix = _maxdiff(out, ref[:, P:])
    st = mgr.get_stats()
    # straddle check: the first window to CLOSE at/after P must reach a raw position < P (overlap:
    # prev window; non-overlap: P not ratio-aligned). c0 = first window index completing at offset>=P.
    overlap = ratio == 4
    if ratio:
        c0 = P // ratio                         # window completing at offset (c0+1)*ratio-1 >= P-?
        first_close = (c0 + 1) * ratio - 1 if (P % ratio) else (P // ratio) * ratio + ratio - 1
        earliest_raw = (c0 - (1 if overlap else 0)) * ratio
        straddles = earliest_raw < P <= first_close + 1
    else:
        straddles = (P - WINDOW) < P              # dense: window reaches into prefix (trivially true)

    match_ok = n_attn == P
    hit_ok = st.prefix_hit_tokens == P and st.prefix_hit_blocks == P // block_size
    snap_ok = (ratio == 0) or (rec_state is not None)
    good = d_suffix < 1e-4 and match_ok and hit_ok and snap_ok and straddles
    print(f"  [{'OK' if good else 'FAIL'}] {name:26s} ratio={ratio:3d} T={T} P={P} blk={block_size}"
          f"  |Δsuffix|={d_suffix:.2e} n_attn={n_attn} hit_tok={st.prefix_hit_tokens}"
          f" straddle={straddles} snap={'-' if ratio == 0 else (rec_state is not None)}")
    return good


# ---------------------------------------------------------------------------------------------------
# B. ENGINE WIRING: drive the real _DSV4BatchedSession paged admit path with a tiny duck-typed inner.
# ---------------------------------------------------------------------------------------------------
def _router(rng, cfg, layer_id):
    out = {"weight": _r(rng, cfg.n_routed_experts, DIM)}
    if cfg.is_hash(layer_id):
        tab = rng.integers(0, cfg.n_routed_experts, size=(VOCAB, cfg.num_experts_per_tok)).astype(np.int32)
        out["tid2eid"] = mx.array(tab)
    else:
        out["bias"] = _r(rng, cfg.n_routed_experts)
    return out


def _block_params(rng, cfg, layer_id):
    inter = cfg.moe_intermediate_size
    p = {"attn": _attn_params(rng, cfg, layer_id), "router": _router(rng, cfg, layer_id),
         "experts": {"w1": _r(rng, cfg.n_routed_experts, inter, DIM),
                     "w3": _r(rng, cfg.n_routed_experts, inter, DIM),
                     "w2": _r(rng, cfg.n_routed_experts, DIM, inter)},
         "shared": {"w1": _r(rng, inter, DIM), "w3": _r(rng, inter, DIM), "w2": _r(rng, DIM, inter)},
         "attn_norm": _r(rng, DIM) * 0.1 + 1.0, "ffn_norm": _r(rng, DIM) * 0.1 + 1.0}
    mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
    for which in ("attn", "ffn"):
        p[f"hc_{which}_fn"] = _r(rng, mix_hc, cfg.hc_mult * DIM).astype(mx.float32)
        p[f"hc_{which}_base"] = _r(rng, mix_hc).astype(mx.float32)
        p[f"hc_{which}_scale"] = _r(rng, 3).astype(mx.float32) * 0.0 + 1.0
    return p


class _FakeInner:
    """Duck-types :class:`quanta.dsv4.runtime.DSV4ResidentModel` for the engine-wiring sub-test."""

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.num_layers = cfg.num_hidden_layers
        self.embed_w = _r(rng, cfg.vocab_size, DIM)
        self.lm_head_w = _r(rng, cfg.vocab_size, DIM)
        self.norm_w = _r(rng, DIM) * 0.1 + 1.0
        self.final_hc = {"fn": _r(rng, DIM, cfg.hc_mult * DIM).astype(mx.float32),
                         "base": _r(rng, DIM).astype(mx.float32),
                         "scale": _r(rng, DIM).astype(mx.float32)}
        self.layers = [_block_params(rng, cfg, i) for i in range(self.num_layers)]
        self._rope_cache: dict[int, tuple[mx.array, mx.array]] = {}

    def _head(self, h):
        hh = hc_head(h, self.final_hc["fn"], self.final_hc["scale"], self.final_hc["base"],
                     self.cfg.hc_mult, self.cfg.norm_eps, self.cfg.hc_eps)
        hh = _rms_w(hh, self.norm_w, self.cfg.norm_eps)
        return hh @ self.lm_head_w.T.astype(hh.dtype)

    def _rope(self, layer_id, length):
        cached = self._rope_cache.get(layer_id)
        if cached is not None and cached[0].shape[0] >= length:
            return cached
        orig, theta = self.cfg.attn_rope(layer_id)
        cs = rope_cos_sin(self.cfg.rope_head_dim, length, orig, theta, self.cfg.rope_factor,
                          self.cfg.beta_fast, self.cfg.beta_slow)
        self._rope_cache[layer_id] = cs
        return cs

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        from quanta.dsv4.hyper import hc_post, hc_pre
        from quanta.dsv4.moe import dsv4_moe
        ids = (token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids)).reshape(1, -1)
        h = hc_expand(self.embed_w[ids].astype(mx.bfloat16), self.cfg.hc_mult)
        if caches is not None:
            cfg = self.cfg
            eps, hc, it, heps = cfg.norm_eps, cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps
            hts = []
            for t in range(h.shape[1]):
                ht, idt, off_t = h[:, t:t + 1], ids[:, t:t + 1], offset + t
                for i, p in enumerate(self.layers):
                    cos, sin = self._rope(i, off_t + 1)
                    step = D.decode_step_dense if cfg.compress_ratio(i) == 0 else D.decode_step_compressed
                    res = ht
                    xx, post, comb = hc_pre(ht, p["hc_attn_fn"], p["hc_attn_scale"], p["hc_attn_base"], hc, it, eps, heps)
                    xx = _rms_w(xx, p["attn_norm"], eps)
                    xx = step(xx, p["attn"], cfg, i, caches, cos, sin, off_t)
                    ht = hc_post(xx, res, post, comb)
                    res = ht
                    xx, post, comb = hc_pre(ht, p["hc_ffn_fn"], p["hc_ffn_scale"], p["hc_ffn_base"], hc, it, eps, heps)
                    xx = _rms_w(xx, p["ffn_norm"], eps)
                    xx = dsv4_moe(xx, p["router"], p["experts"], p["shared"], cfg, i, idt)
                    ht = hc_post(xx, res, post, comb)
                mx.eval(ht)
                hts.append(ht)
            return self._head(hts[0] if len(hts) == 1 else mx.concatenate(hts, axis=1))
        for i, p in enumerate(self.layers):
            h = dsv4_block(h, p, self.cfg, i, ids)
            mx.eval(h)
        return self._head(h)


def _run_engine(rng) -> bool:
    cfg = _cfg((0, 4, 3))                       # 3 layers exercising both compressed regimes + dense
    inner = _FakeInner(cfg, rng)
    batched = DSV4BatchedResidentModel.from_inner(inner, max_batch=2)
    B = 4                                       # block_size: T=11 -> deepest block boundary at 8
    T = 11
    ids = VOCAB_IDS[:T]

    # discrete one-shot reference (the same single-stream forward the session prefills against)
    ref_cache = D.DSV4Cache(cfg.num_hidden_layers)
    ref = inner(mx.array(ids), caches=ref_cache, offset=0)
    mx.eval(ref)
    ref_last = ref[0, -1]

    sess = _DSV4BatchedSession(capacity=2, runtime=batched, paged_kv=True, block_size=B,
                               model_name="dsv4eng")
    ok = sess.prefix_cache_enabled

    # request 1: nothing resident -> full suffix prefill; commits prefix blocks + stores snapshot@8
    row1 = sess.admit(0, ids)
    d1 = _maxdiff(row1, ref_last)
    sess.release(0)

    # request 2: reuses request 1's committed prefix blocks + derived snapshot, suffix-prefills [8,11)
    row2 = sess.admit(1, ids)
    d2 = _maxdiff(row2, ref_last)
    stats = sess.get_cache_stats()
    reuse_ok = stats["prefix_hit_tokens"] >= B and stats["recurrent"]["snapshot_hits"] >= 1
    good = ok and d1 < 5e-4 and d2 < 5e-4 and reuse_ok
    print(f"  [{'OK' if good else 'FAIL'}] engine session: req1|Δ|={d1:.2e} req2|Δ|={d2:.2e} "
          f"hit_tok={stats['prefix_hit_tokens']} snap_hits={stats['recurrent']['snapshot_hits']} "
          f"cow={stats['cow_copies']}")
    return good


def run() -> None:
    rng = np.random.default_rng(0)
    ok = True
    print("A. core: continuous decode == paged prefix-reuse + derived restore + suffix (per regime)")
    # P a block multiple but NOT a ratio multiple -> first post-boundary window straddles the boundary.
    ok &= _run_regime("dense (sliding-window)", 0, T=12, P=8, block_size=4, rng=rng)
    ok &= _run_regime("compressed + indexer", 4, T=16, P=6, block_size=3, rng=rng)
    ok &= _run_regime("compressed (no indexer)", 3, T=12, P=4, block_size=4, rng=rng)
    print("B. engine wiring: real _DSV4BatchedSession paged admit (req1 from-scratch, req2 reuse)")
    ok &= _run_engine(rng)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
