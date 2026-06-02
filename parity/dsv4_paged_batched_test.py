"""Gate: #153 batched-paged KV — ONE block-table scatter/gather across ``B`` streams == the per-stream
paged loop, BIT-EXACT. MODEL-FREE (random tensors, no runtime load, no GPU weights).

Production serves the paged keepers (Nemotron, DSV4, InternLM2.5) through :class:`PagedKVCacheManager`,
whose batched decode still pays a per-stream Python loop: each of ``B`` lock-step streams quantizes its
one new token + slice-assigns it into its own tail block (``write`` / ``write_one``), then a separate
per-stream gather (``gather`` / ``gather_one``) + ``_pad_stack`` re-materializes every stream. #153
replaces that with ONE quantize + ONE fancy-index scatter (``write_*_batched``) + ONE ``mx.take`` gather
(``gather_*_batched``) over the streams' block tables — the paged sibling of the #18 arena loop-kill.
This gate proves the storage primitives are a faithful drop-in BEFORE any stepper is wired (M0), for
BOTH codecs the keepers use (so the SAME primitive serves all three paged keepers):

  1. **single-stream latent** (DSV4, ``single_stream=True``, int8 g128 head_dim=128): seed ``B``
     ragged-length streams via the per-stream prefill write (``write_one``), advance them with the
     batched scatter (``write_one_batched``), and assert each stream reads back (``gather_one``)
     BIT-IDENTICAL to an independent ``_LayerCache`` stream, and the batched read
     (``gather_one_batched``) == ``_pad_stack([gather_one ...])`` on every valid position.
  2. **k/v pair** (Nemotron / InternLM2.5, ``single_stream=False``, int8 g128 head_dim=128, n_kv=2):
     the same round-trip, batched (``write_batched`` / ``gather_batched``) == per-stream
     (``write`` / ``gather``), bit-exact across ragged lengths.
  3. **block-boundary crossing**: prefill lengths + decode steps straddle ``block_size``, so streams
     allocate fresh tail blocks mid-decode and their block ids interleave non-contiguously across
     streams — the test would catch any row/col index confusion in the scatter or the gather padding.
  4. **copy-on-write**: a forked stream's shared partial tail is COW-cloned by the batched writer
     BEFORE the scatter (it never touches a shared block, rule 6); both branches gather correctly and
     the parent's pre-fork stream is untouched.

The transitive anchor to the discrete caches (per-stream paged == ``_LayerCache`` / ``KVCache``) is
already gated in ``parity/dsv4_paged_latent_test.py`` (single) and the Nemotron paged tests (k/v); here
``_LayerCache`` is used directly as the single-stream ground truth, and the per-stream paged path is the
k/v ground truth.

**M1 — the dense decode stepper on the paged store.** ``_PagedKVArena`` (a thin per-layer adapter
presenting :class:`_KVArena`'s ``append_batched`` / ``read_batched`` over the manager's
``write_one_batched`` / ``gather_one_batched``) is driven through :func:`decode_step_dense_batched` and
asserted BIT-identical to the per-stream paged ``lcs`` loop (per-stream ``write_one`` + ``_pad_stack``)
across ``B`` ragged streams and several decode steps — proving the M0 primitive is a faithful drop-in
UNDER the stepper, not just in isolation. The latent read is gather+dequant with no SDPA reorder, and
the SDPA itself is the SAME batched call on both paths, so even ``B >= 2`` is exact (``Δ == 0``). Uses
the tiny DSV4 attn-param fixtures from ``parity.dsv4_batched_test`` (head_dim 8 → bf16 latent, which
also exercises the batched store's bf16 path that M0's int8 round-trip does not).

    uv run --with numpy python -m parity.dsv4_paged_batched_test
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from parity.dsv4_batched_test import _attn_params, _cfg, _r
from quanta.cache_quant import BITS
from quanta.dsv4.attention import rope_cos_sin
from quanta.dsv4.batched_runtime import _latent_quant
from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.decode import (
    _LayerCache,
    _PagedKVArena,
    _PagedLayerCache,
    _pad_stack,
    decode_step_compressed_batched,
    decode_step_dense_batched,
)
from quanta.paged import PagedKVCacheManager

HEAD_DIM = 128
GROUP = 128


def _rand(shape: tuple[int, ...]) -> mx.array:
    """A bf16 normal draw from the current (seeded) RNG — caller seeds once for determinism."""
    return mx.random.normal(shape).astype(mx.bfloat16)


def _eq(a: mx.array, b: mx.array) -> bool:
    """Bit-exact array equality (shape + every element)."""
    return tuple(a.shape) == tuple(b.shape) and bool(mx.all(a == b).item())


def _mgr(*, single_stream: bool, block_size: int, name: str,
         quantized: bool = True, n_layers: int = 1) -> PagedKVCacheManager:
    return PagedKVCacheManager(num_layers=n_layers, block_size=block_size, max_blocks=256, group_size=GROUP,
                               bits=BITS, quantized=quantized, model_name=name, single_stream=single_stream)


def _run_single_stream(*, block_size: int, label: str) -> bool:
    """Seed B ragged rows via per-stream ``write_one``, advance them via ``write_one_batched``, and
    assert each row + the batched read are bit-identical to the per-stream ``_LayerCache`` path."""
    mx.random.seed(0)
    b = 4
    pre_len = [5, 2, 8, 1]                 # ragged prefill lengths (straddle block_size)
    steps = 3                              # decode tokens appended to every stream
    d = HEAD_DIM
    pre = [_rand((1, pre_len[s], d)) for s in range(b)]
    dec = [_rand((1, steps, d)) for s in range(b)]

    # ground truth: B independent _LayerCache streams (prefill chunk, then token-by-token decode).
    lcs = [_LayerCache(quantized=True, group_size=GROUP) for _ in range(b)]
    for s in range(b):
        lcs[s].append_kv(pre[s])
    for t in range(steps):
        for s in range(b):
            lcs[s].append_kv(dec[s][:, t:t + 1])

    # paged manager: per-stream prefill write, then the batched hot-path scatter once per decode step.
    mgr = _mgr(single_stream=True, block_size=block_size, name="m153s")
    seqs = []
    for s in range(b):
        seq = mgr.new_sequence()
        mgr.advance(seq, list(range(pre_len[s])))      # token ids are irrelevant to the KV bytes here
        mgr.write_one(seq, 0, pre[s])
        seqs.append(seq)
    for t in range(steps):
        for s in range(b):
            mgr.advance(seqs[s], [1000 + t])
        kv_step = mx.concatenate([dec[s][:, t:t + 1] for s in range(b)], axis=0)   # [B, 1, D]
        mgr.write_one_batched(seqs, 0, kv_step)

    ok = True
    # (a) per-stream gather of the batched-written blocks == _LayerCache, bit-exact; lengths agree.
    for s in range(b):
        n = pre_len[s] + steps
        ok = ok and seqs[s].n_written[0] == lcs[s].kv_length() == n
        ok = ok and _eq(mgr.gather_one(seqs[s], 0), lcs[s].kv)
    # (b) batched read: [B, L_max, D]; every valid position == per-stream == _pad_stack.
    batched = mgr.gather_one_batched(seqs, 0)
    pad = _pad_stack([mgr.gather_one(seqs[s], 0) for s in range(b)])
    l_max = max(pre_len) + steps
    ok = ok and batched.shape[1] == l_max == pad.shape[1]
    for s in range(b):
        n = pre_len[s] + steps
        ok = ok and _eq(batched[s:s + 1, :n], lcs[s].kv)
        ok = ok and _eq(batched[s:s + 1, :n], pad[s:s + 1, :n])

    print(f"  [{'OK' if ok else 'FAIL'}] {label}: per-row + batched read bit-identical to _LayerCache "
          f"(B={b} lens={[n + steps for n in pre_len]} blk={block_size})")
    return ok


def _run_kv_pair(*, block_size: int, label: str) -> bool:
    """Same round-trip for the k/v-pair codec (Nemotron / InternLM2.5): the batched write+read
    (``write_batched`` / ``gather_batched``) == the per-stream paged loop (``write`` / ``gather``),
    bit-exact. Reference and batched managers see identical k/v; only the decode write path differs."""
    mx.random.seed(1)
    b = 4
    pre_len = [5, 2, 8, 1]
    steps = 3
    n_kv, d = 2, HEAD_DIM
    pre_k = [_rand((1, n_kv, pre_len[s], d)) for s in range(b)]
    pre_v = [_rand((1, n_kv, pre_len[s], d)) for s in range(b)]
    dec_k = [_rand((1, n_kv, steps, d)) for s in range(b)]
    dec_v = [_rand((1, n_kv, steps, d)) for s in range(b)]

    # reference: per-stream write for BOTH prefill and decode.
    ref = _mgr(single_stream=False, block_size=block_size, name="m153kv")
    ref_seqs = []
    for s in range(b):
        seq = ref.new_sequence()
        ref.advance(seq, list(range(pre_len[s])))
        ref.write(seq, 0, pre_k[s], pre_v[s])
        ref_seqs.append(seq)
    for t in range(steps):
        for s in range(b):
            ref.advance(ref_seqs[s], [1000 + t])
            ref.write(ref_seqs[s], 0, dec_k[s][:, :, t:t + 1], dec_v[s][:, :, t:t + 1])

    # batched: per-stream prefill, then the batched scatter per decode step.
    bat = _mgr(single_stream=False, block_size=block_size, name="m153kv")
    bat_seqs = []
    for s in range(b):
        seq = bat.new_sequence()
        bat.advance(seq, list(range(pre_len[s])))
        bat.write(seq, 0, pre_k[s], pre_v[s])
        bat_seqs.append(seq)
    for t in range(steps):
        for s in range(b):
            bat.advance(bat_seqs[s], [1000 + t])
        k_step = mx.concatenate([dec_k[s][:, :, t:t + 1] for s in range(b)], axis=0)   # [B, n_kv, 1, D]
        v_step = mx.concatenate([dec_v[s][:, :, t:t + 1] for s in range(b)], axis=0)
        bat.write_batched(bat_seqs, 0, k_step, v_step)

    ok = True
    # (a) per-stream gather: batched-written == ref-written, bit-exact; lengths agree.
    refs = []
    for s in range(b):
        n = pre_len[s] + steps
        ok = ok and bat_seqs[s].n_written[0] == ref_seqs[s].n_written[0] == n
        rk, rv = ref.gather(ref_seqs[s], 0)            # [1, n_kv, n, D]
        bk, bv = bat.gather(bat_seqs[s], 0)
        ok = ok and _eq(bk, rk) and _eq(bv, rv)
        refs.append((rk, rv))
    # (b) batched read [B, n_kv, L_max, D]: every valid position == per-stream gather.
    bk_all, bv_all = bat.gather_batched(bat_seqs, 0)
    l_max = max(pre_len) + steps
    ok = ok and bk_all.shape[2] == l_max and bv_all.shape[2] == l_max
    for s in range(b):
        n = pre_len[s] + steps
        rk, rv = refs[s]
        ok = ok and _eq(bk_all[s:s + 1, :, :n], rk) and _eq(bv_all[s:s + 1, :, :n], rv)

    print(f"  [{'OK' if ok else 'FAIL'}] {label}: batched k/v == per-stream paged loop "
          f"(B={b} n_kv={n_kv} lens={[n + steps for n in pre_len]} blk={block_size})")
    return ok


def _run_cow(*, block_size: int) -> bool:
    """A forked stream's shared PARTIAL tail block is COW-cloned by the batched writer before the
    scatter: both branches gather correctly, the parent's pre-fork stream is untouched, and at least one
    COW fired (the scatter never mutates a shared block, rule 6)."""
    mx.random.seed(2)
    d = HEAD_DIM
    pre_len = block_size + 1                            # 2 blocks; tail block partial -> shared on fork
    mgr = _mgr(single_stream=True, block_size=block_size, name="m153cow")
    parent = mgr.new_sequence()
    mgr.advance(parent, list(range(pre_len)))
    pre = _rand((1, pre_len, d))
    mgr.write_one(parent, 0, pre)
    parent_before = mgr.gather_one(parent, 0)          # snapshot before fork+write

    child = mgr.fork(parent)                            # shares every block incl. the partial tail (ref=2)
    cow0 = mgr.get_stats().cow_copies

    # batched decode ONE token into BOTH branches -> the shared partial tail must COW to private first.
    tok_p, tok_c = _rand((1, 1, d)), _rand((1, 1, d))
    mgr.advance(parent, [777])
    mgr.advance(child, [888])
    kv_step = mx.concatenate([tok_p, tok_c], axis=0)   # [2, 1, D]
    mgr.write_one_batched([parent, child], 0, kv_step)

    ok = mgr.get_stats().cow_copies >= cow0 + 1        # at least one COW fired
    ok = ok and _eq(mgr.gather_one(parent, 0)[:, :pre_len], parent_before)   # parent prefix untouched
    for seq, tok in ((parent, tok_p), (child, tok_c)):  # each branch == its own _LayerCache
        lc = _LayerCache(quantized=True, group_size=GROUP)
        lc.append_kv(pre)
        lc.append_kv(tok)
        ok = ok and _eq(mgr.gather_one(seq, 0), lc.kv)

    print(f"  [{'OK' if ok else 'FAIL'}] copy-on-write: forked shared tail cloned by the batched writer "
          f"(parent intact, both branches == _LayerCache, blk={block_size})")
    return ok


def _maxdiff(a: mx.array, b: mx.array) -> float:
    """Max abs fp32 difference (``0.0`` == bit-exact)."""
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _rope_full(cfg: DeepSeekV4Config, layer_id: int, length: int) -> tuple[mx.array, mx.array]:
    """Full per-layer RoPE tables for ``[0, length)`` — mirrors ``DSV4ResidentModel._rope``."""
    orig, theta = cfg.attn_rope(layer_id)
    return rope_cos_sin(cfg.rope_head_dim, length, orig, theta,
                        cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)


def _run_dense_stepper(*, block_size: int, pre_len: list[int], steps: int, label: str) -> bool:
    """M1: ``decode_step_dense_batched`` on the paged store via :class:`_PagedKVArena` (ONE block-table
    scatter + ONE gather) == the per-stream paged ``lcs`` loop (per-stream ``write_one`` + ``_pad_stack``),
    BIT-exact across ``B = len(pre_len)`` ragged streams and ``steps`` decode steps. Two managers are
    seeded identically with a raw-latent prefix so the comparison isolates the decode write/read path;
    each step both paths run the SAME ``decode_step_dense_batched`` (so the batched SDPA is identical),
    differing ONLY in how the latent window is materialized — hence the output is bit-identical (``Δ==0``,
    even ``B>=2``) and the per-stream written lengths agree."""
    mx.random.seed(3)
    cfg = _cfg()                                          # layer 0 is dense (RATIOS[0] == 0)
    rng = np.random.default_rng(11)
    p0 = _attn_params(rng, cfg, 0)
    gs, q = _latent_quant(cfg.head_dim)                   # the runtime's latent codec for this head_dim
    b = len(pre_len)
    d = cfg.head_dim
    pre = [_rand((1, pre_len[s], d)) for s in range(b)]   # identical raw-latent prefix per stream (like M0)

    # two managers seeded IDENTICALLY by the per-stream prefill write; only the decode path differs.
    ref_mgr = _mgr(single_stream=True, block_size=block_size, name="m153m1", quantized=q)
    bat_mgr = _mgr(single_stream=True, block_size=block_size, name="m153m1", quantized=q)
    ref_seqs, bat_seqs = [], []
    for s in range(b):
        for mgr, seqs in ((ref_mgr, ref_seqs), (bat_mgr, bat_seqs)):
            seq = mgr.new_sequence()
            mgr.advance(seq, list(range(pre_len[s])))     # open prefill positions (ids irrelevant to KV)
            mgr.write_one(seq, 0, pre[s])
            seqs.append(seq)

    lcs = [_PagedLayerCache(ref_mgr.view_one(ref_seqs[s], 0), quantized=q, group_size=gs)
           for s in range(b)]                             # per-stream paged loop reference (reads live)
    arena = _PagedKVArena(bat_mgr, bat_seqs, 0)           # batched paged adapter (reads live)
    rows = list(range(b))

    worst = 0.0
    len_ok = True
    for t in range(steps):
        offs = [pre_len[s] + t for s in range(b)]         # each stream's next absolute position
        for s in range(b):                                # open the decode position on BOTH managers
            ref_mgr.advance(ref_seqs[s], [9000 + s * 17 + t])
            bat_mgr.advance(bat_seqs[s], [9000 + s * 17 + t])
        x_b = mx.concatenate([_r(rng, 1, 1, cfg.hidden_size) for _ in range(b)], axis=0)   # [B,1,dim]
        cos, sin = _rope_full(cfg, 0, max(offs) + 1)
        ref = decode_step_dense_batched(x_b, p0, cfg, 0, lcs, cos, sin, offs)              # per-stream loop
        bat = decode_step_dense_batched(x_b, p0, cfg, 0, None, cos, sin, offs,
                                        arena=arena, rows=rows)                            # ONE scatter+gather
        mx.eval(ref, bat)
        worst = max(worst, _maxdiff(bat, ref))
        len_ok = len_ok and all(
            bat_seqs[s].n_written[0] == ref_seqs[s].n_written[0] == pre_len[s] + t + 1 for s in range(b))

    ok = worst == 0.0 and len_ok
    print(f"  [{'OK' if ok else 'FAIL'}] {label}: dense paged-batched == per-stream paged loop "
          f"(B={b} lens={[n + steps for n in pre_len]} steps={steps} blk={block_size} "
          f"max|Δ|={worst:.2e} len_ok={len_ok})")
    return ok


def _run_compressed_stepper(*, block_size: int, layer_id: int, pre_len: list[int], steps: int,
                            label: str) -> bool:
    """M2: ``decode_step_compressed_batched`` PAGED-HYBRID path — the LATENT batched through
    :class:`_PagedKVArena` (ONE block-table scatter + ONE gather) while the derived ckv/ikv/ring stay
    per-stream — == the per-stream paged ``lcs`` loop, BIT-exact across ``B`` ragged streams and
    ``steps`` decode steps that close compressor windows (ratio-4 +indexer / ratio-3). Both paths run
    the SAME per-stream :func:`_maybe_pool` on per-object :class:`_PagedLayerCache` derived state seeded
    identically, so the derived is bit-exact; the latent round-trip is M0/M1-bit-exact — hence ``Δ==0``
    even ``B>=2``, and the per-stream written latent lengths AND pooled-token counts (``n_comp``) agree.
    ``layer_id`` selects the compression regime from ``cfg``; the paged store keeps the latent at that
    layer index (a ``layer_id+1``-layer manager). Like :func:`_run_dense_stepper`, two managers are
    seeded identically with a raw-latent prefix so the comparison isolates the decode write/read path."""
    mx.random.seed(4)
    cfg = _cfg()
    rng = np.random.default_rng(12)
    p = _attn_params(rng, cfg, layer_id)                  # compressor (+ indexer on ratio-4) params
    gs, q = _latent_quant(cfg.head_dim)                   # the runtime's latent codec for this head_dim
    b = len(pre_len)
    d = cfg.head_dim
    pre = [_rand((1, pre_len[s], d)) for s in range(b)]   # identical raw-latent prefix per stream

    # two managers seeded IDENTICALLY by the per-stream prefill write; only the decode latent path differs.
    ref_mgr = _mgr(single_stream=True, block_size=block_size, name="m153m2", quantized=q,
                   n_layers=layer_id + 1)
    bat_mgr = _mgr(single_stream=True, block_size=block_size, name="m153m2", quantized=q,
                   n_layers=layer_id + 1)
    ref_seqs, bat_seqs = [], []
    for s in range(b):
        for mgr, seqs in ((ref_mgr, ref_seqs), (bat_mgr, bat_seqs)):
            seq = mgr.new_sequence()
            mgr.advance(seq, list(range(pre_len[s])))     # open prefill positions (ids irrelevant to KV)
            mgr.write_one(seq, layer_id, pre[s])
            seqs.append(seq)

    lcs_ref = [_PagedLayerCache(ref_mgr.view_one(ref_seqs[s], layer_id), quantized=q, group_size=gs)
               for s in range(b)]                         # per-stream paged loop reference
    lcs_bat = [_PagedLayerCache(bat_mgr.view_one(bat_seqs[s], layer_id), quantized=q, group_size=gs)
               for s in range(b)]                         # hybrid: derived ckv/ikv/ring ride these views
    arena = _PagedKVArena(bat_mgr, bat_seqs, layer_id)     # hybrid: latent via ONE scatter + ONE gather
    rows = list(range(b))

    # Seed each stream's raw-hidden ring IDENTICALLY for ref & bat (what a real prefill would leave), so
    # the first window-closing pool sees a full ``coff*ratio`` window — a fresh ring would underflow
    # _pool_one_window. The values are arbitrary: this gate proves ref==bat, not pool realism (the pool
    # arithmetic is gated in dsv4_batched_attention_test §B + dsv4_decode_attn_test).
    ratio = cfg.compress_ratio(layer_id)
    for s in range(b):
        seed_ring = _r(rng, 1, 2 * ratio, cfg.hidden_size)   # raw-hidden ring [1, 2*ratio, dim], f32
        lcs_ref[s].ring = seed_ring
        lcs_bat[s].ring = seed_ring

    worst = 0.0
    len_ok = True
    nc_ok = True
    for t in range(steps):
        offs = [pre_len[s] + t for s in range(b)]         # each stream's next absolute position
        for s in range(b):                                # open the decode position on BOTH managers
            ref_mgr.advance(ref_seqs[s], [9000 + s * 17 + t])
            bat_mgr.advance(bat_seqs[s], [9000 + s * 17 + t])
        x_b = mx.concatenate([_r(rng, 1, 1, cfg.hidden_size) for _ in range(b)], axis=0)   # [B,1,dim]
        cos, sin = _rope_full(cfg, layer_id, max(offs) + 1)
        ref = decode_step_compressed_batched(x_b, p, cfg, layer_id, lcs_ref, cos, sin, offs)  # per-stream
        bat = decode_step_compressed_batched(x_b, p, cfg, layer_id, lcs_bat, cos, sin, offs,
                                             arena=arena, rows=rows)             # latent ONE scatter+gather
        mx.eval(ref, bat)
        worst = max(worst, _maxdiff(bat, ref))
        len_ok = len_ok and all(
            bat_seqs[s].n_written[layer_id] == ref_seqs[s].n_written[layer_id] == pre_len[s] + t + 1
            for s in range(b))
        nc_ok = nc_ok and all(lcs_bat[s].n_comp() == lcs_ref[s].n_comp() for s in range(b))

    ok = worst == 0.0 and len_ok and nc_ok
    print(f"  [{'OK' if ok else 'FAIL'}] {label}: compressed paged-batched == per-stream paged loop "
          f"(B={b} L{layer_id} lens={[n + steps for n in pre_len]} steps={steps} blk={block_size} "
          f"max|Δ|={worst:.2e} len_ok={len_ok} nc_ok={nc_ok})")
    return ok


def run() -> None:
    ok = True
    print("\n=== #153 batched-paged KV: ONE scatter/gather == per-stream paged loop (model-free) ===")
    print("M0 — storage primitive (write_*_batched / gather_*_batched) == per-stream loop:")
    ok &= _run_single_stream(block_size=3, label="single-stream latent int8 g128 hd=128")
    ok &= _run_kv_pair(block_size=3, label="k/v pair         int8 g128 hd=128")
    ok &= _run_cow(block_size=4)
    print("M1 — dense decode stepper on the paged store (_PagedKVArena == per-stream paged lcs loop):")
    ok &= _run_dense_stepper(block_size=3, pre_len=[5, 2, 8, 1], steps=3, label="ragged B=4")
    ok &= _run_dense_stepper(block_size=4, pre_len=[6], steps=2, label="B=1 bit-exact")
    print("M2 — compressed decode stepper on the paged store (latent via _PagedKVArena, derived "
          "ckv/ikv/ring per-stream == per-stream paged lcs loop):")
    ok &= _run_compressed_stepper(block_size=3, layer_id=1, pre_len=[5, 2, 8, 1], steps=5,
                                  label="ratio-4 +indexer ragged B=4")
    ok &= _run_compressed_stepper(block_size=3, layer_id=2, pre_len=[5, 2, 8, 1], steps=5,
                                  label="ratio-3 no-indexer ragged B=4")
    ok &= _run_compressed_stepper(block_size=4, layer_id=1, pre_len=[6], steps=4,
                                  label="ratio-4 +indexer B=1 bit-exact")
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    run()
