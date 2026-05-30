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

    uv run python -m parity.dsv4_paged_batched_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.cache_quant import BITS
from quanta.dsv4.decode import _LayerCache, _pad_stack
from quanta.paged import PagedKVCacheManager

HEAD_DIM = 128
GROUP = 128


def _rand(shape: tuple[int, ...]) -> mx.array:
    """A bf16 normal draw from the current (seeded) RNG — caller seeds once for determinism."""
    return mx.random.normal(shape).astype(mx.bfloat16)


def _eq(a: mx.array, b: mx.array) -> bool:
    """Bit-exact array equality (shape + every element)."""
    return tuple(a.shape) == tuple(b.shape) and bool(mx.all(a == b).item())


def _mgr(*, single_stream: bool, block_size: int, name: str) -> PagedKVCacheManager:
    return PagedKVCacheManager(num_layers=1, block_size=block_size, max_blocks=256, group_size=GROUP,
                               bits=BITS, quantized=True, model_name=name, single_stream=single_stream)


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


def run() -> None:
    ok = True
    print("\n=== #153 batched-paged KV: ONE scatter/gather == per-stream paged loop (model-free) ===")
    ok &= _run_single_stream(block_size=3, label="single-stream latent int8 g128 hd=128")
    ok &= _run_kv_pair(block_size=3, label="k/v pair         int8 g128 hd=128")
    ok &= _run_cow(block_size=4)
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    run()
