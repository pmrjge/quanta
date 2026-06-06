"""Model-free parity gate for Nemotron-H fused batched-decode attention (Approach-1, the default path).

Proves :func:`quanta.nemotron.batched_runtime.batched_decode_step_fused` (one padded
:func:`mx.fast.scaled_dot_product_attention` per GQA layer across ``B`` streams) is equivalent to the
per-stream :func:`batched_decode_step` loop it replaces — on a tiny random-init :class:`NemotronModel`
(``M*EM``: mamba + attention + moe, NO checkpoint, NO GPU). Streams are seeded at **different prefill
lengths** (heterogeneous RoPE offsets — the case batched RoPE must get right) and decoded several steps.

The fusion ONLY changes the GQA attention — the Mamba layers stay per-stream (recurrent state can't
batch) and the MoE layer stays the stacked single call — so ``fused`` vs ``looped`` **isolates** the
attention change (the MoE stacking is identical on both sides). The looped path's own equivalence to the
single-stream decode is gated in ``parity/nemotron_batched_test.py``, so this gate closes the loop.

A. **fused == loop** — ``batched_decode_step_fused`` (form-1: batched Mamba via per-step concat + fused
   attention) vs ``batched_decode_step``, with ``BATCHED_FUSED_SSD_STEP`` **pinned False** (composed SSD
   step on both sides) so this isolates the **attention** fusion: ``B=1`` bit-exact (``L_max == L_1``, no
   padding) + ``B=3`` ragged greedy-exact (the Mamba batching is bit-exact; only the padded-SDPA tiling
   reorders the attention softmax → bf16 ULPs).
B. **native == fused** — ``batched_decode_step_native`` (form-2: persistent ``BatchedMambaState``, no
   per-step concat) vs form-1, at the **default** (graduated fused SSD step on both): **bit-exact** at
   ``B=1`` and ``B=3`` (identical math + identical deterministic kernel, only the recurrent state *storage*
   differs) + a ``BatchedMambaState.scatter_to`` round-trip back to per-stream slots.
B2. **graduated fused SSD step** — the batched native stepper with ``BATCHED_FUSED_SSD_STEP=True`` (the
   default fused one-launch decode kernel) == ``False`` (composed ``ssd_step``), isolating the step kernel
   (attention + MoE byte-identical): **greedy-exact**, ``|Δlogit|`` within the bf16-ULP reorder tol. This
   is the rule-4 output-equivalence proof for the graduated path (+36% @ B=32 — the breakdown bench);
   ``run()`` pins the default ON so a revert fails loud.
C. **dispatch** — ``NemotronBatchedResidentModel.step_batch`` (fused default, via ``from_inner``) vs the
   retained ``batched_decode_step`` reference (one step, ragged offsets).
D. **paged KV loop-kill** (#153-class) — ``_fused_attn_layer`` with ``PagedKVCacheView`` caches +
   ``paged_batched=True`` (ONE ``write_batched`` + ONE ``gather_batched`` over the shared manager, the
   paged sibling of the #18 arena) == ``paged_batched=False`` (the per-stream paged ``.update()`` loop),
   **bit-exact** (``max|Δ|=0``) across ragged streams + steps with block-boundary crossings — both end in
   the same padded SDPA, only the KV store write/read differs (M0 proved batched scatter/gather ==
   per-stream). GRADUATED to ON behind ``NemotronBatchedResidentModel._paged_kv_batched`` (←
   ``NEMOTRON_PAGED_KV_BATCHED_DEFAULT``) — parity-proven here + real-model greedy-exact & +18% @ B=48
   (``parity/nemotron_paged_batched_bench.py``); ``run()`` pins the default-ON (rule 4: a single revert flag).

Arbiter: **greedy-token agreement** (the decode that ships); logits match to bf16 ULPs (padded-SDPA
tiling reorder) — argmax-stable, the [[feedback-batched-rope-bf16]] equivalence class.

    uv run --with numpy python -m parity.nemotron_batched_attention_test
"""

from __future__ import annotations

import mlx.core as mx

import quanta.nemotron.mamba_mixer as mm
from parity.nemotron_batched_test import _randomize, _tiny_cfg
from quanta.cache_quant import BITS
from quanta.nemotron.batched_runtime import (
    NemotronBatchedResidentModel,
    _fused_attn_layer,
    batched_decode_step,
    batched_decode_step_fused,
    make_stream_state,
)
from quanta.nemotron.model import NemotronModel
from quanta.paged import PagedKVCacheManager

STEPS = 6
LOGIT_TOL = 5e-3  # bf16 padded-SDPA tiling reorder; the hard gate is greedy-token agreement


def _prompt(cfg, n: int) -> mx.array:
    return mx.random.randint(0, cfg.vocab_size, (n,))


def _seed(model: NemotronModel, prompt: mx.array) -> tuple[tuple[list, list, list], int]:
    """Prefill ``prompt`` into a fresh per-stream state (int8 KV via make_stream_state); return
    (state, first decode token). Two independent seeds per stream keep fused + looped from sharing
    mutated state."""
    state = make_stream_state(model.cfg)
    caches, ssm, conv = state
    logits, _, _ = model(prompt, caches=caches, ssm=ssm, conv=conv, use_fast=True)
    return state, int(mx.argmax(logits[0, -1]).item())


def _args(model: NemotronModel):
    return (model.layers, model.embed_tokens.weight, model.norm_f.weight, model.lm_head.weight,
            model.cfg.norm_eps)


def _core(b: int, lengths: list[int]) -> tuple[float, bool]:
    """fused == looped over STEPS decode steps at ragged offsets; return (worst |Δlogit|, greedy_match).

    Pins ``BATCHED_FUSED_SSD_STEP=False`` (composed SSD step on BOTH sides) so this stays an
    apples-to-apples **attention-fusion** isolation: the only difference is fused-SDPA vs per-stream SDPA
    (the Mamba step is identical), so ``B=1`` is bit-exact. The graduated fused SSD step is proven
    output-equivalent separately in :func:`_core_grad` (its own greedy-exact gate)."""
    mx.random.seed(0)
    cfg = _tiny_cfg()
    model = NemotronModel(cfg)
    _randomize(model)
    prompts = [_prompt(cfg, n) for n in lengths]
    loop = [_seed(model, p) for p in prompts]
    fuse = [_seed(model, p) for p in prompts]
    l_states = [s for s, _ in loop]
    l_tok = [t for _, t in loop]
    f_states = [s for s, _ in fuse]
    f_tok = [t for _, t in fuse]
    args = _args(model)

    worst = 0.0
    match = True
    saved = mm.BATCHED_FUSED_SSD_STEP
    mm.BATCHED_FUSED_SSD_STEP = False                      # composed both sides → isolate the attention fusion
    try:
        for _ in range(STEPS):
            lb = batched_decode_step(*args, [mx.array([t]) for t in l_tok], l_states)
            fb = batched_decode_step_fused(*args, [mx.array([t]) for t in f_tok], f_states)
            mx.eval(lb, fb)
            for s in range(b):
                lo, fo = lb[s][0, -1], fb[s][0, -1]
                worst = max(worst, float(mx.max(mx.abs(fo - lo)).item()))
                lt, ft = int(mx.argmax(lo).item()), int(mx.argmax(fo).item())
                match = match and (lt == ft)
                l_tok[s], f_tok[s] = lt, ft
    finally:
        mm.BATCHED_FUSED_SSD_STEP = saved
    return worst, match


class _FakeInner:
    """Ducks NemotronResidentModel's surface around a bf16/fp32 NemotronModel so
    NemotronBatchedResidentModel.from_inner can build a batched runtime model-free (no 120B artifact)."""

    def __init__(self, model: NemotronModel) -> None:
        self._m = model
        self.cfg = model.cfg
        self.layers = model.layers
        self.embed_w = model.embed_tokens.weight
        self.norm_f = model.norm_f.weight
        self.lm_head_w = model.lm_head.weight

    @property
    def num_layers(self) -> int:
        return len(self._m.layers)


def _dispatch() -> None:
    """runtime step_batch (fused default) == batched_decode_step (looped), one step, ragged offsets."""
    mx.random.seed(0)
    cfg = _tiny_cfg()
    model = NemotronModel(cfg)
    _randomize(model)
    rt = NemotronBatchedResidentModel.from_inner(_FakeInner(model), max_batch=4)
    assert rt._fused, "batched runtime must default to the fused decode path"

    lengths = [9, 5, 12]
    prompts = [_prompt(cfg, n) for n in lengths]
    fuse = [_seed(model, p) for p in prompts]
    loop = [_seed(model, p) for p in prompts]
    f_states, f_tok = [s for s, _ in fuse], [t for _, t in fuse]
    l_states, l_tok = [s for s, _ in loop], [t for _, t in loop]

    saved = mm.BATCHED_FUSED_SSD_STEP
    mm.BATCHED_FUSED_SSD_STEP = False                      # composed both sides → isolate dispatch + attn fusion
    try:
        fused = rt.step_batch([mx.array([t]) for t in f_tok], f_states)           # fused default (attn)
        looped = batched_decode_step(*_args(model), [mx.array([t]) for t in l_tok], l_states)
        mx.eval(fused, looped)
    finally:
        mm.BATCHED_FUSED_SSD_STEP = saved

    worst = 0.0
    match = True
    for s in range(len(prompts)):
        fo, lo = fused[s][0, -1], looped[s][0, -1]
        worst = max(worst, float(mx.max(mx.abs(fo - lo)).item()))
        match = match and (int(mx.argmax(fo).item()) == int(mx.argmax(lo).item()))
    ok = match and worst < LOGIT_TOL
    print(f"  [{'OK' if ok else 'XX'}] runtime step_batch fused==looped B={len(prompts)} "
          f"greedy_match={match} |Δlogit|={worst:.2e}")
    assert match, "runtime fused step_batch greedy tokens diverged from batched_decode_step"
    assert worst < LOGIT_TOL, f"runtime fused != looped: |Δlogit|={worst:.2e}"


def _core_native(b: int, lengths: list[int]) -> tuple[float, bool]:
    """form-2 native (persistent :class:`BatchedMambaState`) == form-1 fused, from identical seeds —
    expected **BIT-EXACT** (same batched mixer call + same fused SDPA; only the recurrent-state *storage*
    differs: persistent ``[B,...]`` vs reassembled each step). Also round-trips ``scatter_to`` back into
    the per-stream slots and asserts it restores the persistent rows bit-for-bit."""
    mx.random.seed(0)
    cfg = _tiny_cfg()
    model = NemotronModel(cfg)
    _randomize(model)
    rt = NemotronBatchedResidentModel.from_inner(_FakeInner(model), max_batch=max(b, 1))
    prompts = [_prompt(cfg, n) for n in lengths]
    fuse = [_seed(model, p) for p in prompts]
    nat = [_seed(model, p) for p in prompts]
    f_states, f_tok = [s for s, _ in fuse], [t for _, t in fuse]
    n_states, n_tok = [s for s, _ in nat], [t for _, t in nat]
    args = _args(model)
    nat_state = rt.make_batched_state(n_states)          # assemble [B,...] ssm/conv ONCE (form-2)

    worst = 0.0
    match = True
    for _ in range(STEPS):
        fb = batched_decode_step_fused(*args, [mx.array([t]) for t in f_tok], f_states)
        nb = rt.step_batch_native([mx.array([t]) for t in n_tok], nat_state)
        mx.eval(fb, nb)
        for s in range(b):
            fo, no = fb[s][0, -1], nb[s][0, -1]
            worst = max(worst, float(mx.max(mx.abs(fo - no)).item()))
            ft, nt = int(mx.argmax(fo).item()), int(mx.argmax(no).item())
            match = match and (ft == nt)
            f_tok[s], n_tok[s] = ft, nt

    # scatter_to round-trip: the live persistent rows must land bit-for-bit in the per-stream slots.
    nat_state.scatter_to(n_states)
    for i in [li for li, k in enumerate(cfg.layers_block_type) if k == "mamba"]:
        for s in range(b):
            d = float(mx.max(mx.abs(n_states[s][1][i] - nat_state.ssm[i][s:s + 1])).item())
            assert d == 0.0, f"scatter_to ssm mismatch at layer {i} stream {s}: {d:.2e}"
    return worst, match


def _core_grad(b: int, lengths: list[int]) -> tuple[float, bool]:
    """GRADUATION proof (rule 4): the batched native stepper with ``BATCHED_FUSED_SSD_STEP=True`` (the
    fused one-launch SSD decode kernel, the default/graduated path) == ``False`` (the composed ``ssd_step``),
    from identical seeds, over ``STEPS`` at ragged offsets. The ONLY difference is the SSD decode kernel —
    attention (fused SDPA) + MoE (stacked) are byte-identical on both sides — so this **isolates** the
    graduated step. ``ssd_step_fused`` is parity-exact (== ``ssd_step`` ~2e-7, gated in
    ``parity/nemotron_ssd_scan_kernel_test.py``), so the e2e result must be **greedy-exact** with
    ``|Δlogit|`` within the bf16-ULP reorder tolerance. Returns (worst ``|Δlogit|``, greedy_match)."""
    mx.random.seed(0)
    cfg = _tiny_cfg()
    model = NemotronModel(cfg)
    _randomize(model)
    rt = NemotronBatchedResidentModel.from_inner(_FakeInner(model), max_batch=max(b, 1))
    prompts = [_prompt(cfg, n) for n in lengths]
    comp = [_seed(model, p) for p in prompts]
    fuse = [_seed(model, p) for p in prompts]
    c_states, c_tok = [s for s, _ in comp], [t for _, t in comp]
    f_states, f_tok = [s for s, _ in fuse], [t for _, t in fuse]
    c_nat = rt.make_batched_state(c_states)               # composed-step state
    f_nat = rt.make_batched_state(f_states)               # fused-step state
    saved = mm.BATCHED_FUSED_SSD_STEP
    worst, match = 0.0, True
    try:
        for _ in range(STEPS):
            mm.BATCHED_FUSED_SSD_STEP = False
            cb = rt.step_batch_native([mx.array([t]) for t in c_tok], c_nat)
            mm.BATCHED_FUSED_SSD_STEP = True
            fb = rt.step_batch_native([mx.array([t]) for t in f_tok], f_nat)
            mx.eval(cb, fb)
            for s in range(b):
                co, fo = cb[s][0, -1], fb[s][0, -1]
                worst = max(worst, float(mx.max(mx.abs(fo - co)).item()))
                ct, ft = int(mx.argmax(co).item()), int(mx.argmax(fo).item())
                match = match and (ct == ft)
                c_tok[s], f_tok[s] = ct, ft
    finally:
        mm.BATCHED_FUSED_SSD_STEP = saved
    return worst, match


def _core_paged_loopkill(b: int, pre_len: list[int], steps: int) -> float:
    """D (#153-class paged KV loop-kill): ``_fused_attn_layer`` with paged ``PagedKVCacheView`` caches +
    ``paged_batched=True`` (ONE ``write_batched`` + ONE ``gather_batched``) == ``paged_batched=False`` (the
    per-stream paged ``.update()`` loop), **BIT-exact** across ``B`` ragged streams + ``steps`` decode
    steps. Two managers seeded identically with a raw k/v prefix isolate the KV materialization — same
    attention block, same q/k/v projection + RoPE, same padded SDPA; only the store write/read path
    differs. Block size 4 with ragged prefill makes decode cross block boundaries (interleaved block
    ids). Returns the worst ``|Δ|`` over the attention block's output hidden."""
    mx.random.seed(0)
    cfg = _tiny_cfg()
    model = NemotronModel(cfg)
    _randomize(model)
    attn_global = next(i for i, k in enumerate(cfg.layers_block_type) if k == "attention")
    blk = model.layers[attn_global]
    att = blk.mixer
    n_kv, hd = att.nkv, att.hd

    # bf16 KV: the loop-kill wiring (write_batched/gather_batched vs per-stream) is codec-agnostic, and
    # the QUANTIZED k/v batched==per-stream round-trip is already gated in parity/dsv4_paged_batched_test
    # (M0 _run_kv_pair). bf16 keeps this gate head_dim-agnostic (the tiny cfg's head_dim need not be a
    # multiple of a quant group_size). The reference path uses the SAME bf16 manager, so it's apples-to-
    # apples — only the store write/read path differs.
    def _mk() -> PagedKVCacheManager:
        return PagedKVCacheManager(num_layers=1, block_size=4, max_blocks=256, group_size=128,
                                   bits=BITS, quantized=False, model_name="nemo153", single_stream=False)

    ref_mgr, bat_mgr = _mk(), _mk()
    ref_seqs, bat_seqs = [], []
    for s in range(b):                                # identical raw k/v prefix written to BOTH managers
        k_pre = mx.random.normal((1, n_kv, pre_len[s], hd)).astype(mx.bfloat16)
        v_pre = mx.random.normal((1, n_kv, pre_len[s], hd)).astype(mx.bfloat16)
        for mgr, seqs in ((ref_mgr, ref_seqs), (bat_mgr, bat_seqs)):
            seq = mgr.new_sequence()
            mgr.advance(seq, list(range(pre_len[s])))
            mgr.write(seq, 0, k_pre, v_pre)
            seqs.append(seq)
    ref_views = [ref_mgr.view(ref_seqs[s], 0) for s in range(b)]    # per-stream paged loop reference
    bat_views = [bat_mgr.view(bat_seqs[s], 0) for s in range(b)]    # batched loop-kill

    worst = 0.0
    for t in range(steps):
        for s in range(b):                            # open the decode position on BOTH managers
            ref_mgr.advance(ref_seqs[s], [1000 + t])
            bat_mgr.advance(bat_seqs[s], [1000 + t])
        h = mx.random.normal((b, 1, cfg.hidden_size)).astype(mx.bfloat16)
        offs = [pre_len[s] + t for s in range(b)]
        ref = _fused_attn_layer(blk, h, offs, ref_views, paged_batched=False)   # per-stream .update() loop
        bat = _fused_attn_layer(blk, h, offs, bat_views, paged_batched=True)    # ONE scatter + ONE gather
        mx.eval(ref, bat)
        worst = max(worst, float(mx.max(mx.abs(ref - bat)).item()))
    return worst


def run() -> None:
    # Graduation pin (rule 4): the batched decode steppers default to the fused one-launch SSD step
    # (BATCHED_FUSED_SSD_STEP) — proven output-equivalent + a measured +36% @ B=32
    # (parity/nemotron_ultra_decode_step_breakdown.py). A revert that forgets must fail loud here.
    assert mm.BATCHED_FUSED_SSD_STEP is True, ("BATCHED_FUSED_SSD_STEP must default ON (graduated fused "
                                               "SSD decode step) — got False")

    print("A. batched_decode_step_fused == batched_decode_step (looped), ragged offsets:")
    w1, m1 = _core(1, [11])
    exact = "bit-exact" if w1 == 0.0 else f"|Δlogit|={w1:.2e}"
    print(f"  [{'OK' if (m1 and w1 < LOGIT_TOL) else 'XX'}] B=1 {exact} greedy_match={m1}")
    assert m1 and w1 < LOGIT_TOL, f"B=1 fused != looped: |Δlogit|={w1:.2e} match={m1}"

    w3, m3 = _core(3, [13, 7, 10])
    print(f"  [{'OK' if (m3 and w3 < LOGIT_TOL) else 'XX'}] B=3 offsets=[13,7,10] steps={STEPS} "
          f"greedy_match={m3} |Δlogit|={w3:.2e}")
    assert m3 and w3 < LOGIT_TOL, f"B=3 fused != looped: |Δlogit|={w3:.2e} match={m3}"

    print("B. batched_decode_step_native (form-2 persistent) == fused (form-1) — bit-exact + scatter_to:")
    nw1, nm1 = _core_native(1, [11])
    print(f"  [{'OK' if (nm1 and nw1 == 0.0) else 'XX'}] B=1 |Δlogit|={nw1:.2e} greedy_match={nm1} "
          "(persistent vs reassembled state must be identical)")
    assert nm1 and nw1 == 0.0, f"B=1 native != fused: |Δlogit|={nw1:.2e} match={nm1}"

    nw3, nm3 = _core_native(3, [13, 7, 10])
    print(f"  [{'OK' if (nm3 and nw3 == 0.0) else 'XX'}] B=3 offsets=[13,7,10] steps={STEPS} "
          f"|Δlogit|={nw3:.2e} greedy_match={nm3}")
    assert nm3 and nw3 == 0.0, f"B=3 native != fused: |Δlogit|={nw3:.2e} match={nm3}"

    print("B2. GRADUATED fused SSD decode step (BATCHED_FUSED_SSD_STEP True) == composed (False), "
          "native stepper — isolates the step kernel:")
    gw1, gm1 = _core_grad(1, [11])
    print(f"  [{'OK' if (gm1 and gw1 < LOGIT_TOL) else 'XX'}] B=1 |Δlogit|={gw1:.2e} greedy_match={gm1} "
          "(fused one-launch step == composed ssd_step, ~2e-7 fp32 reorder)")
    assert gm1 and gw1 < LOGIT_TOL, f"B=1 graduated fused step != composed: |Δlogit|={gw1:.2e} match={gm1}"

    gw3, gm3 = _core_grad(3, [13, 7, 10])
    print(f"  [{'OK' if (gm3 and gw3 < LOGIT_TOL) else 'XX'}] B=3 offsets=[13,7,10] steps={STEPS} "
          f"|Δlogit|={gw3:.2e} greedy_match={gm3}")
    assert gm3 and gw3 < LOGIT_TOL, f"B=3 graduated fused step != composed: |Δlogit|={gw3:.2e} match={gm3}"

    print("C. runtime dispatch: step_batch(fused default) == batched_decode_step:")
    _dispatch()

    print("D. paged KV loop-kill (#153): _fused_attn_layer paged_batched=True (ONE write_batched + ONE "
          "gather_batched) == per-stream paged .update() loop:")
    for tag, (b, lens) in (("B=1", (1, [6])), ("ragged B=3", (3, [9, 4, 11]))):
        w = _core_paged_loopkill(b, lens, steps=STEPS)
        ok = w == 0.0
        print(f"  [{'OK' if ok else 'XX'}] {tag:>10} blk=4 steps={STEPS}: max|Δ|={w:.2e} "
              f"(paged-batched == per-stream paged loop, bit-exact)")
        assert ok, f"paged loop-kill {tag} != per-stream paged loop: max|Δ|={w:.2e}"

    # The loop-kill has GRADUATED to ON for Nemotron (real-model greedy-exact + +18% @ B=48,
    # parity/nemotron_paged_batched_bench.py): a default-built runtime must engage it (rule 4 — proven,
    # behind a single revert flag). Pin it so a flag revert that forgets this gate fails loud.
    mx.random.seed(0)
    _m = NemotronModel(_tiny_cfg())
    _randomize(_m)
    _rt = NemotronBatchedResidentModel.from_inner(_FakeInner(_m), max_batch=2)
    assert _rt._paged_kv_batched is True, ("NEMOTRON_PAGED_KV_BATCHED_DEFAULT must default the paged KV "
                                           "loop-kill ON (graduated #153) — got _paged_kv_batched=False")
    print("  [OK] default-built runtime engages the paged KV loop-kill (_paged_kv_batched=True, graduated)")

    print("PASS — Nemotron batched Mamba (form-1 concat / form-2 persistent) + fused attention are "
          "per-stream-equivalent (Mamba bit-exact, attention greedy-exact); paged KV loop-kill bit-exact")


if __name__ == "__main__":
    run()
