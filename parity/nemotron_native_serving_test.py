"""Model-free gate: the form-2 **native** serving path == the form-1 path, through the real
``_BaseBatchedSession`` slot adapter (no GPU, no 120B artifact).

``NemotronBatchedResidentModel`` now exposes ``step_batch_native`` over a persistent
:class:`~quanta.nemotron.batched_runtime.BatchedMambaState` (form-2: the Mamba recurrence is held
batched ``[B,...]`` across decode steps, no per-step concat). ``_BaseBatchedSession`` caches that state
for the current alive-slot set and flushes/rebuilds it on any admit/release. This proves the session's
``native_decode=True`` path produces **identical sampled tokens** to ``native_decode=False`` (form-1
``step_batch``) — the runtime equivalence (form-2 == form-1, bit-exact) is gated separately in
``parity/nemotron_batched_attention_test.py``; here we gate the *serving plumbing* on top of it:

  A. **unpaged** — admit 3 distinct-prompt slots, decode, **release one + admit another** (forces a
     flush → scatter-back → rebuild of the batched state), decode more. Greedy tokens must match
     native-on vs native-off for every slot, proving the persistent-state bookkeeping reproduces the
     per-stream semantics across a mid-stream batch-composition change.
  B. **paged** — same but through the PagedKVCacheManager + RecurrentPrefixCache, decoding **across
     block boundaries** so the recurrent snapshot fires via ``BatchedMambaState.recurrent_row`` (form-2)
     instead of the per-stream triple. Tokens must match on vs off, and ≥1 decode-boundary snapshot must
     be stored (the recurrent_row path actually executed).

Runs on the tiny ``M*EM`` :class:`NemotronModel` (2 Mamba + 1 attention + 1 MoE) wrapped by
``from_inner`` — the SAME real runtime under both sessions, so weights are identical.

    uv run --with numpy python -m parity.nemotron_native_serving_test
"""

from __future__ import annotations

import mlx.core as mx

from parity.nemotron_batched_test import _randomize, _tiny_cfg
from quanta.nemotron.batched_runtime import NemotronBatchedResidentModel
from quanta.nemotron.model import NemotronModel
from quanta.paged import PagedKVCacheManager, RecurrentPrefixCache
from quanta.shim.omlx import _NemotronBatchedSession

BLOCK = 4


class _FakeInner:
    """Ducks NemotronResidentModel around a tiny NemotronModel so from_inner builds a real batched
    runtime model-free. ``__call__`` delegates prefill to the tiny model, swallowing the resident-only
    kwargs (``compiled`` / ``mamba_chunked_cont`` / ``capture_layers``) — both sessions share this exact
    prefill, so swallowing chunked-cont can't break the native-on vs native-off equivalence."""

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

    def __call__(self, token_ids, *, caches=None, ssm=None, conv=None, use_fast=True,
                 compiled=True, mamba_chunked_cont=False, capture_layers=None, **_ignored):
        return self._m(token_ids, caches=caches, ssm=ssm, conv=conv, use_fast=use_fast)


def _amax(row: mx.array) -> int:
    return int(mx.argmax(row).item())


def _build_rt():
    mx.random.seed(0)
    cfg = _tiny_cfg()
    model = NemotronModel(cfg)
    _randomize(model)
    return NemotronBatchedResidentModel.from_inner(_FakeInner(model), max_batch=4), cfg


def _drive(session, cfg) -> dict[int, list[int]]:
    """Fixed admit/step/release script (greedy). Returns slot -> sampled-token list. The release+admit in
    the middle changes the alive-slot set, exercising the native path's flush/scatter/rebuild."""
    bos = cfg.bos_token_id
    rec: dict[int, list[int]] = {}
    cur: dict[int, int] = {}

    def adm(slot: int, prompt: list[int]) -> None:
        t = _amax(session.admit(slot, prompt))
        rec.setdefault(slot, []).append(t)
        cur[slot] = t

    def stp(slots: list[int]) -> None:
        out = session.step_batch({s: cur[s] for s in slots})
        for s in slots:
            t = _amax(out[s])
            rec[s].append(t)
            cur[s] = t

    adm(0, [bos, 1, 2, 3, 4, 5])
    adm(1, [bos, 6, 7, 8])
    adm(2, [bos, 9, 10, 11, 12, 13, 14])
    for _ in range(6):
        stp([0, 1, 2])
    session.release(1)
    cur.pop(1, None)
    adm(3, [bos, 15, 16, 17, 18])
    for _ in range(6):
        stp([0, 2, 3])
    return rec


def _unpaged() -> None:
    rt, cfg = _build_rt()
    off = _NemotronBatchedSession(runtime=rt, capacity=4, native_decode=False)
    on = _NemotronBatchedSession(runtime=rt, capacity=4, native_decode=True)
    assert not off._native_decode and on._native_decode, "native_decode flag not wired through session"
    r_off = _drive(off, cfg)
    r_on = _drive(on, cfg)
    same = r_off == r_on
    print(f"  [{'OK' if same else 'XX'}] unpaged native==form-1 over admit/step/release/admit "
          f"(slots={sorted(r_off)}, steps/slot≈{len(r_off[0])})")
    if not same:
        for s in sorted(r_off):
            if r_off[s] != r_on.get(s):
                print(f"      slot {s}: off={r_off[s]} on={r_on.get(s)}")
    assert same, "unpaged native serving diverged from form-1"


def _mk_paged(rt, cfg, native: bool):
    spec = rt.paged_kv_spec
    # The tiny model's head_dim (32) < the default KV group_size (128) that paged_kv_spec reports, so cap
    # the manager's group_size to head_dim (mirrors make_stream_state's cap) — purely a tiny-model test
    # concern; real models (head_dim=128) take spec["group_size"] unchanged.
    gs = min(int(spec["group_size"]), int(cfg.head_dim))
    mgr = PagedKVCacheManager(num_layers=int(spec["n_layers"]), block_size=BLOCK, max_blocks=256,
                              group_size=gs, bits=int(spec["bits"]),
                              quantized=bool(spec["quantized"]), model_name="native-serving-test")
    rec = RecurrentPrefixCache(block_size=BLOCK, model_name="native-serving-test", capacity=256)
    sess = _NemotronBatchedSession(runtime=rt, capacity=4, manager=mgr, rec_cache=rec,
                                   native_decode=native)
    return sess, mgr, rec


def _paged() -> None:
    rt, cfg = _build_rt()
    off, _moff, roff = _mk_paged(rt, cfg, native=False)
    on, _mon, ron = _mk_paged(rt, cfg, native=True)
    assert on._native_decode and not off._native_decode, "paged native_decode flag not wired"
    r_off = _drive(off, cfg)
    r_on = _drive(on, cfg)
    same = r_off == r_on
    # decode crossed block boundaries (prompts of len 6/4/7 then +6 steps) ⇒ recurrent_row snapshots
    snap_on = ron.get_stats().snapshot_stores
    snap_off = roff.get_stats().snapshot_stores
    print(f"  [{'OK' if same else 'XX'}] paged native==form-1 over admit/step/release/admit "
          f"(snapshot_stores on={snap_on} off={snap_off})")
    if not same:
        for s in sorted(r_off):
            if r_off[s] != r_on.get(s):
                print(f"      slot {s}: off={r_off[s]} on={r_on.get(s)}")
    assert same, "paged native serving diverged from form-1"
    assert snap_on >= 1, "native paged decode stored no recurrent-boundary snapshot — recurrent_row path never ran"
    assert snap_on == snap_off, (f"snapshot count differs native={snap_on} form-1={snap_off} — the "
                                 "recurrent_row path is not equivalent to get_recurrent_state")


def run() -> None:
    print("A. unpaged: _NemotronBatchedSession(native_decode) == form-1, with mid-stream release+admit:")
    _unpaged()
    print("B. paged: same through PagedKVCacheManager + RecurrentPrefixCache, across block boundaries:")
    _paged()
    print("PASS — form-2 native serving path is token-identical to form-1 (unpaged + paged)")


if __name__ == "__main__":
    run()
