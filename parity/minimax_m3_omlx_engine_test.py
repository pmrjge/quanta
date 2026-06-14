"""Gate: the oMLX shim drives MiniMax-M3-VL through its own decode stepper + batched session — NO model.

M3-7b wires ``minimax_m3_vl`` into ``quanta.shim.omlx``. The serving hazard this gate exists to catch:
``minimax_m3_vl``.startswith("minimax") is True, so without an M3-first branch the M2.7 ``minimax``
route would silently build the WRONG runtime/cache (a different architecture + a different bos/eos set).
Two M3-specific decode shapes also need bespoke adapters: (a) ``MiniMaxM3ResidentModel.__call__`` reads
the absolute position from each per-layer ``KVCache.offset`` — it has NO ``offset=`` arg, unlike the
GLM/M2.7/Qwen3.5 runtimes the generic ``_SingleTokenStepper`` serves; (b) ``make_caches()`` returns a
per-LAYER cache *list*, so a batched slot's state is ``list[KVCache]`` and the step offset is read off
layer 0. This gates, against tiny stub runtimes (~0 GB — safe alongside a resident job), that:

  (0) ``detect_quanta_artifact`` reads ``minimax_m3_vl`` (top-level model_type; the text_config carries
      none) — the string the routes key off.
  (1) ``_default_runtime_loader`` routes ``minimax_m3_vl`` → the M3 runtime + ``from_pretrained_m3``
      tokenizer (monkeypatched stubs), NOT swallowed by the M2.7 ``minimax`` branch; ``minimax_m2`` still
      routes to the M2.7 runtime (no regression — disjoint both ways).
  (2) ``_make_stepper`` routes ``minimax_m3_vl`` → ``_MiniMaxM3Stepper`` and ``minimax_m2`` →
      ``_SingleTokenStepper`` (+ MiniMaxCache) — the order is correct at the stepper level too.
  (3) ``_MiniMaxM3Stepper`` single-stream: ``stream_generate`` over a fake M3 runtime threads ONE per-
      layer cache list WITHOUT passing ``offset=`` (the fake's ``__call__`` rejects it — a TypeError
      would surface), emits RAW markers for oMLX's parsers, stops on eos; a prompt ``>= chunk_from``
      routes through ``prefill_chunked`` (the M3-5 long-admit path), below it the bit-exact single-shot.
  (4) ``_make_batched_session`` routes ``minimax_m3_vl`` → ``_MiniMaxM3BatchedSession`` (monkeypatched
      batched runtime); the slot adapter passes plain int tokens + per-stream **layer-0** offsets.
  (5) ``batched_stream_generate`` over a stub batched runtime is output-equivalent to single-stream on
      the same prompt (rule 4), freed slots admit the next pending prompt (continuous batching), and the
      per-slot state is the ``list[KVCache]`` shape (offsets read off ``state[0].offset``).
  (6) rule-6 guards: an empty prompt fails loud; an unknown model_type has no stepper.

    uv run python -m parity.minimax_m3_omlx_engine_test
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx

from quanta.shim.omlx import (
    OmlxShimError,
    QuantaOmlxEngine,
    _MiniMaxM3BatchedSession,
    _MiniMaxM3Stepper,
    _RenderChatAdapter,
    _SingleTokenStepper,
    _default_runtime_loader,
    detect_quanta_artifact,
)

EOS = 11
VOCAB = 200
# token id -> literal piece (M3 emits raw markers; the shim's parsers run downstream, not in the engine)
MARKERS = {100: "<mm:think>", 101: "reason", 102: "</mm:think>", 103: "answer", 104: "done"}
# argmax by ABSOLUTE position: prompt encodes to [5,6,7] (len 3); last prompt pos = 2 predicts the first
# generated token, then 3,4,5,... predict the next. Single-stream and batched share this chain.
PRED = {2: 100, 3: 101, 4: 102, 5: 103, 6: 104, 7: EOS}
EXPECT_TOKENS = [100, 101, 102, 103, 104, EOS]
EXPECT_TEXT = "<mm:think>reason</mm:think>answerdone"

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
    assert cond, msg
    _N += 1


def _row(nxt: int) -> mx.array:
    return (mx.arange(VOCAB) == nxt).astype(mx.float32) * 60.0 - 30.0


def _m3_artifact(model_type: str = "minimax_m3_vl") -> str:
    """Synthetic quanta artifact dir: top-level model_type (the M3 text_config carries none), manifest.
    A few bytes; no weights."""
    d = Path(tempfile.mkdtemp(prefix="m3omlx_"))
    (d / "config.json").write_text(json.dumps({"model_type": model_type,
                                               "text_config": {}, "vision_config": {}}))
    (d / "manifest.json").write_text(json.dumps({"format": "quanta", "tensors": {}}))
    return str(d)


# --- fakes ------------------------------------------------------------------------------------------
class _FakeKV:
    """Per-layer KV cache stand-in: only the grown ``offset`` (M3 reads the position from here)."""

    def __init__(self) -> None:
        self.offset = 0


class _FakeM3Runtime:
    """Stands in for MiniMaxM3ResidentModel. ``__call__`` takes (token_ids, *, caches, use_fast, sparse)
    — NOTABLY NO ``offset=`` arg: the per-layer cache list owns the offset. If the stepper passed
    ``offset=`` this would raise TypeError. argmax at absolute last position is ``PRED[off+t-1]``."""

    def __init__(self, n_layers: int = 2) -> None:
        self.num_layers = n_layers
        self.chunked_calls = 0
        self.calls: list[tuple[str, int, int]] = []   # (phase, t, offset_before)

    def make_caches(self) -> list[_FakeKV]:
        return [_FakeKV() for _ in range(self.num_layers)]

    def _logits(self, t: int, off: int) -> mx.array:
        nxt = PRED.get(off + t - 1, EOS)
        return mx.broadcast_to(_row(nxt), (1, t, VOCAB))

    def __call__(self, token_ids, *, caches, use_fast: bool = True, sparse: bool = True) -> mx.array:
        t = int(mx.array(token_ids).reshape(-1).shape[0])
        off = caches[0].offset
        self.calls.append(("step" if t == 1 else "prefill", t, off))
        out = self._logits(t, off)
        for c in caches:
            c.offset = off + t
        return out

    def prefill_chunked(self, token_ids, *, caches, chunk_tokens: int = 4096,
                        use_fast: bool = True, sparse: bool = True) -> mx.array:
        self.chunked_calls += 1
        t = int(mx.array(token_ids).reshape(-1).shape[0])
        off = caches[0].offset
        for c in caches:
            c.offset = off + t
        return self._logits(1, off + t - 1)            # last-position logits [1,1,vocab]


class _FakeM3BatchedRuntime:
    """Stub MiniMaxM3BatchedResidentModel exposing the M3 batched contract the session drives:
    ``make_caches()`` -> per-LAYER list, ``prefill(ids, state)``, ``step_batch(tokens, caches, offsets)``.
    Asserts ``offsets[i] == caches[i][0].offset`` (the session's layer-0 offset override is correct) and
    that each token is a plain int (not an mx.array). Output mirrors the single-stream PRED chain ⇒
    batched is output-equivalent to single-stream (rule 4). ``prefills`` counts admits (lifecycle)."""

    def __init__(self, n_layers: int = 2) -> None:
        self.num_layers = n_layers
        self.prefills = 0
        self.steps: list[tuple[int, ...]] = []

    def make_caches(self) -> list[_FakeKV]:
        return [_FakeKV() for _ in range(self.num_layers)]

    def prefill(self, prompt_ids, state: list[_FakeKV]) -> mx.array:
        ids = list(prompt_ids)
        if not all(isinstance(t, int) for t in ids):
            raise AssertionError("M3 prefill expects plain int ids (the session's _to_prefill_ids)")
        n = len(ids)
        for c in state:
            c.offset = n
        self.prefills += 1
        return mx.broadcast_to(_row(PRED.get(n - 1, EOS)), (1, 1, VOCAB))

    def step_batch(self, stream_token_ids, stream_caches, offsets) -> list[mx.array]:
        toks = [int(t) for t in stream_token_ids]
        if not all(isinstance(t, int) for t in stream_token_ids):
            raise AssertionError("M3 step_batch expects plain int token ids (the session's _to_step_tokens)")
        for i, st in enumerate(stream_caches):
            if int(offsets[i]) != int(st[0].offset):
                raise AssertionError(
                    f"offset[{i}]={offsets[i]} != caches[{i}][0].offset={st[0].offset} "
                    "(the session must read the layer-0 offset)")
        self.steps.append(tuple(toks))
        out: list[mx.array] = []
        for st in stream_caches:
            off = st[0].offset
            for c in st:
                c.offset = off + 1
            out.append(mx.broadcast_to(_row(PRED.get(off, EOS)), (1, 1, VOCAB)))
        return out


class _FakeTok:
    """Minimal tokenizer: fixed 3-token prompt, marker detok via MARKERS. No decode_bytes/n_base ⇒ the
    streaming detok takes its string-fallback path (M3's render_chat tokenizer emits markers verbatim)."""

    eos_id = EOS
    stop_ids = (EOS,)
    bos_id = 1

    def encode(self, text, *, add_bos=False, allow_special=False):
        return [5, 6, 7]

    def decode(self, ids, **kw):
        return "".join(MARKERS.get(int(i), "") for i in ids)


async def _collect(engine, prompt, **kw):
    return [o async for o in engine.stream_generate(prompt, **kw)]


# --- (0)+(1) artifact detection + runtime-loader dispatch order ------------------------------------
class _M3RuntimeStub:
    def __init__(self, root):
        self.root = root
        self.kind = "m3"
        self.num_layers = 2

    def make_caches(self):
        return [_FakeKV(), _FakeKV()]


class _M2RuntimeStub:
    def __init__(self, root):
        self.root = root
        self.kind = "m2"
        self.num_layers = 2


def _check_loader_dispatch() -> None:
    print("\n--- (0)+(1) detect + runtime-loader dispatch order (M3 not swallowed by M2.7) ---")
    import quanta.minimax.runtime as _m2rt
    import quanta.minimax.runtime_m3 as _m3rt
    import quanta.minimax.tokenizer as _tokmod

    art_m3 = _m3_artifact("minimax_m3_vl")
    art_m2 = _m3_artifact("minimax_m2")
    info_m3 = detect_quanta_artifact(art_m3)
    info_m2 = detect_quanta_artifact(art_m2)
    _ck(info_m3 is not None and info_m3.model_type == "minimax_m3_vl",
        f"detect minimax_m3_vl: {info_m3.model_type if info_m3 else None}")
    _ck(info_m2 is not None and info_m2.model_type == "minimax_m2",
        f"detect minimax_m2: {info_m2.model_type if info_m2 else None}")

    saved = {
        "m3rt": _m3rt.MiniMaxM3ResidentModel,
        "m2rt": _m2rt.MiniMaxResidentModel,
        "fp_m3": _tokmod.MiniMaxTokenizer.from_pretrained_m3,
        "fp_m2": _tokmod.MiniMaxTokenizer.from_pretrained,
    }
    try:
        _m3rt.MiniMaxM3ResidentModel = _M3RuntimeStub
        _m2rt.MiniMaxResidentModel = _M2RuntimeStub
        _tokmod.MiniMaxTokenizer.from_pretrained_m3 = classmethod(lambda cls, p: _FakeTok())
        _tokmod.MiniMaxTokenizer.from_pretrained = classmethod(lambda cls, p: _FakeTok())

        rt_m3, tok_m3 = _default_runtime_loader(Path(art_m3))
        rt_m2, tok_m2 = _default_runtime_loader(Path(art_m2))
        _ck(isinstance(rt_m3, _M3RuntimeStub) and getattr(rt_m3, "kind", None) == "m3",
            f"minimax_m3_vl → M3 runtime (not swallowed by M2.7): {type(rt_m3).__name__}")
        _ck(isinstance(tok_m3, _RenderChatAdapter),
            f"minimax_m3_vl tokenizer wrapped in _RenderChatAdapter: {type(tok_m3).__name__}")
        _ck(isinstance(rt_m2, _M2RuntimeStub) and getattr(rt_m2, "kind", None) == "m2",
            f"minimax_m2 → M2.7 runtime (no regression): {type(rt_m2).__name__}")
    finally:
        _m3rt.MiniMaxM3ResidentModel = saved["m3rt"]
        _m2rt.MiniMaxResidentModel = saved["m2rt"]
        _tokmod.MiniMaxTokenizer.from_pretrained_m3 = saved["fp_m3"]
        _tokmod.MiniMaxTokenizer.from_pretrained = saved["fp_m2"]
        shutil.rmtree(art_m3, ignore_errors=True)
        shutil.rmtree(art_m2, ignore_errors=True)


# --- (2) stepper dispatch on model_type -----------------------------------------------------------
def _check_stepper_dispatch() -> None:
    print("\n--- (2) _make_stepper dispatch (minimax_m3 → _MiniMaxM3Stepper, minimax_m2 → MiniMaxCache) ---")
    art_m3 = _m3_artifact("minimax_m3_vl")
    art_m2 = _m3_artifact("minimax_m2")
    try:
        eng_m3 = QuantaOmlxEngine(art_m3, runtime=_FakeM3Runtime(), tokenizer=_FakeTok(),
                                  eos_token_ids={EOS})
        st_m3 = eng_m3._make_stepper(quantized_kv=True)
        _ck(isinstance(st_m3, _MiniMaxM3Stepper),
            f"minimax_m3_vl → {type(st_m3).__name__}")

        # M2.7 routes to _SingleTokenStepper with a MiniMaxCache — the runtime needs num_layers.
        class _M2Rt:
            num_layers = 2

            def __call__(self, *a, **k):
                return mx.zeros((1, 1, VOCAB))
        eng_m2 = QuantaOmlxEngine(art_m2, runtime=_M2Rt(), tokenizer=_FakeTok(), eos_token_ids={EOS})
        st_m2 = eng_m2._make_stepper(quantized_kv=True)
        from quanta.minimax.decode import MiniMaxCache
        _ck(isinstance(st_m2, _SingleTokenStepper) and isinstance(st_m2._cache, MiniMaxCache),
            f"minimax_m2 → {type(st_m2).__name__} + {type(st_m2._cache).__name__}")
    finally:
        shutil.rmtree(art_m3, ignore_errors=True)
        shutil.rmtree(art_m2, ignore_errors=True)


# --- (3) single-stream stepper: threading + raw markers + eos + chunked routing --------------------
def _check_single_stream() -> None:
    print("\n--- (3) _MiniMaxM3Stepper single-stream (cache-offset threading, raw markers, eos) ---")
    art = _m3_artifact("minimax_m3_vl")
    try:
        fake = _FakeM3Runtime()
        eng = QuantaOmlxEngine(art, runtime=fake, tokenizer=_FakeTok(), eos_token_ids={EOS})
        outs = asyncio.run(_collect(eng, "hi", max_tokens=20, temperature=0.0))
        last = outs[-1]
        # prefill seeds the whole prompt in ONE cached forward (t=3, off=0), then single-token steps.
        threaded = (fake.calls and fake.calls[0] == ("prefill", 3, 0)
                    and all(c[0] == "step" and c[1] == 1 for c in fake.calls[1:]))
        _ck(last.tokens == EXPECT_TOKENS and last.finished and last.finish_reason == "stop",
            f"stream_generate tokens={last.tokens} finish={last.finish_reason!r}")
        _ck(last.text == EXPECT_TEXT and "<mm:think>" in last.text and "</mm:think>" in last.text,
            f"raw markers pass through verbatim: {last.text!r}")
        _ck(threaded, f"per-layer cache threaded WITHOUT offset= (prefill-then-steps): {fake.calls}")
        _ck(fake.chunked_calls == 0, "short prompt uses the bit-exact single-shot path (no chunked)")

        # chunked routing: a direct stepper with a low chunk_from sends the prompt through prefill_chunked
        rc = _FakeM3Runtime()
        st_chunk = _MiniMaxM3Stepper(rc, chunk_from=2)
        _ = st_chunk.prefill([5, 6, 7])
        _ck(rc.chunked_calls == 1, f"prompt len 3 >= chunk_from 2 → prefill_chunked (M3-5): {rc.chunked_calls}")
        rs = _FakeM3Runtime()
        st_single = _MiniMaxM3Stepper(rs, chunk_from=10)
        _ = st_single.prefill([5, 6, 7])
        _ck(rs.chunked_calls == 0, "prompt len 3 < chunk_from 10 → single-shot (no chunked)")
        # empty prompt fails loud (rule 6)
        try:
            _MiniMaxM3Stepper(_FakeM3Runtime()).prefill([])
            loud = False
        except OmlxShimError:
            loud = True
        _ck(loud, "empty prompt → OmlxShimError (rule 6)")
    finally:
        shutil.rmtree(art, ignore_errors=True)


# --- (4) batched-session route -------------------------------------------------------------------
def _check_batched_route() -> None:
    print("\n--- (4) _make_batched_session route (minimax_m3 → _MiniMaxM3BatchedSession) ---")
    import quanta.minimax.batched_runtime_m3 as _bm3
    art = _m3_artifact("minimax_m3_vl")
    saved = _bm3.MiniMaxM3BatchedResidentModel

    class _Stub:
        def __init__(self, root, *, max_batch=32, **kw):
            self.root = root
            self.max_batch = max_batch
            self.num_layers = 2
            self.kind = "m3-batched"
    try:
        _bm3.MiniMaxM3BatchedResidentModel = _Stub
        eng = QuantaOmlxEngine(art, runtime=_FakeM3Runtime(), tokenizer=_FakeTok(),
                               eos_token_ids={EOS}, paged_kv=False)
        sess = eng._make_batched_session(capacity=4)
        _ck(isinstance(sess, _MiniMaxM3BatchedSession),
            f"minimax_m3_vl → {type(sess).__name__}")
        _ck(getattr(sess._rt, "kind", None) == "m3-batched" and sess._rt.max_batch == 4,
            f"_make_runtime built MiniMaxM3BatchedResidentModel(max_batch=4): {type(sess._rt).__name__}")
    finally:
        _bm3.MiniMaxM3BatchedResidentModel = saved
        shutil.rmtree(art, ignore_errors=True)


# --- (5) batched equivalence + continuous batching -----------------------------------------------
def _check_batched_equivalence() -> None:
    print("\n--- (5) batched_stream_generate == single-stream + continuous batching ---")
    art = _m3_artifact("minimax_m3_vl")
    try:
        # single-stream reference
        single_last = asyncio.run(_collect(
            QuantaOmlxEngine(art, runtime=_FakeM3Runtime(), tokenizer=_FakeTok(), eos_token_ids={EOS}),
            "hi", max_tokens=20, temperature=0.0))[-1]
        _ck(single_last.tokens == EXPECT_TOKENS, f"single-stream reference tokens={single_last.tokens}")

        # B=4 identical streams — each equals single-stream (rule 4)
        b = 4
        brt = _FakeM3BatchedRuntime()
        sess = _MiniMaxM3BatchedSession(root=None, capacity=b, runtime=brt)
        eng_b = QuantaOmlxEngine(art, runtime=brt, tokenizer=_FakeTok(), eos_token_ids={EOS},
                                 batched_session=sess)

        async def _collect_b():
            chunks: dict[int, list] = {i: [] for i in range(b)}
            async for sidx, chunk in eng_b.batched_stream_generate(
                    ["hi"] * b, max_tokens=20, temperature=0.0):
                chunks[sidx].append(chunk)
            return chunks
        chunks = asyncio.run(_collect_b())
        per_stream_eq = all(
            chunks[i] and chunks[i][-1].tokens == single_last.tokens
            and chunks[i][-1].text == single_last.text
            and chunks[i][-1].finish_reason == single_last.finish_reason for i in range(b))
        _ck(per_stream_eq, f"B=4 per-stream == single-stream (each {chunks[0][-1].tokens})")
        _ck(brt.prefills == b, f"session admitted all {b} streams (prefills={brt.prefills})")
        _ck(not sess._caches, f"every slot released after finishing (caches_left={len(sess._caches)})")

        # continuous batching: 5 prompts, capacity 2 ⇒ freed slots admit the next pending prompt
        n, cap = 5, 2
        cbrt = _FakeM3BatchedRuntime()
        cbsess = _MiniMaxM3BatchedSession(root=None, capacity=cap, runtime=cbrt)
        eng_cb = QuantaOmlxEngine(art, runtime=cbrt, tokenizer=_FakeTok(), eos_token_ids={EOS},
                                  batched_session=cbsess)

        async def _collect_cb():
            seen: set[int] = set()
            finals: dict[int, Any] = {}
            async for sidx, chunk in eng_cb.batched_stream_generate(
                    ["p"] * n, max_tokens=20, temperature=0.0, batch_size=cap):
                seen.add(sidx)
                if chunk.finished:
                    finals[sidx] = chunk
            return seen, finals
        seen, finals = asyncio.run(_collect_cb())
        cb_ok = (seen == set(range(n)) and len(finals) == n
                 and all(finals[i].tokens == single_last.tokens for i in range(n))
                 and cbrt.prefills == n and not cbsess._caches)
        _ck(cb_ok, f"continuous batching cap={cap}: streams={sorted(seen)} prefills={cbrt.prefills}")
    finally:
        shutil.rmtree(art, ignore_errors=True)


# --- (6) unknown model_type fails loud ------------------------------------------------------------
def _check_unknown_loud() -> None:
    print("\n--- (6) unknown model_type → no stepper (rule 6) ---")
    art = _m3_artifact("llama_surprise")
    try:
        eng = QuantaOmlxEngine(art, runtime=_FakeM3Runtime(), tokenizer=_FakeTok(), eos_token_ids={EOS})
        try:
            eng._make_stepper(quantized_kv=True)
            loud = False
        except OmlxShimError:
            loud = True
        _ck(loud, "unknown model_type → OmlxShimError (no silent fallback)")
    finally:
        shutil.rmtree(art, ignore_errors=True)


def run() -> None:
    _check_loader_dispatch()
    _check_stepper_dispatch()
    _check_single_stream()
    _check_batched_route()
    _check_batched_equivalence()
    _check_unknown_loud()
    print(f"\nPARITY-CHECKS: {_N}")
    print("PASS")


if __name__ == "__main__":
    run()
