"""Gate: the oMLX shim drives Qwen3.5 — parsers contract + batched serving + multi-step MTP.

Two new agentic-loop capabilities the Qwen3.5 oMLX path surfaces — and the parsers contract
formalized across all models — are gated here MODEL-FREE (stubs for the in-flight batched runtime
and ``spec_generate_k`` so this passes before those sibling agents merge). Mirrors
``parity/dsv4_omlx_engine_test.py`` + ``parity/nemotron_omlx_engine_test.py``. Asserts:

  (a) **Parsers contract conformance** — :class:`quanta.shim.tool_parsers.Qwen3ReasoningParser`,
      :class:`Qwen3ToolParser`, and :class:`KimiToolParser` explicitly implement the
      :class:`ReasoningParser` / :class:`ToolParser` ``@runtime_checkable`` Protocols (``isinstance``
      holds at the surface). Round-trip tested: a Qwen tool-call text -> ``parse_tool_calls`` ->
      ``format_tool_response`` preserves the id.

  (b) **Batched engine equivalence** — with a STUB batched runtime that returns deterministic
      per-stream logits, driving the engine with ``B=4`` identical prompts emits the same per-stream
      output the single-stream stub run on the same prompt would emit. Proves the engine's
      stream-multiplexing dispatch is correct without the real runtime (the real runtime is gated in
      ``parity/qwen35_batched_test.py``).

  (c) **Multi-step MTP hook** — a stub ``spec_generate_k`` records the ``k`` arg and returns a fixed
      token list; the engine calls it with ``k=spec_k`` when ``spec_k > 1`` and falls back to plain
      ``spec_generate`` when ``spec_k == 1``. Verifies the dispatch logic + the loud-fail when the
      sibling symbol is absent.

Tiny tensors (few KB), no checkpoint loaded — SAFE to run alongside a resident GPU job.

    uv run --with numpy python -m parity.qwen35_omlx_engine_test
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx

from quanta.shim.omlx import (
    OmlxShimError,
    Qwen35BatchedEngine,
    _SingleTokenStepper,
)
from quanta.shim.tool_parsers import (
    KimiToolParser,
    Qwen3ReasoningParser,
    Qwen3ToolParser,
    ReasoningParser,
    ToolParser,
)

EOS = 11
VOCAB_SIZE = 200
# token id -> literal piece; reasoning + tool markers (ordinary Qwen tokens — bytes-level BPE)
VOCAB = {100: "<think>", 101: "reason", 102: "</think>", 103: "<tool_call>", 104: "done"}
# argmax at absolute offset: prompt encodes to [5,6,7] (len 3); the forward at last prompt pos
# (offset 2) predicts the first generated token; subsequent decodes 3,4,5,... predict the next.
PRED = {2: 100, 3: 101, 4: 102, 5: 103, 6: 104, 7: EOS}
EXPECT_TOKENS = [100, 101, 102, 103, 104, EOS]


def _fake_qwen_artifact() -> str:
    """Synthetic quanta artifact dir: config + manifest, model_type = qwen3_5_moe_text. A few bytes."""
    d = Path(tempfile.mkdtemp(prefix="qwen35omlx_"))
    (d / "config.json").write_text(json.dumps({"text_config": {"model_type": "qwen3_5_moe_text"}}))
    (d / "manifest.json").write_text(json.dumps({"format": "quanta", "tensors": {}}))
    return str(d)


class _FakeSingleStreamRuntime:
    """Stands in for ``Qwen35ResidentModel``: single-token ``__call__(token_ids, *, caches, offset)``
    -> ``[1,T,vocab]`` + ``num_layers`` + ``make_caches``. argmax at absolute ``offset`` is
    ``PRED[offset]`` (default eos). Records each offset for the test to verify prompt-seed +
    decode-step pattern."""

    def __init__(self, *, n_layers: int = 2) -> None:
        self.num_layers = n_layers
        self.offsets: list[int] = []
        # mock attrs the batched runtime's from_inner uses (kept tiny — not loaded)
        self.layers: list[Any] = []
        self.embed_w = mx.zeros((1, 1))
        self.norm_w = mx.zeros((1,))
        self.lm_head_w = mx.zeros((1, 1))
        self.cfg = SimpleNamespace(num_hidden_layers=n_layers)

    def make_caches(self) -> SimpleNamespace:
        return SimpleNamespace(tag="hybrid-cache")

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        t = int(mx.array(token_ids).reshape(-1).shape[0])
        self.offsets.append(int(offset))
        nxt = PRED.get(int(offset), EOS)
        row = (mx.arange(VOCAB_SIZE) == nxt).astype(mx.float32) * 60.0 - 30.0
        return mx.broadcast_to(row, (1, t, VOCAB_SIZE))


class _FakeTok:
    """Minimal tokenizer for the engine path: fixed 3-token prompt + marker detok via VOCAB. No
    ``decode_bytes``/``n_base`` so ``_Detok`` takes the string-fallback path."""

    eos_id = EOS
    stop_ids = (EOS,)

    def encode(self, text, *, add_bos=False, allow_special=False):
        return [5, 6, 7]

    def decode(self, ids, **kw):
        return "".join(VOCAB.get(int(i), "") for i in ids)


# --- (a) parsers contract conformance --------------------------------------------------------------
QWEN_TOOL_TEXT = ('here we go <tool_call>\n{"name": "get_weather", '
                  '"arguments": {"location": "Tokyo"}}\n</tool_call> end')
KIMI_TOOL_TEXT = ("ok<|tool_calls_section_begin|><|tool_call_begin|>"
                  "functions.get_weather:0<|tool_call_argument_begin|>{\"location\": \"SF\"}"
                  "<|tool_call_end|><|tool_calls_section_end|>")


def _check_parsers_contract() -> bool:
    print("\n--- (a) parsers contract conformance ---")
    ok = True

    rp = Qwen3ReasoningParser()
    qtp = Qwen3ToolParser()
    ktp = KimiToolParser()

    # Protocol isinstance — the runtime_checkable Protocols match by method presence
    rp_is = isinstance(rp, ReasoningParser)
    qtp_is = isinstance(qtp, ToolParser)
    ktp_is = isinstance(ktp, ToolParser)
    good = rp_is and qtp_is and ktp_is
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] Protocol isinstance: "
          f"Qwen3ReasoningParser->ReasoningParser={rp_is} "
          f"Qwen3ToolParser->ToolParser={qtp_is} KimiToolParser->ToolParser={ktp_is}")

    # ReasoningParser.parse: explicit block, bare opener, truncated, none
    r1 = rp.parse("Sure <think>let me think</think> The answer is 42.")
    g1 = r1.get("reasoning") == "let me think" and r1.get("answer") == "Sure  The answer is 42."
    r2 = rp.parse("let me think</think>The answer is 42.")   # bare-opener fallback shape
    g2 = r2.get("reasoning") == "let me think" and r2.get("answer") == "The answer is 42."
    r3 = rp.parse("<think>just thinking, no closer")          # truncated
    g3 = r3.get("reasoning") == "just thinking, no closer" and r3.get("answer") == ""
    r4 = rp.parse("plain answer with no think")               # none
    g4 = r4.get("reasoning") is None and r4.get("answer") == "plain answer with no think"
    good = g1 and g2 and g3 and g4
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] Qwen3ReasoningParser.parse: "
          f"explicit={g1} bare={g2} truncated={g3} none={g4}")

    # ToolParser.parse_tool_calls: Qwen JSON + Kimi special tokens both extracted
    qtc = qtp.parse_tool_calls(QWEN_TOOL_TEXT)
    qg = (len(qtc) == 1 and qtc[0]["name"] == "get_weather"
          and json.loads(qtc[0]["arguments"]) == {"location": "Tokyo"})
    ktc = ktp.parse_tool_calls(KIMI_TOOL_TEXT)
    kg = (len(ktc) == 1 and ktc[0]["name"] == "get_weather"
          and json.loads(ktc[0]["arguments"]) == {"location": "SF"})
    # empty list when no markup (NOT None — the contract distinguishes "no calls" from "parse error")
    eg_q = qtp.parse_tool_calls("just prose") == []
    eg_k = ktp.parse_tool_calls("just prose") == []
    good = qg and kg and eg_q and eg_k
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] ToolParser.parse_tool_calls: "
          f"qwen={qg} kimi={kg} empty_q={eg_q} empty_k={eg_k}")

    # ToolParser.format_tool_response: shapes + id round-trip + loud failure on empty id
    qresp = qtp.format_tool_response(qtc[0]["id"], '{"temp": 22}')
    kresp = ktp.format_tool_response(ktc[0]["id"], '{"temp": 18}')
    qg_resp = "<tool_response>" in qresp and "</tool_response>" in qresp and '"temp": 22' in qresp
    kg_resp = ("<|tool_response_begin|>" in kresp and "<|tool_response_end|>" in kresp
               and ktc[0]["id"] in kresp and '"temp": 18' in kresp)
    try:
        qtp.format_tool_response("", "x")
        loud = False
    except ValueError:
        loud = True
    good = qg_resp and kg_resp and loud
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] ToolParser.format_tool_response: "
          f"qwen_shape={qg_resp} kimi_shape={kg_resp} empty_id_raises={loud}")

    return ok


# --- (b) batched engine equivalence ----------------------------------------------------------------
class _StubBatchedModel:
    """Records each ``step_batch`` call (token ids + offsets). Constructed via ``from_inner`` by the
    engine. Per-stream logits are emitted independent of state — the engine only checks dispatch /
    plumbing here; full step semantics are gated in ``parity/qwen35_batched_test.py`` (real runtime)."""

    def __init__(self, *, max_batch: int = 8) -> None:
        self.max_batch = int(max_batch)
        self.batch_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []  # (toks, offsets)
        self.cfg = SimpleNamespace(num_hidden_layers=2)
        self.num_layers = 2

    def make_caches(self):
        # one cache per stream; the stub doesn't track per-stream state (offsets are passed in)
        return SimpleNamespace(offset=0)

    def prefill(self, prompt_ids, state):
        # the orchestrator calls this once per stream; just return the last-position logits.
        last = list(prompt_ids)[-1]
        # the *next* greedy token is the deterministic chain (last + 1) % VOCAB_SIZE
        nxt = (int(last) + 1) % VOCAB_SIZE
        row = (mx.arange(VOCAB_SIZE) == nxt).astype(mx.float32) * 60.0 - 30.0
        state.offset = len(list(prompt_ids))
        return mx.broadcast_to(row, (1, 1, VOCAB_SIZE))

    def step_batch(self, stream_token_ids, stream_caches, offsets):
        toks = tuple(int(t) for t in stream_token_ids)
        offs = tuple(int(o) for o in offsets)
        self.batch_calls.append((toks, offs))
        # deterministic chain per stream: next = (tok + 1) % vocab
        out = []
        for tok in toks:
            nxt = (int(tok) + 1) % VOCAB_SIZE
            row = (mx.arange(VOCAB_SIZE) == nxt).astype(mx.float32) * 60.0 - 30.0
            out.append(mx.broadcast_to(row, (1, 1, VOCAB_SIZE)))
        return out


def _install_stub_batched_modules(stub_model_cls=_StubBatchedModel,
                                  spec_generate_k_fn=None) -> list[str]:
    """Install fake ``quanta.qwen35.batched_runtime`` + ``batched_generate`` modules so the engine's
    lazy import resolves them without the in-flight task #147 code being on disk. Returns the list
    of installed module names so the caller can clean them up. The stub spec_generate_k (when
    provided) is patched onto :mod:`quanta.qwen35.spec` for the spec_k>1 test."""
    installed: list[str] = []

    # batched_runtime stub
    br = types.ModuleType("quanta.qwen35.batched_runtime")
    # Mirror the REAL from_inner keyword-only contract (max_batch / kv_quantized / kv_group_size /
    # packed / loopkill — the last three landed with #153 option-B + the packed-experts work); the
    # stub consumes only max_batch but must ACCEPT the full set or the engine's real call
    # (omlx.py: from_inner(..., max_batch=, packed=)) raises. Kept explicit (not **kwargs) so a future
    # signature growth fails loud here instead of silently drifting.
    br.Qwen35BatchedResidentModel = type(
        "Qwen35BatchedResidentModel", (stub_model_cls,),
        {"from_inner": classmethod(
            lambda cls, layers, embed_w, norm_w, lm_head_w, cfg, *, max_batch=32,
            kv_quantized=False, kv_group_size=64, packed=False, loopkill=None:
            cls(max_batch=max_batch))})
    sys.modules["quanta.qwen35.batched_runtime"] = br
    installed.append("quanta.qwen35.batched_runtime")

    # batched_generate stub — drives step_batch in a continuous-batching loop. Per-stream output
    # mirrors the deterministic chain so test equivalence is well-defined.
    bg = types.ModuleType("quanta.qwen35.batched_generate")

    def generate_batched(model, prompts, *, max_new_tokens, temperature=0.0, top_k=0, top_p=1.0,
                         min_p=0.0, eos_id=None, seeds=0):
        del temperature, top_k, top_p, min_p, seeds  # unused by the deterministic chain
        # normalize eos to a set of ints (mirrors the real batched_generate contract)
        stop = set()
        if eos_id is not None:
            if isinstance(eos_id, int):
                stop.add(int(eos_id))
            else:
                stop.update(int(s) for s in eos_id if s is not None)
        prompts = [list(p) for p in prompts]
        b = len(prompts)
        caches = [model.make_caches() for _ in range(b)]
        # prefill: sample first token from the prefill logits (greedy)
        outs: list[list[int]] = [[] for _ in range(b)]
        next_toks: list[int] = [0] * b
        offsets: list[int] = [0] * b
        done: list[bool] = [False] * b
        for i, p in enumerate(prompts):
            lg = model.prefill(p, caches[i])
            tok = int(mx.argmax(lg[0, -1]).item())
            offsets[i] = len(p)
            if tok in stop:
                done[i] = True
            else:
                outs[i].append(tok)
                next_toks[i] = tok
                if len(outs[i]) >= max_new_tokens:
                    done[i] = True
        for _step in range(max_new_tokens):
            active = [i for i in range(b) if not done[i]]
            if not active:
                break
            toks = [next_toks[i] for i in active]
            acs = [caches[i] for i in active]
            offs = [offsets[i] for i in active]
            per_stream = model.step_batch(toks, acs, offs)
            for j, i in enumerate(active):
                tok = int(mx.argmax(per_stream[j][0, -1]).item())
                offsets[i] += 1
                if tok in stop:
                    done[i] = True
                    continue
                outs[i].append(tok)
                if len(outs[i]) >= max_new_tokens:
                    done[i] = True
                    continue
                next_toks[i] = tok
        return outs

    bg.generate_batched = generate_batched
    sys.modules["quanta.qwen35.batched_generate"] = bg
    installed.append("quanta.qwen35.batched_generate")

    # spec_generate_k stub (optional) — patched onto quanta.qwen35.spec so the engine's lazy import
    # finds it under the existing module name.
    if spec_generate_k_fn is not None:
        import quanta.qwen35.spec as qspec  # safe — this module exists pre-task-149
        # restore on caller side
        installed.append(f"__patched_spec_k__={getattr(qspec, 'spec_generate_k', '__missing__')}")
        qspec.spec_generate_k = spec_generate_k_fn
    return installed


def _uninstall_stub_batched_modules(names: list[str]) -> None:
    for n in names:
        if n.startswith("__patched_spec_k__="):
            import quanta.qwen35.spec as qspec
            prev = n.split("=", 1)[1]
            if prev == "__missing__":
                qspec.__dict__.pop("spec_generate_k", None)
            else:
                # cannot serialize a function back; just remove the patch (the test caller knows it)
                qspec.__dict__.pop("spec_generate_k", None)
            continue
        sys.modules.pop(n, None)


def _greedy_single_stream(prompt: list[int], max_new: int) -> list[int]:
    """The deterministic chain greedy reference: next = (last + 1) % vocab, terminate at EOS or
    max_new. Mirrors the stub batched ``generate_batched``: EOS terminates the stream and is NOT
    emitted (matches the engine's eos behavior — the eos token is consumed but not visible in the
    stream's text). Per-stream batched output must match this run on the same prompt."""
    out: list[int] = []
    last = prompt[-1]
    for _ in range(max_new):
        nxt = (last + 1) % VOCAB_SIZE
        if nxt == EOS:
            break                                  # eos consumed, not emitted (engine stop semantics)
        out.append(nxt)
        last = nxt
    return out


def _check_batched_engine_equivalence() -> bool:
    print("\n--- (b) batched engine stream-multiplexing equivalence ---")
    ok = True
    art = _fake_qwen_artifact()
    installed = _install_stub_batched_modules()
    try:
        eng = Qwen35BatchedEngine(art, runtime=_FakeSingleStreamRuntime(),
                                  tokenizer=_FakeTok(), max_batch=4,
                                  eos_token_ids={EOS})

        # construction validation: max_batch and spec_k must be >= 1
        try:
            Qwen35BatchedEngine(art, runtime=_FakeSingleStreamRuntime(),
                                tokenizer=_FakeTok(), max_batch=0)
            loud_mb = False
        except OmlxShimError:
            loud_mb = True
        try:
            Qwen35BatchedEngine(art, runtime=_FakeSingleStreamRuntime(),
                                tokenizer=_FakeTok(), spec_k=0)
            loud_sk = False
        except OmlxShimError:
            loud_sk = True
        good = loud_mb and loud_sk
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] construct validation: "
              f"max_batch=0_raises={loud_mb} spec_k=0_raises={loud_sk}")

        # batched_generate: B=4 identical prompts must each yield the same greedy chain
        prompts = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
        outs = asyncio.run(eng.batched_generate(prompts, max_new_tokens=8))
        expected = _greedy_single_stream([1, 2, 3], max_new=8)
        good = len(outs) == 4 and all(o == expected for o in outs)
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] B=4 identical prompts: each stream matches single "
              f"(expected={expected[:4]}... got[0]={outs[0][:4]}...)")

        # batched_generate: distinct prompts → distinct per-stream chains (no cross-stream leakage)
        prompts2 = [[1, 2, 3], [4, 5, 6]]
        outs2 = asyncio.run(eng.batched_generate(prompts2, max_new_tokens=4))
        e1 = _greedy_single_stream([1, 2, 3], max_new=4)
        e2 = _greedy_single_stream([4, 5, 6], max_new=4)
        good = outs2 == [e1, e2]
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] distinct prompts independent: out={outs2} expected=[{e1},{e2}]")

        # batched_generate: B > max_batch fails loud (no silent rebatching)
        too_many = [[1, 2, 3]] * 5
        try:
            asyncio.run(eng.batched_generate(too_many, max_new_tokens=2))
            loud = False
        except OmlxShimError:
            loud = True
        ok = ok and loud
        print(f"  [{'OK' if loud else 'FAIL'}] B>max_batch raises (no silent rebatching): {loud}")

        # empty prompts list returns []
        good = asyncio.run(eng.batched_generate([], max_new_tokens=4)) == []
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] empty prompts -> [] (no-op): {good}")

        # stop()/start() invalidates the cached batched model so a stale runtime is never used
        # (oMLX engine pool stops + restarts engines on cache eviction)
        _ = asyncio.run(eng.batched_generate([[1, 2, 3]], max_new_tokens=2))
        cached_before = eng._batched_model is not None
        asyncio.run(eng.stop())
        cleared_after_stop = eng._batched_model is None
        # restart with a fresh injected runtime so we can prove the next batched call rebuilds
        eng._runtime = _FakeSingleStreamRuntime()
        eng._tokenizer = _FakeTok()
        eng._loaded = True
        _ = asyncio.run(eng.batched_generate([[1, 2, 3]], max_new_tokens=2))
        rebuilt = eng._batched_model is not None and eng._batched_model.max_batch == 4
        good = cached_before and cleared_after_stop and rebuilt
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] stop() invalidates _batched_model "
              f"(cached_before={cached_before} cleared={cleared_after_stop} rebuilt={rebuilt})")

        # non-Qwen3.5 inner runtime → _ensure_batched_model raises loud (not AttributeError)
        class _BadRuntime:
            """Stand-in for a Nemotron-shaped runtime: missing the Qwen3.5 attrs."""
            num_layers = 2
            cfg = SimpleNamespace(num_hidden_layers=2)
            # no `layers`, no `embed_w` / `norm_w` / `lm_head_w` — the Qwen3.5 surface batched needs
            def make_caches(self): return SimpleNamespace(offset=0)
            def __call__(self, *a, **kw): return mx.zeros((1, 1, VOCAB_SIZE))

        eng_bad = Qwen35BatchedEngine(art, runtime=_BadRuntime(), tokenizer=_FakeTok(),
                                      eos_token_ids={EOS})
        try:
            asyncio.run(eng_bad.batched_generate([[1, 2, 3]], max_new_tokens=2))
            loud = False
        except OmlxShimError as e:
            loud = "missing attrs" in str(e) and "qwen3_5" in str(e).lower()
        ok = ok and loud
        print(f"  [{'OK' if loud else 'FAIL'}] non-Qwen3.5 inner runtime -> OmlxShimError "
              f"(not AttributeError): {loud}")

    finally:
        _uninstall_stub_batched_modules(installed)
        shutil.rmtree(art, ignore_errors=True)
    return ok


def _check_factory_model_type_guard() -> bool:
    """The load_qwen35_batched_engine factory must refuse non-Qwen3.5 artifacts loudly."""
    print("\n--- factory model_type guard ---")
    from quanta.shim.omlx import load_qwen35_batched_engine

    ok = True
    # a Qwen3.5 artifact loads fine
    qart = _fake_qwen_artifact()
    try:
        eng = load_qwen35_batched_engine(qart, max_batch=2, spec_k=1)
        good = isinstance(eng, Qwen35BatchedEngine) and eng.max_batch == 2 and eng.spec_k == 1
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] Qwen3.5 artifact loads: {good}")
    finally:
        shutil.rmtree(qart, ignore_errors=True)

    # a non-Qwen3.5 artifact (e.g. Nemotron / Kimi) raises loud
    for mt in ("nemotron_h", "kimi_k2", "deepseek_v4", "glm_moe_dsa", "minimax_m2"):
        d = Path(tempfile.mkdtemp(prefix=f"qwenfac_{mt}_"))
        (d / "config.json").write_text(json.dumps({"text_config": {"model_type": mt}}))
        (d / "manifest.json").write_text(json.dumps({"format": "quanta", "tensors": {}}))
        try:
            load_qwen35_batched_engine(str(d))
            loud = False
        except OmlxShimError as e:
            loud = "not Qwen3.5" in str(e) or "qwen3_5" in str(e).lower()
        finally:
            shutil.rmtree(d, ignore_errors=True)
        ok = ok and loud
        print(f"  [{'OK' if loud else 'FAIL'}] non-Qwen3.5 model_type={mt!r} -> OmlxShimError: {loud}")
    return ok


# --- (c) multi-step MTP hook dispatch --------------------------------------------------------------
def _check_spec_k_dispatch() -> bool:
    print("\n--- (c) multi-step MTP hook (spec_k) dispatch ---")
    ok = True
    art = _fake_qwen_artifact()

    # stub spec_generate_k that records k + returns a fixed tokens list
    recorded: dict[str, Any] = {}

    def _stub_spec_generate_k(model, mtp, embed, head, prompt_ids, *, k, max_new, eos_id=None):
        recorded["k"] = int(k)
        recorded["max_new"] = int(max_new)
        recorded["eos_id"] = eos_id
        recorded["prompt"] = list(prompt_ids)
        return [42, 43, 44], {"rounds": 1, "tokens": 3, "mean_accept": 1.0, "max_accept": 1, "k": int(k)}

    installed = _install_stub_batched_modules(spec_generate_k_fn=_stub_spec_generate_k)
    try:
        # spec_k > 1: dispatches to spec_generate_k with the right k
        eng = Qwen35BatchedEngine(art, runtime=_FakeSingleStreamRuntime(),
                                  tokenizer=_FakeTok(), spec_k=3,
                                  eos_token_ids={EOS})
        mtp = lambda *a, **kw: None  # noqa: E731 — unused by the stub
        embed = mx.zeros((4, 4))
        head = mx.zeros((4, 4))
        toks, stats = asyncio.run(eng.spec_generate_batched(mtp, embed, head, [1, 2, 3],
                                                            max_new=10, eos_id=EOS))
        good = (recorded.get("k") == 3 and recorded.get("max_new") == 10
                and recorded.get("eos_id") == EOS and recorded.get("prompt") == [1, 2, 3]
                and toks == [42, 43, 44] and stats["k"] == 3)
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] spec_k=3 dispatches to spec_generate_k: "
              f"recorded={recorded} tokens={toks} stats={stats}")

        # per-call override of spec_k
        recorded.clear()
        toks2, stats2 = asyncio.run(eng.spec_generate_batched(mtp, embed, head, [9, 8, 7],
                                                              max_new=5, spec_k=2))
        good = recorded.get("k") == 2 and stats2["k"] == 2
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] per-call spec_k=2 override: recorded k={recorded.get('k')}")

        # spec_k == 1: bypasses spec_generate_k entirely (uses spec_generate). The stub's
        # spec_generate_k is NOT called — recorded should NOT capture this call.
        recorded.clear()

        # patch quanta.qwen35.spec.spec_generate to record k=1 calls (the existing path)
        import quanta.qwen35.spec as qspec
        orig_sg = qspec.spec_generate
        sg_recorded: dict[str, Any] = {}

        def _stub_spec_generate(model, mtp, embed, head, prompt_ids, *, max_new, eos_id=None):
            sg_recorded["called"] = True
            sg_recorded["max_new"] = int(max_new)
            return [7, 8, 9], {"rounds": 1, "tokens": 3, "mean_accept": 1.0, "max_accept": 1, "k": 1}

        qspec.spec_generate = _stub_spec_generate
        try:
            eng1 = Qwen35BatchedEngine(art, runtime=_FakeSingleStreamRuntime(),
                                       tokenizer=_FakeTok(), spec_k=1,
                                       eos_token_ids={EOS})
            toks3, stats3 = asyncio.run(eng1.spec_generate_batched(mtp, embed, head, [1, 2, 3],
                                                                   max_new=6))
            # spec_generate was called; spec_generate_k was NOT (recorded stayed empty)
            good = (sg_recorded.get("called") is True and sg_recorded.get("max_new") == 6
                    and not recorded and toks3 == [7, 8, 9] and stats3["k"] == 1)
            ok = ok and good
            print(f"  [{'OK' if good else 'FAIL'}] spec_k=1 -> spec_generate (not spec_generate_k): "
                  f"sg_called={sg_recorded.get('called')} sk_recorded_empty={not recorded}")
        finally:
            qspec.spec_generate = orig_sg

        # spec_k < 1 on a per-call basis raises loud
        try:
            asyncio.run(eng.spec_generate_batched(mtp, embed, head, [1, 2, 3], max_new=4, spec_k=0))
            loud = False
        except OmlxShimError:
            loud = True
        ok = ok and loud
        print(f"  [{'OK' if loud else 'FAIL'}] spec_k=0 raises (no silent fallback): {loud}")

    finally:
        _uninstall_stub_batched_modules(installed)
        shutil.rmtree(art, ignore_errors=True)

    # spec_k >= 2 with NO spec_generate_k installed -> loud OmlxShimError (the in-flight task #149
    # symbol absent). Test in a fresh artifact + fresh engine without the stub patched in.
    art2 = _fake_qwen_artifact()
    import quanta.qwen35.spec as qspec
    had_k = "spec_generate_k" in qspec.__dict__
    saved_k = qspec.__dict__.get("spec_generate_k")
    qspec.__dict__.pop("spec_generate_k", None)
    try:
        eng2 = Qwen35BatchedEngine(art2, runtime=_FakeSingleStreamRuntime(),
                                   tokenizer=_FakeTok(), spec_k=2, eos_token_ids={EOS})
        try:
            asyncio.run(eng2.spec_generate_batched(lambda *a, **kw: None, mx.zeros((4, 4)),
                                                   mx.zeros((4, 4)), [1, 2, 3], max_new=4))
            loud = False
        except OmlxShimError:
            loud = True
        ok = ok and loud
        print(f"  [{'OK' if loud else 'FAIL'}] missing spec_generate_k raises (no silent k=1 fallback): {loud}")
    finally:
        if had_k:
            qspec.spec_generate_k = saved_k
        shutil.rmtree(art2, ignore_errors=True)

    return ok


# --- baseline: the existing Qwen3.5 single-stream engine path still works -------------------------
def _check_existing_single_stream() -> bool:
    """No-regression: the inherited single-stream stream_generate must still produce the existing
    raw-output stream — same as ``parity/omlx_shim_models_test.py``'s Qwen branch."""
    print("\n--- no-regression: single-stream path unchanged ---")
    ok = True
    art = _fake_qwen_artifact()
    try:
        rt = _FakeSingleStreamRuntime()
        eng = Qwen35BatchedEngine(art, runtime=rt, tokenizer=_FakeTok(),
                                  eos_token_ids={EOS})
        st = eng._make_stepper(quantized_kv=True)
        good = isinstance(st, _SingleTokenStepper)
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] inherited stepper dispatch: "
              f"qwen3_5 -> {type(st).__name__}")

        outs: list[Any] = []

        async def _drive():
            async for o in eng.stream_generate("hello", max_tokens=20, temperature=0.0):
                outs.append(o)
        asyncio.run(_drive())
        last = outs[-1]
        good = (last.tokens == EXPECT_TOKENS and last.finished and last.finish_reason == "stop"
                and last.text == "<think>reason</think><tool_call>done")
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] single-stream stream_generate: tokens={last.tokens} "
              f"text={last.text!r}")
    finally:
        shutil.rmtree(art, ignore_errors=True)
    return ok


def run() -> None:
    ok_a = _check_parsers_contract()
    ok_b = _check_batched_engine_equivalence()
    ok_c = _check_spec_k_dispatch()
    ok_d = _check_existing_single_stream()
    ok_e = _check_factory_model_type_guard()
    ok = ok_a and ok_b and ok_c and ok_d and ok_e
    print("\nPASS" if ok else "\nFAIL")


if __name__ == "__main__":
    run()
