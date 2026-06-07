"""oMLX-compatible engine for quanta baked artifacts.

Importable without oMLX: subclasses oMLX's ``BaseEngine`` when present (so it passes the
server's ``isinstance(engine, BaseEngine)`` gate and is dispatched by the engine pool) and
falls back to a plain class otherwise (standalone / tests). The engine owns the quanta runtime —
it loads the model's resident runtime + tokenizer from the artifact (dispatched on ``model_type``)
and decodes through a model-specific **stepper**: the KV-cached absorbed-MLA path for Kimi /
DeepSeek-V3, the Hyper-Connections + compressed-KV single-token path for DeepSeek-V4-Flash, or the
threaded ``(ssm, conv)`` recurrence for the Nemotron-H hybrid — while oMLX provides the
OpenAI/Anthropic server surface. mlx-lm is never imported.

Every generation kwarg from oMLX (temperature, top_p, top_k, min_p, repetition/frequency/presence
penalties, stop strings, and a per-request ``seed`` for reproducible sampling) is applied — for both
model classes, since only the decode stepper differs and the sampling/stop loop is shared. Generation
always stops on the model's eos/stop token ids and the ``max_tokens`` cap, so it can never run
unbounded. Sparse XAttention prefill is on by default.

The engine is a thin **raw-output** token producer: it decodes ordinary text byte-accurately and
renders special-token markers (``</think>``, ``<|tool_calls_section_begin|>`` …) as their literal
strings, but does **not** parse the model output itself. Reasoning and tool-call extraction are
oMLX's responsibility — its server splits ``<think>…</think>`` with its own thinking parser and reads
tool calls via its own parsers — so we conform to oMLX's contract rather than pre-parsing here.

Beyond the single-stream stepper loop, the engine also exposes two **multi-stream** entry points
that mirror the same kwargs surface:

* :meth:`QuantaOmlxEngine.batched_stream_generate` / :meth:`QuantaOmlxEngine.batched_generate` drive
  ``B`` requests through a model-specific batched runtime (``quanta.dsv4.batched_runtime`` /
  ``quanta.nemotron.batched_runtime`` — lazy-imported so the shim still loads when the sibling
  modules are not yet merged). Per-stream sampling / eos / max_new are honored independently; freed
  slots admit the next pending prompt (continuous batching), and the per-stream finished/finish-reason
  state matches what the single-stream loop yields.
* A ``spec_k`` kwarg on ``generate`` / ``stream_generate`` switches the single-stream loop to native
  multi-step MTP self-speculation through ``quanta.{dsv4,nemotron}.spec.spec_generate_k`` (lazy
  imports — DSV4's lives on main, Nemotron's is in flight). Because every emitted token is the main
  model's own greedy/sampled token (the draft is kept only when it matches the main model's pick, and
  the bonus is the main model's pick outright), the output is **bit-identical** to the ``spec_k == 1``
  path regardless of MTP quality — spec changes only *speed*, never correctness.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx

from quanta.cache import MLACache
from quanta.modeling.xattention import DEFAULT_SPARSE

try:  # subclass oMLX's engine ABC when the host is present; stay importable without it
    from omlx.engine.base import BaseEngine as _OmlxBaseEngine
except Exception:  # pragma: no cover - standalone use
    _OmlxBaseEngine = object

# Request kwargs that drive the chat TEMPLATE (not sampling) — forwarded to apply_chat_template.
# Kimi's template reads ``thinking``/``preserve_thinking``; Nemotron's reads ``enable_thinking``
# (+ ``truncate_history_thinking`` via chat_template_kwargs); ``reasoning_effort`` is forwarded for
# templates that use it (e.g. DSV4) and harmlessly ignored by templates that don't.
_TEMPLATE_KEYS = ("thinking", "enable_thinking", "preserve_thinking", "reasoning_effort")

# Per-model batched-decode operating point: the measured **throughput knee** — the largest B where
# aggregate tok/s still climbs meaningfully and peak memory stays safely under the ~490 GiB working
# set (criterion fixed with the user). Measured per model in ``parity/<model>_batched_bench.py``;
# the per-model DEFAULT serving capacity (concurrent decode slots) when the caller passes no explicit
# ``batch_size`` (an explicit ``batch_size`` wins up to the uniform :data:`SERVING_BATCH_CAP` for a
# throughput worker — sweep past it via a direct session). Two kinds of
# entry: a throughput WORKER's measured knee (rule 6 — only a knee we have ACTUALLY benched appears;
# never an invented one), and the latency-first ORCHESTRATOR operating point (Qwen3.5), a deliberately
# LOW B chosen because the agentic loop is single-stream-ish, not a throughput regime — so it is NOT a
# measured knee and makes no aggregate-tok/s claim. A model absent here ⇒ the generic fallback.
# Prefix-matched to mirror ``_make_batched_session``.
BEST_BATCH: tuple[tuple[str, int], ...] = (
    ("deepseek_v4", 48),  # DSV4 worker (#19): agg ~92.8 tok/s @48, ~11.5x over the per-stream loop; B=64
    #                       OOMs the looped path's ~490 GiB lazy graph and agg has flattened by 48. This
    #                       is the benched KNEE; serving HARD-caps it to SERVING_BATCH_CAP=32 (uniform).
    ("nemotron", 32),     # Nemotron worker (#20): agg peaks 136.6 tok/s @32 (2.46x); REGRESSES at 48
    #                       (131.4 tok/s) though memory is fine (108 GiB) — the knee is below 48. The
    #                       #153 paged KV loop-kill (default since 7f49bd9) REAFFIRMS 32: decode-only
    #                       loopkill ~146 tok/s @32 ≈ ~144 @48 (the loop-kill flattened the >32 decay) but
    #                       32 uses ~16 GiB less KV (99.7 vs 115.5, parity/nemotron_paged_batched_bench.py).
    #                       ULTRA-550B (U4/Stream-A) confirms 32: agg PLATEAUS ~48 tok/s from B=32
    #                       (48.0/47.8/47.9/48.3 @ B=32/48/64/80; per-stream 1.50→0.60),
    #                       B>32 buys ZERO aggregate — only latency + ~1.92 GiB/stream (per-stream
    #                       Mamba SSD-step caps amortization, not memory/MoE).
    #                       parity/nemotron_ultra_decode_scale.py.
    ("internlm2", 32),    # InternLM2.5 worker (#21): agg peaks 243.1 tok/s @32 (4.49x over the loop);
    #                       REGRESSES to 213.6 @48 then plateaus ~210-221 through 128 — KV-light but the
    #                       knee is still 32 (per-stream decay outpaces B past 32; memory flat ~9 GiB).
    #                       HARD cap (like Nemotron): the #153 loop-kill bench reaffirms 332→322 @32→48.
    ("qwen3_5", 4),       # Qwen3.5 orchestrator (#26): latency-first low-B default, NOT a throughput
    #                       knee. The agentic loop runs single-stream-ish (capacity clamps to prompts
    #                       on hand, so a lone request still decodes at B=1); B=4 caps concurrency low
    #                       to keep per-token latency down while still amortizing the routed-MoE read
    #                       across a few overlapping agent traces. Pin an explicit batch_size to sweep.
)
DEFAULT_BATCH_CAPACITY = 8  # generic fallback for a model class with no measured operating point
SERVING_BATCH_CAP = 32      # UNIFORM hard ceiling on concurrent decode slots for every throughput WORKER
#   (user directive 2026-05-30: "B=32 as hard-coded default ... do the same for nemotron and deepseek").
#   Serving clamps DOWN to this even on an explicit ``batch_size`` — DECOUPLED from the measured
#   :data:`BEST_BATCH` knee, so DSV4's benched knee (48; it scales there per #18 M5) stays honest in
#   ``BEST_BATCH`` while serving is held at 32; Nemotron (#20) / InternLM2.5 (#21) regress past 32 anyway.
#   The Qwen3.5 ORCHESTRATOR is EXEMPT (its B=4 is a sweepable latency pin, not a ceiling; rule 6). A
#   bench that must sweep B>32 drives the batched SESSION directly (``DSV4BatchedResidentModel.step_batch``),
#   bypassing this engine-layer policy — e.g. ``parity/dsv4_batched_bench.py``, ``parity/dsv4_b48_noise.py``.


def _best_batch_for(model_type: str | None) -> int | None:
    """The measured best-B (throughput knee) for ``model_type`` via first-matching prefix, or ``None``
    when the model's knee has not been measured. Prefixes mirror the ``_make_batched_session``
    dispatch so a best-B is only ever declared for a model that actually has a batched runtime."""
    mt = model_type or ""
    for prefix, best in BEST_BATCH:
        if mt.startswith(prefix):
            return best
    return None


class OmlxShimError(RuntimeError):
    """Raised when the quanta oMLX engine cannot load or run an artifact."""


class TokenizerLike(Protocol):
    def encode(self, text: str, *a: Any, **k: Any) -> Sequence[int]: ...
    def decode(self, ids: Sequence[int], *a: Any, **k: Any) -> str: ...


class RuntimeLike(Protocol):
    num_layers: int
    def __call__(self, token_ids: mx.array, **kwargs: Any) -> mx.array: ...


@dataclass(frozen=True, slots=True)
class QuantaArtifactInfo:
    root: Path
    model_type: str | None


@dataclass(slots=True)
class OmlxGenerationOutput:
    """Structural match for ``omlx.engine.base.GenerationOutput``."""

    text: str
    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = "stop"
    new_text: str = ""
    finished: bool = True
    tool_calls: list[dict[str, Any]] | None = None
    cached_tokens: int = 0


def detect_quanta_artifact(path: str | Path) -> QuantaArtifactInfo | None:
    """Return artifact info when ``path`` is a quanta bake (manifest.json format=='quanta')."""
    root = Path(path).expanduser().resolve(strict=False)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        if json.loads(manifest.read_text()).get("format") != "quanta":
            return None
        cfg = json.loads((root / "config.json").read_text())
        mt = cfg.get("text_config", {}).get("model_type") or cfg.get("model_type")
    except (OSError, ValueError):
        return None
    return QuantaArtifactInfo(root=root, model_type=mt if isinstance(mt, str) else None)


# ``quanta.dsv4.encoding.encode_chat`` (the DSV4 prompt renderer) accepts only these kwargs; the
# engine forwards a superset of chat controls (add_generation_prompt / tools / enable_thinking / …)
# that must be filtered out before calling it, not passed through.
_DSV4_CHAT_KEYS = ("thinking_mode", "reasoning_effort", "context", "drop_thinking", "add_default_bos_token")


class _DSV4TokenizerAdapter:
    """Adapt :class:`~quanta.dsv4.tokenizer.DeepSeekV4Tokenizer` to the engine's tokenizer contract.

    The DSV4 tokenizer predates this contract: its ``encode`` has no ``allow_special`` flag (it always
    interprets added-token strings, like the Nemotron tokenizer), and it renders chat via
    :func:`quanta.dsv4.encoding.encode_chat` rather than an ``apply_chat_template``. This thin wrapper
    bridges both *without touching the parity-gated tokenizer*: ``encode`` accepts and ignores
    ``allow_special``; ``apply_chat_template`` renders through ``encode_chat`` with the request's chat
    kwargs filtered to the ones that renderer understands (mapping the cross-model ``enable_thinking``
    toggle onto DSV4's ``thinking_mode``). ``decode`` / ``eos_id`` / ``stop_ids`` / ``bos_id`` delegate.
    It deliberately exposes no ``decode_bytes`` / ``n_base`` / ``id_to_special``, so the streaming
    detokenizer takes its string-fallback path — DSV4's ``decode`` already emits special-token markers
    verbatim, which is the engine's raw-output contract."""

    def __init__(self, tokenizer: Any) -> None:
        self._tk = tokenizer
        self.bos_id = getattr(tokenizer, "bos_id", None)
        self.eos_id = getattr(tokenizer, "eos_id", None)
        self.stop_ids = getattr(tokenizer, "stop_ids", ())

    def encode(self, text: str, *, add_bos: bool = False, allow_special: bool = False) -> list[int]:
        del allow_special  # DSV4 always interprets added-token strings (see class docstring)
        return list(self._tk.encode(text, add_bos=add_bos))

    def decode(self, ids: Sequence[int], **kw: Any) -> str:
        return str(self._tk.decode(ids, **kw))

    def apply_chat_template(self, messages: list[dict[str, Any]], *, tokenize: bool = False,
                            add_generation_prompt: bool = True, tools: Any = None,
                            **kwargs: Any) -> Any:
        """Render the DSV4 chat prompt (string by default, ids when ``tokenize=True``).

        ``add_generation_prompt`` / ``tools`` are accepted for contract compatibility and ignored:
        DSV4's format always appends the assistant generation prompt and has no tool-call template."""
        from quanta.dsv4.encoding import encode_chat

        del add_generation_prompt, tools
        params = {k: kwargs[k] for k in _DSV4_CHAT_KEYS if k in kwargs}
        if "thinking_mode" not in params:
            et = kwargs.get("enable_thinking", kwargs.get("thinking"))
            if et is not None:
                params["thinking_mode"] = "thinking" if et else "chat"
        text = encode_chat(messages, **params)
        return self.encode(text, add_bos=False) if tokenize else text


class _RenderChatAdapter:
    """Adapt a quanta tokenizer that renders chat via ``render_chat`` (MiniMax-M2.7 / Qwen3.5) to the
    engine's tokenizer contract.

    Like :class:`_DSV4TokenizerAdapter`, these tokenizers predate the contract: ``encode`` has no
    ``allow_special`` flag (they always interpret added-token strings) and they expose
    ``render_chat``/``encode_chat`` rather than ``apply_chat_template``. This thin wrapper bridges both
    *without touching the parity-gated tokenizer*: ``encode`` accepts and ignores ``allow_special``;
    ``apply_chat_template`` renders through ``render_chat`` with the request's chat kwargs (mapping the
    cross-model ``thinking`` toggle onto ``enable_thinking``). ``decode``/``eos_id``/``stop_ids``/
    ``bos_id`` delegate. It deliberately exposes no ``decode_bytes`` / ``n_base`` / ``id_to_special``,
    so the streaming detokenizer takes its string-fallback path — these tokenizers' ``decode`` already
    emits special-token markers verbatim (the engine's raw-output contract)."""

    def __init__(self, tokenizer: Any) -> None:
        self._tk = tokenizer
        self.bos_id = getattr(tokenizer, "bos_id", None)
        self.eos_id = getattr(tokenizer, "eos_id", None)
        self.stop_ids = getattr(tokenizer, "stop_ids", None) or getattr(tokenizer, "_stop_ids", ())

    def encode(self, text: str, *, add_bos: bool = False, allow_special: bool = False) -> list[int]:
        del allow_special  # these tokenizers always interpret added-token strings (see class docstring)
        return list(self._tk.encode(text, add_bos=add_bos))

    def decode(self, ids: Sequence[int], **kw: Any) -> str:
        return str(self._tk.decode(ids, **kw))

    def apply_chat_template(self, messages: list[dict[str, Any]], *, tokenize: bool = False,
                            add_generation_prompt: bool = True, tools: Any = None,
                            **kwargs: Any) -> Any:
        """Render the chat prompt (text by default, ids when ``tokenize=True``) via ``render_chat``.

        The cross-model ``thinking`` toggle is mapped onto ``enable_thinking`` (Qwen reads it directly;
        MiniMax's template ignores it harmlessly). Extra template kwargs ride through to the renderer's
        ``**kwargs`` → jinja, which silently ignores any it does not reference."""
        params = dict(kwargs)
        if "enable_thinking" not in params and "thinking" in params:
            params["enable_thinking"] = params["thinking"]
        params.pop("thinking", None)
        text = self._tk.render_chat(messages, add_generation_prompt=add_generation_prompt,
                                    tools=tools, **params)
        return self.encode(text, add_bos=False) if tokenize else text


def _default_runtime_loader(root: Path) -> tuple[RuntimeLike, TokenizerLike]:
    """Build the resident runtime + tokenizer for an artifact, dispatched on ``model_type``.

    Refuses to load an artifact whose model class has no resident runtime rather than
    silently building the wrong one (e.g. a Nemotron bake through the Kimi path).
    """
    info = detect_quanta_artifact(root)
    mt = (info.model_type if info else None) or ""
    # DeepSeek-V4-Flash (HC + compressed-KV; GPT-2 byte-level BPE) — checked before the V3 ``deepseek``
    # prefix below, which the ``deepseek_v4`` model_type would otherwise be swallowed by.
    if mt.startswith("deepseek_v4"):
        from quanta.dsv4.runtime import DSV4ResidentModel
        from quanta.dsv4.tokenizer import DeepSeekV4Tokenizer

        tok = _DSV4TokenizerAdapter(DeepSeekV4Tokenizer.from_pretrained(str(root)))
        return DSV4ResidentModel(root), tok
    if mt.startswith("kimi") or mt.startswith("deepseek"):  # Kimi / DeepSeek-V3 family (MLA)
        from quanta.runtime import ResidentModel
        from quanta.tokenizer import KimiTokenizer

        rm = ResidentModel(root)
        return rm, KimiTokenizer(root, bos_id=rm.cfg.bos_token_id)
    if mt.startswith("nemotron"):
        from quanta.nemotron.runtime import NemotronResidentModel
        from quanta.nemotron.tokenizer import NemotronTokenizer

        return NemotronResidentModel(root), NemotronTokenizer(root)
    if mt.startswith("glm"):  # GLM-5.1 (MLA + DSA Lightning-Indexer); tokenizer conforms directly
        from quanta.glm.runtime import GLMResidentModel
        from quanta.glm.tokenizer import GLMTokenizer

        return GLMResidentModel(root), GLMTokenizer.from_pretrained(str(root))
    if mt.startswith("minimax"):  # MiniMax-M2.7 (GQA); render_chat tokenizer needs the adapter
        from quanta.minimax.runtime import MiniMaxResidentModel
        from quanta.minimax.tokenizer import MiniMaxTokenizer

        return MiniMaxResidentModel(root), _RenderChatAdapter(MiniMaxTokenizer.from_pretrained(str(root)))
    if mt.startswith("qwen3_5") or mt.startswith("qwen3.5"):  # Qwen3.5 hybrid (DeltaNet + gated GQA)
        from quanta.qwen35.runtime import Qwen35ResidentModel
        from quanta.qwen35.tokenizer import Qwen35Tokenizer

        return Qwen35ResidentModel(root), _RenderChatAdapter(Qwen35Tokenizer.from_pretrained(str(root)))
    if mt == "qwen2" or mt.startswith("qwen2."):  # Qwen2.5-1M (dense GQA + QKV biases + DCA)
        from quanta.qwen25.runtime import Qwen25ResidentModel
        from quanta.qwen25.tokenizer import Qwen25Tokenizer

        return Qwen25ResidentModel(root), _RenderChatAdapter(Qwen25Tokenizer.from_pretrained(str(root)))
    if mt == "internlm2" or mt.startswith("internlm2"):  # InternLM2.5-7B-1M (dense GQA + dynamic-NTK)
        from quanta.internlm2.runtime import InternLM2ResidentModel
        from quanta.internlm2.tokenizer import InternLM2Tokenizer

        return (InternLM2ResidentModel(root),
                _RenderChatAdapter(InternLM2Tokenizer.from_pretrained(str(root))))
    raise OmlxShimError(
        f"no resident runtime for quanta artifact model_type={mt!r} "
        "(supported: kimi, deepseek_v3, deepseek_v4, nemotron, glm, minimax, qwen2, qwen3_5, internlm2)"
    )


def _apply_penalties(logits: mx.array, prev: Sequence[int] | None, rep: float, freq: float,
                     pres: float) -> mx.array:
    """Repetition (CTRL/HF multiplicative) + frequency/presence (OpenAI additive) penalties.

    ``rep`` divides (logit>0) / multiplies (logit<0) the logit of any previously-emitted token; ``freq``
    subtracts ``count(token)·freq`` (scales with how often it was emitted); ``pres`` subtracts a flat
    ``pres`` for any token emitted at least once. All three touch only tokens that already appeared, so
    unseen tokens are unchanged. ``freq``/``pres`` discourage the repetition that makes models loop.
    """
    if not prev or (rep == 1.0 and freq == 0.0 and pres == 0.0):
        return logits
    toks = mx.array([int(t) for t in prev], dtype=mx.int32)
    counts = mx.zeros(logits.shape[0]).at[toks].add(1.0)  # per-token emission count (repeats accumulate)
    seen = counts > 0
    out = logits
    if rep != 1.0:
        out = mx.where(seen, mx.where(out > 0, out / rep, out * rep), out)
    if freq != 0.0 or pres != 0.0:
        out = out - counts * freq - seen.astype(out.dtype) * pres
    return out


def _apply_top_p(logits: mx.array, top_p: float) -> mx.array:
    if not 0.0 < top_p < 1.0:
        return logits
    order = mx.argsort(-logits)
    ordered = logits[order]
    cmass = mx.cumsum(mx.softmax(ordered)) - mx.softmax(ordered)
    cutoff = mx.min(mx.where(cmass < top_p, ordered, mx.array(mx.inf)))
    return mx.where(logits < cutoff, mx.array(-mx.inf), logits)


def _apply_min_p(logits: mx.array, min_p: float) -> mx.array:
    if min_p <= 0.0:
        return logits
    probs = mx.softmax(logits)
    return mx.where(probs < min_p * mx.max(probs), mx.array(-mx.inf), logits)


def _earliest_stop(text: str, stops: Sequence[str]) -> int | None:
    """Index of the earliest occurrence of any non-empty stop string in ``text``, else None."""
    hits = [i for i in (text.find(s) for s in stops if s) if i >= 0]
    return min(hits) if hits else None


class _Detok:
    """Incremental **raw-output** detokenizer for streaming.

    Special tokens (``</think>``, ``<|tool_calls_section_begin|>`` …) are emitted as their literal
    marker strings (via the tokenizer's ``id_to_special`` map) so the oMLX server's reasoning/tool
    parsers can act on them — the eos tokens are handled by the caller and never reach here. Ordinary
    tokens use the byte-accurate UTF-8 path (``decode_bytes``): accumulate raw bytes and flush only
    complete UTF-8, so a multi-byte token split across steps never surfaces as a replacement char.
    Falls back to delta-on-decoded-string for tokenizers that expose only ``decode`` (test stubs).
    ``text`` is the cumulative decoded output.
    """

    def __init__(self, tokenizer: TokenizerLike | None) -> None:
        self._tk = tokenizer
        self._bytes_ok = tokenizer is not None and hasattr(tokenizer, "decode_bytes")
        self._n_base = getattr(tokenizer, "n_base", None)
        self._id2sp = getattr(tokenizer, "id_to_special", None) or {}
        self._buf = b""
        self._ids: list[int] = []
        self.text = ""

    def add(self, tid: int) -> str:
        if self._tk is None:
            raise OmlxShimError("detokenizer has no tokenizer")
        tid = int(tid)
        if self._n_base is not None and tid >= self._n_base:  # special token → literal marker (raw)
            piece = ""
            if self._buf:  # a partial ordinary byte can't complete across a special; flush it
                piece += self._buf.decode("utf-8", "replace")
                self._buf = b""
            piece += self._id2sp.get(tid, "")
            self.text += piece
            return piece
        if self._bytes_ok:
            self._buf += bytes(self._tk.decode_bytes([tid]))
            try:
                piece = self._buf.decode("utf-8")
                self._buf = b""
            except UnicodeDecodeError as e:  # keep the incomplete tail buffered for next token
                piece = self._buf[: e.start].decode("utf-8")
                rest = self._buf[e.start :]
                if len(rest) >= 4:  # longer than any valid incomplete tail ⇒ genuinely invalid
                    piece += rest.decode("utf-8", "replace")
                    rest = b""
                self._buf = rest
            self.text += piece
            return piece
        self._ids.append(tid)  # string fallback: re-decode, emit the delta
        full = str(self._tk.decode(self._ids))
        piece = full[len(self.text) :]
        self.text = full
        return piece


class _DecodeStepper(Protocol):
    """Per-request decode session. ``prefill`` runs the prompt, ``step`` runs one token; both return
    the **last-position** logits row ``[vocab]``. This is the only model-specific seam in generation —
    it hides the call convention (MLA KV cache + absorbed decode vs. the Nemotron hybrid's threaded
    ``(ssm, conv)`` state) so the sampling / detok / stop loop below is shared across model classes."""

    def prefill(self, prompt_ids: list[int]) -> mx.array: ...
    def step(self, tok: int) -> mx.array: ...


class _MLAStepper:
    """Kimi / DeepSeek (MLA): one :class:`MLACache` per layer, sparse prefill, absorbed single-token
    decode. The caches are mutated in place; the offset is read back from cache 0 each step."""

    def __init__(self, runtime: RuntimeLike, *, quantized_kv: bool, sparse: Any) -> None:
        self._rt = runtime
        self._sparse = sparse
        self._caches = [MLACache(quantized=quantized_kv) for _ in range(runtime.num_layers)]

    def prefill(self, prompt_ids: list[int]) -> mx.array:
        return self._rt(mx.array(prompt_ids), caches=self._caches, sparse=self._sparse)[0, -1]

    def step(self, tok: int) -> mx.array:
        return self._rt(mx.array([tok]), caches=self._caches,
                        offset=self._caches[0].offset, absorbed=True)[0, -1]


class _NemotronStepper:
    """Nemotron-H hybrid: thread per-layer state — a growing ``KVCache`` on attention layers, the
    ``(ssm, conv)`` recurrence on mamba layers, nothing on MoE. The runtime returns
    ``(logits, ssm, conv)``; prefill runs the chunked SSD, each step the O(1) recurrence."""

    def __init__(self, runtime: Any) -> None:
        from quanta.nemotron.generate import attn_caches

        self._rt = runtime
        self._caches = attn_caches(runtime)  # KVCache per attention layer, None elsewhere
        self._ssm: list[Any] | None = None
        self._conv: list[Any] | None = None

    def prefill(self, prompt_ids: list[int]) -> mx.array:
        logits, self._ssm, self._conv = self._rt(mx.array(prompt_ids), caches=self._caches)
        return logits[0, -1]

    def step(self, tok: int) -> mx.array:
        logits, self._ssm, self._conv = self._rt(
            mx.array([tok]), caches=self._caches, ssm=self._ssm, conv=self._conv)
        return logits[0, -1]


class _DSV4Stepper:
    """DeepSeek-V4-Flash: one :class:`~quanta.dsv4.decode.DSV4Cache` shared across all layers (the HC
    residual carries no cache; the per-layer attention KV / compressed-KV / indexer state does).

    ``prefill`` seeds the cache by stepping the prompt **one token at a time** — bounded memory, never
    materializing ``[1,T,vocab]`` for a long prompt — and ``step`` decodes one token at the running
    absolute offset (tracked here, mirroring :func:`quanta.dsv4.generate.generate`; the runtime's
    single-token forward grows the cache in place). Both return the last-position logits row ``[vocab]``.
    """

    def __init__(self, runtime: RuntimeLike, *, cache: Any | None = None) -> None:
        from quanta.dsv4.decode import DSV4Cache

        self._rt = runtime
        self._cache = cache if cache is not None else DSV4Cache(runtime.num_layers)
        self._offset = 0

    def prefill(self, prompt_ids: list[int]) -> mx.array:
        if not prompt_ids:
            raise OmlxShimError("DSV4 prefill got an empty prompt")
        row = None
        for pos, tid in enumerate(prompt_ids):  # seed positions 0..len-1, keep only the last logits
            row = self._rt(mx.array([tid]), caches=self._cache, offset=pos)[0, -1]
        self._offset = len(prompt_ids)
        mx.eval(row)
        return row

    def step(self, tok: int) -> mx.array:
        row = self._rt(mx.array([tok]), caches=self._cache, offset=self._offset)[0, -1]
        self._offset += 1
        return row


class _SingleTokenStepper:
    """Generic single-token decode session for the GLM-5.1 / MiniMax-M2.7 / Qwen3.5 resident runtimes.

    All three expose the same single-token contract as DSV4 (``__call__(token_ids, *, caches, offset)``
    grows a per-layer decode cache in place), so one stepper serves them: seed the cache by stepping the
    prompt one token at a time (bounded memory — never materialize ``[1,T,vocab]`` for a long prompt),
    then decode one token per step at the running absolute offset. This mirrors each model's standalone
    ``generate`` exactly. Returns the last-position logits row ``[vocab]``. The cache (built by
    :meth:`QuantaOmlxEngine._make_stepper` for the model class) is mutated in place."""

    def __init__(self, runtime: RuntimeLike, cache: Any) -> None:
        self._rt = runtime
        self._cache = cache
        self._offset = 0

    def prefill(self, prompt_ids: list[int]) -> mx.array:
        if not prompt_ids:
            raise OmlxShimError("prefill got an empty prompt")
        row = None
        for pos, tid in enumerate(prompt_ids):  # seed positions 0..len-1, keep only the last logits
            row = self._rt(mx.array([tid]), caches=self._cache, offset=pos)[0, -1]
        self._offset = len(prompt_ids)
        mx.eval(row)
        return row

    def step(self, tok: int) -> mx.array:
        row = self._rt(mx.array([tok]), caches=self._cache, offset=self._offset)[0, -1]
        self._offset += 1
        return row


# --- Reasoning-parser contract (LOCAL STUB) ---------------------------------------------------------
# stub — replaced when #150 (Qwen oMLX shim agent) lands the formal `ReasoningParser` Protocol in
# `quanta.shim.tool_parsers` (or `quanta.shim.parsers_contract`). We define this stub so DSV4 /
# Nemotron code can structurally conform NOW; once the real Protocol is importable, the stub becomes
# a no-op (the runtime check still passes — both are duck-typed Protocols over the same surface).
#
# AUDIT (this commit): DSV4 has no custom reasoning parser in `quanta.shim.tool_parsers` (the engine
# emits raw ``<think>…</think>`` text and oMLX's stock ``thinking.extract_thinking`` handles it — see
# the per-model contract docstring at the top of this file). Nemotron likewise has no custom parser
# ("Kimi tool-patch, Nemotron none" — project memory). So the conformance surface below has nothing
# to register today; it's plumbed only so a future parser added by either model would conform without
# touching this file.
try:  # prefer the formal Protocol when the Qwen agent (#150) has landed it
    from quanta.shim.tool_parsers import ReasoningParser as _ReasoningParser  # type: ignore[attr-defined]
except ImportError:  # stub Protocol — same surface as the eventual real one
    class _ReasoningParser(Protocol):
        """Minimal reasoning-parser contract used until #150 lands the formal Protocol.

        A reasoning parser, given the model's raw output, returns ``(reasoning, content)`` — the
        ``<think>…</think>`` body and the remainder. Mirrors :func:`omlx.api.thinking.extract_thinking`
        so a quanta parser can be a drop-in replacement for the stock one when the model's markup is
        non-standard (DSV4 / Nemotron currently use the stock markup, so neither registers here)."""

        def extract(self, text: str) -> tuple[str, str]: ...


def _conformant_reasoning_parsers() -> tuple[_ReasoningParser, ...]:
    """Return the tuple of reasoning parsers the engine knows about for DSV4 / Nemotron — currently
    empty (both use oMLX's stock parser). When #150 lands, the formal Protocol replaces ``_ReasoningParser``
    and any model-specific parser registered in :mod:`quanta.shim.tool_parsers` would be added here.
    The contract conformance check (the ``hasattr`` below) stays exact under both Protocols."""
    parsers: tuple[_ReasoningParser, ...] = ()
    for p in parsers:  # pragma: no cover - empty today; enforces shape when populated
        if not hasattr(p, "extract"):
            raise OmlxShimError(
                f"reasoning parser {type(p).__name__} does not conform to ReasoningParser (no .extract)")
    return parsers


# --- Batched-decode sessions ------------------------------------------------------------------------
class _BatchedSession(Protocol):
    """Per-engine batched decode session, fronts a model-specific batched runtime.

    The session is a thin adapter over ``quanta.{dsv4,nemotron}.batched_runtime`` (lazy-imported, so
    the shim still loads when the sibling modules are not merged). It owns ``capacity`` slots, each
    holding one in-flight request's decode state; the engine drives admission / step / release while
    sampling / eos / detok stay per-stream in :meth:`QuantaOmlxEngine.batched_stream_generate`.

    ``capacity`` is the static batch size the underlying runtime was built for; the engine never
    admits more concurrent streams than that. ``num_layers`` is forwarded for parity with the
    single-stream stepper's surface.
    """

    capacity: int
    num_layers: int

    def admit(self, slot: int, prompt_ids: list[int], max_new: int | None = None) -> mx.array: ...
    def step_batch(self, slot_to_token: dict[int, int]) -> dict[int, mx.array]: ...
    def release(self, slot: int) -> None: ...

    # paged-KV stats surface (#152): default OFF; a session owning a live PagedKVCacheManager
    # (#174/#175) overrides these so the engine can report real prefix-reuse stats to oMLX.
    prefix_cache_enabled: bool

    def get_cache_stats(self) -> dict[str, Any] | None: ...


class _BaseBatchedSession:
    """Continuous-batching session: owns one decode cache per slot and drives a model's batched
    runtime through its real **caller-owned-state** API — ``make_cache`` / ``prefill(ids, cache)`` /
    ``step_batch(token_ids, caches, offsets)``. The engine
    (:meth:`QuantaOmlxEngine.batched_stream_generate`) only ever sees the slot contract
    (``admit`` / ``step_batch(slot->token)`` / ``release``), so this class IS the
    slot <-> per-stream-state adapter; the runtime stays a pure compute unit (its per-stream caches
    live here, which is also where #152's paged cache will swap in).

    ``admit`` builds a fresh per-slot cache and prefills the prompt into it (returning the final-
    position logits row the engine samples). ``step_batch`` gathers the alive slots' caches + tokens
    and runs one fused batched decode step. ``release`` drops the slot's cache. Subclasses supply five
    model-specific hooks: runtime construction, the per-stream cache factory, prompt/token formatting,
    and how a step reports each stream's absolute offset."""

    def __init__(self, root: str | Path | None = None, *, capacity: int,
                 runtime: Any | None = None, manager: Any | None = None,
                 rec_cache: Any | None = None, paged_kv: bool = False,
                 block_size: int = 32, max_blocks: int = 4096, rec_capacity: int = 4096,
                 model_name: str = "", native_decode: bool = True) -> None:
        if runtime is None:
            if root is None:
                raise OmlxShimError(f"{type(self).__name__}: artifact root required when runtime is unset")
            runtime = self._make_runtime(root, capacity)
        self._rt = runtime
        self.capacity = int(capacity)
        self.num_layers = getattr(runtime, "num_layers", 0)
        # --- form-2 native decode (#153): if the runtime exposes the persistent-batched-state API
        # (Nemotron's step_batch_native + make_batched_state), hold ONE BatchedMambaState across steps
        # for the current alive-slot set so the Mamba recurrence is never re-concatenated per step. Other
        # runtimes (no native API) transparently fall back to the form-1 step_batch dispatch. The state is
        # flushed back into the per-stream triples on any slot-set change (admit/release) — see
        # _flush_native / _run_native. Gated model-free in parity/nemotron_native_serving_test.py.
        self._native_capable = (hasattr(runtime, "step_batch_native")
                                and hasattr(runtime, "make_batched_state"))
        self._native_decode = bool(native_decode) and self._native_capable
        self._nat: Any | None = None                       # cached BatchedMambaState | None
        self._nat_slots: tuple[int, ...] = ()              # the alive slots it was assembled for
        self._caches: dict[int, Any] = {}                  # unpaged: slot -> per-stream cache
        # --- paged mode (#152): a manager may be injected (tests) or built here from the runtime's
        # paged spec when ``paged_kv`` is on. Engaged iff a PagedKVCacheManager ends up present. ------
        if manager is None and paged_kv:
            from quanta.paged import PagedKVCacheManager, RecurrentPrefixCache  # lazy (no oMLX dep)
            spec = getattr(runtime, "paged_kv_spec", None)
            if spec is None:
                raise OmlxShimError(
                    f"{type(self).__name__}: paged_kv=True but runtime {type(runtime).__name__} "
                    "exposes no 'paged_kv_spec'")
            manager = PagedKVCacheManager(
                num_layers=int(spec["n_layers"]), block_size=int(block_size), max_blocks=int(max_blocks),
                group_size=int(spec["group_size"]), bits=int(spec["bits"]),
                quantized=bool(spec["quantized"]), model_name=model_name,
                single_stream=bool(spec.get("single_stream", False)))
            if rec_cache is None and bool(getattr(runtime, "has_recurrent_state", False)):
                rec_cache = RecurrentPrefixCache(block_size=int(block_size), model_name=model_name,
                                                 capacity=int(rec_capacity))
        self._manager = manager                            # PagedKVCacheManager | None
        self._rec = rec_cache                              # RecurrentPrefixCache | None
        self._slots: dict[int, tuple[Any, Any]] = {}       # paged: slot -> (SeqHandle, paged_state)
        self._block_size = int(manager.block_size) if manager is not None else 0
        self._has_recurrent = False
        if manager is not None:
            # rule 6: a paged runtime MUST expose the paged contract — never silently half-page.
            for attr in ("make_paged_state", "prefill_paged", "has_recurrent_state"):
                if not hasattr(runtime, attr):
                    raise OmlxShimError(
                        f"{type(self).__name__}: paged runtime {type(runtime).__name__} missing {attr!r}")
            self._has_recurrent = bool(runtime.has_recurrent_state)
            if self._has_recurrent and not hasattr(runtime, "get_recurrent_state"):
                raise OmlxShimError(
                    f"{type(self).__name__}: recurrent paged runtime missing 'get_recurrent_state'")

    # --- model-specific hooks (subclasses override) --------------------------
    def _make_runtime(self, root: str | Path, capacity: int) -> Any:
        raise NotImplementedError

    def _new_cache(self) -> Any:
        raise NotImplementedError

    def _to_prefill_ids(self, prompt_ids: list[int]) -> Any:
        return mx.array(prompt_ids)

    def _to_step_tokens(self, tokens: list[int]) -> list[Any]:
        return [mx.array([t]) for t in tokens]

    def _step_offsets(self, caches: list[Any]) -> list[int] | None:
        return [c.offset for c in caches]

    def _configure_cache(self, cache: Any, prompt_ids: list[int], max_new: int | None) -> None:
        """Per-request cache setup after construction, BEFORE prefill. Default no-op; a subclass whose
        cache carries request-lifetime state (Qwen3.5's dynamic-YaRN factor) overrides this to fix that
        state from the request's known span (``len(prompt_ids) + max_new``) so prefill and every decode
        step share it. ``max_new`` is the stream's generation budget (``None`` when the caller did not
        budget the request)."""

    # --- form-2 native decode helpers (#153) ---------------------------------
    def _state_for(self, slot: int) -> Any:
        """The per-stream state triple for a slot, in whichever mode is active (paged or unpaged)."""
        return self._slots[slot][1] if self._manager is not None else self._caches[slot]

    def _flush_native(self) -> None:
        """Write the cached batched recurrent state back into its slots' per-stream triples and drop it.
        Called whenever the alive-slot set is about to change (admit/release) or differs from the cached
        set, so a subsequent rebuild reads CURRENT per-stream state. KV lives in the (shared-by-ref)
        per-stream caches so only ssm/conv need un-batching; cheap (slice views, MLX-immutable)."""
        if self._nat is None:
            return
        self._nat.scatter_to([self._state_for(s) for s in self._nat_slots])
        self._nat = None
        self._nat_slots = ()

    def _run_native(self, slots: list[int], states: list[Any], tokens: list[Any]) -> list[Any]:
        """One form-2 decode step over ``slots``: reuse the cached BatchedMambaState if it matches the
        current alive-slot set, else flush the old one and reassemble from the (now-current) per-stream
        states. The recurrence threads in place across steps with no per-step concat (the IO win)."""
        if self._nat is None or self._nat_slots != tuple(slots):
            self._flush_native()
            self._nat = self._rt.make_batched_state(states)
            self._nat_slots = tuple(slots)
        return self._rt.step_batch_native(tokens, self._nat)

    # --- engine-facing slot contract -----------------------------------------
    def admit(self, slot: int, prompt_ids: list[int], max_new: int | None = None) -> mx.array:
        if not prompt_ids:
            raise OmlxShimError(f"{type(self).__name__}: admit got an empty prompt")
        if not 0 <= slot < self.capacity:
            raise OmlxShimError(
                f"{type(self).__name__}: admit slot {slot} out of range [0, {self.capacity})")
        if slot in self._caches or slot in self._slots:
            raise OmlxShimError(f"{type(self).__name__}: admit slot {slot} already in use")
        self._flush_native()                               # alive-slot set changes -> rebuild next step
        if self._manager is not None:
            return self._admit_paged(slot, prompt_ids)     # paged keepers carry no request-lifetime state
        cache = self._new_cache()
        self._configure_cache(cache, prompt_ids, max_new)  # request-lifetime cache state (YaRN pin) pre-prefill
        logits = self._rt.prefill(self._to_prefill_ids(prompt_ids), cache)
        self._caches[slot] = cache
        row = logits[0, -1]  # [vocab] at the final prompt position
        mx.eval(row)
        return row

    def step_batch(self, slot_to_token: dict[int, int]) -> dict[int, mx.array]:
        if not slot_to_token:
            return {}
        if self._manager is not None:
            return self._step_paged(slot_to_token)
        slots = sorted(slot_to_token)
        missing = [s for s in slots if s not in self._caches]
        if missing:
            raise OmlxShimError(f"{type(self).__name__}: step_batch on unadmitted slots {missing}")
        caches = [self._caches[s] for s in slots]
        tokens = self._to_step_tokens([slot_to_token[s] for s in slots])
        if self._native_decode:
            outs = self._run_native(slots, caches, tokens)               # form-2 persistent state
        else:
            outs = self._rt.step_batch(tokens, caches, self._step_offsets(caches))
        if len(outs) != len(slots):
            raise OmlxShimError(
                f"{type(self).__name__}: step_batch returned {len(outs)} rows for {len(slots)} slots")
        result = {s: outs[i][0, -1] for i, s in enumerate(slots)}  # [vocab] per slot
        mx.eval(list(result.values()))
        return result

    def release(self, slot: int) -> None:
        self._flush_native()                               # write back before the slot's triple is dropped
        if self._manager is not None:
            entry = self._slots.pop(slot, None)
            if entry is not None:
                self._manager.free(entry[0])     # ref-- (prefix blocks stay resident for LRU reuse)
            return
        self._caches.pop(slot, None)

    # --- paged slot contract (#152) ------------------------------------------
    def _admit_paged(self, slot: int, prompt_ids: list[int]) -> mx.array:
        """Paged admit: re-reference the resident attention-KV prefix blocks + restore the recurrent
        boundary state, then prefill ONLY the uncached suffix. A whole prompt that matches is trimmed
        by one block so the last token is recomputed — KV reuse alone can't yield the last-position
        logits the sampler needs (logits are not cached)."""
        mgr, rec = self._manager, self._rec
        seq = mgr.new_sequence()
        # match against prompt_ids[:-1] so at least the final token is always recomputed.
        n_attn = mgr.match_prefix(seq, prompt_ids[:-1]) if len(prompt_ids) > 1 else 0
        rec_state = None
        if n_attn and self._has_recurrent:
            rec_state = rec.lookup_at(prompt_ids, n_attn) if rec is not None else None
            if rec_state is None:
                # no recurrent boundary snapshot here -> can't suffix-skip a hybrid; recompute in full.
                mgr.truncate(seq, 0)
                n_attn = 0
        suffix = prompt_ids[n_attn:]
        mgr.advance(seq, suffix)                                   # opens [n_attn, len) for KV writes
        state = self._rt.make_paged_state(mgr, seq)
        logits, new_boundaries = self._rt.prefill_paged(
            self._to_prefill_ids(suffix), state,
            offset=n_attn, recurrent_in=rec_state, block_size=self._block_size)
        mgr.commit(seq)                                            # content-hash any block that filled
        if rec is not None:
            for pos, payload in new_boundaries:                   # store recurrent boundary snapshots
                rec.store_at(prompt_ids, pos, payload)
        self._slots[slot] = (seq, state)
        row = logits[0, -1]
        mx.eval(row)
        return row

    def _step_paged(self, slot_to_token: dict[int, int]) -> dict[int, mx.array]:
        """Paged decode step: open one position per alive slot (``advance``), run the unchanged batched
        forward (attention writes through the paged views; recurrent state mutates per-stream), then
        ``commit`` and snapshot the recurrent state at any block boundary that just filled."""
        slots = sorted(slot_to_token)
        missing = [s for s in slots if s not in self._slots]
        if missing:
            raise OmlxShimError(f"{type(self).__name__}: step_batch on unadmitted slots {missing}")
        mgr, rec = self._manager, self._rec
        for s in slots:
            mgr.advance(self._slots[s][0], [int(slot_to_token[s])])
        states = [self._slots[s][1] for s in slots]
        offsets = [self._slots[s][0].length - 1 for s in slots]    # abs position of each new token
        tokens = self._to_step_tokens([slot_to_token[s] for s in slots])
        if self._native_decode:
            outs = self._run_native(slots, states, tokens)               # form-2 persistent state
        else:
            outs = self._rt.step_batch(tokens, states, offsets)
        if len(outs) != len(slots):
            raise OmlxShimError(
                f"{type(self).__name__}: step_batch returned {len(outs)} rows for {len(slots)} slots")
        for i, s in enumerate(slots):
            seq = self._slots[s][0]
            mgr.commit(seq)
            if self._has_recurrent and rec is not None and seq.length % self._block_size == 0:
                # form-2 holds the live recurrent state batched -> snapshot row i, not the stale triple.
                payload = (self._nat.recurrent_row(i) if self._native_decode
                           else self._rt.get_recurrent_state(states[i]))
                rec.store_at(seq.token_ids, seq.length, payload)
        result = {s: outs[i][0, -1] for i, s in enumerate(slots)}
        mx.eval(list(result.values()))
        return result

    # --- paged-KV stats (real once a PagedKVCacheManager is wired in — #174/#175) -------------
    @property
    def prefix_cache_enabled(self) -> bool:
        return self._manager is not None

    def get_cache_stats(self) -> dict[str, Any] | None:
        if self._manager is None:
            return None
        stats = self._manager.get_stats().to_dict()
        if self._rec is not None:
            stats["recurrent"] = self._rec.get_stats().__dict__
        return stats


class _DSV4BatchedSession(_BaseBatchedSession):
    """DeepSeek-V4-Flash batched session — one per-slot decode cache, fused batched MoE across the alive
    slots. With the runtime's #18 KV arena on (default), ``make_cache`` hands out an
    ``_ArenaCacheHandle`` leasing one ``max_batch`` row (latent + compressed extras); a discrete
    ``DSV4Cache`` otherwise. DSV4's ``step_batch`` validates each stream's declared offset against its
    cache, so the offsets come straight from the cache's ``offset``. Lazy-imports the batched runtime
    so the shim still loads when that module is absent (loud on use, rule 6)."""

    def _make_runtime(self, root: str | Path, capacity: int) -> Any:
        from quanta.dsv4 import batched_runtime as _dbr  # lazy

        return _dbr.DSV4BatchedResidentModel(root, max_batch=capacity)

    def _new_cache(self) -> Any:
        return self._rt.make_cache()

    def release(self, slot: int) -> None:
        """Free the slot AND return its #18 KV-arena row to the runtime's free-list. With ``kv_arena``
        on (default), ``make_cache`` leases one ``max_batch`` row per stream; that row MUST be returned
        on release so a later admit can reuse it (else the arena caps total served requests at
        ``max_batch``). No-op for a discrete ``DSV4Cache`` (no ``row``) or the paged path (its state
        lives in ``_slots``, freed by the base via the manager) — keyed off the cache having a ``row``,
        so a stub runtime without ``free_cache`` is never called into."""
        cache = self._caches.get(slot)
        super().release(slot)
        if cache is not None and getattr(cache, "row", None) is not None:
            self._rt.free_cache(cache)


class _NemotronBatchedSession(_BaseBatchedSession):
    """Nemotron-H batched session — one ``(caches, ssm, conv)`` triple per slot (KV on the 8 attention
    layers, recurrent state on the 40 Mamba layers). Nemotron's ``step_batch`` reads each attention
    layer's ``KVCache.offset`` internally, so the session passes ``offsets=None``."""

    def _make_runtime(self, root: str | Path, capacity: int) -> Any:
        from quanta.nemotron import batched_runtime as _nbr  # lazy

        return _nbr.NemotronBatchedResidentModel(root, max_batch=capacity)

    def _new_cache(self) -> Any:
        return self._rt.make_stream_state()

    def _step_offsets(self, caches: list[Any]) -> list[int] | None:
        return None  # Nemotron derives per-stream offset from each KVCache internally


class _InternLM2BatchedSession(_BaseBatchedSession):
    """InternLM2.5-7B-Chat-1M batched session — one :class:`quanta.internlm2.decode.InternLM2Cache` per
    slot (dense GQA, int8 KV on all 32 layers, NO recurrent state). The clean #152 paged case: every
    layer's k/v pair is paged and nothing is content-addressed at a block boundary
    (``has_recurrent_state=False`` ⇒ the base ``_admit_paged`` skips the recurrent branch). ``step_batch``
    reads each ``InternLM2Cache.offset`` internally, so the unpaged path passes ``offsets=None`` (the
    paged path supplies explicit per-token offsets). Lazy-imports the batched runtime (loud on use, rule 6)."""

    def _make_runtime(self, root: str | Path, capacity: int) -> Any:
        from quanta.internlm2 import batched_runtime as _ibr  # lazy

        return _ibr.InternLM2BatchedResidentModel(root, max_batch=capacity)

    def _new_cache(self) -> Any:
        return self._rt.new_cache()

    def _step_offsets(self, caches: list[Any]) -> list[int] | None:
        return None  # InternLM2Cache.offset is authoritative (paged step passes explicit offsets)


class _Qwen35BatchedSession(_BaseBatchedSession):
    """Qwen3.5 batched session — one :class:`quanta.qwen35.decode.Qwen35Cache` per slot (int8 KV on the
    15 full-attn layers, recurrent GatedDeltaNet state on the 45 linear-attn layers). Qwen3.5's
    ``step_batch`` takes plain int token ids + per-stream offsets (``Qwen35Cache.offset``)."""

    def _make_runtime(self, root: str | Path, capacity: int) -> Any:
        from quanta.qwen35 import batched_runtime as _qbr  # lazy

        return _qbr.Qwen35BatchedResidentModel(root, max_batch=capacity)

    def _new_cache(self) -> Any:
        return self._rt.make_caches()

    def _to_prefill_ids(self, prompt_ids: list[int]) -> Any:
        return list(prompt_ids)

    def _to_step_tokens(self, tokens: list[int]) -> list[Any]:
        return [int(t) for t in tokens]

    def _configure_cache(self, cache: Any, prompt_ids: list[int], max_new: int | None) -> None:
        """Pin the dynamic-YaRN factor to the request's largest absolute position
        (``len(prompt_ids) + max_new``) BEFORE prefill, so the rotate-then-cache roped K use ONE
        consistent factor across prefill and every decode step. Serving past the native window REQUIRES
        this — :meth:`Qwen35Cache.yarn_seq` refuses to derive a drifting factor from the live position
        (rule 6: never silently corrupt the cached KV); at or below native it is a harmless no-op
        (factor 1.0). With no budget (``max_new is None``) the pin is skipped: the live-position path
        then serves correctly to the native window and fails loud past it (still never silent)."""
        if max_new is None:
            return
        cache.pin_yarn(len(prompt_ids) + int(max_new))


@dataclass(slots=True)
class _StreamState:
    """Per-stream state for the batched decode loop — isolates sampler / eos / detok / RNG / generated
    history so one stream finishing doesn't poison its slot-mate's history. ``next_key`` provides a
    fresh PRNG subkey per step (None for global RNG when ``rng`` is None, matching the single-stream
    loop's reproducibility contract)."""

    prompt_ids: list[int]
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repetition_penalty: float
    frequency_penalty: float
    presence_penalty: float
    eos: set[int]
    stop: list[str] | None
    rng: mx.array | None
    detok: "_Detok"
    generated: list[int]

    def next_key(self) -> mx.array | None:
        if self.rng is None:
            return None
        self.rng, sub = mx.random.split(self.rng)
        return sub


class QuantaOmlxEngine(_OmlxBaseEngine):
    """oMLX ``BaseEngine`` backed by the quanta resident runtime (model-specific decode stepper)."""

    def __init__(self, model_name: str, *, runtime: RuntimeLike | None = None,
                 tokenizer: TokenizerLike | None = None, runtime_loader=None,
                 output_cls: type[Any] = OmlxGenerationOutput, eos_token_ids: set[int] | None = None,
                 batched_session: Any | None = None, paged_kv: bool | None = None,
                 eagle: tuple[Any, mx.array, mx.array, tuple[int, ...], int | None] | None = None,
                 eagle_bits: int | None = 4, eagle_group_size: int = 64) -> None:
        self._model_name = model_name
        self._root = Path(model_name).expanduser().resolve(strict=False)
        self._runtime = runtime
        self._injected_runtime = runtime is not None  # explicit runtime (standalone/test) vs artifact load
        self._tokenizer = tokenizer
        self._runtime_loader = runtime_loader or _default_runtime_loader
        self._output_cls = output_cls
        self._eos = set(eos_token_ids or ())
        self._loaded = runtime is not None and tokenizer is not None
        self._active = 0
        # injected batched session (tests / advanced callers); None ⇒ built lazily on first use.
        self._batched_session = batched_session
        # #152 prefix-sharing paged KV — OFF until parity-green (rule 4); None ⇒ the package default.
        from quanta.paged import PAGED_KV_DEFAULT
        self._paged_kv = PAGED_KV_DEFAULT if paged_kv is None else bool(paged_kv)
        # InternLM2 EAGLE-3 spec-decode state (drafter, embed, head, capture_layers, head_bits) —
        # injected directly (tests) or lazily loaded from the artifact's embedded eagle/ sidecar by
        # :meth:`_ensure_eagle` (the only keeper whose spec path is a trained drafter, not native MTP).
        # ``eagle_bits`` is the drafter PTQ for serving (M3 operating point: int4 g64 → 1.42× lossless
        # @ k=2; the EAGLE guarantee holds for any drafter quality — quant only changes draft *speed*).
        self._eagle = eagle
        self._eagle_bits = eagle_bits
        self._eagle_group_size = eagle_group_size

    # --- BaseEngine properties -------------------------------------------------
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self._tokenizer

    @property
    def model_type(self) -> str | None:
        info = detect_quanta_artifact(self._root)
        return info.model_type if info else None

    @property
    def grammar_compiler(self):
        return None

    @property
    def prefix_cache_enabled(self) -> bool:
        """True only while a batched session with a live paged-KV manager is active (the #152 prefix
        win); False for single-stream decode and the unpaged batched path. Delegates to the active
        batched session — the paged path is OFF by default until parity-green (rule 4), so this stays
        False until #174/#175 wire a PagedKVCacheManager into the session."""
        sess = self._batched_session
        return bool(sess is not None and sess.prefix_cache_enabled)

    def has_active_requests(self) -> bool:
        return self._active > 0

    def get_stats(self) -> dict[str, Any]:
        return {"engine_type": "quanta", "model_name": self._model_name, "loaded": self._loaded,
                "active_requests": self._active}

    def get_cache_stats(self) -> dict[str, Any] | None:
        """oMLX cache-stats hook: forward the active batched session's paged-KV stats (block reuse /
        hit rate, shaped like oMLX's ``BaseCacheStats``); ``None`` when no batched session is live or
        its paged cache is off (the default until #174/#175 wire the manager in)."""
        sess = self._batched_session
        return sess.get_cache_stats() if sess is not None else None

    # --- token accounting ------------------------------------------------------
    def count_chat_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        is_partial: bool | None = None,
    ) -> int:
        """Count prompt tokens after rendering ``messages`` through the tokenizer's chat template.

        Mirrors :meth:`omlx.engine.batched.BatchedEngine.count_chat_tokens` — oMLX's Anthropic
        ``/v1/messages``, OpenAI ``/v1/chat/completions``, and ``/v1/messages/count_tokens`` handlers
        all call this method for input-token accounting + context-window validation. Without it
        every request 500s with ``AttributeError: 'QuantaOmlxEngine' object has no attribute
        'count_chat_tokens'`` (the gap surfaced by the task #162 smoke test against the int4-g64
        resident Nemotron). Implementation matches the batched-engine reference: render via
        ``apply_chat_template(messages, tokenize=False, add_generation_prompt=not is_partial)`` then
        return ``len(tokenizer.encode(prompt))``. ``tools`` / ``chat_template_kwargs`` ride through
        so the count reflects the same prompt the request will run on. Returns ``0`` if the
        tokenizer isn't loaded yet or lacks ``apply_chat_template`` (defensive; the engine.start()
        contract guarantees the tokenizer by the time the server reaches a /v1 handler)."""
        tk = self._tokenizer
        if tk is None or not hasattr(tk, "apply_chat_template"):
            return 0
        kw: dict[str, Any] = dict(chat_template_kwargs or {})
        add_gen = not bool(is_partial)
        try:
            prompt = tk.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_gen,
                tools=tools, **kw,
            )
        except TypeError:
            # Adapter / tokenizer doesn't accept one of the kwargs — retry minimally so a
            # missing-feature template never collapses the whole request to a 500.
            prompt = tk.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_gen,
            )
        if not isinstance(prompt, str):
            return 0
        return len(tk.encode(prompt))

    # --- lifecycle -------------------------------------------------------------
    async def start(self) -> None:
        if self._loaded:
            return
        self._runtime, self._tokenizer = self._runtime_loader(self._root)
        eos = getattr(self._tokenizer, "eos_id", None) or getattr(self._tokenizer, "eos_token_id", None)
        if isinstance(eos, int):
            self._eos.add(eos)
        # widen with the tokenizer's full stop set ([EOS], <|im_end|>, [EOT] for Kimi)
        self._eos |= {int(i) for i in getattr(self._tokenizer, "stop_ids", ()) if isinstance(i, int)}
        self._loaded = True

    async def stop(self) -> None:
        self._runtime = self._tokenizer = None
        self._loaded = False

    # --- generation ------------------------------------------------------------
    async def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0,
                       top_p: float = 0.9, top_k: int = 0, min_p: float = 0.0,
                       repetition_penalty: float = 1.0, frequency_penalty: float = 0.0,
                       presence_penalty: float = 0.0, seed: int | None = None,
                       stop: list[str] | None = None, **kwargs: Any) -> Any:
        last = None
        async for chunk in self.stream_generate(prompt, max_tokens=max_tokens, temperature=temperature,
                                                top_p=top_p, top_k=top_k, min_p=min_p,
                                                repetition_penalty=repetition_penalty,
                                                frequency_penalty=frequency_penalty,
                                                presence_penalty=presence_penalty, seed=seed,
                                                stop=stop, **kwargs):
            last = chunk
        return last or self._output_cls(text="", finish_reason="length")

    async def stream_generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0,
                              top_p: float = 0.9, top_k: int = 0, min_p: float = 0.0,
                              repetition_penalty: float = 1.0, frequency_penalty: float = 0.0,
                              presence_penalty: float = 0.0, seed: int | None = None,
                              stop: list[str] | None = None, **kwargs: Any) -> AsyncIterator[Any]:
        """Stream tokens for ``prompt``; yields one ``output_cls`` chunk per emitted token.

        ``spec_k`` (default 1): when ``>1``, route the single-stream loop through
        :meth:`_dispatch_spec_k` (``quanta.{dsv4,nemotron}.spec.spec_generate_k``) for multi-step MTP
        self-speculation. Output is **bit-identical** to ``spec_k == 1`` greedy regardless of
        ``spec_k`` (the spec verify arbitrates losslessly — the head only changes *speed*); spec
        currently requires ``temperature == 0`` (greedy verify) and yields one bulk chunk at the end,
        since spec emits the full token list in one call rather than per-step. The trade-off:
        ``spec_k > 1`` reduces main-model forwards per emitted token (1 forward → up to ``k+1``
        tokens) at the cost of one extra MTP forward per round.
        """
        await self.start()
        add_bos = bool(kwargs.pop("add_bos", True))
        allow_special = bool(kwargs.pop("allow_special", False))
        quantized_kv = bool(kwargs.pop("quantized_kv", True))  # int8 latent KV by default (ppl-gated)
        spec_k = int(kwargs.pop("spec_k", 1))
        if spec_k < 1:
            raise OmlxShimError(f"spec_k must be >= 1 (got {spec_k})")
        prompt_ids = self._encode(prompt, add_bos=add_bos, allow_special=allow_special)
        if spec_k > 1:
            async for chunk in self._stream_via_spec(prompt_ids=prompt_ids, spec_k=spec_k,
                                                    max_tokens=max_tokens, temperature=temperature,
                                                    stop=stop):
                yield chunk
            return
        stepper = self._make_stepper(quantized_kv=quantized_kv)
        logits_row = stepper.prefill(prompt_ids)  # [vocab] at the last prompt position
        rng = mx.random.key(seed) if seed is not None else None  # per-request PRNG -> reproducible sampling
        detok = _Detok(self._tokenizer)
        generated: list[int] = []
        self._active += 1
        try:
            for step in range(max_tokens):
                if rng is not None:
                    rng, sub = mx.random.split(rng)  # one fresh subkey per step (deterministic from seed)
                else:
                    sub = None
                tok = self._sample(logits_row, temperature, top_k, top_p, min_p,
                                   repetition_penalty, frequency_penalty, presence_penalty, generated, sub)
                generated.append(tok)
                finished, reason, new_text = False, None, ""
                if tok in self._eos:
                    finished, reason = True, "stop"  # eos itself has no visible text
                else:
                    new_text = detok.add(tok)  # raw output incl. </think> and tool-section markers
                text = detok.text
                if not finished and stop:  # stop string: cut it (and anything after) from the output
                    cut = _earliest_stop(text, stop)
                    if cut is not None:
                        start = len(text) - len(new_text)
                        new_text = text[start:cut] if cut > start else ""
                        text, finished, reason = text[:cut], True, "stop"
                if not finished and step == max_tokens - 1:
                    finished, reason = True, "length"
                yield self._output_cls(
                    text=text, tokens=list(generated), prompt_tokens=len(prompt_ids),
                    completion_tokens=len(generated), new_text=new_text,
                    finish_reason=reason, finished=finished)
                if finished:
                    break
                logits_row = stepper.step(tok)
        finally:
            self._active -= 1

    async def _stream_via_spec(self, *, prompt_ids: list[int], spec_k: int, max_tokens: int,
                                temperature: float, stop: list[str] | None) -> AsyncIterator[Any]:
        """Multi-step MTP path used when ``spec_k > 1``: call ``spec_generate_k`` once, detok the
        full result, then yield ONE final chunk (spec returns the full token list in one shot, so
        per-token streaming would require an additional callback the sibling spec API doesn't
        expose). The output text is the same as the ``spec_k == 1`` greedy path under the same
        prompt + eos (losslessness); the only observable difference is timing + a single output
        chunk instead of one per token.

        Sampling beyond greedy is not yet supported through spec — the spec verify pins each emitted
        token to the main model's ``argmax``, so non-zero temperature on this path would silently
        diverge from the requested distribution (rule 6: never silently emit something different
        from what the caller asked for)."""
        if temperature > 0.0:
            raise OmlxShimError(
                "spec_k>1 currently supports only greedy decode (temperature=0); spec verify pins "
                "each emitted token to the main model's argmax")
        # spec_generate_k's eos_id contract differs by model (DSV4: ``int | None``; Nemotron: also a
        # collection via ``_as_stop_set``). To stay compatible with both, pass the full set when there
        # are multiple stop ids (Nemotron handles it; DSV4 ignores it which is harmless under the
        # post-spec EOS-trim below); pass an int when there's exactly one (DSV4 accepts; Nemotron
        # treats it as a single-element stop set). ``None`` when ``self._eos`` is empty.
        if not self._eos:
            eos_id: Any = None
        elif len(self._eos) == 1:
            eos_id = next(iter(self._eos))
        else:
            eos_id = set(self._eos)
        self._active += 1
        try:
            raw_tokens = self._dispatch_spec_k(prompt_ids=prompt_ids, spec_k=spec_k,
                                                max_new=max_tokens, eos_id=eos_id)
            # Defensive EOS-trim — even if spec_generate_k didn't honor our eos_id (e.g. it was a set
            # passed to a DSV4-style ``int | None`` signature), we trim at the first EOS in our own
            # stop set so the chunk we yield always matches the single-stream ``spec_k == 1`` path.
            cut_idx = None
            for i, tok in enumerate(raw_tokens):
                if tok in self._eos:
                    cut_idx = i + 1   # inclusive: keep the eos in the token list, drop nothing after
                    break
            tokens = raw_tokens[:cut_idx] if cut_idx is not None else list(raw_tokens)
            finished, reason = False, None
            detok = _Detok(self._tokenizer)
            for tok in tokens:
                if tok in self._eos:
                    finished, reason = True, "stop"
                    break  # eos itself has no visible text
                detok.add(tok)
            text = detok.text
            if not finished and stop:
                cut = _earliest_stop(text, stop)
                if cut is not None:
                    text, finished, reason = text[:cut], True, "stop"
            if not finished:
                # spec asked for max_new tokens; if we got back fewer than max_new without an eos,
                # spec already exhausted its budget internally — still a "length" termination from
                # the caller's perspective (no clean eos was reached).
                finished, reason = True, "length"
            yield self._output_cls(
                text=text, tokens=list(tokens), prompt_tokens=len(prompt_ids),
                completion_tokens=len(tokens), new_text=text,
                finish_reason=reason, finished=finished)
        finally:
            self._active -= 1

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        tmpl_kw = self._pop_template_kwargs(kwargs)  # thinking/enable_thinking/... -> template, not sampler
        prompt, templated = self._format(messages, kwargs.pop("tools", None), tmpl_kw)
        kwargs.setdefault("add_bos", not templated)  # the chat template carries its own structure
        kwargs.setdefault("allow_special", templated)  # map its control tokens to special ids
        return await self.generate(prompt, **kwargs)

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> AsyncIterator[Any]:
        tmpl_kw = self._pop_template_kwargs(kwargs)
        prompt, templated = self._format(messages, kwargs.pop("tools", None), tmpl_kw)
        kwargs.setdefault("add_bos", not templated)
        kwargs.setdefault("allow_special", templated)
        async for out in self.stream_generate(prompt, **kwargs):
            yield out

    # --- batched generation (DSV4 / Nemotron) ----------------------------------
    async def batched_generate(self, prompts: Sequence[str], **kwargs: Any) -> list[Any]:
        """Drive ``len(prompts)`` requests through the batched runtime; return the final per-stream
        ``output_cls`` for each, in the same order as ``prompts``.

        Per-request kwargs may be passed positionally via ``per_request`` (a list of dicts, one per
        prompt, override the shared sampler/eos/max_new); shared ``**kwargs`` are applied to every
        request that does not override. Equivalent to running ``generate(prompt, **req_kwargs)`` on
        each prompt independently *modulo* the shared batched runtime — the per-stream outputs are
        order-equivalent (rule 4: a batched optimization is output-equivalent to the naive path,
        gated by ``parity/{dsv4,nemotron}_omlx_engine_test.py``)."""
        outs: dict[int, Any] = {}
        async for idx, chunk in self.batched_stream_generate(prompts, **kwargs):
            # keep the most-recent chunk per stream — the final ``finished=True`` chunk replaces any
            # earlier partials, and an early break (consumer cancelled) leaves the most-recent partial
            # in place rather than dropping the stream entirely.
            outs[idx] = chunk
        return [outs.get(i, self._output_cls(text="", finish_reason="length")) for i in range(len(prompts))]

    async def batched_stream_generate(self, prompts: Sequence[str], *,
                                      per_request: Sequence[dict[str, Any]] | None = None,
                                      max_tokens: int = 256, temperature: float = 0.0,
                                      top_p: float = 0.9, top_k: int = 0, min_p: float = 0.0,
                                      repetition_penalty: float = 1.0, frequency_penalty: float = 0.0,
                                      presence_penalty: float = 0.0,
                                      stop: list[str] | None = None,
                                      **kwargs: Any) -> AsyncIterator[tuple[int, Any]]:
        """Multi-stream decoder: drive up to ``capacity`` concurrent requests through the batched
        runtime; yield ``(stream_idx, output_cls)`` per emitted token.

        ``prompts``: ordered prompts (``len(prompts)`` may exceed ``capacity`` — extra prompts are
        admitted as slots free up; this is the continuous-batching surface). ``per_request``: a list
        of dicts (one per prompt) overriding the shared sampler / ``eos`` / ``max_tokens`` /
        ``seed`` / ``stop`` for that stream — anything not in the dict falls back to the kwarg
        default. The yielded ``stream_idx`` is the index into ``prompts``.

        Per-stream sampling / eos / max_new are independent (the engine builds one PRNG key + one
        :class:`_Detok` per stream); freed slots admit the next pending prompt. ``spec_k`` is NOT
        forwarded — multi-step MTP is single-stream only today (the spec verify drives one main
        forward at a time; batching it across streams is a separate optimization)."""
        if not prompts:
            return
        await self.start()
        per_request = list(per_request or [])
        if per_request and len(per_request) != len(prompts):
            raise OmlxShimError(
                f"per_request length {len(per_request)} != len(prompts) {len(prompts)}")
        capacity = self._resolve_capacity(len(prompts), kwargs.pop("batch_size", None))
        session = self._make_batched_session(capacity=capacity)
        capacity = session.capacity  # the session may clamp to its own static batch_size
        # per-stream config (sampler / eos / stop / max_new / seed / detok)
        streams = [self._build_stream_state(prompts[i], per_request[i] if per_request else None,
                                            kwargs=kwargs, default_max=max_tokens,
                                            default_temperature=temperature, default_top_p=top_p,
                                            default_top_k=top_k, default_min_p=min_p,
                                            default_rep=repetition_penalty,
                                            default_freq=frequency_penalty,
                                            default_pres=presence_penalty, default_stop=stop)
                   for i in range(len(prompts))]
        # match single-stream semantics: a stream with max_tokens<=0 emits nothing — never even
        # admitted. Done before allocating sessions so a request for ``max_tokens=0`` is a no-op.
        pending = [i for i in range(len(prompts)) if streams[i].max_tokens > 0]
        slot_owner: dict[int, int] = {}      # slot -> stream idx
        slot_logits: dict[int, mx.array] = {}
        free_slots = list(range(capacity))
        if not pending:
            return
        self._active += 1
        try:
            while pending or slot_owner:
                # admit as many pending streams as we have free slots for
                while pending and free_slots:
                    sidx = pending.pop(0)
                    slot = free_slots.pop(0)
                    slot_owner[slot] = sidx
                    # pass the request's generation budget so a YaRN-scaled model (Qwen3.5) can pin its
                    # dynamic factor across prefill + decode at admit (no-op for the other sessions).
                    slot_logits[slot] = session.admit(slot, streams[sidx].prompt_ids,
                                                      max_new=streams[sidx].max_tokens)
                if not slot_owner:
                    break
                # sample one token per alive slot from its prefilled / stepped logits row
                emit_tokens: dict[int, int] = {}
                for slot, sidx in slot_owner.items():
                    st = streams[sidx]
                    tok = self._sample(slot_logits[slot], st.temperature, st.top_k, st.top_p,
                                       st.min_p, st.repetition_penalty, st.frequency_penalty,
                                       st.presence_penalty, st.generated, st.next_key())
                    emit_tokens[slot] = tok
                # yield per-stream chunks; mark finished streams to release after the yields
                to_release: list[int] = []
                for slot, sidx in list(slot_owner.items()):
                    tok = emit_tokens[slot]
                    st = streams[sidx]
                    st.generated.append(tok)
                    finished, reason, new_text = False, None, ""
                    if tok in st.eos:
                        finished, reason = True, "stop"
                    else:
                        new_text = st.detok.add(tok)
                    text = st.detok.text
                    if not finished and st.stop:
                        cut = _earliest_stop(text, st.stop)
                        if cut is not None:
                            start = len(text) - len(new_text)
                            new_text = text[start:cut] if cut > start else ""
                            text, finished, reason = text[:cut], True, "stop"
                    if not finished and len(st.generated) >= st.max_tokens:
                        finished, reason = True, "length"
                    chunk = self._output_cls(
                        text=text, tokens=list(st.generated), prompt_tokens=len(st.prompt_ids),
                        completion_tokens=len(st.generated), new_text=new_text,
                        finish_reason=reason, finished=finished)
                    yield sidx, chunk
                    if finished:
                        to_release.append(slot)
                # step the still-alive slots; release the finished ones (freeing their slot for
                # the next pending prompt — continuous batching).
                alive_tokens = {slot: tok for slot, tok in emit_tokens.items() if slot not in to_release}
                if alive_tokens:
                    new_logits = session.step_batch(alive_tokens)
                    if set(new_logits.keys()) != set(alive_tokens.keys()):
                        raise OmlxShimError(
                            f"batched runtime returned logits for slots {sorted(new_logits)} "
                            f"but step_batch was called with {sorted(alive_tokens)}")
                    slot_logits.update(new_logits)
                for slot in to_release:
                    session.release(slot)
                    slot_owner.pop(slot, None)
                    slot_logits.pop(slot, None)
                    free_slots.append(slot)
        finally:
            # release any slots still held (e.g. cancelled mid-stream / consumer broke early) —
            # never leak a slot, which would silently starve the next batched_stream_generate call
            # for that engine (rule 6: no silent failures).
            for slot in list(slot_owner):
                try:
                    session.release(slot)
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass
            self._active -= 1

    def _build_stream_state(self, prompt: str, overrides: dict[str, Any] | None, *,
                            kwargs: dict[str, Any], default_max: int, default_temperature: float,
                            default_top_p: float, default_top_k: int, default_min_p: float,
                            default_rep: float, default_freq: float, default_pres: float,
                            default_stop: list[str] | None) -> "_StreamState":
        """Build the per-stream state for one batched stream — keeps sampler / eos / max_new / seed /
        detok / generated isolated between streams so a finished slot can be released independently."""
        ov = overrides or {}
        add_bos = bool(ov.get("add_bos", kwargs.get("add_bos", True)))
        allow_special = bool(ov.get("allow_special", kwargs.get("allow_special", False)))
        prompt_ids = self._encode(prompt, add_bos=add_bos, allow_special=allow_special)
        stream_eos: set[int] = set(self._eos)
        if "eos_token_ids" in ov:
            stream_eos = {int(t) for t in ov["eos_token_ids"]}
        if "eos_id" in ov and ov["eos_id"] is not None:
            stream_eos.add(int(ov["eos_id"]))
        seed = ov.get("seed", kwargs.get("seed"))
        return _StreamState(
            prompt_ids=prompt_ids,
            max_tokens=int(ov.get("max_tokens", default_max)),
            temperature=float(ov.get("temperature", default_temperature)),
            top_p=float(ov.get("top_p", default_top_p)),
            top_k=int(ov.get("top_k", default_top_k)),
            min_p=float(ov.get("min_p", default_min_p)),
            repetition_penalty=float(ov.get("repetition_penalty", default_rep)),
            frequency_penalty=float(ov.get("frequency_penalty", default_freq)),
            presence_penalty=float(ov.get("presence_penalty", default_pres)),
            eos=stream_eos,
            stop=list(ov.get("stop", default_stop) or []) or None,
            rng=mx.random.key(int(seed)) if seed is not None else None,
            detok=_Detok(self._tokenizer),
            generated=[],
        )

    # --- internals -------------------------------------------------------------
    @staticmethod
    def _pop_template_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Split chat-template controls out of the request kwargs so they reach ``apply_chat_template``
        instead of the sampler: a ``chat_template_kwargs`` dict, plus any well-known flags in
        ``_TEMPLATE_KEYS`` (``thinking`` / ``enable_thinking`` / ``preserve_thinking`` /
        ``reasoning_effort``). Forwarding a superset is safe — both tokenizers accept ``**kwargs`` and
        ignore vars their template never references."""
        tmpl = dict(kwargs.pop("chat_template_kwargs", None) or {})
        for k in _TEMPLATE_KEYS:
            if k in kwargs:
                tmpl[k] = kwargs.pop(k)
        return tmpl

    def _format(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                template_kwargs: dict[str, Any] | None = None) -> tuple[str, bool]:
        """Return (prompt, templated): templated=True when the tokenizer's chat template was used.

        The tokenizer's ``apply_chat_template`` renders the OpenAI messages + tools into the model's
        native prompt; ``template_kwargs`` (thinking / enable_thinking / reasoning_effort) are forwarded
        so reasoning can be toggled per request. The engine does not pre-parse output, so reasoning/tool
        extraction is left to the oMLX server (which owns the OpenAI/Anthropic response shaping)."""
        tk = self._tokenizer
        if tk is not None and hasattr(tk, "apply_chat_template"):
            text = tk.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                          tools=tools or None, **(template_kwargs or {}))
            return str(text), True
        body = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        return body + "\nassistant:", False

    def _encode(self, text: str, *, add_bos: bool = True, allow_special: bool = False) -> list[int]:
        if self._tokenizer is None:
            raise OmlxShimError("engine has no tokenizer")
        return [int(t) for t in self._tokenizer.encode(text, add_bos=add_bos, allow_special=allow_special)]

    def _make_stepper(self, *, quantized_kv: bool) -> _DecodeStepper:
        """Build the per-request decode session for this artifact's model class (``model_type``).

        Refuses an unknown model class loudly rather than guessing a decode convention (CLAUDE.md #6).
        """
        mt = self.model_type or ""
        if mt.startswith("deepseek_v4"):  # before the V3 ``deepseek`` prefix it would collide with
            return _DSV4Stepper(self._runtime)
        if mt.startswith("kimi") or mt.startswith("deepseek"):
            return _MLAStepper(self._runtime, quantized_kv=quantized_kv, sparse=DEFAULT_SPARSE)
        if mt.startswith("nemotron"):
            return _NemotronStepper(self._runtime)
        if mt.startswith("glm"):
            from quanta.glm.decode import GLMCache

            # int8 MLA latent by default — Kimi pattern (#47). Caller passes a bf16 cache directly
            # if they need it (e.g. tight parity testing); oMLX serving always wants int8.
            return _SingleTokenStepper(self._runtime, GLMCache(self._runtime.num_layers, quantized=True))
        if mt.startswith("minimax"):
            from quanta.minimax.decode import MiniMaxCache

            return _SingleTokenStepper(self._runtime,
                                       MiniMaxCache(self._runtime.num_layers, quantized=True))
        if mt.startswith("qwen3_5") or mt.startswith("qwen3.5"):  # hybrid cache needs cfg → use factory
            return _SingleTokenStepper(self._runtime, self._runtime.make_caches())
        if mt == "qwen2" or mt.startswith("qwen2."):  # dense Qwen2.5-1M (DCA + int8 intra-K cache)
            # ``make_caches()`` honors the runtime's ``quantized_kv`` default (True → int8 g64),
            # the #122 inference policy — at 1M context cuts ~96 GB off the resident KV footprint.
            return _SingleTokenStepper(self._runtime, self._runtime.make_caches())
        if mt == "internlm2" or mt.startswith("internlm2"):  # dense InternLM2.5-1M (dynamic-NTK + int8 KV)
            return _SingleTokenStepper(self._runtime, self._runtime.make_caches())
        if not mt and self._injected_runtime:
            # standalone / injected runtime with no artifact model_type (tests, embedding the engine
            # directly): use the generic single-token decoder. A *real* artifact always carries a
            # model_type, so an unrecognized one still fails loud below (rule 6 — no mis-routing).
            return _SingleTokenStepper(self._runtime, None)
        raise OmlxShimError(f"no decode stepper for quanta artifact model_type={mt!r}")

    def _default_capacity(self, n_prompts: int) -> int:
        """Concurrent decode slots when the caller passes no ``batch_size``: this model's measured
        throughput knee (:data:`BEST_BATCH`) if known, else :data:`DEFAULT_BATCH_CAPACITY`, then clamped
        to the uniform serving ceiling (:meth:`_hard_batch_cap`) and to the prompts on hand. So a worker
        whose knee exceeds the cap (DSV4 knee 48) still defaults to the cap (32)."""
        knee = _best_batch_for(self.model_type) or DEFAULT_BATCH_CAPACITY
        hard = self._hard_batch_cap()
        if hard is not None:
            knee = min(knee, hard)
        return min(n_prompts, knee)

    def _hard_batch_cap(self) -> int | None:
        """The HARD batch ceiling for this model class, or ``None``. Every throughput WORKER is clamped to
        the uniform :data:`SERVING_BATCH_CAP` (32) — even on an explicit ``batch_size``, serving never
        exceeds it (user directive). This is **decoupled** from each model's measured :data:`BEST_BATCH`
        knee: DSV4's benched knee is 48 (it scales there — #18 M5 / worker #19) yet serving is held at 32;
        Nemotron (#20) and InternLM2.5 (#21) regress past 32 anyway, so the cap also sits at their knees.
        The **Qwen3.5 orchestrator is EXEMPT** — its B=4 is a sweepable latency pin, not a throughput
        ceiling (rule 6: never clamp a soft default). A model with no measured knee (generic fallback) has
        no batched worker, so nothing to cap (``None``)."""
        mt = self.model_type or ""
        if mt.startswith("qwen3_5") or mt.startswith("qwen3.5"):
            return None  # orchestrator latency pin — sweepable, never a ceiling
        return SERVING_BATCH_CAP if _best_batch_for(mt) is not None else None

    def _resolve_capacity(self, n_prompts: int, requested: int | None) -> int:
        """Final concurrent-decode-slot count for a ``batched_stream_generate`` call. ``requested`` is the
        caller's explicit ``batch_size`` (``None`` ⇒ this model's measured default via
        :meth:`_default_capacity`). An explicit batch_size over this model's HARD ceiling
        (:meth:`_hard_batch_cap`) is clamped DOWN to it — warned, never silent (rule 6: the caller asked
        for more slots than will help). Always clamped to the prompts on hand; ``< 1`` fails loud."""
        if requested is None:
            cap = self._default_capacity(n_prompts)
        else:
            cap = int(requested)
            hard = self._hard_batch_cap()
            if hard is not None and cap > hard:
                warnings.warn(
                    f"{self.model_type} batch_size={cap} exceeds the uniform serving batch cap {hard}; "
                    f"clamping to {hard} (serving never exceeds it — drive the batched session directly "
                    f"to sweep a larger B).", stacklevel=3)
                cap = hard
        if cap < 1:
            raise OmlxShimError(f"batched_stream_generate: batch_size must be >= 1 (got {cap})")
        return min(cap, n_prompts)

    def _make_batched_session(self, *, capacity: int) -> _BatchedSession:
        """Build the engine's batched decode session for this artifact's model class (``model_type``).

        DSV4, Nemotron and Qwen3.5 expose a batched runtime (#145/#146/#147); each is driven through
        the same :class:`_BaseBatchedSession` slot adapter (the #152 unification — previously Qwen3.5
        was a separate :class:`Qwen35BatchedEngine` subclass). Other model classes fail loud here
        (rule 6) until their batched runtime is implemented — never silently fall back to a serial loop
        pretending to be batched (the caller would think it has ``B`` concurrent slots when it actually
        has one). When a session was injected at engine construction (tests / advanced callers), it is
        returned as-is. ``self._paged_kv`` (off by default, rule 4) is threaded to the session, which
        builds the shared :class:`~quanta.paged.PagedKVCacheManager` from the runtime's paged spec; a
        model whose batched runtime lacks the paged contract raises loudly when paged_kv is on."""
        if self._batched_session is not None:
            return self._batched_session
        mt = self.model_type or ""
        kw = {"capacity": capacity, "paged_kv": self._paged_kv, "model_name": self._model_name}
        if mt.startswith("deepseek_v4"):
            self._batched_session = _DSV4BatchedSession(self._root, **kw)
        elif mt.startswith("nemotron"):
            self._batched_session = _NemotronBatchedSession(self._root, **kw)
        elif mt.startswith("qwen3_5") or mt.startswith("qwen3.5"):
            # Qwen3.5 is OUT of the #152 paged scope (user directive; its batched runtime has no
            # paged contract). Force unpaged even when PAGED_KV_DEFAULT is True — rule 6: never hand
            # paged_kv=True to a runtime with no paged_kv_spec. The 3 keepers honor self._paged_kv.
            self._batched_session = _Qwen35BatchedSession(self._root, **{**kw, "paged_kv": False})
        elif mt == "internlm2" or mt.startswith("internlm2"):
            self._batched_session = _InternLM2BatchedSession(self._root, **kw)
        else:
            raise OmlxShimError(
                f"no batched runtime for quanta artifact model_type={mt!r} "
                "(DSV4 / Nemotron / InternLM2.5 only)")
        return self._batched_session

    def _ensure_eagle(self) -> tuple[Any, mx.array, mx.array, tuple[int, ...], int | None]:
        """Lazy-load the InternLM2.5 EAGLE-3 spec-decode state ``(drafter, embed, head, capture_layers,
        head_bits)``.

        Injected via the ``eagle=`` ctor arg (tests / advanced callers) takes precedence; otherwise the
        trained drafter is loaded from the artifact's **embedded** ``eagle/`` sidecar
        (:func:`quanta.eagle.artifact.load_eagle`, portable in-artifact refs only), PTQ'd to the M3
        serving operating point (``eagle_bits`` g``eagle_group_size`` — int4 g64 ⇒ 1.42× lossless @
        k=2), and paired with the runtime's frozen ``embed_head()`` (bf16). Refuses loudly (rule 6) if no
        sidecar is embedded — never silently downgrades a spec request to plain decode."""
        if self._eagle is not None:
            return self._eagle
        import mlx.nn as nn

        from quanta.eagle.artifact import load_eagle
        try:
            drafter, layers = load_eagle(self._root)
        except FileNotFoundError as e:
            raise OmlxShimError(
                "spec_k>1 on InternLM2.5 needs a trained EAGLE-3 drafter embedded in the artifact "
                f"({e}). Embed one with quanta.eagle.artifact.embed_eagle(<art_dir>, <drafter."
                "safetensors>, capture_layers=quanta.internlm2.eagle.DEFAULT_CAPTURE_LAYERS, "
                "drafter_cfg=quanta.internlm2.eagle.INTERNLM2_DRAFTER_CFG).") from e
        if self._eagle_bits:                                   # PTQ the drafter body (lossless-safe)
            nn.quantize(drafter, group_size=self._eagle_group_size, bits=self._eagle_bits)
        embed, head = self._runtime.embed_head()
        embed, head = embed.astype(mx.bfloat16), head.astype(mx.bfloat16)
        mx.eval(drafter.parameters(), embed, head)
        self._eagle = (drafter, embed, head, layers, self._eagle_bits or None)
        return self._eagle

    def _dispatch_spec_k(self, *, prompt_ids: list[int], spec_k: int, max_new: int,
                         eos_id: Any) -> list[int]:
        """Dispatch a single-stream request to speculative decode (``spec_k > 1``) — native MTP
        self-speculation (DSV4 / Nemotron) or an EAGLE-3 trained drafter (InternLM2.5).

        Lazy-imports the per-model spec entry so the engine compiles even when a sibling agent hasn't
        merged — the failure point is the call, not the module import. Output is **bit-identical to
        plain greedy** for every path (the target's own verify arbitrates losslessly; the draft head /
        drafter only changes *speed*). Refuses loudly if the model class has no spec path or its
        prerequisites are missing (rule 6 — never silently emit a non-spec stream)."""
        mt = self.model_type or ""
        rt = self._runtime
        if mt == "internlm2" or mt.startswith("internlm2"):    # EAGLE-3 drafter (not native MTP)
            from quanta.internlm2.eagle import spec_generate as _eagle_spec

            drafter, embed, head, layers, head_bits = self._ensure_eagle()
            tokens, _stats = _eagle_spec(rt, drafter, embed, head, prompt_ids, max_new=max_new,
                                         k=spec_k, layers=layers, eos_id=eos_id, head_bits=head_bits)
            return [int(t) for t in tokens]
        # native MTP self-speculation (DSV4 / Nemotron) — needs the MTP head + embed + lm_head.
        for attr in ("mtp", "embed_w", "lm_head_w"):
            if not hasattr(rt, attr):
                raise OmlxShimError(
                    f"spec_k>1 requires runtime.{attr} (MTP head + embed + lm_head); missing on "
                    f"{type(rt).__name__}")
        if mt.startswith("deepseek_v4"):
            from quanta.dsv4 import spec as _spec  # lazy: spec_generate_k landed on main earlier

            fn = getattr(_spec, "spec_generate_k", None)
            if fn is None:
                raise OmlxShimError("quanta.dsv4.spec.spec_generate_k missing — rebuild the DSV4 module")
            tokens, _stats = fn(rt, rt.mtp, rt.embed_w, rt.lm_head_w, prompt_ids,
                                max_new=max_new, eos_id=eos_id, k=spec_k)
            return [int(t) for t in tokens]
        if mt.startswith("nemotron"):
            from quanta.nemotron import spec as _spec  # lazy: spec_generate_k in flight (#148)

            fn = getattr(_spec, "spec_generate_k", None)
            if fn is None:
                raise OmlxShimError(
                    "quanta.nemotron.spec.spec_generate_k not yet available — rebuild the Nemotron "
                    "module (sibling agent #148)")
            tokens, _stats = fn(rt, rt.mtp, rt.embed_w, rt.lm_head_w, prompt_ids,
                                max_new=max_new, eos_id=eos_id, k=spec_k)
            return [int(t) for t in tokens]
        raise OmlxShimError(
            f"spec_k>1 not wired for quanta artifact model_type={mt!r} "
            "(DSV4 / Nemotron native MTP, or InternLM2.5 EAGLE-3)")

    def _sample(self, logits: mx.array, temperature: float, top_k: int, top_p: float, min_p: float,
                rep: float, freq: float, pres: float, prev: Sequence[int],
                key: mx.array | None = None) -> int:
        lg = _apply_penalties(logits.astype(mx.float32), prev, rep, freq, pres)
        if temperature <= 0.0:
            tok = mx.argmax(lg)
        else:
            lg = lg / temperature
            if top_k > 0:
                kth = mx.sort(lg)[lg.shape[0] - top_k]
                lg = mx.where(lg < kth, mx.array(-mx.inf), lg)
            lg = _apply_min_p(_apply_top_p(lg, top_p), min_p)
            tok = mx.random.categorical(lg, key=key)  # key=None -> global RNG; a key -> reproducible
        mx.eval(tok)
        return int(tok.item())


def load_quanta_engine(model_name: str, **_: Any) -> QuantaOmlxEngine:
    """Factory used by the oMLX engine pool when a quanta artifact is selected."""
    if detect_quanta_artifact(model_name) is None:
        raise OmlxShimError(f"not a quanta artifact: {model_name}")
    return QuantaOmlxEngine(model_name)


# --- Qwen3.5 agentic serving: batched runtime + multi-step MTP ------------------------------------
#
# Two new capabilities the Qwen3.5 path surfaces for serving — both wrap proven single-stream
# contracts so the per-stream output is bit-identical to the existing generate path:
#
#   * ``Qwen35BatchedEngine`` — many concurrent agent traces share resident weights; the routed-MoE
#     weight reads amortize across B streams via the batched runtime's ``step_batch`` (target ~10×
#     aggregate throughput vs B=1 at B=32). Continuous batching: finished streams free their slot.
#   * ``spec_k`` parameter — drives multi-step native MTP (k>=2 chains ``k-1`` extra drafts off the
#     MTP's own post-block hidden; k==1 is the plain :func:`quanta.qwen35.spec.spec_generate`).
#     Output is bit-identical to greedy regardless of ``k`` (the verify arbitrates losslessly);
#     ``k`` is a SPEED lever only — accept rate drops past step 1 (off-distribution chained drafts)
#     but a longer verify window amortizes routed-expert weight loads.
#
# Both are lazy-imported (the runtime / spec_k symbols come from sibling task agents that may not
# be on main yet); when the import fails this engine raises a loud, named error rather than
# silently falling back to a non-batched / non-multi-step path (CLAUDE.md #6).


def _import_qwen35_batched():
    """Lazy import of ``quanta.qwen35.batched_runtime`` + ``quanta.qwen35.batched_generate``.

    Returns ``(Qwen35BatchedResidentModel, generate_batched)``. Raises :class:`OmlxShimError` (NOT
    ImportError) when the in-flight sibling modules are not yet on disk — so the shim still loads on
    a checkout without them, but a serving call requesting the batched path fails loud."""
    try:
        from quanta.qwen35 import batched_runtime as _qbr
        from quanta.qwen35 import batched_generate as _qbg
    except ImportError as e:
        raise OmlxShimError(
            "Qwen35BatchedEngine needs quanta.qwen35.batched_runtime + batched_generate "
            f"(task #147 sibling): {e}") from e
    return _qbr.Qwen35BatchedResidentModel, _qbg.generate_batched


def _import_qwen35_spec_k():
    """Lazy import of :func:`quanta.qwen35.spec.spec_generate_k`. Falls back to :func:`spec_generate`
    if ``spec_generate_k`` is not yet on disk — the k=1 path is equivalent (the k-chained code path is
    the in-flight task #149 addition). When the spec module itself is missing, raises
    :class:`OmlxShimError`."""
    try:
        from quanta.qwen35 import spec as _qsp
    except ImportError as e:
        raise OmlxShimError(f"Qwen35BatchedEngine needs quanta.qwen35.spec: {e}") from e
    return getattr(_qsp, "spec_generate_k", None), _qsp.spec_generate


class Qwen35BatchedEngine(QuantaOmlxEngine):
    """Qwen3.5 oMLX engine extended for agentic-loop serving — batched B>1 + multi-step MTP.

    Inherits the single-stream ``stream_generate`` / ``chat`` path from :class:`QuantaOmlxEngine`
    (so behavior is identical for B=1 + spec_k=1) and adds:

    * :meth:`batched_generate` — drive many concurrent prompts through one resident copy of the
      weights, dispatching ``B`` tokens through one MoE call per step. Per-request sampling
      kwargs (temperature/top_k/top_p/min_p) honored per stream; per-request ``max_new`` / eos
      respected per stream. Finished streams free their slot (continuous batching — the underlying
      :func:`quanta.qwen35.batched_generate.generate_batched` already implements this).

    * ``spec_k`` parameter on :meth:`spec_generate_batched` — when ``spec_k > 1`` the engine
      dispatches per-stream through :func:`quanta.qwen35.spec.spec_generate_k` (multi-step MTP),
      else through plain :func:`quanta.qwen35.spec.spec_generate`. Output is bit-identical to
      plain greedy decode regardless of ``spec_k`` (the verify arbitrates losslessly), so
      ``spec_k`` is a speed lever only:

        * ``spec_k=1`` — single MTP draft per round; safe baseline (the existing parity-gated path);
        * ``spec_k>=2`` — chain ``spec_k-1`` extra drafts off the MTP's own post-block hidden.
          Chained drafts are off-distribution → accept rate drops fast past step 1, but a longer
          verify window amortizes the routed-expert weight load. Numbers TBD from orchestrator benches.

    Lazy imports — :class:`quanta.qwen35.batched_runtime.Qwen35BatchedResidentModel` and
    :func:`quanta.qwen35.spec.spec_generate_k` come from sibling agents that may not be merged yet;
    a serving call hitting either path while the sibling is absent fails loud (no silent fallback to
    the non-batched / non-multi-step path; rule 6).
    """

    def __init__(self, model_name: str, *, max_batch: int = 32, spec_k: int = 1,
                 **kwargs: Any) -> None:
        """``max_batch`` caps concurrent streams in :meth:`batched_generate`; ``spec_k`` sets the
        default MTP chain length for :meth:`spec_generate_batched` (each call may override).
        Forwarded ``**kwargs`` go to :class:`QuantaOmlxEngine`."""
        if max_batch < 1:
            raise OmlxShimError(f"max_batch must be >= 1 (got {max_batch})")
        if spec_k < 1:
            raise OmlxShimError(f"spec_k must be >= 1 (got {spec_k})")
        super().__init__(model_name, **kwargs)
        self.max_batch = int(max_batch)
        self.spec_k = int(spec_k)
        # the batched runtime wraps the single-stream Qwen35ResidentModel — lazily built on first
        # batched call so a B=1 + spec_k=1 deployment never touches the batched code path.
        self._batched_model: Any = None

    # --- batched lifecycle ----------------------------------------------------
    async def stop(self) -> None:
        """Drop the cached batched model alongside the inner runtime/tokenizer.

        Without this override the parent's :meth:`QuantaOmlxEngine.stop` nulls ``_runtime`` /
        ``_tokenizer`` but leaves ``_batched_model`` dangling — and a subsequent ``start()`` builds a
        fresh ``_runtime`` while the next batched call short-circuits on the cached ``_batched_model``
        that still holds the OLD runtime's ``layers`` / ``embed_w`` / ``lm_head_w`` references
        (oMLX's engine pool stops and restarts engines on cache eviction — exactly this case)."""
        self._batched_model = None
        await super().stop()

    def _ensure_batched_model(self) -> Any:
        """Return the batched runtime, building it once on first batched call. The single-stream
        :class:`QuantaOmlxEngine.start` has already loaded :attr:`_runtime` via the artifact loader;
        the batched model wraps an *already-loaded* inner runtime via ``from_inner`` so the artifact
        is read once (rule 8 — one layer resident at load time, not duplicated).

        Validates that the inner runtime carries the Qwen3.5 surface the batched runtime's
        ``from_inner`` needs (``layers`` / ``embed_w`` / ``norm_w`` / ``lm_head_w`` / ``cfg``) — fails
        loud (CLAUDE.md #6) when the engine was constructed over a non-Qwen3.5 artifact rather than
        crashing later with an opaque ``AttributeError`` from the runtime mismatch.
        """
        if self._batched_model is not None:
            return self._batched_model
        if self._runtime is None:
            raise OmlxShimError("Qwen35BatchedEngine: call await engine.start() before batched serving")
        rt = self._runtime
        missing = [a for a in ("layers", "embed_w", "norm_w", "lm_head_w", "cfg") if not hasattr(rt, a)]
        if missing:
            raise OmlxShimError(
                f"Qwen35BatchedEngine: inner runtime {type(rt).__name__} is missing attrs "
                f"{missing} (Qwen3.5 surface). Check model_type={self.model_type!r} — only "
                "qwen3_5 artifacts are supported by this engine.")
        Qwen35BatchedResidentModel, _ = _import_qwen35_batched()
        # the inner runtime is the single-stream resident model already loaded by start(); reuse its
        # weights via from_inner so we never double-load the artifact. Inherit the inner's packed-ness
        # (#153 option B: loop-kill ⇒ packed; the inner Qwen35ResidentModel exposes ``.packed``).
        self._batched_model = Qwen35BatchedResidentModel.from_inner(
            rt.layers, rt.embed_w, rt.norm_w, rt.lm_head_w, rt.cfg, max_batch=self.max_batch,
            packed=getattr(rt, "packed", False))
        return self._batched_model

    # --- batched generation ---------------------------------------------------
    async def batched_generate(self, prompts: Sequence[Sequence[int]], *,
                               max_new_tokens: int = 256,
                               temperature: float | Sequence[float] = 0.0,
                               top_p: float | Sequence[float] = 1.0,
                               top_k: int | Sequence[int] = 0,
                               min_p: float | Sequence[float] = 0.0,
                               eos_id: Any = None,
                               seeds: int | Sequence[int] = 0) -> list[list[int]]:
        """Generate up to ``max_new_tokens`` per prompt over ``B=len(prompts)`` concurrent streams.

        ``prompts`` are TOKEN-ID iterables (not strings — the orchestrator owns prompt rendering, so
        each agent trace can carry its own chat template / system prompt). Per-stream sampling
        kwargs (temperature/top_k/top_p/min_p) accept a scalar (applied to every stream) OR a
        per-stream sequence of length ``B``; ``seeds`` likewise accepts a single int (offset per
        stream so each gets its own key) or a per-stream sequence.

        Per-stream output is bit-identical to running :func:`quanta.qwen35.generate.generate` on
        each prompt alone with the same args (the batched runtime's step is parity-gated equal to
        the single-stream step, and the sampler is the very same function — gated in
        ``parity/qwen35_batched_test.py``).

        Fails loud (:class:`OmlxShimError`) when ``B > max_batch`` rather than silently rebatching
        — the caller picks the queue policy."""
        await self.start()
        b = len(prompts)
        if b == 0:
            return []
        if b > self.max_batch:
            raise OmlxShimError(
                f"batched_generate: B={b} exceeds max_batch={self.max_batch} "
                "(raise max_batch on construction or split the queue)")
        model = self._ensure_batched_model()
        _, generate_batched = _import_qwen35_batched()
        # eos default: the engine's own configured stop set (tokenizer eos + stop_ids) so the batched
        # path matches the single-stream stop semantics. Caller may override with an explicit eos_id.
        stop_set = eos_id if eos_id is not None else (self._eos or None)
        self._active += b
        try:
            return generate_batched(model, prompts, max_new_tokens=max_new_tokens,
                                    temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
                                    eos_id=stop_set, seeds=seeds)
        finally:
            self._active -= b

    # --- multi-step MTP -------------------------------------------------------
    async def spec_generate_batched(self, mtp: Any, embed: mx.array, head: mx.array,
                                    prompt_ids: Sequence[int], *, max_new: int,
                                    eos_id: Any = None, spec_k: int | None = None
                                    ) -> tuple[list[int], dict]:
        """Single-stream native-MTP self-speculative generation, ``k = spec_k`` (default
        :attr:`spec_k`). Output is bit-identical to plain greedy regardless of ``spec_k``.

        ``mtp`` is the native MTP head (callable contract per
        :class:`quanta.qwen35.mtp.MTPHead`); ``embed`` / ``head`` are the shared embedding / LM-head
        matrices the combine + readout need. ``prompt_ids`` is the prompt's token ids.

        Dispatch:

        * ``spec_k == 1`` -> :func:`quanta.qwen35.spec.spec_generate` (the existing parity-tested
          single-draft path);
        * ``spec_k >= 2`` -> :func:`quanta.qwen35.spec.spec_generate_k` (chained drafts, in-flight
          sibling task #149). When that symbol is not yet on disk, raises :class:`OmlxShimError`.

        Returns ``(tokens, stats)`` with ``stats['k'] == spec_k`` so the orchestrator can compare
        accept-rate / throughput across ``spec_k`` settings.
        """
        await self.start()
        k = self.spec_k if spec_k is None else int(spec_k)
        if k < 1:
            raise OmlxShimError(f"spec_k must be >= 1 (got {k})")
        spec_generate_k_fn, spec_generate_fn = _import_qwen35_spec_k()
        self._active += 1
        try:
            if k == 1:
                # bit-identical to the existing single-draft path (parity-gated in
                # parity/qwen35_mtp_spec_test.py); use spec_generate directly.
                return spec_generate_fn(self._runtime, mtp, embed, head, prompt_ids,
                                        max_new=max_new, eos_id=eos_id)
            if spec_generate_k_fn is None:
                raise OmlxShimError(
                    f"spec_k={k} needs quanta.qwen35.spec.spec_generate_k (in-flight task #149); "
                    "fall back to spec_k=1 or wait for the sibling agent to merge")
            return spec_generate_k_fn(self._runtime, mtp, embed, head, prompt_ids,
                                      k=k, max_new=max_new, eos_id=eos_id)
        finally:
            self._active -= 1


def load_qwen35_batched_engine(model_name: str, *, max_batch: int = 32, spec_k: int = 1,
                               **_: Any) -> Qwen35BatchedEngine:
    """Factory for an agentic-loop Qwen3.5 deployment (batched + multi-step MTP).

    Refuses non-Qwen3.5 artifacts loudly (rule 6 — never guess a decode convention): the batched
    runtime is Qwen3.5-specific (its ``step_batch`` threads the GDN recurrent state + GQA KV cache
    per stream), so a Nemotron / Kimi / DSV4 artifact routed here would crash later with an
    AttributeError from the runtime mismatch."""
    info = detect_quanta_artifact(model_name)
    if info is None:
        raise OmlxShimError(f"not a quanta artifact: {model_name}")
    mt = (info.model_type or "")
    if not (mt.startswith("qwen3_5") or mt.startswith("qwen3.5")):
        raise OmlxShimError(
            f"load_qwen35_batched_engine: artifact model_type={mt!r} is not Qwen3.5 "
            "(only qwen3_5 / qwen3.5 artifacts are supported by the batched engine)")
    return Qwen35BatchedEngine(model_name, max_batch=max_batch, spec_k=spec_k)
