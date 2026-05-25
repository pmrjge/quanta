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
"""

from __future__ import annotations

import json
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
    raise OmlxShimError(
        f"no resident runtime for quanta artifact model_type={mt!r} "
        "(supported: kimi, deepseek_v3, deepseek_v4, nemotron)"
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


class QuantaOmlxEngine(_OmlxBaseEngine):
    """oMLX ``BaseEngine`` backed by the quanta resident runtime (model-specific decode stepper)."""

    def __init__(self, model_name: str, *, runtime: RuntimeLike | None = None,
                 tokenizer: TokenizerLike | None = None, runtime_loader=None,
                 output_cls: type[Any] = OmlxGenerationOutput, eos_token_ids: set[int] | None = None) -> None:
        self._model_name = model_name
        self._root = Path(model_name).expanduser().resolve(strict=False)
        self._runtime = runtime
        self._tokenizer = tokenizer
        self._runtime_loader = runtime_loader or _default_runtime_loader
        self._output_cls = output_cls
        self._eos = set(eos_token_ids or ())
        self._loaded = runtime is not None and tokenizer is not None
        self._active = 0

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
        return False

    def has_active_requests(self) -> bool:
        return self._active > 0

    def get_stats(self) -> dict[str, Any]:
        return {"engine_type": "quanta", "model_name": self._model_name, "loaded": self._loaded,
                "active_requests": self._active}

    def get_cache_stats(self) -> dict[str, Any] | None:
        return None

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
        await self.start()
        add_bos = bool(kwargs.pop("add_bos", True))
        allow_special = bool(kwargs.pop("allow_special", False))
        quantized_kv = bool(kwargs.pop("quantized_kv", True))  # int8 latent KV by default (ppl-gated)
        prompt_ids = self._encode(prompt, add_bos=add_bos, allow_special=allow_special)
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
        raise OmlxShimError(f"no decode stepper for quanta artifact model_type={mt!r}")

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
