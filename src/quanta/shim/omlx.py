"""oMLX-compatible engine for quanta baked artifacts.

Importable without oMLX: subclasses oMLX's ``BaseEngine`` when present (so it passes the
server's ``isinstance(engine, BaseEngine)`` gate and is dispatched by the engine pool) and
falls back to a plain class otherwise (standalone / tests). The engine owns the quanta
runtime — it loads :class:`ResidentModel` + :class:`KimiTokenizer` from the artifact and
decodes with the KV-cached, absorbed-MLA path — while oMLX provides the OpenAI/Anthropic
server surface. mlx-lm is never imported.

Every generation kwarg from oMLX (temperature, top_p, top_k, min_p, repetition/presence
penalties, stop) is applied; sparse XAttention prefill is on by default via the runtime.
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


def _default_runtime_loader(root: Path) -> tuple[RuntimeLike, TokenizerLike]:
    from quanta.runtime import ResidentModel
    from quanta.tokenizer import KimiTokenizer

    rm = ResidentModel(root)
    return rm, KimiTokenizer(root, bos_id=rm.cfg.bos_token_id)


def _apply_penalties(logits: mx.array, prev: Sequence[int] | None, rep: float, pres: float) -> mx.array:
    if not prev or (rep == 1.0 and pres == 0.0):
        return logits
    idx = mx.array(sorted({int(t) for t in prev}), dtype=mx.int32)
    seen = mx.zeros(logits.shape[0]).at[idx].add(1.0) > 0
    pen = mx.where(logits > 0, logits / rep, logits * rep) if rep != 1.0 else logits
    pen = pen - pres if pres != 0.0 else pen
    return mx.where(seen, pen, logits)


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


class QuantaOmlxEngine(_OmlxBaseEngine):
    """oMLX ``BaseEngine`` backed by the quanta resident runtime (KV-cached absorbed decode)."""

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
        self._loaded = True

    async def stop(self) -> None:
        self._runtime = self._tokenizer = None
        self._loaded = False

    # --- generation ------------------------------------------------------------
    async def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0,
                       top_p: float = 0.9, top_k: int = 0, min_p: float = 0.0,
                       repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
                       stop: list[str] | None = None, **kwargs: Any) -> Any:
        last = None
        async for chunk in self.stream_generate(prompt, max_tokens=max_tokens, temperature=temperature,
                                                top_p=top_p, top_k=top_k, min_p=min_p,
                                                repetition_penalty=repetition_penalty,
                                                presence_penalty=presence_penalty, stop=stop, **kwargs):
            last = chunk
        return last or self._output_cls(text="", finish_reason="length")

    async def stream_generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0,
                              top_p: float = 0.9, top_k: int = 0, min_p: float = 0.0,
                              repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
                              stop: list[str] | None = None, **kwargs: Any) -> AsyncIterator[Any]:
        await self.start()
        prompt_ids = self._encode(prompt)
        caches = [MLACache() for _ in range(self._runtime.num_layers)]
        logits = self._runtime(mx.array(prompt_ids), caches=caches, sparse=DEFAULT_SPARSE)  # prefill
        generated: list[int] = []
        self._active += 1
        try:
            for _ in range(max_tokens):
                tok = self._sample(logits[0, -1], temperature, top_k, top_p, min_p,
                                   repetition_penalty, presence_penalty, generated)
                generated.append(tok)
                text = self._decode(generated)
                finished = tok in self._eos or bool(stop and any(s in text for s in stop))
                yield self._output_cls(text=text, tokens=list(generated), prompt_tokens=len(prompt_ids),
                                       completion_tokens=len(generated), new_text=self._decode([tok]),
                                       finish_reason="stop" if finished else None, finished=finished)
                if finished:
                    break
                logits = self._runtime(mx.array([tok]), caches=caches, offset=caches[0].offset, absorbed=True)
        finally:
            self._active -= 1

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return await self.generate(self._format(messages, kwargs.pop("tools", None)), **kwargs)

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> AsyncIterator[Any]:
        async for out in self.stream_generate(self._format(messages, kwargs.pop("tools", None)), **kwargs):
            yield out

    # --- internals -------------------------------------------------------------
    def _format(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
        tk = self._tokenizer
        if tk is not None and hasattr(tk, "apply_chat_template"):
            return tk.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools or None)
        body = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        return body + "\nassistant:"

    def _encode(self, text: str) -> list[int]:
        if self._tokenizer is None:
            raise OmlxShimError("engine has no tokenizer")
        return [int(t) for t in self._tokenizer.encode(text)]

    def _decode(self, ids: Sequence[int]) -> str:
        if self._tokenizer is None:
            raise OmlxShimError("engine has no tokenizer")
        return str(self._tokenizer.decode([int(t) for t in ids]))

    def _sample(self, logits: mx.array, temperature: float, top_k: int, top_p: float, min_p: float,
                rep: float, pres: float, prev: Sequence[int]) -> int:
        lg = _apply_penalties(logits.astype(mx.float32), prev, rep, pres)
        if temperature <= 0.0:
            tok = mx.argmax(lg)
        else:
            lg = lg / temperature
            if top_k > 0:
                kth = mx.sort(lg)[lg.shape[0] - top_k]
                lg = mx.where(lg < kth, mx.array(-mx.inf), lg)
            lg = _apply_min_p(_apply_top_p(lg, top_p), min_p)
            tok = mx.random.categorical(lg)
        mx.eval(tok)
        return int(tok.item())


def load_quanta_engine(model_name: str, **_: Any) -> QuantaOmlxEngine:
    """Factory used by the oMLX engine pool when a quanta artifact is selected."""
    if detect_quanta_artifact(model_name) is None:
        raise OmlxShimError(f"not a quanta artifact: {model_name}")
    return QuantaOmlxEngine(model_name)
