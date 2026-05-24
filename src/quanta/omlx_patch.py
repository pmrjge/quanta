"""oMLX integration: register the quanta engine and route quanta artifacts to it.

oMLX has no plugin seam — ``engine_pool._load_engine`` dispatches on a hard-coded
``engine_type`` chain — so we (1) keep an engine **registry** (``register_engine``) mapping an
engine_type to a factory, and (2) lazily monkeypatch two oMLX modules once they import:
``omlx.model_discovery`` (tag quanta artifacts with ``engine_type='quanta'``) and
``omlx.engine_pool`` (route registered engine_types to their factory, owning the runtime).
Importing this module does not import oMLX; the patch applies via an import hook when oMLX
loads. ``main`` is the ``quanta-omlx`` console script: arm the patch, then hand off to oMLX's CLI.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

PATCH_VERSION = "0.1.0"
PATCH_MARKER = "__quanta_omlx_autopatch__"
_ENV_FLAG = "QUANTA_OMLX_AUTOPATCH"
_TARGETS = frozenset({"omlx.model_discovery", "omlx.engine_pool", "omlx.api.tool_calling"})

# engine registry: engine_type -> (detect(path)->bool, factory(model_path)->BaseEngine)
EngineFactory = Callable[[str], Any]
ArtifactDetector = Callable[[str | Path], bool]
ENGINE_REGISTRY: dict[str, tuple[ArtifactDetector, EngineFactory]] = {}

__all__ = ["ENGINE_REGISTRY", "PATCH_VERSION", "apply_now", "enabled", "install",
           "main", "register_engine"]


def register_engine(engine_type: str, detector: ArtifactDetector, factory: EngineFactory) -> None:
    """Register a model-class engine: artifacts matching ``detector`` load via ``factory``."""
    ENGINE_REGISTRY[engine_type] = (detector, factory)


def _register_quanta() -> None:
    from quanta.shim.omlx import detect_quanta_artifact, load_quanta_engine

    register_engine("quanta", lambda p: detect_quanta_artifact(p) is not None, load_quanta_engine)


_register_quanta()


def enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "1").lower() not in {"0", "false", "no", "off"}


def install() -> bool:
    """Install the lazy import hook and patch any already-imported oMLX modules."""
    if not enabled():
        return False
    if not any(isinstance(f, _OmlxPatchFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _OmlxPatchFinder())
    _patch_loaded()
    return True


def apply_now() -> None:
    if enabled():
        _patch_loaded()


def main() -> None:
    """``quanta-omlx`` console script: arm the patch, then run oMLX's CLI."""
    install()
    from omlx.cli import main as omlx_main

    omlx_main()


def _engine_type_for(path: str | Path) -> str | None:
    for engine_type, (detect, _) in ENGINE_REGISTRY.items():
        try:
            if detect(path):
                return engine_type
        except Exception:
            continue
    return None


def _patch_loaded() -> None:
    for name in _TARGETS:
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, PATCH_MARKER, None) != PATCH_VERSION:
            _patch_module(name, mod)


def _patch_module(name: str, module: ModuleType) -> None:
    {"omlx.model_discovery": _patch_model_discovery,
     "omlx.engine_pool": _patch_engine_pool,
     "omlx.api.tool_calling": _patch_tool_calling}[name](module)
    setattr(module, PATCH_MARKER, PATCH_VERSION)


class _OmlxPatchFinder(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self._resolving = False

    def find_spec(self, fullname: str, path: Any = None, target: ModuleType | None = None):
        if fullname not in _TARGETS or self._resolving:
            return None
        self._resolving = True
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            self._resolving = False
        if spec and spec.loader:
            spec.loader = _OmlxPatchLoader(fullname, spec.loader)
        return spec


class _OmlxPatchLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, wrapped: importlib.abc.Loader) -> None:
        self._fullname = fullname
        self._wrapped = wrapped

    def create_module(self, spec):
        return getattr(self._wrapped, "create_module", lambda s: None)(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        _patch_module(self._fullname, module)


def _patch_model_discovery(module: ModuleType) -> None:
    orig_detect = module.detect_model_type
    orig_is_dir = module._is_model_dir
    orig_register = module._register_model

    def detect_model_type(model_path: Path):
        return "llm" if _engine_type_for(model_path) else orig_detect(model_path)

    def _is_model_dir(path: Path) -> bool:
        return _engine_type_for(path) is not None or orig_is_dir(path)

    def _register_model(models: dict[str, Any], model_dir: Path, model_id: str) -> None:
        et = _engine_type_for(model_dir)
        orig_register(models, model_dir, model_id)
        if et and model_id in models:
            models[model_id].model_type = "llm"
            models[model_id].engine_type = et

    module.detect_model_type = detect_model_type
    module._is_model_dir = _is_model_dir
    module._register_model = _register_model


def _patch_engine_pool(module: ModuleType) -> None:
    orig_load = module.EnginePool._load_engine

    async def _load_engine(self: Any, model_id: str, force_lm: bool = False) -> None:
        entry = self._entries[model_id]
        et = getattr(entry, "engine_type", None)
        reg = ENGINE_REGISTRY.get(et) if et in ENGINE_REGISTRY else None
        if reg is None and _engine_type_for(entry.model_path):
            reg = ENGINE_REGISTRY[_engine_type_for(entry.model_path)]
        if reg is None:
            return await orig_load(self, model_id, force_lm=force_lm)
        _, factory = reg
        engine = factory(entry.model_path)
        await engine.start()
        entry.engine = engine
        entry.engine_type = et or "quanta"
        if hasattr(entry, "last_access"):
            entry.last_access = module.time.time()
        if hasattr(self, "_current_model_memory"):
            self._current_model_memory += getattr(entry, "estimated_size", 0)

    module.EnginePool._load_engine = _load_engine


def _patch_tool_calling(module: ModuleType) -> None:
    """Teach oMLX's ``parse_tool_calls`` the Kimi-K2.6 tool-call format.

    oMLX's built-in registry has no Kimi parser, and our custom tokenizer exposes no mlx-lm tool
    hooks, so Kimi tool markup falls through unparsed. The raw-output engine surfaces the markers as
    literal text (``clean_special_tokens`` keeps them), so we wrap ``parse_tool_calls`` to extract
    Kimi calls first and delegate everything else to the original parser."""
    from quanta.shim.kimi_tools import parse_kimi_tool_calls

    orig = module.parse_tool_calls

    def parse_tool_calls(text: str, tokenizer: Any = None, tools: Any = None):
        parsed = parse_kimi_tool_calls(text)
        if parsed is None:  # not Kimi markup → original registry (xml/json/gemma/glm/qwen/...)
            return orig(text, tokenizer, tools)
        from omlx.api.openai_models import FunctionCall, ToolCall

        cleaned, calls = parsed
        tcs = [ToolCall(id=c["id"], type="function",
                        function=FunctionCall(name=c["name"], arguments=c["arguments"]))
               for c in calls] or None
        return cleaned, tcs

    module.parse_tool_calls = parse_tool_calls
