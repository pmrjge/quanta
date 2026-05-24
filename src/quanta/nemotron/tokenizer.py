"""Nemotron-H tokenizer — thin wrapper over the HF ``tokenizers`` fast tokenizer.

Loads ``tokenizer.json`` (BPE + pre-tokenizer + special tokens). Encode/decode plus the
generation stop set. **Two eos** (the Kimi lesson repeats): the tokenizer/chat eos is
``<|im_end|>`` (11) while ``</s>`` is 2; ``generation_config`` lists both, so serving stops
on ``{2, 11}`` and ``<|im_end|>`` is what the model emits to end a turn. ``add_bos_token`` is
false upstream, so :meth:`encode` defaults to no BOS. :meth:`apply_chat_template` renders the
upstream ``chat_template.jinja`` via Jinja2 (lazy import, transformers-equivalent env).

Requires ``tokenizers`` (run with ``uv run --with tokenizers``); never on the runtime hot path.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


def _tojson(obj: Any, indent: int | None = None, separators: Any = None,
            sort_keys: bool = False) -> str:
    """``tojson`` Jinja filter, matching the one transformers injects for chat templates."""
    return json.dumps(obj, indent=indent, separators=separators, sort_keys=sort_keys,
                      ensure_ascii=False)


class NemotronTokenizer:
    def __init__(self, model_dir: str | Path) -> None:
        self._dir = Path(model_dir)
        self._tk = Tokenizer.from_file(str(self._dir / "tokenizer.json"))
        self.bos_id = self._tk.token_to_id("<s>")
        self.eos_id = self._tk.token_to_id("<|im_end|>")  # chat/generation eos
        self.stop_ids = {
            i for i in (self._tk.token_to_id("</s>"), self._tk.token_to_id("<|im_end|>"))
            if isinstance(i, int)
        }

    def encode(self, text: str, *, add_bos: bool = False, allow_special: bool = False) -> list[int]:
        # add_special_tokens=False: no template wrapping; control tokens present in the text are still
        # recognized to their ids (the HF tokenizer always splits added tokens), so ``allow_special`` is
        # accepted for the oMLX shim's calling convention but is a no-op here (always recognized).
        del allow_special
        ids = self._tk.encode(text, add_special_tokens=False).ids
        return [self.bos_id, *ids] if add_bos else list(ids)

    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        return self._tk.decode([int(i) for i in ids], skip_special_tokens=skip_special)

    @cached_property
    def _chat_template(self) -> str | None:
        f = self._dir / "chat_template.jinja"
        return f.read_text() if f.is_file() else None

    def apply_chat_template(self, conversation: list[dict[str, Any]], *,
                            tools: list[dict[str, Any]] | None = None, tokenize: bool = False,
                            add_generation_prompt: bool = True, **kwargs: Any) -> str | list[int]:
        """Render the upstream ``chat_template.jinja`` (Jinja2, transformers-equivalent env).

        Returns the rendered string, or token ids when ``tokenize=True`` (no BOS — the chat
        template carries its own structure and the model uses ``add_bos_token=false``).
        """
        tmpl = self._chat_template
        if tmpl is None:
            raise FileNotFoundError(f"no chat_template.jinja in {self._dir}")
        try:
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError as e:  # pragma: no cover - serving dep
            raise RuntimeError("apply_chat_template needs jinja2 (ships with the 'omlx' extra)") from e
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        env.filters["tojson"] = _tojson
        rendered = env.from_string(tmpl).render(
            messages=conversation, tools=tools or [], add_generation_prompt=add_generation_prompt,
            bos_token="<s>", eos_token="<|im_end|>", **kwargs,
        )
        if tokenize:
            return self.encode(rendered, add_bos=False)
        return rendered
