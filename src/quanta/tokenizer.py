"""Kimi-K2.6 tokenizer — tiktoken BPE with the upstream pat_str + special tokens.

Replicates the ``tiktoken.Encoding`` built by the source ``tokenization_kimi.py``:
the BPE ranks from ``tiktoken.model`` (163,584 base ids) plus the 256 reserved
special tokens at ``[n_base, n_base + 256)`` — named from ``tokenizer_config.json``'s
``added_tokens_decoder`` (``[BOS]``=163584, ``[EOS]``=163585, ``<|im_end|>``=163586,
``[EOT]``=163593, ``[PAD]``=163839; unfilled slots → ``<|reserved_token_i|>``).

Two encode modes keep the parity/eval path byte-identical while enabling faithful
serving:
  * ``allow_special=False`` (default): control-token strings are encoded as ordinary
    text (never mapped to a special id, never raised) — exactly the plain-text
    behaviour the perplexity and bake-calibration paths depend on.
  * ``allow_special=True``: control-token strings map to their special ids — used by
    the chat path (``apply_chat_template`` output is encoded this way).

``apply_chat_template`` renders the upstream ``chat_template.jinja`` with Jinja2
(lazy-imported; needed only for serving, never on the eval/hot path) under the same
environment settings transformers uses, so the output matches the reference.

Requires ``tiktoken`` (run with ``uv run --with tiktoken``); never on the runtime hot path.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any

import tiktoken
from tiktoken.load import load_tiktoken_bpe

_PAT_STR = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

_NUM_RESERVED_SPECIAL_TOKENS = 256


def _tojson(obj: Any, indent: int | None = None, separators: Any = None, sort_keys: bool = False) -> str:
    """``tojson`` Jinja filter, matching the one transformers injects for chat templates."""
    return json.dumps(obj, indent=indent, separators=separators, sort_keys=sort_keys, ensure_ascii=False)


class KimiTokenizer:
    def __init__(self, model_dir: str | Path, bos_id: int = 163584, eos_id: int = 163586) -> None:
        self._dir = Path(model_dir)
        ranks = load_tiktoken_bpe(str(self._dir / "tiktoken.model"))
        self.n_base = len(ranks)
        # The 256 reserved specials occupy [n_base, n_base+256); name them from the source
        # added_tokens_decoder, filling gaps with reserved placeholders (mirrors tokenization_kimi.py).
        named = self._named_specials()
        self.special_tokens: dict[str, int] = {
            named.get(i, f"<|reserved_token_{i}|>"): i
            for i in range(self.n_base, self.n_base + _NUM_RESERVED_SPECIAL_TOKENS)
        }
        self.encoding = tiktoken.Encoding(
            name="kimi", pat_str=_PAT_STR, mergeable_ranks=ranks, special_tokens=self.special_tokens
        )
        self.id_to_special: dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        self.bos_id = bos_id
        # Generation eos = <|im_end|> (config.json / generation_config.json eos_token_id == 163586),
        # which differs from the tokenizer's nominal [EOS]=163585. Stop on both, plus end-of-turn [EOT].
        self.eos_id = eos_id
        self.stop_ids: set[int] = {
            i
            for i in (
                self.special_tokens.get("[EOS]"),
                self.special_tokens.get("<|im_end|>"),
                self.special_tokens.get("[EOT]"),
                eos_id,
            )
            if isinstance(i, int)
        }

    def _named_specials(self) -> dict[int, str]:
        cfg = self._dir / "tokenizer_config.json"
        if not cfg.is_file():
            return {}
        atd = json.loads(cfg.read_text()).get("added_tokens_decoder", {})
        return {int(k): v["content"] for k, v in atd.items() if isinstance(v, dict) and "content" in v}

    def encode(self, text: str, *, add_bos: bool = True, allow_special: bool = False) -> list[int]:
        if allow_special:
            ids = self.encoding.encode(text, allowed_special="all")
        else:  # treat control-token strings as ordinary text (byte-identical to the eval path)
            ids = self.encoding.encode(text, allowed_special=set(), disallowed_special=())
        return [self.bos_id, *ids] if add_bos else list(ids)

    def decode(self, ids: list[int]) -> str:
        return self.encoding.decode([int(i) for i in ids if int(i) < self.n_base])

    def decode_bytes(self, ids: list[int]) -> bytes:
        """Raw bytes for ``ids`` (specials dropped) — for byte-accurate incremental detok.

        A single BPE token can be a partial UTF-8 sequence; streaming must flush only
        complete characters, so the serving path detokenizes over bytes, not strings.
        """
        return self.encoding.decode_bytes([int(i) for i in ids if int(i) < self.n_base])

    @cached_property
    def _chat_template(self) -> str | None:
        f = self._dir / "chat_template.jinja"
        return f.read_text() if f.is_file() else None

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        thinking: bool = True,
        preserve_thinking: bool = False,
        **kwargs: Any,
    ) -> str | list[int]:
        """Render the upstream ``chat_template.jinja`` (Jinja2, transformers-equivalent env).

        Tool declarations render through the template's JSON branch unless a precomputed
        ``tools_ts_str`` is supplied (the upstream TypeScript encoder is offline tooling we
        don't vendor). Returns the rendered string, or token ids when ``tokenize=True``
        (no BOS by default — the chat template carries its own structure).
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
            messages=conversation,
            tools=tools,
            tools_ts_str=kwargs.get("tools_ts_str"),
            add_generation_prompt=add_generation_prompt,
            thinking=thinking,
            preserve_thinking=preserve_thinking,
        )
        if tokenize:
            return self.encode(rendered, add_bos=bool(kwargs.get("add_bos", False)), allow_special=True)
        return rendered
