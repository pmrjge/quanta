"""InternLM2 SentencePiece tokenizer (text <-> ids) — pure-Python wrapper around the source
``tokenizer.model`` (binary SentencePiece protobuf) + the added-tokens table parsed from
``tokenizer_config.json``.

Unlike Qwen2.5/Qwen3.5 (byte-level BPE driven off ``tokenizer.json``), InternLM2 ships a classic
SentencePiece model — so we lean on the ``sentencepiece`` runtime library (already a project
dep — see ``pyproject.toml``) for the BPE proper and bolt on:

* **Added/special token handling**: the eight tokens listed in ``tokenizer_config.json``
  (``<unk>``/``<s>``/``</s>`` + ``<|plugin|>``/``<|interpreter|>``/``<|action_end|>``/
  ``<|action_start|>``/``<|im_end|>``/``<|im_start|>``) are matched and emitted as raw ids
  *before* the surrounding text is handed to SentencePiece — the SP model itself doesn't know
  about them.
* **BOS injection**: tokenizer_config sets ``add_bos_token=True``; :meth:`encode` auto-prepends
  ``<s>`` (id 1) unless ``add_bos=False`` is passed. The chat template emits ``{{ bos_token }}``
  at the very start so :meth:`encode_chat` does NOT double-prepend.
* **Chat template**: parsed from ``tokenizer_config.json:chat_template`` (the InternLM2 form
  ``<|im_start|>{role}\\n{content}<|im_end|>\\n`` chunks; no reasoning / no tools by default).
  Rendered via ``jinja2``.

Generation stop set and pad id are read from ``generation_config.json`` /
``tokenizer_config.json`` — nothing hardcoded. No ``transformers`` runtime dependency (rule-5);
``sentencepiece`` is a small pure-binary library, not the HF stack.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

# InternLM2 chat eos — the assistant emits this to end a turn (id 92542)
CHAT_EOS_TOKEN = "<|im_end|>"


class InternLM2Tokenizer:
    """SentencePiece BPE for InternLM2, with the added-token + chat-template layer InternLM2 needs."""

    def __init__(self, tokenizer_model_path: str) -> None:
        # Imported lazily so a bare `import quanta.internlm2.tokenizer` doesn't pay the SP cost
        # until the tokenizer is actually constructed.
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_model_path)
        # Defaults — overridden by :meth:`_load_token_policy` from the sibling config files.
        self.bos_id: int = int(self.sp.bos_id()) if self.sp.bos_id() >= 0 else 1
        self.eos_id: int = int(self.sp.eos_id()) if self.sp.eos_id() >= 0 else 2
        self.pad_id: int = self.eos_id
        self.unk_id: int = int(self.sp.unk_id()) if self.sp.unk_id() >= 0 else 0
        self.add_bos_token: bool = True
        # Added/special tokens — populated by :meth:`from_pretrained` (a bare path-only
        # construction has no tokenizer_config.json to read).
        self.special_tokens: dict[str, int] = {}
        self._skip_ids: frozenset[int] = frozenset()
        self._special_re: re.Pattern | None = None
        self._stop_ids: tuple[int, ...] = (self.eos_id,)
        self._chat_template: str | None = None

    @classmethod
    def from_pretrained(cls, path: str) -> "InternLM2Tokenizer":
        """Load from a checkpoint directory (or a direct path to ``tokenizer.model``).

        When ``path`` is a directory, sidecar config files (``tokenizer_config.json``,
        ``generation_config.json``, ``special_tokens_map.json``) are read to populate the
        added-tokens table, the chat template, the stop set, and the pad id.
        """
        if os.path.isdir(path):
            d, sp_path = path, os.path.join(path, "tokenizer.model")
        else:
            d, sp_path = os.path.dirname(path), path
        tok = cls(sp_path)
        tok._load_token_policy(d)
        return tok

    def _load_token_policy(self, d: str) -> None:
        """Read added tokens, chat template, stop set, pad id, and ``add_bos_token`` from the configs."""
        tcfg_path = os.path.join(d, "tokenizer_config.json")
        if os.path.isfile(tcfg_path):
            with open(tcfg_path, encoding="utf-8") as f:
                tcfg = json.load(f)
            # Added tokens — InternLM2 keys them by string id under "added_tokens_decoder".
            added: dict[str, dict[str, Any]] = tcfg.get("added_tokens_decoder", {}) or {}
            self.special_tokens = {meta["content"]: int(tid) for tid, meta in added.items()}
            self._skip_ids = frozenset(int(tid) for tid, meta in added.items()
                                       if meta.get("special"))
            if isinstance(tcfg.get("chat_template"), str):
                self._chat_template = tcfg["chat_template"]
            self.add_bos_token = bool(tcfg.get("add_bos_token", True))

        # Generation stop set + pad — generation_config.json wins; tokenizer_config.json fallback.
        stop: list[int] = []
        gen_path = os.path.join(d, "generation_config.json")
        if os.path.isfile(gen_path):
            with open(gen_path, encoding="utf-8") as f:
                gen = json.load(f)
            eos = gen.get("eos_token_id")
            if isinstance(eos, (list, tuple)):
                stop = [int(x) for x in eos]
            elif eos is not None:
                stop = [int(eos)]
            pad = gen.get("pad_token_id")
            if pad is not None:
                self.pad_id = int(pad)
        if not stop:
            stop = [self.eos_id]
        # Always include the chat eos so server-side generation stops on a turn boundary, even
        # if generation_config.json was minimal.
        chat_eos = self.special_tokens.get(CHAT_EOS_TOKEN)
        if chat_eos is not None and chat_eos not in stop:
            stop.append(chat_eos)
        self._stop_ids = tuple(dict.fromkeys(stop))

        # Compile a special-token splitter — longest-first to handle e.g. ``<|action_start|>`` vs
        # ``<|action_end|>`` correctly.
        if self.special_tokens:
            specials = sorted(self.special_tokens, key=len, reverse=True)
            self._special_re = re.compile("|".join(re.escape(s) for s in specials))

    # --- accessors -----------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return int(self.sp.vocab_size())

    @property
    def stop_ids(self) -> tuple[int, ...]:
        """Generation stop set — ``</s>`` + ``<|im_end|>`` for InternLM2.5."""
        return self._stop_ids

    # --- encode --------------------------------------------------------------
    def _split_special(self, text: str) -> Iterable[tuple[str, bool]]:
        """Yield ``(chunk, is_special)`` pairs splitting on the added-token strings."""
        if self._special_re is None or not text:
            if text:
                yield text, False
            return
        last = 0
        for m in self._special_re.finditer(text):
            s, e = m.span()
            if s > last:
                yield text[last:s], False
            yield text[s:e], True
            last = e
        if last < len(text):
            yield text[last:], False

    def encode(self, text: str, *, add_bos: bool | None = None) -> list[int]:
        """Encode ``text`` → token ids. Added-token strings map to their ids directly; the
        surrounding text is encoded with SentencePiece.

        ``add_bos`` defaults to the tokenizer's :attr:`add_bos_token` (``True`` for InternLM2.5);
        pass ``False`` for chat encoding (the chat template emits ``{{ bos_token }}`` itself, so
        :meth:`encode_chat` calls with ``add_bos=False`` to avoid double-prepending).
        """
        prepend_bos = self.add_bos_token if add_bos is None else add_bos
        ids: list[int] = [self.bos_id] if (prepend_bos and self.bos_id is not None) else []
        for chunk, is_special in self._split_special(text):
            if is_special:
                ids.append(self.special_tokens[chunk])
            elif chunk:
                # SentencePiece will not add bos/eos itself (we manage that above + via chunks).
                ids.extend(int(x) for x in self.sp.EncodeAsIds(chunk))
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        """Decode ids → text. SentencePiece detokenizes contiguous runs of non-special ids; added
        tokens emit their string verbatim (or are skipped when ``skip_special_tokens=True``)."""
        out: list[str] = []
        buf: list[int] = []

        def flush_buf() -> None:
            if buf:
                out.append(self.sp.DecodeIds(buf))
                buf.clear()

        special_id_to_str = {tid: s for s, tid in self.special_tokens.items()}
        for raw in ids:
            i = int(raw)
            if i in special_id_to_str:
                flush_buf()
                if not (skip_special_tokens and i in self._skip_ids):
                    out.append(special_id_to_str[i])
            else:
                buf.append(i)
        flush_buf()
        return "".join(out)

    # --- chat ----------------------------------------------------------------
    def render_chat(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool = True,
                    tools: list[Any] | None = None, **kwargs: Any) -> str:
        """Render the InternLM2 chat prompt via the checkpoint's chat template (jinja).

        The InternLM2 template is the plain ``<|im_start|>{role}\\n{content}<|im_end|>\\n`` form
        with an optional trailing ``<|im_start|>assistant\\n`` for the generation prompt — no
        reasoning block, no tool-call structure (the added ``<|action_start|>`` / ``<|plugin|>``
        tokens exist but the default template doesn't emit them; callers wanting agent traces
        can pass a custom template via ``kwargs``-driven jinja extension if/when wired up).
        """
        tmpl = self._chat_template
        if tmpl is None:
            raise RuntimeError("no chat template loaded; use InternLM2Tokenizer.from_pretrained(dir)")
        from jinja2 import Template

        # The InternLM2 template references ``bos_token`` and ``eos_token`` as Jinja variables.
        bos_str = next((s for s, tid in self.special_tokens.items() if tid == self.bos_id),
                       "<s>")
        eos_str = next((s for s, tid in self.special_tokens.items() if tid == self.eos_id),
                       "</s>")
        return Template(tmpl).render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            bos_token=bos_str,
            eos_token=eos_str,
            **kwargs,
        )

    def encode_chat(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool = True,
                    tools: list[Any] | None = None, **kwargs: Any) -> list[int]:
        """Render the chat prompt and encode it. ``add_bos=False`` — the template carries
        ``{{ bos_token }}`` at the start so we don't want :meth:`encode` to inject a second one."""
        text = self.render_chat(messages, add_generation_prompt=add_generation_prompt,
                                tools=tools, **kwargs)
        return self.encode(text, add_bos=False)
