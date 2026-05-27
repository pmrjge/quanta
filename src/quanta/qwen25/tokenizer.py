"""Qwen2.5 BPE tokenizer (text <-> ids) — a self-contained pure-Python port of the checkpoint's
HuggingFace ``tokenizer.json`` (GPT-2-style byte-level BPE, ``Qwen2Tokenizer``).

Same byte-level BPE machinery as :class:`quanta.qwen35.tokenizer.Qwen35Tokenizer` — driven entirely
off ``tokenizer.json`` (vocab, merges, added/special tokens, optional NFC normalizer, ``Split`` pre-
tokenizer regex). The two divergences from the Qwen3.5 case:

* Qwen2.5 has a **BOS id** (151643 = ``<|endoftext|>``) but its tokenizer is ``add_bos_token=False``
  / ``bos_token=null``, so ``encode`` never prepends one (same effective behavior as Qwen3.5).
* Qwen2.5's chat template lives **inline** in ``tokenizer_config.json:chat_template`` rather than a
  sibling ``chat_template.jinja`` file. :meth:`from_pretrained` picks it up from either location.

Generation stop set, eos, and pad are read from ``generation_config.json`` / ``tokenizer_config.json``
— nothing hardcoded. No ``tokenizers``/``transformers`` runtime dependency (rule-5).
"""

from __future__ import annotations

import json
import os
import unicodedata
from typing import Any, Iterable, Iterator

import regex

EOS_TOKEN = "<|im_end|>"  # the generation eos the model emits (151645)


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 reversible byte->unicode table — every byte 0..255 → printable unicode, no unk needed."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _get_pairs(word: list[str]) -> set[tuple[str, str]]:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class Qwen25Tokenizer:
    """Byte-level BPE for Qwen2.5, parsed from a checkpoint ``tokenizer.json``."""

    def __init__(self, tokenizer_json_path: str) -> None:
        with open(tokenizer_json_path, encoding="utf-8") as f:
            data = json.load(f)

        model = data["model"]
        if model.get("type") != "BPE":
            raise ValueError(f"expected a BPE model, got {model.get('type')!r}")
        if model.get("byte_fallback"):
            raise ValueError("byte_fallback BPE is not supported (none expected for Qwen2.5)")

        self.encoder: dict[str, int] = model["vocab"]
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(model["merges"]):
            a, b = (m if isinstance(m, (list, tuple)) else m.split(" "))
            self.bpe_ranks[(a, b)] = i

        self.special_tokens: dict[str, int] = {a["content"]: a["id"] for a in data["added_tokens"]}
        self._added_ids: frozenset[int] = frozenset(self.special_tokens.values())
        self._skip_ids: frozenset[int] = frozenset(
            a["id"] for a in data["added_tokens"] if a.get("special"))

        self.decoder: dict[int, str] = {i: t for t, i in self.encoder.items()}
        self.decoder.update({a["id"]: a["content"] for a in data["added_tokens"]})

        # Normalizer: Qwen2.5 historically ships NFC (same as Qwen3.5); refuse other types (rule-6).
        self._nfc = False
        norm = data.get("normalizer")
        if norm is not None:
            ntype = norm.get("type")
            if ntype == "NFC":
                self._nfc = True
            elif ntype is not None:
                raise ValueError(f"unsupported normalizer {ntype!r}")

        self._split_pats: list[regex.Pattern] = []
        self._add_prefix_space = False
        pt = data.get("pre_tokenizer") or {}
        subs = pt.get("pretokenizers", [pt]) if pt.get("type") == "Sequence" else [pt]
        for sub in subs:
            stype = sub.get("type")
            if stype == "Split":
                self._split_pats.append(regex.compile(sub["pattern"]["Regex"]))
            elif stype == "ByteLevel":
                self._add_prefix_space = bool(sub.get("add_prefix_space", False))
        if self._add_prefix_space:
            raise ValueError("add_prefix_space=true is not supported (Qwen2.5 uses false)")

        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}

        specials = sorted(self.special_tokens, key=len, reverse=True)
        self._special_re = regex.compile("|".join(regex.escape(s) for s in specials)) if specials else None

        self.eos_id = self.special_tokens[EOS_TOKEN]
        self.bos_id: int | None = None  # Qwen2.5: add_bos_token=False ⇒ never prepended
        self.pad_id: int = self.eos_id
        self._stop_ids: tuple[int, ...] = (self.eos_id,)
        self._cache: dict[str, list[str]] = {}
        self._chat_template: str | None = None

    @classmethod
    def from_pretrained(cls, path: str) -> "Qwen25Tokenizer":
        """Load from a checkpoint directory (or a direct path to ``tokenizer.json``).

        When ``path`` is a directory, the chat template is sourced from (in order):
          1. ``chat_template.jinja`` (sibling file — Qwen3.5 convention)
          2. ``tokenizer_config.json:chat_template`` (inline string — Qwen2.5 convention)

        The generation stop set / pad id come from ``generation_config.json`` /
        ``tokenizer_config.json``. Nothing tokenizer-policy-related is hardcoded.
        """
        if os.path.isdir(path):
            d, json_path = path, os.path.join(path, "tokenizer.json")
        else:
            d, json_path = os.path.dirname(path), path
        tok = cls(json_path)

        ct = os.path.join(d, "chat_template.jinja")
        if os.path.isfile(ct):
            with open(ct, encoding="utf-8") as f:
                tok._chat_template = f.read()

        tok._load_token_policy(d)
        return tok

    def _load_token_policy(self, d: str) -> None:
        """Read the stop set, pad id, and (Qwen2.5-style) inline chat template from the config files."""
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

        tcfg_path = os.path.join(d, "tokenizer_config.json")
        if os.path.isfile(tcfg_path):
            with open(tcfg_path, encoding="utf-8") as f:
                tcfg = json.load(f)
            if not stop:
                eos_tok = tcfg.get("eos_token")
                if eos_tok is not None:
                    stop = [self._token_to_id(eos_tok)]
            pad_tok = tcfg.get("pad_token")
            if pad_tok is not None:
                self.pad_id = self._token_to_id(pad_tok)
            # Qwen2.5 ships its chat template inline in tokenizer_config.json (Qwen3.5 uses a
            # sibling .jinja file). Prefer the .jinja file (already loaded) if present; otherwise
            # fall back to the inline string here.
            if self._chat_template is None and isinstance(tcfg.get("chat_template"), str):
                self._chat_template = tcfg["chat_template"]

        if not stop:
            stop = [self.eos_id]
        self._stop_ids = tuple(dict.fromkeys(stop))

    def _token_to_id(self, token: str) -> int:
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.encoder:
            return self.encoder[token]
        raise KeyError(f"token {token!r} is not in the vocabulary")

    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

    @property
    def stop_ids(self) -> tuple[int, ...]:
        """Generation stop set — ``<|im_end|>`` + ``<|endoftext|>`` for Qwen2.5."""
        return self._stop_ids

    # --- pre-tokenization -----------------------------------------------------
    @staticmethod
    def _split_isolated(text: str, pat: regex.Pattern) -> list[str]:
        out: list[str] = []
        last = 0
        for mobj in pat.finditer(text):
            s, e = mobj.span()
            if e == s:
                continue
            if s > last:
                out.append(text[last:s])
            out.append(text[s:e])
            last = e
        if last < len(text):
            out.append(text[last:])
        return out

    def _pretokenize(self, chunk: str) -> list[str]:
        pieces = [chunk]
        for pat in self._split_pats:
            nxt: list[str] = []
            for p in pieces:
                nxt.extend(self._split_isolated(p, pat))
            pieces = nxt
        return pieces

    def _split_special(self, text: str) -> Iterator[tuple[str, bool]]:
        if self._special_re is None:
            if text:
                yield text, False
            return
        last = 0
        for mobj in self._special_re.finditer(text):
            s, e = mobj.span()
            if s > last:
                yield text[last:s], False
            yield text[s:e], True
            last = e
        if last < len(text):
            yield text[last:], False

    # --- BPE ------------------------------------------------------------------
    def _bpe(self, token: str) -> list[str]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = list(token)
        pairs = _get_pairs(word)
        if not pairs:
            self._cache[token] = word
            return word
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                if j < len(word) - 1 and word[j + 1] == second:
                    new_word.append(first + second)
                    i = j + 2
                else:
                    new_word.append(word[j])
                    i = j + 1
            word = new_word
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        self._cache[token] = word
        return word

    def _encode_ordinary(self, text: str) -> list[int]:
        if self._nfc:
            text = unicodedata.normalize("NFC", text)
        ids: list[int] = []
        for piece in self._pretokenize(text):
            mapped = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(mapped):
                ids.append(self.encoder[tok])
        return ids

    # --- public API -----------------------------------------------------------
    def encode(self, text: str, *, add_bos: bool = False) -> list[int]:
        """Encode ``text`` to token ids. Special-token strings (``<|im_start|>``, ``<|im_end|>``,
        chat markers from :meth:`encode_chat`) map to their ids directly.

        ``add_bos`` is a no-op for Qwen2.5 (``add_bos_token=False``); accepted for engine-contract
        parity with the other tokenizers.
        """
        ids: list[int] = [self.bos_id] if (add_bos and self.bos_id is not None) else []
        for chunk, is_special in self._split_special(text):
            if is_special:
                ids.append(self.special_tokens[chunk])
            else:
                ids.extend(self._encode_ordinary(chunk))
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        """Decode ids to text (lossy UTF-8 reassembly of byte-level pieces, like the reference)."""
        out: list[str] = []
        buf = bytearray()
        for i in ids:
            i = int(i)
            tok = self.decoder.get(i)
            if tok is None:
                raise KeyError(f"id {i} is out of vocabulary")
            if i in self._added_ids:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                if not (skip_special_tokens and i in self._skip_ids):
                    out.append(tok)
            else:
                buf.extend(self.byte_decoder[c] for c in tok)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def render_chat(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool = True,
                    tools: list[Any] | None = None, **kwargs: Any) -> str:
        """Render the Qwen2.5 chat prompt via the checkpoint's chat template (jinja).

        Qwen2.5 (unlike Qwen3.5) has **no** ``<think>`` reasoning block — straightforward
        ``<|im_start|>{role}\\n{content}<|im_end|>`` chunks. ``tools`` is forwarded (the standard
        OpenAI-style function-call format the template supports).
        """
        tmpl = self._chat_template
        if tmpl is None:
            raise RuntimeError("no chat template loaded; use Qwen25Tokenizer.from_pretrained(dir)")
        from jinja2 import Template

        return Template(tmpl).render(
            messages=messages, add_generation_prompt=add_generation_prompt,
            tools=tools, **kwargs)

    def encode_chat(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool = True,
                    tools: list[Any] | None = None, **kwargs: Any) -> list[int]:
        """Render the chat prompt and BPE-encode it (no BOS prefix — the template carries structure)."""
        text = self.render_chat(messages, add_generation_prompt=add_generation_prompt,
                                tools=tools, **kwargs)
        return self.encode(text, add_bos=False)
