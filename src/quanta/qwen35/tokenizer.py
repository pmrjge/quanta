"""Qwen3.5-397B-A17B BPE tokenizer (text <-> ids) — a self-contained pure-Python port of the
checkpoint's HuggingFace ``tokenizer.json`` (a GPT-2-style byte-level BPE, ``Qwen2Tokenizer``).

Mirrors :class:`quanta.glm.tokenizer.GLMTokenizer` / :class:`quanta.dsv4.tokenizer.DeepSeekV4Tokenizer`:
driven **entirely** off ``tokenizer.json`` (vocab, merges, added/special tokens, the ``NFC`` normalizer,
and the ``Split`` pre-tokenizer regex(es)), with the only hardcoded table the universal GPT-2
``bytes_to_unicode`` map — so nothing is hand-transcribed and a re-release is picked up automatically.
No ``tokenizers``/``transformers`` runtime dependency (rule 5); tokenization is a preprocessing
boundary, not the compute hot path (rule 3 permits the bounded Python iteration). Gated bit-exact
against the HF ``tokenizers`` reference in ``parity/qwen35_tokenizer_test.py``.

Qwen3.5 specifics:

* **No BOS.** ``tokenizer_config.json`` has ``add_bos_token=false`` / ``bos_token=null`` and the
  ByteLevel post-processor adds no prefix — ``encode`` never prepends a BOS.
* **NFC normalization.** Unlike GLM/DeepSeek-V4, ``tokenizer.json`` carries a ``{"type": "NFC"}``
  normalizer; it is applied to every ordinary (non-special) chunk before byte-mapping so the port is
  bit-exact with the reference. Any other normalizer type fails loud (rule 6).
* **eos = ``<|im_end|>``; the generation stop set is ``{<|im_end|>, <|endoftext|>}``** (the
  ``generation_config.json`` ``eos_token_id`` list); ``<|endoftext|>`` is also the pad token. These ids
  are read from the checkpoint's config files, not hardcoded.
* **Chat / reasoning.** The chat format uses ``<|im_start|>`` / ``<|im_end|>`` role markers and a
  ``<think>`` reasoning block that is ON by default — :meth:`encode_chat` renders the checkpoint's
  ``chat_template.jinja`` (jinja2, offline). Tool calls use the qwen3_coder XML form.
"""

from __future__ import annotations

import json
import os
import unicodedata
from typing import Any, Iterable, Iterator

import regex

EOS_TOKEN = "<|im_end|>"  # tokenizer_config.json eos_token (the model's generation eos)


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 reversible byte->unicode table: every byte 0..255 maps to a printable unicode char, so a
    byte-level BPE never needs an unk token."""
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


class Qwen35Tokenizer:
    """Byte-level BPE for Qwen3.5, parsed from a checkpoint ``tokenizer.json``."""

    def __init__(self, tokenizer_json_path: str) -> None:
        with open(tokenizer_json_path, encoding="utf-8") as f:
            data = json.load(f)

        model = data["model"]
        if model.get("type") != "BPE":
            raise ValueError(f"expected a BPE model, got {model.get('type')!r}")
        if model.get("byte_fallback"):
            raise ValueError("byte_fallback BPE is not supported (none expected for Qwen3.5)")

        self.encoder: dict[str, int] = model["vocab"]               # token str -> id
        # merge rank: each entry is "a b" (a real space inside a token is impossible — byte-level maps
        # it to 'Ġ') or, in a newer tokenizer.json, a ["a","b"] pair. Unpacking fails loud on malformed.
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(model["merges"]):
            a, b = (m if isinstance(m, (list, tuple)) else m.split(" "))
            self.bpe_ranks[(a, b)] = i

        # Added tokens are *atomic* (isolated on encode, emitted verbatim on decode). Only the
        # ``special=true`` subset is stripped by ``skip_special_tokens``; the rest (``<think>``,
        # ``<tool_call>``, fim/repo markers, …) are kept. ``special_tokens`` drives the encode-time
        # pre-split (all added tokens, special or not).
        self.special_tokens: dict[str, int] = {a["content"]: a["id"] for a in data["added_tokens"]}
        self._added_ids: frozenset[int] = frozenset(self.special_tokens.values())
        self._skip_ids: frozenset[int] = frozenset(
            a["id"] for a in data["added_tokens"] if a.get("special"))

        # id -> token str: BPE vocab overlaid with added tokens (which the BPE vocab omits).
        self.decoder: dict[int, str] = {i: t for t, i in self.encoder.items()}
        self.decoder.update({a["id"]: a["content"] for a in data["added_tokens"]})

        # Normalizer: Qwen3.5 ships NFC. Apply it to ordinary chunks so we match the reference; refuse
        # any other normalizer rather than silently producing wrong ids (rule 6).
        self._nfc = False
        norm = data.get("normalizer")
        if norm is not None:
            ntype = norm.get("type")
            if ntype == "NFC":
                self._nfc = True
            else:
                raise ValueError(f"unsupported normalizer {ntype!r} (Qwen3.5 uses NFC)")

        # Pre-tokenizer: read the Split regex(es) (in order) and the ByteLevel flags straight from file.
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
            raise ValueError("add_prefix_space=true is not supported (Qwen3.5 uses false)")

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}

        # Special-token splitter: alternation of all special strings, longest-first so the longest
        # match wins at any position.
        specials = sorted(self.special_tokens, key=len, reverse=True)
        self._special_re = regex.compile("|".join(regex.escape(s) for s in specials)) if specials else None

        self.eos_id = self.special_tokens[EOS_TOKEN]
        self.bos_id: int | None = None  # Qwen3.5 has no BOS (add_bos_token=false, bos_token=null)
        # The pad id and the generation stop set come from the checkpoint config (set by
        # :meth:`from_pretrained`); default to a single-eos stop set when constructed bare.
        self.pad_id: int = self.eos_id
        self._stop_ids: tuple[int, ...] = (self.eos_id,)
        self._cache: dict[str, list[str]] = {}
        self._chat_template: str | None = None

    @classmethod
    def from_pretrained(cls, path: str) -> "Qwen35Tokenizer":
        """Load from a checkpoint directory (or a direct path to ``tokenizer.json``).

        When ``path`` is a directory, also picks up the sibling ``chat_template.jinja`` (for
        :meth:`encode_chat`) and reads ``generation_config.json`` / ``tokenizer_config.json`` for the
        generation stop set and pad id (so nothing tokenizer-policy-related is hardcoded)."""
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
        """Read the generation stop set and pad id from the checkpoint config files.

        ``generation_config.json`` ``eos_token_id`` is the authoritative stop set (a single id or a
        list); ``tokenizer_config.json`` provides ``pad_token`` and an ``eos_token`` fallback. Every id
        is resolved through the tokenizer's own tables (fails loud on an unknown token name)."""
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

        if not stop:
            stop = [self.eos_id]
        # de-dup, preserve order
        self._stop_ids = tuple(dict.fromkeys(stop))

    def _token_to_id(self, token: str) -> int:
        """Resolve a token *string* to its id via the added-token / BPE tables; fail loud if unknown."""
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
        """Generation stop ids (``generation_config.json`` ``eos_token_id``): ``<|im_end|>`` plus
        ``<|endoftext|>`` for Qwen3.5."""
        return self._stop_ids

    # --- pre-tokenization -----------------------------------------------------
    @staticmethod
    def _split_isolated(text: str, pat: regex.Pattern) -> list[str]:
        """HF ``Split`` with ``behavior="Isolated"``: every match becomes its own piece and the gaps
        between matches are kept as pieces too (all non-empty)."""
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
        """Yield ``(chunk, is_special)`` spans; special chunks are exact added-token matches (matched on
        the raw text, before NFC, exactly as HF extracts added tokens)."""
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
        """Vanilla GPT-2 byte-level BPE: repeatedly merge the lowest-rank adjacent pair."""
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
        """Encode ``text`` to token ids. Special-token strings embedded in ``text`` (the chat markers
        produced by :meth:`encode_chat`, ``<think>``, ``<tool_call>``, …) map to their ids directly.

        ``add_bos`` prepends ``bos_id`` only when one is configured — Qwen3.5 has none
        (``add_bos_token=false``), so this is a no-op; the parameter exists for engine-contract parity
        with the other tokenizers."""
        ids: list[int] = [self.bos_id] if (add_bos and self.bos_id is not None) else []
        for chunk, is_special in self._split_special(text):
            if is_special:
                ids.append(self.special_tokens[chunk])
            else:
                ids.extend(self._encode_ordinary(chunk))
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        """Decode ids to text. Byte-level pieces are reassembled and UTF-8 decoded (lossy, like the
        reference); special tokens are emitted verbatim (or skipped if ``skip_special_tokens``)."""
        out: list[str] = []
        buf = bytearray()
        for i in ids:
            i = int(i)
            tok = self.decoder.get(i)
            if tok is None:
                raise KeyError(f"id {i} is out of vocabulary")
            if i in self._added_ids:                     # atomic token: not byte-level decodable
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

    def encode_chat(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool = True,
                    enable_thinking: bool = True, tools: list[Any] | None = None,
                    **kwargs: Any) -> list[int]:
        """Render the Qwen3.5 chat prompt via the checkpoint's ``chat_template.jinja`` and BPE-encode it.

        ``add_generation_prompt`` appends the ``<|im_start|>assistant`` opener; with reasoning on
        (``enable_thinking=True``, the default) the template emits a bare ``<think>\\n`` opener, and with
        ``enable_thinking=False`` it emits an empty ``<think>\\n\\n</think>\\n\\n`` block. ``tools`` (the
        qwen3_coder XML tool-call form) is forwarded to the template. Needs ``jinja2`` (offline). The
        template is loaded by :meth:`from_pretrained`; if unset (constructed bare), this raises."""
        tmpl = self._chat_template
        if tmpl is None:
            raise RuntimeError("no chat_template.jinja loaded; use Qwen35Tokenizer.from_pretrained(dir)")
        from jinja2 import Template

        text = Template(tmpl).render(
            messages=messages, add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking, tools=tools, **kwargs)
        return self.encode(text, add_bos=False)
