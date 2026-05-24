"""DeepSeek-V4 BPE tokenizer (text <-> ids) — a self-contained pure-Python port of the checkpoint's
HuggingFace ``tokenizer.json`` (a GPT-2-style byte-level BPE).

The checkpoint ships a ``PreTrainedTokenizerFast`` ``tokenizer.json``; this module reproduces it
*exactly* without depending on ``tokenizers``/``transformers`` at runtime (rule 5 keeps those offline).
It is driven **entirely** off ``tokenizer.json`` — vocab, merges, added (special) tokens, **and** the
three pre-tokenizer ``Split`` regexes are all read from the file, so nothing is hand-transcribed; the
only hardcoded table is the universal GPT-2 ``bytes_to_unicode`` map. Gated bit-exact against the HF
``tokenizers`` reference in ``parity/dsv4_tokenizer_test.py``.

Pipeline (matching the file's ``pre_tokenizer`` Sequence, ``model`` BPE, and ByteLevel decoder):

1. **Special split.** Carve out exact occurrences of the 1283 ``added_tokens`` (all ``normalized=false``
   -> matched against raw text, longest-first); each becomes its special id directly.
2. **Pre-tokenize** each ordinary chunk through the three ``Split`` (behavior ``Isolated``) regexes in
   order — digit isolation ``\\p{N}{1,3}``, CJK/kana runs, then the big GPT word/punct/space regex —
   each subsequent split refining the previous pieces.
3. **Byte-map** each piece's UTF-8 bytes through the GPT-2 byte->unicode table (ByteLevel,
   ``add_prefix_space=false``, ``use_regex=false`` -> pure mapping, no further split).
4. **BPE-merge** each mapped piece by ascending merge rank (vanilla GPT-2; ``byte_fallback=false`` and
   full 256-byte coverage mean no unk is ever produced), then map each merged token to its vocab id.

Tokenization is a preprocessing boundary, not the compute hot path (rule 3 permits the bounded Python
iteration here); ``_bpe`` is memoized. Decoding reverses step 3 (byte-level -> UTF-8, lossy like the
reference) and emits special tokens verbatim (or skips them).
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Iterator

import regex

from quanta.dsv4.encoding import bos_token, eos_token


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


class DeepSeekV4Tokenizer:
    """Byte-level BPE for DeepSeek-V4, parsed from a checkpoint ``tokenizer.json``."""

    def __init__(self, tokenizer_json_path: str):
        with open(tokenizer_json_path, encoding="utf-8") as f:
            data = json.load(f)

        model = data["model"]
        if model.get("type") != "BPE":
            raise ValueError(f"expected a BPE model, got {model.get('type')!r}")
        if model.get("byte_fallback"):
            raise ValueError("byte_fallback BPE is not supported (none expected for DeepSeek-V4)")

        self.encoder: dict[str, int] = model["vocab"]                       # token str -> id
        # merge rank: "a b" -> (a, b); split on the single separating space (a real space inside a
        # token is impossible — byte-level maps it to 'Ġ'). Unpacking fails loud on a malformed merge.
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(model["merges"]):
            a, b = m.split(" ")
            self.bpe_ranks[(a, b)] = i

        # Added tokens — all 1283 are *atomic* (isolated on encode, emitted verbatim on decode). Only
        # the ``special=true`` subset (bos/eos/pad/place-holders/...) is stripped by
        # ``skip_special_tokens``; the rest (``<｜User｜>``, ``<think>``, ``｜DSML｜``, task tokens, ...)
        # are kept. ``special_tokens`` drives the encode-time pre-split (all added tokens).
        self.special_tokens: dict[str, int] = {a["content"]: a["id"] for a in data["added_tokens"]}
        self._added_ids: frozenset[int] = frozenset(self.special_tokens.values())
        self._skip_ids: frozenset[int] = frozenset(
            a["id"] for a in data["added_tokens"] if a.get("special"))

        # id -> token str: BPE vocab overlaid with added tokens (covers ids the BPE vocab omits,
        # e.g. 128000..129279). Together they span the full embedding range.
        self.decoder: dict[int, str] = {i: t for t, i in self.encoder.items()}
        self.decoder.update({a["id"]: a["content"] for a in data["added_tokens"]})

        # Pre-tokenizer: read the Split regexes (in order) and the ByteLevel flags straight from file.
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
            raise ValueError("add_prefix_space=true is not supported (DeepSeek-V4 uses false)")

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}

        # Special-token splitter: alternation of all special strings, longest-first so the longest
        # match wins at any position.
        specials = sorted(self.special_tokens, key=len, reverse=True)
        self._special_re = regex.compile("|".join(regex.escape(s) for s in specials)) if specials else None

        self.bos_id = self.special_tokens[bos_token]
        self.eos_id = self.special_tokens[eos_token]
        self.stop_ids: frozenset[int] = frozenset({self.eos_id})
        self._cache: dict[str, list[str]] = {}

    @classmethod
    def from_pretrained(cls, path: str) -> "DeepSeekV4Tokenizer":
        """Load from a checkpoint directory (or a direct path to ``tokenizer.json``)."""
        if os.path.isdir(path):
            path = os.path.join(path, "tokenizer.json")
        return cls(path)

    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

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
        """Yield ``(chunk, is_special)`` spans; special chunks are exact added-token matches."""
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
        ids: list[int] = []
        for piece in self._pretokenize(text):
            mapped = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(mapped):
                ids.append(self.encoder[tok])
        return ids

    # --- public API -----------------------------------------------------------
    def encode(self, text: str, *, add_bos: bool = False) -> list[int]:
        """Encode ``text`` to token ids. Special-token strings embedded in ``text`` (e.g. the chat
        markers produced by :mod:`quanta.dsv4.encoding`) map to their ids directly. ``add_bos``
        prepends the BOS id — leave it ``False`` for chat strings, which already contain the BOS marker."""
        ids: list[int] = [self.bos_id] if add_bos else []
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

    def encode_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
        """Encode a chat conversation: render the prompt via :func:`quanta.dsv4.encoding.encode_chat`
        (``reasoning_effort="max"`` by default), then BPE-encode it. The rendered prompt already begins
        with the BOS marker, so BOS is not added again."""
        from quanta.dsv4.encoding import encode_chat as _encode_chat

        return self.encode(_encode_chat(messages, **kwargs), add_bos=False)
