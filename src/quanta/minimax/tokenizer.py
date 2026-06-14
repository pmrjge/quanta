"""MiniMax-M2.7 BPE tokenizer (text <-> ids) — a self-contained pure-Python port of the checkpoint's
HuggingFace ``tokenizer.json`` (a GPT-2-style byte-level BPE).

Mirrors :class:`quanta.glm.tokenizer.GLMTokenizer` / :class:`quanta.dsv4.tokenizer.DeepSeekV4Tokenizer`:
driven **entirely** off ``tokenizer.json`` (vocab, merges, added/special tokens, the ``NFC`` normalizer,
and the ``Split`` pre-tokenizer regex + ``ByteLevel`` flags), with the only hardcoded table the
universal GPT-2 ``bytes_to_unicode`` map — so nothing is hand-transcribed and a re-release is picked up
automatically. No ``tokenizers``/``transformers`` runtime dependency (rule 5); tokenization is a
preprocessing boundary, not the compute hot path (rule 3 permits the bounded Python iteration here).
Token ids (bos/eos/stop set) come from :class:`quanta.minimax.config.MiniMaxConfig` — the authoritative
source — not from the tokenizer_config strings (whose ``bos_token`` is a different marker than the
generation ``bos_token_id``). Gated bit-exact against the HF ``tokenizers`` reference in
``parity/minimax_tokenizer_test.py``.

MiniMax specifics that differ from the GLM/DSV4 ports (all read from the file, not assumed):

* **NFC normalizer.** ``tokenizer.json`` declares a ``normalizer`` of type ``NFC``; ordinary (non-
  special) text is NFC-normalized before byte-mapping, matching HF. Special-token strings are matched
  against raw text first (HF applies the normalizer only to non-special spans).
* **``Split`` is ``behavior="Removed", invert=true``** — i.e. the regex is a GPT-2 *tokenizing* pattern
  whose matches ARE the pieces (``regex.findall``), not delimiters isolated between gaps (the GLM/DSV4
  ``Isolated`` form). We read both flags and dispatch on them, failing loud on any other combination.
* **bos 200019 / eos 200020 / vocab from config.** ``encode`` follows the checkpoint's ``add_bos_token``
  (absent in tokenizer_config -> default ``False``): the chat BOS marker is injected by the chat
  template, not by ``encode``.
"""

from __future__ import annotations

import json
import os
import unicodedata
from typing import Any, Iterable, Iterator

import regex

from quanta.minimax.config import MiniMaxConfig


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


class MiniMaxTokenizer:
    """Byte-level BPE for MiniMax-M2.7, parsed from a checkpoint ``tokenizer.json``."""

    def __init__(self, tokenizer_json_path: str, config: MiniMaxConfig) -> None:
        with open(tokenizer_json_path, encoding="utf-8") as f:
            data = json.load(f)

        model = data["model"]
        if model.get("type") != "BPE":
            raise ValueError(f"expected a BPE model, got {model.get('type')!r}")
        if model.get("byte_fallback"):
            raise ValueError("byte_fallback BPE is not supported (none expected for MiniMax-M2.7)")

        self.config = config

        self.encoder: dict[str, int] = model["vocab"]                       # token str -> id
        # merge rank: each entry is "a b" (a real space inside a token is impossible — byte-level maps
        # it to 'Ġ') or, in newer tokenizer.json, a ["a","b"] pair. Unpacking fails loud on malformed.
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(model["merges"]):
            a, b = (m if isinstance(m, (list, tuple)) else m.split(" "))
            self.bpe_ranks[(a, b)] = i

        # Added tokens are atomic (isolated on encode, emitted verbatim on decode); only the
        # ``special=true`` subset is stripped by ``skip_special_tokens``.
        self.special_tokens: dict[str, int] = {a["content"]: a["id"] for a in data["added_tokens"]}
        self._added_ids: frozenset[int] = frozenset(self.special_tokens.values())
        self._skip_ids: frozenset[int] = frozenset(
            a["id"] for a in data["added_tokens"] if a.get("special"))

        self.decoder: dict[int, str] = {i: t for t, i in self.encoder.items()}
        self.decoder.update({a["id"]: a["content"] for a in data["added_tokens"]})

        # Normalizer: read its type from the file. MiniMax declares NFC; we apply it to ordinary text.
        norm = data.get("normalizer") or {}
        ntype = norm.get("type")
        if ntype not in (None, "NFC"):
            raise ValueError(f"unsupported normalizer {ntype!r} (only NFC/none handled for MiniMax)")
        self._normalize = bool(ntype == "NFC")

        # Pre-tokenizer: read the Split regex + ByteLevel flags straight from the file. MiniMax uses a
        # single GPT-2 Split with behavior=Removed,invert=true (== regex.findall on the pattern).
        self._split_pat: regex.Pattern | None = None
        self._split_findall = False
        self._add_prefix_space = False
        pt = data.get("pre_tokenizer") or {}
        subs = pt.get("pretokenizers", [pt]) if pt.get("type") == "Sequence" else [pt]
        for sub in subs:
            stype = sub.get("type")
            if stype == "Split":
                if self._split_pat is not None:
                    raise ValueError("multiple Split pre-tokenizers not supported (MiniMax has one)")
                self._split_pat = regex.compile(sub["pattern"]["Regex"])
                behavior, invert = sub.get("behavior"), bool(sub.get("invert", False))
                if behavior == "Removed" and invert:
                    self._split_findall = True       # matches ARE the pieces
                elif behavior == "Isolated" and not invert:
                    self._split_findall = False      # matches + gaps are pieces (GLM/DSV4 form)
                else:
                    raise ValueError(
                        f"unsupported Split behavior={behavior!r} invert={invert} "
                        "(only Removed+invert or Isolated handled)")
            elif stype == "ByteLevel":
                self._add_prefix_space = bool(sub.get("add_prefix_space", False))
        if self._split_pat is None:
            raise ValueError("no Split pre-tokenizer found in tokenizer.json (cannot pre-tokenize)")
        if self._add_prefix_space:
            raise ValueError("add_prefix_space=true is not supported (MiniMax-M2.7 uses false)")

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}

        # Special-token splitter: alternation of all special strings, longest-first so the longest
        # match wins at any position.
        specials = sorted(self.special_tokens, key=len, reverse=True)
        self._special_re = regex.compile(
            "|".join(regex.escape(s) for s in specials)) if specials else None

        # Token ids come from config (the authoritative generation ids), not tokenizer_config strings.
        self.bos_id: int = config.bos_token_id
        self.eos_id: int = config.eos_token_id
        self.pad_id: int = self.eos_id
        self._add_bos_token: bool = False  # tokenizer_config has no add_bos_token -> GPT2 default False
        # Generation stop set: the model's eos id(s) from generation_config (config.eos_token_ids).
        self.stop_ids: tuple[int, ...] = tuple(dict.fromkeys(config.eos_token_ids))
        self._cache: dict[str, list[str]] = {}
        self._chat_template: str | None = None

    @classmethod
    def from_pretrained(cls, path: str) -> "MiniMaxTokenizer":
        """Load from a checkpoint directory (or a direct path to ``tokenizer.json``). Reads token ids
        from :meth:`MiniMaxConfig.from_pretrained` and the chat template from
        ``tokenizer_config.json["chat_template"]`` (preferred) or the sibling ``chat_template.jinja``."""
        if os.path.isdir(path):
            d, json_path = path, os.path.join(path, "tokenizer.json")
        else:
            d, json_path = os.path.dirname(path), path
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"tokenizer.json not found at {json_path}")
        tok = cls(json_path, MiniMaxConfig.from_pretrained(d))
        tok._chat_template = _load_chat_template(d)
        return tok

    @classmethod
    def from_pretrained_m3(cls, path: str) -> "MiniMaxTokenizer":
        """Load for a **MiniMax-M3-VL** artifact (the oMLX serving path). Identical BPE machinery to
        :meth:`from_pretrained`, but the token ids come from :class:`quanta.minimax.config_m3.\
MiniMaxM3Config` — the M3 bos/eos set (200019/200020) differs from M2.7's, and the M2.7
        :class:`MiniMaxConfig` parse would silently mis-read the nested ``minimax_m3_vl`` config
        (rule 6: never derive the stop set from the wrong config). The chat template (the M3 markup —
        ``]<]minimax[>[`` sections, ``<mm:think>`` reasoning) loads from ``tokenizer_config.json`` /
        ``chat_template.jinja`` exactly as for M2.7."""
        from quanta.minimax.config_m3 import MiniMaxM3Config  # noqa: PLC0415 — avoid an import cycle
        if os.path.isdir(path):
            d, json_path = path, os.path.join(path, "tokenizer.json")
        else:
            d, json_path = os.path.dirname(path), path
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"tokenizer.json not found at {json_path}")
        tok = cls(json_path, MiniMaxM3Config.from_pretrained(d))
        tok._chat_template = _load_chat_template(d)
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

    # --- pre-tokenization -----------------------------------------------------
    @staticmethod
    def _split_isolated(text: str, pat: regex.Pattern) -> list[str]:
        """HF ``Split`` with ``behavior="Isolated"``: every match is its own piece and the gaps between
        matches are kept too (all non-empty)."""
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
        if self._split_findall:
            # behavior=Removed, invert=true: the regex matches are exactly the pieces (GPT-2 findall).
            return self._split_pat.findall(chunk)
        return self._split_isolated(chunk, self._split_pat)

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
        if self._normalize:
            text = unicodedata.normalize("NFC", text)
        ids: list[int] = []
        for piece in self._pretokenize(text):
            mapped = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(mapped):
                ids.append(self.encoder[tok])
        return ids

    # --- public API -----------------------------------------------------------
    def encode(self, text: str, *, add_bos: bool | None = None) -> list[int]:
        """Encode ``text`` to token ids. Special-token strings embedded in ``text`` map to their ids
        directly. ``add_bos`` prepends ``bos_id``; when left ``None`` it follows the checkpoint's
        ``add_bos_token`` (``False`` for MiniMax — the chat template injects the BOS marker itself)."""
        prepend_bos = self._add_bos_token if add_bos is None else add_bos
        ids: list[int] = [self.bos_id] if prepend_bos else []
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

    def render_chat(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool = True,
                    tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> str:
        """Render the MiniMax chat prompt via the checkpoint's chat template (needs ``jinja2``).

        The template injects its own BOS/role markers and (with ``add_generation_prompt``) the trailing
        ``]~b]ai\\n<think>\\n`` assistant opener. The template path is set by :meth:`from_pretrained`; if
        unset (constructed directly from ``tokenizer.json``), this raises."""
        tmpl = self._chat_template
        if tmpl is None:
            raise RuntimeError(
                "no chat template loaded; use MiniMaxTokenizer.from_pretrained(dir)")
        from jinja2 import Environment, BaseLoader
        from jinja2.exceptions import TemplateError

        def _raise(msg: str) -> Any:
            raise TemplateError(msg)

        def _tojson(obj: Any, ensure_ascii: bool = True, **kw: Any) -> str:
            return json.dumps(obj, ensure_ascii=ensure_ascii, **kw)

        env = Environment(loader=BaseLoader())
        env.filters["tojson"] = _tojson         # HF tojson accepts ensure_ascii; jinja2's builtin does not
        env.globals["raise_exception"] = _raise
        return env.from_string(tmpl).render(
            messages=messages, add_generation_prompt=add_generation_prompt, tools=tools, **kwargs)

    def encode_chat(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool = True,
                    tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> list[int]:
        """Render the chat prompt (:meth:`render_chat`) and BPE-encode it. The rendered prompt already
        begins with the BOS marker, so ``add_bos`` is forced off here."""
        text = self.render_chat(messages, add_generation_prompt=add_generation_prompt,
                                tools=tools, **kwargs)
        return self.encode(text, add_bos=False)


def _load_chat_template(d: str) -> str | None:
    """Prefer ``tokenizer_config.json["chat_template"]``; fall back to the sibling ``chat_template.jinja``
    (MiniMax-M2.7 ships the template as the standalone ``.jinja`` file, not inside tokenizer_config)."""
    cfg_path = os.path.join(d, "tokenizer_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            ct = json.load(f).get("chat_template")
        if isinstance(ct, str) and ct:
            return ct
    jinja_path = os.path.join(d, "chat_template.jinja")
    if os.path.isfile(jinja_path):
        with open(jinja_path, encoding="utf-8") as f:
            return f.read()
    return None
