"""Kimi-K2.6 tokenizer — thin wrapper over tiktoken (offline tooling only).

Replicates the ``tiktoken.Encoding`` built by the source ``tokenization_kimi.py``
(BPE ranks from ``tiktoken.model`` + the Kimi ``pat_str``). For perplexity/eval we
only need plain-text encode/decode plus the BOS id, so special tokens are omitted
from the encoder (base ids occupy ``[0, n_base)``; specials live at ``n_base+``).
Requires ``tiktoken`` (run with ``uv run --with tiktoken``); never on the runtime
hot path.
"""

from __future__ import annotations

from pathlib import Path

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


class KimiTokenizer:
    def __init__(self, model_dir: str | Path, bos_id: int = 163584, eos_id: int = 163586) -> None:
        ranks = load_tiktoken_bpe(str(Path(model_dir) / "tiktoken.model"))
        self.n_base = len(ranks)
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.encoding = tiktoken.Encoding(
            name="kimi", pat_str=_PAT_STR, mergeable_ranks=ranks, special_tokens={}
        )

    def encode(self, text: str, *, add_bos: bool = True) -> list[int]:
        ids = self.encoding.encode(text)
        return [self.bos_id, *ids] if add_bos else ids

    def decode(self, ids: list[int]) -> str:
        return self.encoding.decode([i for i in ids if i < self.n_base])
