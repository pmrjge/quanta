"""Model-agnostic speculative-decoding infrastructure shared across the per-model spec.py loops.

Currently this package ships the *structural* pieces for **tree drafting** (EAGLE-2 style, task
#157). The per-model verify path — passing a tree-causal attention mask into each runtime's
attention regime — is the follow-on plumbing each model owns; see :mod:`quanta.spec.tree` for the
contract those entry points consume.
"""

from quanta.spec.tree import (
    TreeDraft,
    build_tree,
    longest_accepted_path,
    tree_causal_mask,
    tree_size,
)

__all__ = [
    "TreeDraft",
    "build_tree",
    "longest_accepted_path",
    "tree_causal_mask",
    "tree_size",
]
