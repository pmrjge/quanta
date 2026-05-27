"""Model-agnostic speculative-decoding infrastructure shared across the per-model spec.py loops.

This package ships the structural pieces for **tree drafting** (EAGLE-2 style, task #157). Two
verify regimes coexist:

* the single-forward tree-causal-mask path (pure-attention main models) consumes
  :func:`tree_causal_mask` + :func:`longest_accepted_path`;
* the W-parallel chain-verify path (hybrid models — Qwen3.5 GDN, Nemotron-H Mamba — whose recurrent
  state can't be tree-masked losslessly) consumes :func:`enumerate_paths`.

Both share :func:`build_tree` (BFS-expand top-``W`` MTP children ``D`` levels deep) and
:func:`tree_size`. See :mod:`quanta.spec.tree` for the full contract each per-model entry consumes.
"""

from quanta.spec.tree import (
    TreeDraft,
    build_tree,
    enumerate_paths,
    longest_accepted_path,
    tree_causal_mask,
    tree_size,
)

__all__ = [
    "TreeDraft",
    "build_tree",
    "enumerate_paths",
    "longest_accepted_path",
    "tree_causal_mask",
    "tree_size",
]
