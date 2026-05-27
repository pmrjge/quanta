"""Tree-drafting infrastructure for native MTP speculative decoding (EAGLE-2 style).

The per-model ``spec_generate_tree`` entry points consume the four primitives below to drive a
single-forward verify of a (``width``, ``depth``)-tree of MTP-drafted continuations:

  1. :func:`build_tree` — BFS-expand the MTP head ``depth`` levels, taking the top-``width`` children
     of every node. Returns a :class:`TreeDraft` with the flat BFS-order token list and parent-index
     table; the verify forward consumes them as ``[root, d_1, d_2, ..., d_{T-1}]``.
  2. :func:`tree_causal_mask` — the ``[T, T]`` boolean mask each main-model attention layer needs so
     each draft token attends ONLY to its ancestor path (and itself), preserving the lossless verify
     semantics — siblings must NOT see each other or the verify would commit a token that depends on
     unrelated draft context.
  3. :func:`longest_accepted_path` — walk root-to-leaf picking the child whose token equals the main
     model's greedy at the parent's position, accepting the longest such prefix.
  4. :func:`tree_size` — total node count for a (``width``, ``depth``) tree: ``1 + sum(W^d for d in
     1..D)``. Useful for cache pre-sizing and parity assertions.

Losslessness still rests on the main model arbitrating every emitted token: ``accept iff
draft == greedy``, ``bonus = greedy``. The tree only changes *which candidates are presented* to
verify in a single forward, never *which tokens get committed* — so output is bit-identical to plain
greedy regardless of MTP quality (CLAUDE.md rule 4).

This module is pure-Python + tiny MLX boolean arrays; it never loads a model, never allocates
anything per-vocab-size, and is gated model-free in ``parity/tree_spec_test.py`` against hand-built
trees. The verify-side plumbing (passing the mask into ``model(...)``) lives in each model's
``spec.py`` and is NOT this module's responsibility — those entry points currently raise
``NotImplementedError`` with the follow-on task named.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx


def tree_size(width: int, depth: int) -> int:
    """Total node count for a (``width``, ``depth``) tree, INCLUDING the root.

    A ``(W, D)`` tree has ``1 + W + W^2 + ... + W^D = (W^(D+1) - 1) / (W - 1)`` nodes for ``W > 1``,
    or ``D + 1`` for ``W == 1`` (a chain). The number of DRAFT tokens (children of the root, used by
    verify) is ``tree_size(W, D) - 1``."""
    if width < 1 or depth < 0:
        raise ValueError(f"width must be >= 1 and depth >= 0 (got width={width}, depth={depth})")
    if width == 1:
        return depth + 1
    return (width ** (depth + 1) - 1) // (width - 1)


@dataclass(frozen=True)
class TreeDraft:
    """A BFS-linearized tree of MTP drafts.

    Fields:

    * ``tokens`` — length-``T`` list of token ids in BFS order. ``tokens[0]`` is the verified root
      (the already-committed token at the current verify offset; it is NOT a draft, just the context
      the tree branches off of). ``tokens[1:]`` are the ``T-1`` drafted tokens, each indexed by its
      flat BFS position. The verify forward consumes the full ``tokens`` sequence in one call.
    * ``parents`` — length-``T`` list of flat-tree parent indices. ``parents[0] == -1`` (the root has
      no parent). For ``i >= 1`` ``parents[i]`` is the BFS index of node ``i``'s parent; siblings
      share their parent. The parent table fully encodes the tree topology and is the input to
      :func:`tree_causal_mask` and :func:`longest_accepted_path`.

    The class is frozen so a built draft cannot be mutated under a spec-decode round mid-loop.
    """

    tokens: list[int]
    parents: list[int]

    def __post_init__(self) -> None:
        # Cheap structural checks — catches a malformed input early so the verify-side plumbing does
        # not silently consume a bad tree (CLAUDE.md rule 6 / no silent failures).
        if len(self.tokens) != len(self.parents):
            raise ValueError(
                f"tokens / parents length mismatch ({len(self.tokens)} vs {len(self.parents)})"
            )
        if not self.parents or self.parents[0] != -1:
            raise ValueError("parents[0] must be -1 (the root has no parent)")
        for i, p in enumerate(self.parents[1:], start=1):
            if not 0 <= p < i:
                raise ValueError(
                    f"parents[{i}] = {p} is invalid: must be a strict ancestor (0 <= p < {i})"
                )

    @property
    def size(self) -> int:
        """Total node count (= ``len(tokens)`` = ``len(parents)``)."""
        return len(self.tokens)

    @property
    def num_drafts(self) -> int:
        """Number of DRAFT tokens (``size - 1``; excludes the verified root)."""
        return self.size - 1


def build_tree(
    expand: Callable[[Any, int], list[tuple[int, Any]]],
    root_hidden: Any,
    root_token: int,
    *,
    width: int,
    depth: int,
) -> TreeDraft:
    """BFS-expand the top-``width`` MTP children per node, ``depth`` levels deep.

    ``expand(prev_hidden, token) -> [(child_token, child_hidden), ...]`` is the per-model callable
    that runs ONE MTP step and returns the top-``width`` predicted children, ordered most-likely
    first. The ``hidden`` type is opaque to this module — for DSV4 / Qwen3.5 it's the HC residual
    ``[1,1,hc,d]`` (or ``[1,1,d]``) from ``mtp_forward(..., return_hidden=True)``; for Nemotron it's
    the post-MTP-block hidden returned by ``NemotronMTP.__call__``. The per-model
    ``spec_generate_tree`` wraps its native MTP head to provide this callable.

    Returns a :class:`TreeDraft` with ``tokens[0] = root_token`` and ``tokens[1:]`` the BFS-flat
    drafts. The size is :func:`tree_size` ``(width, depth)``.

    The expand callable's hidden return is consumed ONLY to seed the next-level MTP step here; the
    hidden state used by the post-verify accept/bonus path is the MAIN MODEL's hidden at the last
    accepted position, captured during the verify forward (not these MTP hiddens). So we discard the
    hidden list once the build is done — the per-model loop does not see them.
    """
    if width < 1 or depth < 0:
        raise ValueError(f"width must be >= 1 and depth >= 0 (got width={width}, depth={depth})")

    tokens: list[int] = [int(root_token)]
    parents: list[int] = [-1]
    hiddens: list[Any] = [root_hidden]

    # Indices of the current expansion frontier (BFS layer). Start with just the root.
    layer_start, layer_end = 0, 1
    for level in range(depth):
        for parent_idx in range(layer_start, layer_end):
            children = expand(hiddens[parent_idx], tokens[parent_idx])
            if len(children) != width:
                raise ValueError(
                    f"expand must return exactly width={width} children at depth-{level + 1} "
                    f"expansion of node {parent_idx}, got {len(children)}"
                )
            for child_tok, child_h in children:
                tokens.append(int(child_tok))
                parents.append(parent_idx)
                hiddens.append(child_h)
        layer_start, layer_end = layer_end, len(tokens)

    return TreeDraft(tokens=tokens, parents=parents)


def tree_causal_mask(parents: Sequence[int]) -> mx.array:
    """The ``[T, T]`` boolean attention mask for the tree-linearized verify forward.

    ``mask[i, j] == True`` iff key ``j`` is an ancestor of query ``i`` (or ``j == i``). Each draft
    node attends to its full ancestor path back to the root (which carries the prompt's KV state via
    the model's own KV cache; that cache window is FULLY visible to every query — only the in-tree
    keys are gated). Siblings never see each other, and uncle / cousin nodes are masked, which is
    what makes the verify lossless: the main model's greedy at each in-tree position depends only on
    that node's ancestors, exactly as if its branch had been the only one drafted.

    Pure-Python ancestor walk per node (``O(T * depth)`` work, tiny mask materialization). Returns
    ``mx.bool_`` of shape ``[T, T]``. The per-model spec entry converts this to the SDPA-additive
    form (``where(mask, 0, -inf)`` or equivalent) at the attention call site — convention varies
    across DSV4 dense / compressed / indexed regimes and Nemotron / Qwen3.5 GQA paths.
    """
    T = len(parents)
    if T == 0:
        raise ValueError("parents must be non-empty")
    if parents[0] != -1:
        raise ValueError("parents[0] must be -1 (root has no parent)")
    # Build a dense bool grid with a per-row ancestor walk. T <= ~hundreds in practice (W=4 D=3 = 85;
    # W=8 D=2 = 73), so this stays trivially cheap.
    rows: list[list[bool]] = []
    for i in range(T):
        row = [False] * T
        j = i
        while j >= 0:
            row[j] = True
            j = parents[j]
        rows.append(row)
    return mx.array(rows, dtype=mx.bool_)


def longest_accepted_path(
    parents: Sequence[int],
    tokens: Sequence[int],
    greedy_predictions: Sequence[int],
) -> tuple[list[int], int]:
    """Walk root-to-leaf, picking at each step the child whose token equals
    ``greedy_predictions[parent]``. Accept the longest such prefix.

    Inputs:

    * ``parents`` / ``tokens`` — the :class:`TreeDraft` topology (length ``T``).
    * ``greedy_predictions`` — length ``T``; ``greedy_predictions[i]`` is the main model's argmax at
      the verify-forward position corresponding to ``tokens[i]``. Each model's verify call returns
      ``argmax(logits, axis=-1)`` over the linearized window; this list IS that vector.

    Returns ``(accepted_indices, bonus_idx)``:

    * ``accepted_indices`` — flat-tree indices of the accepted DRAFT nodes (``i >= 1`` only; the
      root is the already-verified context, never re-emitted). Length ``j`` in ``[0, depth]``.
    * ``bonus_idx`` — the flat-tree index whose ``greedy_predictions[bonus_idx]`` is the BONUS
      token (i.e., the main model's greedy at the first un-accepted position). Equals the last
      accepted index when ``j >= 1``, or ``0`` (the root) when ``j == 0``. The caller emits
      ``greedy_predictions[bonus_idx]`` after the accepted drafts; together they form the
      ``j + 1`` tokens committed by this round — matching the linear / chained spec contract.

    Tie-breaking: if multiple siblings share the target token (rare with a real MTP since top-N
    ranks them by probability, but possible with a stub), the first one in BFS order wins.
    """
    T = len(parents)
    if len(tokens) != T or len(greedy_predictions) != T:
        raise ValueError(
            f"parents / tokens / greedy_predictions length mismatch "
            f"({T} / {len(tokens)} / {len(greedy_predictions)})"
        )

    # Children map: parent_idx -> [child_idx, ...] in BFS order (= construction order).
    children: dict[int, list[int]] = {}
    for i, p in enumerate(parents):
        if p >= 0:
            children.setdefault(p, []).append(i)

    accepted: list[int] = []
    current = 0  # the root — its greedy_predictions[0] is the FIRST greedy continuation
    while True:
        target = int(greedy_predictions[current])
        next_node: int | None = None
        for c in children.get(current, ()):
            if int(tokens[c]) == target:
                next_node = c
                break
        if next_node is None:
            break  # no child matches — the bonus is greedy_predictions[current]
        accepted.append(next_node)
        current = next_node
    return accepted, current
