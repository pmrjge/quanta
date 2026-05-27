"""Model-free parity tests for the shared tree-drafting primitives (task #157).

Exercises :mod:`quanta.spec.tree` with hand-built trees and stub MTP callables; never loads a model,
never allocates per-vocab-size, runs in <100ms on CPU. Tests:

  - :func:`tree_size` — closed-form node count for ``(W, D)``.
  - :class:`TreeDraft` validation — ill-formed parent tables fail loud (rule 6).
  - :func:`build_tree` — BFS construction order, topology, ``W=1`` chain edge case.
  - :func:`tree_causal_mask` — each row is the ancestor set; siblings are masked from each other.
  - :func:`longest_accepted_path` — perfect-tree, no-match, partial-match, sibling tie-break.

The shared module is the *structural* piece of #157; the per-model ``spec_generate_tree`` entry
points (which require the per-model verify-side attention-mask plumbing) are NOT exercised here
beyond their import path — see each model's ``spec.py`` for the follow-on.

Run:  ``uv run --with numpy python -m parity.tree_spec_test``
"""

from __future__ import annotations

import mlx.core as mx

from quanta.spec.tree import (
    TreeDraft,
    build_tree,
    longest_accepted_path,
    tree_causal_mask,
    tree_size,
)


def _stub_expand_topn(start_token: int = 1000):
    """A deterministic stub ``expand`` for :func:`build_tree`.

    The first call returns children ``[start, start+1, ..., start+width-1]`` paired with hiddens
    ``[(start,0), (start,1), ...]``; subsequent calls return contiguous next-block ids so every node
    in the BFS-flat tree gets a unique token, making the topology assertions exact. The 'hidden' is
    just a tuple — opaque to :func:`build_tree`, which only feeds it back into the next expand call.
    """
    counter = [start_token]

    def expand(prev_hidden, parent_token: int):
        del prev_hidden, parent_token  # not consumed by the stub
        base = counter[0]
        children = [(base + i, ("h", base + i)) for i in range(_stub_expand_topn.width)]
        counter[0] = base + _stub_expand_topn.width
        return children

    return expand


_stub_expand_topn.width = 0  # type: ignore[attr-defined]  # set per-call before passing the stub


def test_tree_size() -> None:
    """Closed-form node counts for the canonical (W, D) shapes."""
    assert tree_size(1, 0) == 1  # root only
    assert tree_size(1, 3) == 4  # chain: root + 3
    assert tree_size(2, 2) == 7  # 1 + 2 + 4
    assert tree_size(4, 2) == 21  # 1 + 4 + 16  (EAGLE-2 paper default)
    assert tree_size(8, 2) == 73  # 1 + 8 + 64
    assert tree_size(4, 3) == 85  # 1 + 4 + 16 + 64

    # Errors: caller-side bounds.
    try:
        tree_size(0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("tree_size(0, 1) must raise")
    try:
        tree_size(2, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("tree_size(2, -1) must raise")
    print("[OK] tree_size")


def test_treedraft_validation() -> None:
    """Frozen :class:`TreeDraft` rejects ill-formed parent tables (no silent malformed trees)."""
    # OK: minimal root-only.
    _ = TreeDraft(tokens=[42], parents=[-1])

    # OK: a width-2 depth-1 tree.
    _ = TreeDraft(tokens=[0, 1, 2], parents=[-1, 0, 0])

    # Length mismatch.
    try:
        TreeDraft(tokens=[0, 1], parents=[-1])
    except ValueError:
        pass
    else:
        raise AssertionError("length mismatch must raise")

    # Wrong root parent.
    try:
        TreeDraft(tokens=[0], parents=[0])
    except ValueError:
        pass
    else:
        raise AssertionError("non-(-1) root parent must raise")

    # Forward reference (parent > child).
    try:
        TreeDraft(tokens=[0, 1], parents=[-1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("forward parent reference must raise")

    # Self-reference (parents[i] == i).
    try:
        TreeDraft(tokens=[0, 1], parents=[-1, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("self-referential parent must raise")
    print("[OK] TreeDraft validation")


def test_build_tree_width2_depth2() -> None:
    """(W=2, D=2): root + 2 level-1 + 4 level-2. Token ids assigned by the stub in BFS order."""
    _stub_expand_topn.width = 2
    draft = build_tree(_stub_expand_topn(start_token=100), root_hidden=("h", "root"),
                       root_token=99, width=2, depth=2)
    assert draft.size == tree_size(2, 2) == 7
    assert draft.tokens == [99, 100, 101, 102, 103, 104, 105]
    # Parents: root=-1; children of root (idx 1,2) -> parent 0; children of node 1 (idx 3,4) ->
    # parent 1; children of node 2 (idx 5,6) -> parent 2.
    assert draft.parents == [-1, 0, 0, 1, 1, 2, 2]
    assert draft.num_drafts == 6
    print("[OK] build_tree W=2 D=2")


def test_build_tree_width4_depth2() -> None:
    """(W=4, D=2): the EAGLE-2 paper's default tree shape (21 nodes, 20 drafts)."""
    _stub_expand_topn.width = 4
    draft = build_tree(_stub_expand_topn(start_token=200), root_hidden=None,
                       root_token=199, width=4, depth=2)
    assert draft.size == 21
    # Root + 4 level-1 + 16 level-2.
    assert draft.tokens[0] == 199
    assert draft.tokens[1:5] == [200, 201, 202, 203]            # level-1 children of root
    assert draft.tokens[5:9] == [204, 205, 206, 207]            # children of node 1
    assert draft.tokens[9:13] == [208, 209, 210, 211]           # children of node 2
    assert draft.parents[0] == -1
    assert draft.parents[1:5] == [0, 0, 0, 0]
    assert draft.parents[5:9] == [1, 1, 1, 1]
    assert draft.parents[9:13] == [2, 2, 2, 2]
    print("[OK] build_tree W=4 D=2")


def test_build_tree_chain() -> None:
    """(W=1, D=3): degenerate chain — same as linear chained draft (k=3)."""
    _stub_expand_topn.width = 1
    draft = build_tree(_stub_expand_topn(start_token=10), root_hidden=None,
                       root_token=9, width=1, depth=3)
    assert draft.size == 4
    assert draft.tokens == [9, 10, 11, 12]
    assert draft.parents == [-1, 0, 1, 2]
    print("[OK] build_tree chain (W=1)")


def test_build_tree_bad_expand() -> None:
    """An expand callable that returns the wrong number of children fails loud (rule 6)."""
    def bad_expand(prev_hidden, token):
        return [(1, None), (2, None), (3, None)]  # returns 3 even though width=2 was requested

    try:
        build_tree(bad_expand, root_hidden=None, root_token=0, width=2, depth=1)
    except ValueError as e:
        assert "width=2" in str(e)
    else:
        raise AssertionError("bad expand width must raise")
    print("[OK] build_tree bad expand fails loud")


def test_tree_causal_mask_simple() -> None:
    """Hand-built (W=2, D=2) tree: explicit ancestor-set mask."""
    # Topology (BFS):
    #            0 (root)
    #           / \
    #          1   2
    #         / \ / \
    #        3  4 5  6
    parents = [-1, 0, 0, 1, 1, 2, 2]
    mask = tree_causal_mask(parents)
    assert mask.shape == (7, 7)
    arr = mask.tolist()  # python lists for clean equality
    # Row 0 (root): only sees itself.
    assert arr[0] == [True, False, False, False, False, False, False]
    # Row 1 (left child of root): sees {0, 1}.
    assert arr[1] == [True, True, False, False, False, False, False]
    # Row 2 (right child): sees {0, 2}.
    assert arr[2] == [True, False, True, False, False, False, False]
    # Row 3 (leftmost leaf, child of 1): sees {0, 1, 3}.
    assert arr[3] == [True, True, False, True, False, False, False]
    # Row 4 (second leaf, child of 1): sees {0, 1, 4} — masks its sibling 3 and its uncle 2.
    assert arr[4] == [True, True, False, False, True, False, False]
    # Row 5 (child of 2): sees {0, 2, 5} — masks sibling 6 AND uncle/cousins 1/3/4.
    assert arr[5] == [True, False, True, False, False, True, False]
    # Row 6 (child of 2): sees {0, 2, 6}.
    assert arr[6] == [True, False, True, False, False, False, True]
    print("[OK] tree_causal_mask (W=2 D=2)")


def test_tree_causal_mask_root_only() -> None:
    """Trivial root-only tree: 1x1 ``[[True]]`` mask."""
    mask = tree_causal_mask([-1])
    assert mask.shape == (1, 1)
    assert mask.tolist() == [[True]]
    print("[OK] tree_causal_mask (root only)")


def test_tree_causal_mask_chain() -> None:
    """Chain (W=1): mask is the standard lower-triangular causal mask (every node attends to all
    predecessors and itself, no off-path nodes to exclude)."""
    parents = [-1, 0, 1, 2]  # root -> A -> B -> C
    mask = tree_causal_mask(parents)
    expected = [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]
    assert mask.tolist() == expected
    print("[OK] tree_causal_mask chain == lower-triangular")


def test_longest_accepted_perfect_tree() -> None:
    """Greedy matches the first child at every level — accept the full leftmost path."""
    # (W=2, D=2) — same topology as test_tree_causal_mask_simple.
    parents = [-1, 0, 0, 1, 1, 2, 2]
    tokens = [99, 100, 101, 102, 103, 104, 105]
    # Greedy at each node's position == that node's first-child token.
    # root (idx 0) -> wants tokens[1] = 100  ✓ accept node 1
    # node 1 -> wants tokens[3] = 102 ✓ accept node 3
    # node 3 -> wants ??? (idx 3 is a leaf — its greedy is the bonus)
    greedy = [100, 102, 105, 999, 998, 997, 996]
    accepted, bonus_idx = longest_accepted_path(parents, tokens, greedy)
    assert accepted == [1, 3]
    assert bonus_idx == 3
    assert greedy[bonus_idx] == 999
    print("[OK] longest_accepted_path perfect tree")


def test_longest_accepted_no_match() -> None:
    """Greedy at root matches no child — accept 0 drafts; bonus is greedy at root."""
    parents = [-1, 0, 0, 1, 1, 2, 2]
    tokens = [99, 100, 101, 102, 103, 104, 105]
    # root wants 555, which is none of {100, 101}
    greedy = [555, 102, 105, 999, 998, 997, 996]
    accepted, bonus_idx = longest_accepted_path(parents, tokens, greedy)
    assert accepted == []
    assert bonus_idx == 0
    assert greedy[bonus_idx] == 555
    print("[OK] longest_accepted_path no match")


def test_longest_accepted_partial() -> None:
    """Greedy matches the right branch at level 1 but no match at level 2 — accept exactly 1."""
    parents = [-1, 0, 0, 1, 1, 2, 2]
    tokens = [99, 100, 101, 102, 103, 104, 105]
    # root -> wants 101 -> matches node 2; node 2's greedy is 8888 -> no match among {104, 105}.
    greedy = [101, 999, 8888, 999, 999, 999, 999]
    accepted, bonus_idx = longest_accepted_path(parents, tokens, greedy)
    assert accepted == [2]
    assert bonus_idx == 2
    assert greedy[bonus_idx] == 8888
    print("[OK] longest_accepted_path partial accept (right branch)")


def test_longest_accepted_sibling_tiebreak() -> None:
    """When two children share the target token, the first BFS sibling wins."""
    parents = [-1, 0, 0]
    tokens = [0, 42, 42]   # two identical children — ill-typed for a real MTP but legal for the helper
    greedy = [42, 0, 0]    # root wants 42 — both nodes 1 and 2 match
    accepted, bonus_idx = longest_accepted_path(parents, tokens, greedy)
    assert accepted == [1]    # node 1 (the first sibling) wins
    assert bonus_idx == 1
    print("[OK] longest_accepted_path sibling tie-break (BFS order)")


def test_longest_accepted_chain_full_accept() -> None:
    """Chain (W=1, D=3): all 3 drafts match — accept all, bonus is greedy at the leaf."""
    parents = [-1, 0, 1, 2]
    tokens = [10, 11, 12, 13]
    greedy = [11, 12, 13, 14]  # every step matches; bonus = greedy[3] = 14
    accepted, bonus_idx = longest_accepted_path(parents, tokens, greedy)
    assert accepted == [1, 2, 3]
    assert bonus_idx == 3
    assert greedy[bonus_idx] == 14
    print("[OK] longest_accepted_path chain full accept")


def test_longest_accepted_length_mismatch() -> None:
    """Mismatched input lengths fail loud (rule 6 — never silently consume a malformed tree)."""
    try:
        longest_accepted_path([-1, 0], [1, 2, 3], [1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("length mismatch must raise")
    print("[OK] longest_accepted_path length mismatch fails loud")


def test_per_model_dispatch_stubs_import() -> None:
    """Per-model ``spec_generate_tree`` stubs are importable and raise the documented contract.

    We do not call the model paths (they'd need a model) — we only assert the stubs exist with the
    documented signature and raise ``NotImplementedError`` with the follow-on task named, so the
    contract is visible to the spec_decode benches before the per-model attention-mask plumbing
    lands.
    """
    from quanta.dsv4.spec import spec_generate_tree as dsv4_tree
    from quanta.nemotron.spec import spec_generate_tree as nemo_tree
    from quanta.qwen35.spec import spec_generate_tree as qwen_tree

    for fn, name in (
        (dsv4_tree, "dsv4"),
        (nemo_tree, "nemotron"),
        (qwen_tree, "qwen35"),
    ):
        try:
            fn(None, None, None, None, [1, 2, 3], width=2, depth=2, max_new=4)
        except NotImplementedError as e:
            assert "attention-mask plumbing" in str(e) or "spec_generate_tree" in str(e), (
                f"{name}: expected the contract message to name the follow-on, got: {e}"
            )
        else:
            raise AssertionError(f"{name}: spec_generate_tree must raise NotImplementedError")
    print("[OK] per-model spec_generate_tree stubs import + raise on contract")


def main() -> int:
    tests = [
        test_tree_size,
        test_treedraft_validation,
        test_build_tree_width2_depth2,
        test_build_tree_width4_depth2,
        test_build_tree_chain,
        test_build_tree_bad_expand,
        test_tree_causal_mask_simple,
        test_tree_causal_mask_root_only,
        test_tree_causal_mask_chain,
        test_longest_accepted_perfect_tree,
        test_longest_accepted_no_match,
        test_longest_accepted_partial,
        test_longest_accepted_sibling_tiebreak,
        test_longest_accepted_chain_full_accept,
        test_longest_accepted_length_mismatch,
        test_per_model_dispatch_stubs_import,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    # Touch MLX so the import is exercised (mask helpers return mx.array); cheap.
    _ = mx.array([0])
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
