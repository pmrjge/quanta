"""Native MTP self-speculative decoding for DeepSeek-V4-Flash — lossless, 1 draft head.

DSV4 ships ONE native MTP head (``num_nextn_predict_layers == 1``), so each round drafts exactly
**one** token. The structure mirrors :mod:`quanta.eagle.spec` specialized to ``k == 1``:

  1. draft 1 token with the MTP head from ``(main hidden at p, embed[token_{p+1}])`` → ``token_{p+2}``;
  2. verify in **one** main-model forward over ``[cur, draft]`` at the current offset;
  3. accept the draft iff it equals the main model's greedy token at that position, and always emit
     the bonus (the main model's greedy token at the next position);
  4. ``truncate`` the decode cache to drop a rejected draft so the KV state is bit-identical to never
     having fed it.

Because every emitted token is the main model's own ``argmax`` (the draft is kept only when it
*matches* that argmax, and the bonus is the argmax outright), the output is **bit-identical to plain
greedy decode** — the MTP head changes only *speed* (tokens per main forward), never correctness
(CLAUDE.md rule-4 / losslessness). ``stats`` reports ``mean_accept`` (mean tokens emitted per main
forward; 1 = no speedup, 2 = every draft accepted) and ``rounds``.

Consumed contracts (siblings — not reimplemented here):

* :class:`quanta.dsv4.runtime.DSV4ResidentModel` — ``model(token_ids, *, caches, offset,
  capture_layers)`` returns ``logits`` or ``(logits, {layer: hidden})`` where the capture is the HC
  residual stream ``[T, hc, dim]`` after the layer; ``model.cfg`` / ``model.num_layers``.
* :class:`quanta.dsv4.decode.DSV4Cache` — the per-layer decode cache; ``truncate(length)`` rolls back
  rejected drafts, ``offset`` is the consumed position. Constructed via ``model.make_caches()`` when
  the model provides it, else ``DSV4Cache(model.num_layers)``.
* :func:`quanta.dsv4.mtp.mtp_forward` / :class:`quanta.dsv4.mtp.MTPHead` — the MTP draft forward,
  called as ``mtp(prev_hidden, next_ids, embed, head) -> logits``.

Gated model-free (stub main model + stub MTP + stub cache) in ``parity/dsv4_mtp_spec_test.py``.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.dsv4.decode import DSV4Cache
from quanta.spec.tree import build_tree, enumerate_paths


def _make_caches(model, *, max_rollback: int = 1):
    """A fresh decode cache for ``model`` — its own factory if it has one (called with ``max_rollback``
    when supported, else without — so older factories keep working), else a default ``DSV4Cache``
    sized for ``max_rollback``. ``max_rollback`` = the maximum number of tokens a single ``truncate``
    can drop in one call (= ``k`` for :func:`spec_generate_k`); the cache's raw-hidden ring scales
    with it so deeper rollbacks don't blow past the bounded window-pooling state."""
    factory = getattr(model, "make_caches", None)
    if factory is not None:
        try:
            return factory(max_rollback=max_rollback)
        except TypeError:
            return factory()
    return DSV4Cache(model.num_layers, max_rollback=max_rollback)


def spec_generate(model, mtp, embed: mx.array, head: mx.array, prompt_ids,
                  *, max_new: int, eos_id: int | None = None) -> tuple[list[int], dict]:
    """Lossless native-MTP self-speculation (1 draft head). Returns ``(tokens, stats)``.

    ``model``: the resident DSV4 model; ``mtp``: the MTP draft head (``mtp(prev_hidden, next_ids,
    embed, head) -> logits``); ``embed``/``head``: the shared embedding ``[vocab, dim]`` and LM-head
    ``[vocab, dim]`` matrices the MTP combine + readout need; ``prompt_ids``: the prompt token ids.
    ``stats['mean_accept']`` is the mean number of tokens emitted per main-model forward (1 = the MTP
    never helped, 2 = every draft was accepted)."""
    last = model.cfg.num_hidden_layers - 1   # capture the final-layer HC residual (the MTP feature)
    prompt_ids = list(prompt_ids)
    if not prompt_ids:
        raise ValueError("spec_generate needs a non-empty prompt")
    caches = _make_caches(model)

    # --- prefill: verified token at position q = len(prompt)-1, plus its feature ---
    logits, caps = model(mx.array(prompt_ids), caches=caches, offset=0, capture_layers=(last,))
    mx.eval(logits, caps[last])
    q = len(prompt_ids) - 1
    prev_hidden = caps[last][-1][None, None]            # [1,1,hc,dim] main hidden at position q
    cur = int(mx.argmax(logits[0, -1]).item())          # verified token at q+1
    out = [cur]
    accept_lens: list[int] = []
    stop = eos_id is not None and cur == eos_id

    while len(out) < max_new and not stop:
        # --- draft 1 token: MTP(feat_q, embed[cur]) -> token_{q+2} ---
        dlog = mtp(prev_hidden, mx.array([[cur]]), embed, head)   # [1,1,vocab]
        draft = int(mx.argmax(dlog[0, 0]).item())

        # --- verify both in ONE main forward over [cur, draft] at offset q+1 ---
        vlog, vcaps = model(mx.array([cur, draft]), caches=caches, offset=q + 1,
                            capture_layers=(last,))
        bpred = mx.argmax(vlog[0], axis=-1)             # [2]; bpred[0]=greedy@q+2, bpred[1]=greedy@q+3
        mx.eval(bpred, vcaps[last])
        b0, b1 = int(bpred[0].item()), int(bpred[1].item())

        j = 1 if draft == b0 else 0                     # accept the draft iff it matches main greedy
        bonus = b1 if j == 1 else b0                    # main greedy token at the first unverified pos
        if j == 1:
            out.append(draft)
        out.append(bonus)
        accept_lens.append(j + 1)

        caches.truncate((q + 1) + (j + 1))              # keep cur + j accepted drafts; drop the rest
        prev_hidden = vcaps[last][j][None, None]        # main hidden at the last accepted position
        q = q + 1 + j
        cur = bonus
        # stop iff an eos was emitted *this round* (the accepted draft or the bonus) — bounded check;
        # any earlier eos would already have stopped the loop, so this is exact.
        if eos_id is not None and (bonus == eos_id or (j == 1 and draft == eos_id)):
            stop = True

    out = out[:max_new]
    if eos_id is not None and eos_id in out:            # terminate at the first eos (inclusive),
        out = out[: out.index(eos_id) + 1]              # so the stream matches greedy's eos stop
    return out, {
        "rounds": len(accept_lens),
        "tokens": len(out),
        "mean_accept": (sum(accept_lens) / len(accept_lens)) if accept_lens else 0.0,
        "max_accept": max(accept_lens) if accept_lens else 0,
        "k": 1,
    }


def spec_generate_k(model, mtp, embed: mx.array, head: mx.array, prompt_ids,
                    *, k: int, max_new: int, eos_id: int | None = None
                    ) -> tuple[list[int], dict]:
    """Lossless self-speculation with ``k`` **chained** MTP drafts per round.

    For each round: draft 1 token with ``MTP(main_hidden, [cur])``, then chain ``k-1`` more drafts
    each consuming the *previous MTP step's own post-block hidden* as ``prev_hidden`` (the head was
    trained on main-model hidden — the chain runs off-distribution, so acceptance drops fast past
    step 1). Verify all ``k+1`` positions in **one** main-model forward over ``[cur, d_1, ..., d_k]``
    at the current offset; accept the longest greedy-matching prefix of ``[d_1, ..., d_k]`` and
    always emit the bonus (the main model's greedy at the first unverified position). The cache is
    truncated to drop the rejected tail so the KV state is bit-identical to never having fed it
    (rule-4 / losslessness — output is identical to plain greedy).

    Why ``k≥2`` at all when chained drafts are off-distribution? On bandwidth-bound MoE decode the
    verify cost per token drops sub-linearly with verify length (see ``parity/dsv4_int4_bench.py``):
    even with low chained-accept rates a longer verify window can amortize expert weight loads
    enough to net a win — that's the bet this entry point makes measurable. ``k=1`` is equivalent to
    :func:`spec_generate` (kept as the parity-tested code path; this entry is the speed lever).

    Mirrors :func:`spec_generate` shape (same model/cache/MTP contracts, same eos stop semantics)."""
    if k < 1:
        raise ValueError(f"k must be >= 1 (got {k})")
    if k == 1:
        return spec_generate(model, mtp, embed, head, prompt_ids, max_new=max_new, eos_id=eos_id)

    last = model.cfg.num_hidden_layers - 1
    prompt_ids = list(prompt_ids)
    if not prompt_ids:
        raise ValueError("spec_generate_k needs a non-empty prompt")
    caches = _make_caches(model, max_rollback=k)    # ring must support full-suffix rejection

    logits, caps = model(mx.array(prompt_ids), caches=caches, offset=0, capture_layers=(last,))
    mx.eval(logits, caps[last])
    q = len(prompt_ids) - 1
    prev_hidden = caps[last][-1][None, None]            # [1,1,hc,dim] main hidden at position q
    cur = int(mx.argmax(logits[0, -1]).item())
    out = [cur]
    accept_lens: list[int] = []
    stop = eos_id is not None and cur == eos_id

    while len(out) < max_new and not stop:
        # --- draft k tokens by chaining the MTP head; first step uses MAIN hidden, subsequent ---
        # steps feed the MTP block's own post-block hidden back in (off-distribution; accept rate
        # decays — verify still arbitrates so this only shifts cost, never correctness).
        drafts: list[int] = []
        d_log, d_h = mtp(prev_hidden, mx.array([[cur]]), embed, head, return_hidden=True)
        d_tok = int(mx.argmax(d_log[0, 0]).item())
        drafts.append(d_tok)
        for _step in range(k - 1):
            d_log, d_h = mtp(d_h, mx.array([[d_tok]]), embed, head, return_hidden=True)
            d_tok = int(mx.argmax(d_log[0, 0]).item())
            drafts.append(d_tok)

        # --- verify [cur, d_1, ..., d_k] in ONE main forward at offset q+1 ---
        verify_seq = [cur, *drafts]
        vlog, vcaps = model(mx.array(verify_seq), caches=caches, offset=q + 1,
                            capture_layers=(last,))
        bpred = mx.argmax(vlog[0], axis=-1)              # [k+1]
        mx.eval(bpred, vcaps[last])
        bp = [int(bpred[i].item()) for i in range(k + 1)]  # bp[i] = main greedy at position q+2+i

        # accept the longest greedy-matching prefix of drafts (j tokens accepted, bonus = bp[j])
        j = 0
        while j < k and drafts[j] == bp[j]:
            j += 1
        bonus = bp[j]
        out.extend(drafts[:j])
        out.append(bonus)
        accept_lens.append(j + 1)

        caches.truncate((q + 1) + (j + 1))               # keep cur + j accepted drafts; drop rest
        prev_hidden = vcaps[last][j][None, None]         # main hidden at the last accepted position
        q = q + 1 + j
        cur = bonus

        if eos_id is not None:
            tail = drafts[:j] + [bonus]
            if eos_id in tail:
                stop = True

    out = out[:max_new]
    if eos_id is not None and eos_id in out:
        out = out[: out.index(eos_id) + 1]
    return out, {
        "rounds": len(accept_lens),
        "tokens": len(out),
        "mean_accept": (sum(accept_lens) / len(accept_lens)) if accept_lens else 0.0,
        "max_accept": max(accept_lens) if accept_lens else 0,
        "k": k,
    }


def _mtp_top_w(logits_row: mx.array, width: int) -> list[int]:
    """Top-``width`` token ids from a ``[vocab]`` MTP logit row, most-likely first.

    Used by :func:`spec_generate_tree`'s ``expand`` callable to seed the BFS tree-build. For DSV4's
    vocab (``163_840``) and the canonical ``width ∈ {2, 4, 8}``, a single ``argsort`` is trivially
    cheap (~µs); we never materialize a per-vocab tensor anywhere else in the spec loop."""
    if width == 1:
        return [int(mx.argmax(logits_row).item())]
    idx = mx.argsort(-logits_row.astype(mx.float32))   # descending order
    return [int(idx[i].item()) for i in range(width)]


def spec_generate_tree(model, mtp, embed: mx.array, head: mx.array, prompt_ids,
                       *, width: int, depth: int, max_new: int,
                       eos_id: int | None = None) -> tuple[list[int], dict]:
    """**W-parallel chain-verify** tree drafting over the native DSV4 MTP head — lossless (#157).

    Each round drafts a ``(width, depth)`` tree from the MTP head, enumerates the ``W ** D``
    root-to-leaf draft paths, and verifies each as a chain ``[cur, *path_drafts]`` through the main
    model — with the decode cache rolled back losslessly between paths via
    :meth:`quanta.dsv4.decode.DSV4Cache.truncate`. The longest greedy-matching prefix across paths
    is committed (accepted drafts + bonus); a final commit-forward re-feeds the accepted drafts to
    advance the cache to the committed offset. Output is **bit-identical to plain greedy decode**
    for any MTP quality (CLAUDE.md rule 4) — the head only changes which candidate set is presented
    to verify, never which tokens commit.

    Why W-parallel chain verify (and not the single-forward tree-causal-mask form)? DSV4 is
    pure-attention (no recurrent state) and *could* admit the EAGLE-2 single-forward tree-mask form
    — but its three attention regimes (``attention_dense`` for ratio-0 layers, the compressed-KV
    path, and the indexed sparse path) each need explicit tree-mask plumbing through their SDPA
    calls AND through every decode stepper in :mod:`quanta.dsv4.decode`. The W-parallel form is
    interface-uniform with Qwen3.5 / Nemotron (also #157), reuses the existing decode steppers as-is,
    and is the natural drop-in target for the planned batched verify (B = ``W ** D`` in ONE forward
    via :mod:`quanta.dsv4.batched_runtime`) that flips the economics in tree drafting's favor.

    **Economics.** This implementation runs ``W ** D + 1`` main-model forwards per round
    (``W ** D`` per-path verifies + 1 commit-forward to advance the cache to the committed offset).
    At ``W=2, D=2`` that is 5 forwards; ``mean_accept`` is bounded by ``depth + 1``, so PER ACCEPTED
    TOKEN this is ~2–3× the cost of :func:`spec_generate_k` (chained k=2: 1 forward at
    ``mean_accept ≈ 2.78``). USE :func:`spec_generate_k` FOR ACTUAL DECODE SPEEDUP UNTIL THE
    BATCHED-VERIFY OR SINGLE-FORWARD TREE-MASK FORM LANDS. This entry point is kept for (a) lossless
    contract correctness — it admits a ``(W, D)`` tree on DSV4 without per-regime mask plumbing;
    (b) interface uniformity with the hybrid-model paths (Qwen3.5 / Nemotron) — same call site;
    (c) the natural follow-on batched-verify integration via :mod:`quanta.dsv4.batched_runtime`
    that collapses the ``W ** D`` sequential forwards into ONE batched forward.

    Returns ``(tokens, stats)`` where ``stats`` reports ``mean_accept`` (tokens emitted per round),
    ``rounds``, ``max_accept``, ``width``, ``depth``, and ``paths_per_round`` (= ``W ** D``).
    Mirrors :func:`spec_generate_k`'s eos / max_new semantics. ``width=1`` degenerates to a chain
    (= :func:`spec_generate_k` with ``k=depth``) and short-circuits to that proven path.
    """
    if width < 1 or depth < 1:
        raise ValueError(f"width must be >= 1, depth must be >= 1 (got width={width}, depth={depth})")
    if width == 1:
        # A width-1 tree is a length-``depth`` chain — defer to the proven chained spec_generate_k
        # rather than reinvent its accept/rollback logic here.
        return spec_generate_k(model, mtp, embed, head, prompt_ids,
                               k=depth, max_new=max_new, eos_id=eos_id)

    last = model.cfg.num_hidden_layers - 1
    prompt_ids = list(prompt_ids)
    if not prompt_ids:
        raise ValueError("spec_generate_tree needs a non-empty prompt")
    # max_rollback = depth + 1: per-path verify pushes ``depth+1`` new tokens; the round-start
    # (q+1) state must remain reachable for the per-path truncate. The +1 covers the commit-forward
    # in the same ring window.
    caches = _make_caches(model, max_rollback=depth + 1)

    # --- prefill: verified token at position q = len(prompt)-1, plus its feature ---
    logits, caps = model(mx.array(prompt_ids), caches=caches, offset=0, capture_layers=(last,))
    mx.eval(logits, caps[last])
    q = len(prompt_ids) - 1
    prev_hidden = caps[last][-1][None, None]            # [1,1,hc,dim] main hidden at position q
    cur = int(mx.argmax(logits[0, -1]).item())          # verified token at q+1
    out = [cur]
    accept_lens: list[int] = []
    stopped = eos_id is not None and cur == eos_id

    paths_per_round = width ** depth

    while len(out) < max_new and not stopped:
        # --- build the (W, D) MTP draft tree ---
        # ``expand(parent_hidden, parent_token)`` runs ONE MTP step and returns the top-``width``
        # children as ``[(child_token, child_hidden), ...]``. The MTP's post-block hidden is fed to
        # the next level (off-distribution past depth 1 — the head was trained on MAIN hidden;
        # acceptance drops fast past step 1, but the chain remains lossless because verify still
        # arbitrates). All ``width`` children of one parent share the SAME post-MTP hidden — the
        # MTP head only sees the parent context, not the per-child token-id branching.
        def _expand(parent_hidden: mx.array, parent_token: int):
            d_log, d_h = mtp(parent_hidden, mx.array([[int(parent_token)]]),
                             embed, head, return_hidden=True)
            top = _mtp_top_w(d_log[0, 0], width)
            return [(tok, d_h) for tok in top]

        draft = build_tree(_expand, root_hidden=prev_hidden, root_token=cur,
                           width=width, depth=depth)
        paths = enumerate_paths(draft.parents, draft.tokens)  # W^D paths of length D each

        # --- per-path verify with cache snapshot/restore ---
        # Track the best path's (accept length, bonus token, last-accepted-position hidden, the
        # path itself). Captured during verify BEFORE the round-end truncate — those tensors are
        # already materialized; the cache truncate doesn't invalidate them.
        best_j = -1
        best_bonus = cur                                  # safe sentinel; overwritten on first path
        best_path: list[int] = []
        best_hidden = prev_hidden

        for path_drafts in paths:
            verify_seq = [cur, *path_drafts]
            vlog, vcaps = model(mx.array(verify_seq), caches=caches, offset=q + 1,
                                capture_layers=(last,))
            bpred = mx.argmax(vlog[0], axis=-1)            # [depth+1]
            mx.eval(bpred, vcaps[last])
            bp = [int(bpred[i].item()) for i in range(depth + 1)]

            # Longest matching prefix of ``path_drafts`` against ``bp[:depth]``.
            # ``bp[i]`` is the main-greedy at verify position ``i`` = the token that should follow
            # ``verify_seq[i]``; so a draft ``path_drafts[i]`` is accepted iff
            # ``bp[i] == path_drafts[i]``.
            j = 0
            while j < depth and path_drafts[j] == bp[j]:
                j += 1

            if j > best_j:
                best_j = j
                best_bonus = bp[j]
                best_path = path_drafts
                best_hidden = vcaps[last][j][None, None]   # hidden at last accepted draft / cur

            # Roll back the cache so the next path verifies from the same round-start state. KV
            # slice + compressor pooling remainder restore + indexer state — DSV4Cache.truncate
            # handles all three regimes uniformly (gated in parity/dsv4_decode_attn_test.py).
            # Within-bound rollback is guaranteed because max_rollback was sized to depth + 1 at
            # make_caches time.
            caches.truncate(q + 1)

        # --- commit the best path ---
        # Re-feed [cur, *best_path[:best_j]] starting at offset q+1 to advance the cache to the
        # committed offset. best_j == 0 commits only cur (cache stays at q+1 — no extra forward).
        commit_seq = [cur, *best_path[:best_j]]
        if commit_seq:
            _ = model(mx.array(commit_seq), caches=caches, offset=q + 1)
            mx.eval(caches.offset)                         # materialize the cache update

        # Emit accepted drafts + the bonus (the main model's greedy at the first un-accepted tree
        # position along the best path). Mirrors :func:`spec_generate_k`'s emit pattern.
        out.extend(best_path[:best_j])
        out.append(best_bonus)
        accept_lens.append(best_j + 1)

        q = q + 1 + best_j
        cur = best_bonus
        prev_hidden = best_hidden

        # Bounded eos check: only the tokens emitted *this round* could have hit an eos that wasn't
        # caught earlier; check them inclusive of the bonus and terminate the loop after emitting.
        if eos_id is not None:
            tail = best_path[:best_j] + [best_bonus]
            if eos_id in tail:
                stopped = True

    out = out[:max_new]
    if eos_id is not None and eos_id in out:            # terminate at the first eos (inclusive)
        out = out[: out.index(eos_id) + 1]
    return out, {
        "rounds": len(accept_lens),
        "tokens": len(out),
        "mean_accept": (sum(accept_lens) / len(accept_lens)) if accept_lens else 0.0,
        "max_accept": max(accept_lens) if accept_lens else 0,
        "width": width,
        "depth": depth,
        "paths_per_round": paths_per_round,
    }
