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


def _resolve_caches(model, make_state, max_rollback: int):
    """The decode cache for a spec run: the caller's ``make_state`` factory when given, else the
    model's own :func:`_make_caches`.

    ``make_state`` is the tree-spec-over-paged hook (#158-160): a callable ``make_state(max_rollback)
    -> cache`` the serving layer wires to build a **paged** :class:`~quanta.dsv4.decode.PagedDSV4Cache`
    (latent in the shared block pool; ``replicate(B)`` forks the sequence COW) instead of the discrete
    per-stream :class:`~quanta.dsv4.decode.DSV4Cache`. The spec loop is otherwise identical — it
    prefills the prompt into whatever cache it gets and drives the SAME ``replicate``/``truncate``/
    ``offset`` contract (the paged cache satisfies it via M0). ``make_state=None`` (default) is
    byte-identical to the pre-#158 path (rule 4).

    The returned cache MUST be fresh (``offset == 0``) — the spec loop owns the prefill; a pre-filled
    cache would double-count positions. Fail loud (rule 6) rather than silently mis-offset."""
    if make_state is None:
        return _make_caches(model, max_rollback=max_rollback)
    caches = make_state(max_rollback=max_rollback)
    if caches.offset != 0:
        raise ValueError(
            f"spec make_state must return a FRESH cache (offset 0); got offset={caches.offset}. "
            f"The spec loop owns the prefill — pass an empty paged cache, not a pre-filled one.")
    return caches


# --- paged forward lifecycle bracket (#158-160) -------------------------------------------------
# A PAGED cache (:class:`~quanta.dsv4.decode.PagedDSV4Cache`) needs the manager ``advance`` (open the
# positions) driven BEFORE a forward writes them and ``commit`` AFTER, plus ``free`` on a discarded
# verify replica. The discrete cache writes latent directly, so its ``begin_forward``/``end_forward``/
# ``release`` are no-ops (rule 4: the discrete spec path is byte-identical to pre-#158). The spec loop
# brackets every forward through these helpers; ``getattr`` keeps the model-free stub caches (which
# track their own length and never define the hooks) unaffected.
def _begin_forward(cache, token_ids) -> None:
    fn = getattr(cache, "begin_forward", None)
    if fn is not None:
        fn(token_ids)


def _end_forward(cache) -> None:
    fn = getattr(cache, "end_forward", None)
    if fn is not None:
        fn()


def _release(cache) -> None:
    fn = getattr(cache, "release", None)
    if fn is not None:
        fn()


def spec_generate(model, mtp, embed: mx.array, head: mx.array, prompt_ids,
                  *, max_new: int, eos_id: int | None = None, make_state=None
                  ) -> tuple[list[int], dict]:
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
    caches = _resolve_caches(model, make_state, max_rollback=1)

    # --- prefill: verified token at position q = len(prompt)-1, plus its feature ---
    _begin_forward(caches, prompt_ids)                  # paged: advance; discrete: no-op
    logits, caps = model(mx.array(prompt_ids), caches=caches, offset=0, capture_layers=(last,))
    _end_forward(caches)                                # paged: commit; discrete: no-op
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
        _begin_forward(caches, [cur, draft])
        vlog, vcaps = model(mx.array([cur, draft]), caches=caches, offset=q + 1,
                            capture_layers=(last,))
        _end_forward(caches)
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
                    *, k: int, max_new: int, eos_id: int | None = None, make_state=None
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
        return spec_generate(model, mtp, embed, head, prompt_ids, max_new=max_new, eos_id=eos_id,
                             make_state=make_state)

    last = model.cfg.num_hidden_layers - 1
    prompt_ids = list(prompt_ids)
    if not prompt_ids:
        raise ValueError("spec_generate_k needs a non-empty prompt")
    caches = _resolve_caches(model, make_state, max_rollback=k)  # ring must support full-suffix reject

    _begin_forward(caches, prompt_ids)                  # paged: advance; discrete: no-op
    logits, caps = model(mx.array(prompt_ids), caches=caches, offset=0, capture_layers=(last,))
    _end_forward(caches)
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
        _begin_forward(caches, verify_seq)
        vlog, vcaps = model(mx.array(verify_seq), caches=caches, offset=q + 1,
                            capture_layers=(last,))
        _end_forward(caches)
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


def batch_verify(model, cache: DSV4Cache, cur: int, paths: list[list[int]],
                 *, depth: int, q: int, last_layer: int
                 ) -> tuple[int, int, list[int], mx.array]:
    """**B-wide batched verify** of ``B = len(paths)`` candidate draft paths against the main model.

    The :func:`spec_generate_tree`'s ``batched=True`` hot loop. Replicates the prefix decode cache
    ``B`` times (structural sharing via :meth:`DSV4Cache.replicate` — zero-cost under MLX's
    immutable arrays), then advances all ``B`` replicas through :meth:`DSV4BatchedResidentModel.batch_step`
    one tree-position at a time — every verify position is ONE batched-MoE call over all ``B``
    candidates, vs ``W ** D`` sequential per-path forwards in the non-batched form. Output is
    bit-identical to the sequential form (same paths considered, same longest-matching-prefix
    selection, same commit-forward); only the verify cost economy changes (see
    docs/batched_tree_verify.md for the cost table).

    Inputs:

    * ``model`` — a :class:`quanta.dsv4.batched_runtime.DSV4BatchedResidentModel` (the only
      runtime with the ``batch_step`` API). Single-stream models lack ``batch_step`` and will fail
      loudly (rule 6).
    * ``cache`` — the prefix decode cache (un-replicated, read-only here).
    * ``cur`` — the verified token at position ``q + 1`` (verify root).
    * ``paths`` — ``B = W ** D`` candidate draft paths, each of length ``depth``.
    * ``q`` — current consumed position (``cache.offset = q + 1``).
    * ``last_layer`` — the layer index whose HC residual to capture for the next MTP build.

    Returns ``(best_j, best_bonus, best_path, best_hidden)``:

    * ``best_j`` — accepted-draft count for the best path, in ``[0, depth]``.
    * ``best_bonus`` — main-model greedy token at the first un-accepted tree position (BFS-leftmost
      tie-break, matching the sequential form).
    * ``best_path`` — the candidate path that achieved ``best_j``.
    * ``best_hidden`` — main hidden ``[1, 1, hc, dim]`` at the last-accepted position along
      ``best_path`` — the ``prev_hidden`` for the next round's MTP tree-build.

    The replicas are discarded after pick; the commit-forward (in :func:`spec_generate_tree`) re-feeds
    ``[cur, *best_path[:best_j]]`` through the ORIGINAL (un-replicated) cache, exactly as today.
    """
    if not hasattr(model, "batch_step"):
        raise TypeError(
            "batch_verify needs a model with .batch_step (DSV4BatchedResidentModel). Single-stream "
            "DSV4ResidentModel does not implement it; use spec_generate_tree(..., batched=False) or "
            "construct a DSV4BatchedResidentModel via .from_inner around your single-stream model."
        )
    b = len(paths)
    if b < 1:
        raise ValueError(f"batch_verify needs >= 1 path (got {b})")
    if any(len(p) != depth for p in paths):
        raise ValueError(f"batch_verify: every path must have length depth={depth}; got "
                         f"lengths {[len(p) for p in paths]}")
    if cache.offset != q + 1:
        raise ValueError(f"batch_verify: cache.offset={cache.offset} != q+1={q + 1} "
                         f"(prefix cache must sit at verify-root position)")

    # Per-path verify sequence: each [cur, *path_drafts] of length depth+1 — same as sequential.
    verify_seqs: list[list[int]] = [[cur, *p] for p in paths]

    # Structural-sharing replication: B caches all referencing the prefix arrays (no copy cost).
    replicas = cache.replicate(b)

    # Drive depth+1 batched steps. At step t every replica reads its own row t of verify_seqs (a
    # different draft token across replicas) and writes that position's KV into its own (now
    # diverged) suffix. The MoE call runs ONCE over the stacked [B,1,hc,dim] — the win.
    per_pos_logits: list[mx.array] = []                   # depth+1 × [B, 1, vocab]
    per_pos_hidden: list[mx.array | None] = []            # depth+1 × [B, 1, hc, dim]
    for t in range(depth + 1):
        toks = [seq[t] for seq in verify_seqs]
        for r, tk in zip(replicas, toks, strict=True):    # paged: advance each replica; discrete: no-op
            _begin_forward(r, [tk])
        logits_t, hidden_t = model.batch_step(
            tokens=toks, caches=replicas, offset=q + 1 + t, capture_layer=last_layer,
        )
        for r in replicas:                                # paged: commit each replica; discrete: no-op
            _end_forward(r)
        per_pos_logits.append(logits_t)
        per_pos_hidden.append(hidden_t)
    # Single eval at the end — covers logits + every captured hidden + every replica's grown state.
    mx.eval(*per_pos_logits, *(h for h in per_pos_hidden if h is not None))

    # --- interpret outputs row-by-row (cheap Python loop over B ≤ ~64) -------
    # bp[t][b] = main-model greedy at verify position t for replica b — argmaxed once per (t, b).
    bp: list[list[int]] = []
    for t in range(depth + 1):
        argmax_t = mx.argmax(per_pos_logits[t][:, 0], axis=-1)                  # [B]
        mx.eval(argmax_t)
        bp.append([int(argmax_t[k].item()) for k in range(b)])

    # BFS-leftmost tie-break == sequential form: walk paths in build_tree order, keep STRICT improvement.
    best_j = -1
    best_bonus = cur                                       # sentinel; overwritten on first path
    best_path: list[int] = []
    best_hidden_b = 0                                      # row of per_pos_hidden to pluck
    for k in range(b):
        path_k = paths[k]
        # bp[i][k] = greedy at position i for stream k = the token that should follow verify_seqs[k][i].
        # path_k[i] is accepted iff bp[i][k] == path_k[i].
        j = 0
        while j < depth and path_k[j] == bp[j][k]:
            j += 1
        if j > best_j:
            best_j = j
            best_bonus = bp[j][k]
            best_path = path_k
            best_hidden_b = k

    # Materialize the picked replica's last-accepted-position hidden as the next prev_hidden.
    if per_pos_hidden[best_j] is None:                     # capture_layer was out of range; caller's bug
        raise RuntimeError(
            f"batch_verify: capture_layer={last_layer} did not return a captured hidden "
            f"(every batch_step at depth t={best_j} returned None). Spec loop relies on the MTP "
            f"feature; refusing to silently drop it (rule 6).")
    best_hidden = per_pos_hidden[best_j][best_hidden_b:best_hidden_b + 1]       # [1,1,hc,dim]
    mx.eval(best_hidden)                       # materialize BEFORE freeing the replicas' paged blocks
    for r in replicas:                         # paged: free COW blocks (rule 6, no pool leak); discrete: no-op
        _release(r)

    return best_j, best_bonus, best_path, best_hidden


def spec_generate_tree(model, mtp, embed: mx.array, head: mx.array, prompt_ids,
                       *, width: int, depth: int, max_new: int,
                       eos_id: int | None = None,
                       batched: bool = True, make_state=None) -> tuple[list[int], dict]:
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

    **Economics.** Per round: ``W ** D`` per-position batched verifies (one batched main-model
    forward in the default ``batched=True`` path) + 1 commit-forward to advance the cache to the
    committed offset. At ``W=2, D=2`` the per-position verify is a ``B=4`` stacked forward.
    Real-model parity + bench (``parity/dsv4_batched_tree_verify_real.py``, commit 5 of
    ``docs/batched_tree_verify.md``) measured at ``max_new=64`` with native MTP:

      * ``spec_generate_k(k=2)``: 175.67s, 0.36 tok/s, mean_accept 2.52/3
      * ``spec_generate_tree(W=2,D=2, batched=False)``: 486.39s, 0.13 tok/s, mean_accept 2.74/3
      * ``spec_generate_tree(W=2,D=2, batched=True)``: 46.65s, 1.37 tok/s, mean_accept 2.74/3

    → batched-tree is **3.77× chained k=2** and **10.43× sequential tree** while keeping the
    higher accept rate. ``batched=True`` is the default since commit 5 landed; ``batched=False``
    stays as the proven reference path for parity bisection.

    Returns ``(tokens, stats)`` where ``stats`` reports ``mean_accept`` (tokens emitted per round),
    ``rounds``, ``max_accept``, ``width``, ``depth``, ``paths_per_round`` (= ``W ** D``), and
    ``batched`` (the flag's value for this run, for stat collation). Mirrors
    :func:`spec_generate_k`'s eos / max_new semantics. ``width=1`` degenerates to a chain
    (= :func:`spec_generate_k` with ``k=depth``) and short-circuits to that proven path.

    ``batched=True`` (default) replaces the per-path ``W ** D`` sequential verify forwards with ONE
    ``B = W ** D`` batched forward via :func:`batch_verify` — the routed-MoE weight reads amortize
    across all ``B`` candidate paths. Output is bit-identical to ``batched=False`` (real-model
    parity gated at ``W=2, D=2, max_new=32`` — see the parity script linked above). Requires
    ``model`` to be a :class:`quanta.dsv4.batched_runtime.DSV4BatchedResidentModel` (the only
    runtime with ``batch_step``); single-stream :class:`DSV4ResidentModel` callers must pass
    ``batched=False`` or wrap their inner via ``DSV4BatchedResidentModel.from_inner``.

    ``make_state`` (default ``None``) is the **tree-spec-over-paged** hook (#158-160): a callable
    ``make_state(max_rollback) -> cache`` the serving layer wires to build a **paged**
    :class:`~quanta.dsv4.decode.PagedDSV4Cache` (latent in the shared block pool, ``replicate(B)``
    forks the sequence COW) instead of the discrete per-stream cache. The spec loop then prefills the
    prompt into that paged cache and drives the SAME ``replicate``/``truncate``/``offset`` contract;
    output is bit-identical to the discrete path (the paged cache is a behavior-exact substitute —
    gated in ``parity/dsv4_spec_paged_test.py`` model-free + a real-weight gate). ``None`` is
    byte-identical to the pre-#158 path (rule 4). See :meth:`DSV4BatchedResidentModel.make_paged_state`
    for the factory the serving layer passes.
    """
    if width < 1 or depth < 1:
        raise ValueError(f"width must be >= 1, depth must be >= 1 (got width={width}, depth={depth})")
    if width == 1:
        # A width-1 tree is a length-``depth`` chain — defer to the proven chained spec_generate_k
        # rather than reinvent its accept/rollback logic here. ``batched`` has no effect on a
        # chain (no path-level fan-out to batch), so the short-circuit ignores the flag and the
        # gate ``test_width1_matches_spec_generate_k`` covers it for both flag values.
        return spec_generate_k(model, mtp, embed, head, prompt_ids,
                               k=depth, max_new=max_new, eos_id=eos_id, make_state=make_state)

    last = model.cfg.num_hidden_layers - 1
    prompt_ids = list(prompt_ids)
    if not prompt_ids:
        raise ValueError("spec_generate_tree needs a non-empty prompt")
    # max_rollback = depth + 1: per-path verify pushes ``depth+1`` new tokens; the round-start
    # (q+1) state must remain reachable for the per-path truncate. The +1 covers the commit-forward
    # in the same ring window. ``make_state`` (when given) builds a paged cache at this depth (#158-160).
    caches = _resolve_caches(model, make_state, max_rollback=depth + 1)

    # --- prefill: verified token at position q = len(prompt)-1, plus its feature ---
    _begin_forward(caches, prompt_ids)                  # paged: advance; discrete: no-op
    logits, caps = model(mx.array(prompt_ids), caches=caches, offset=0, capture_layers=(last,))
    _end_forward(caches)
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

        if batched:
            # --- B-wide batched verify (one batched forward per tree position) ---
            # The replicas advance lock-step under ``model.batch_step``; the original ``caches`` is
            # NOT mutated (zero-cost structural-sharing replication keeps the prefix read-only).
            # The commit-forward below re-feeds the accepted prefix through ``caches`` — same as
            # the sequential form. Output is bit-identical to the per-path verify.
            best_j, best_bonus, best_path, best_hidden = batch_verify(
                model, caches, cur, paths, depth=depth, q=q, last_layer=last,
            )
        else:
            # --- per-path sequential verify with cache snapshot/restore ---
            # Track the best path's (accept length, bonus token, last-accepted-position hidden, the
            # path itself). Captured during verify BEFORE the round-end truncate — those tensors are
            # already materialized; the cache truncate doesn't invalidate them.
            best_j = -1
            best_bonus = cur                              # safe sentinel; overwritten on first path
            best_path = []
            best_hidden = prev_hidden

            for path_drafts in paths:
                verify_seq = [cur, *path_drafts]
                _begin_forward(caches, verify_seq)
                vlog, vcaps = model(mx.array(verify_seq), caches=caches, offset=q + 1,
                                    capture_layers=(last,))
                _end_forward(caches)
                bpred = mx.argmax(vlog[0], axis=-1)        # [depth+1]
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
            _begin_forward(caches, commit_seq)             # paged: advance the ORIGINAL seq; discrete: no-op
            _ = model(mx.array(commit_seq), caches=caches, offset=q + 1)
            _end_forward(caches)
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
        "batched": batched,
    }
