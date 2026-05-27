"""Native MTP self-speculative decoding for Nemotron-H — lossless, 1 draft head + multi-step k>=2.

Nemotron-H ships ONE native MTP head (``num_nextn_predict_layers == 1``; two stacked sub-blocks, see
:mod:`quanta.nemotron.mtp`), so each round drafts exactly **one** token with :func:`spec_generate`.
:func:`spec_generate_k` extends this to ``k >= 2`` by CHAINING the single MTP head against its own
post-block hidden — each round drafts ``k`` tokens in series (off-distribution past step 1 → chained-
accept rate drops, but verify amortizes over a longer window). The structure mirrors
:mod:`quanta.dsv4.spec` / :mod:`quanta.glm.spec` (k == 1):

  1. draft k tokens with the MTP head: step 0 uses main hidden at p; steps 1..k-1 chain on the MTP
     block's OWN post-block hidden;
  2. verify all ``k + 1`` tokens (``[cur, d_1, ..., d_k]``) in **one** main-model forward at the
     current offset;
  3. accept the longest greedy-matching prefix (largest ``j`` such that ``d_i == greedy(cur, d_1, ...,
     d_{i-1})`` for all ``i in [1, j]``), and always emit the bonus (the main model's greedy token at
     the first unverified position);
  4. roll the per-attention-layer KV caches and the Mamba recurrence state back so the post-round
     state is bit-identical to having only fed ``[cur, d_1, ..., d_j]``.

Every emitted token is the main model's own ``argmax`` (each kept draft *matches* that argmax; the
bonus is the argmax outright), so the output is **bit-identical to plain greedy decode** (CLAUDE.md
rule 4 / losslessness) regardless of MTP quality — the head changes only *speed*. ``stats`` reports
``mean_accept`` (mean tokens emitted per **round**; ``1`` = no chained draft kept on any round,
``k+1`` = every chained draft kept; perfect-chain rounds run one main forward, partial-reject rounds
run two — verify + a bounded re-run that advances the recurrent state) and ``rounds``.

**Rollback (the hard part for hybrid Mamba).** The KV cache (attention layers) is per-position and
slices losslessly. The Mamba recurrence state ``(ssm, conv)`` is a *summary* of every consumed token
— it cannot be sliced. For ``k >= 2`` we **snapshot** the ``(ssm, conv)`` lists BEFORE each verify
forward (cheap: ``list(...)`` is O(layers), MLX arrays are immutable so the captured references stay
bit-exact); on partial acceptance (``j < k``) we restore the snapshot AND re-run the main model on
just the accepted prefix ``[cur, d_1, ..., d_j]`` to advance state to the committed offset. The
re-run also re-grows the (truncated-back) KV cache to the committed length, so the post-round state
is bit-exact. ``j == k`` (perfect chain) avoids the re-run; ``j == 0`` (all rejected) re-runs only
``[cur]``. For ``k == 1`` :func:`spec_generate_k` delegates to :func:`spec_generate` so the
parity-tested single-step path is unchanged.

Consumed contracts (siblings — not reimplemented here):

* The resident Nemotron model — ``model(token_ids, *, caches, ssm, conv, offset, capture_layers)``
  returning ``logits`` or ``(logits, {layer: hidden})``; the hybrid threads per-layer KV caches
  (attention) + ``(ssm, conv)`` (mamba) state, and ``model.cfg`` / ``model.num_layers``. The
  per-attention-layer KV cache must be built with ``max_rollback >= k`` (see
  :func:`quanta.nemotron.generate.attn_caches`) — a deeper truncate fails LOUD (rule 6).
* :class:`quanta.nemotron.mtp.NemotronMTP` — the MTP draft head, called as
  ``mtp(prev_hidden, token_emb, head) -> (logits, new_hidden)``. The MTP's optional ``draft_topk``
  attribute makes the routed moe sub-block lighter than the main model's (the speed lever).

Gated model-free (stub main model + stub MTP) in ``parity/nemotron_mtp_spec_test.py`` for ``k=1``,
``k=2`` and ``k=3`` (perfect/wrong MTPs, eos stops, k=1 shim, cache rollback pattern).
"""

from __future__ import annotations

import mlx.core as mx


def _as_stop_set(eos_id) -> set[int]:
    """Normalize ``eos_id`` (an ``int``, a collection, or ``None``) into a set of stop ids."""
    if eos_id is None:
        return set()
    if isinstance(eos_id, int):
        return {int(eos_id)}
    return {int(s) for s in eos_id}


def _capture_state(model):
    """A fresh capture-state for ``model`` — its own factory if it has one, else ``None`` triples.

    Returns ``(caches, ssm, conv)``: the per-attention-layer KV caches and the per-mamba-layer
    recurrence state. A model that does not expose ``make_caches`` runs with ``None`` state (fresh
    prefill each call) — which the model-free stub uses (its logits depend only on the input tokens)."""
    factory = getattr(model, "make_caches", None)
    if factory is not None:
        st = factory()
        # accept either a bare caches list or a (caches, ssm, conv) triple
        if isinstance(st, tuple) and len(st) == 3:
            return st
        return st, None, None
    n = getattr(model, "num_layers", None)
    caches = [None] * n if n is not None else None
    return caches, None, None


def _forward(model, token_ids, *, caches, ssm, conv, offset, capture_layers):
    """Call ``model`` on the documented spec contract; return ``(logits, caps)`` (``caps`` the capture
    dict, or ``None``). Contract: ``model(token_ids, *, caches, ssm, conv, offset, capture_layers) ->
    logits | (logits, caps)`` — what the model-free stub implements. The resident Nemotron model is
    wired to this through a thin adapter built at real-model integration (deferred): its native
    ``__call__`` manages ``offset`` via the caches and returns ``(logits, ssm, conv)``, so the adapter
    re-surfaces the captured hidden as ``caps`` and threads ``ssm``/``conv`` itself.

    No signature-probing fallback: a contract mismatch raises loudly (rule 6) rather than being
    silently retried with a narrower call (which could mask a genuine TypeError from inside a
    correctly-signatured model body)."""
    out = model(token_ids, caches=caches, ssm=ssm, conv=conv, offset=offset,
                capture_layers=capture_layers)
    if isinstance(out, tuple) and out and isinstance(out[0], mx.array):
        # a (logits, caps) capture surface vs a (logits, ssm, conv) state surface: caps is a dict.
        if len(out) >= 2 and isinstance(out[1], dict):
            return out[0], out[1]
        return out[0], None
    return out, None


def spec_generate(model, mtp, embed: mx.array, head: mx.array, prompt_ids,
                  *, max_new: int, eos_id=None) -> tuple[list[int], dict]:
    """Lossless native-MTP self-speculation (1 draft head). Returns ``(tokens, stats)``.

    ``model``: the resident Nemotron model; ``mtp``: the MTP draft head (called
    ``mtp(prev_hidden, token_emb, head) -> (logits, new_hidden)``); ``embed``/``head``: the shared
    embedding ``[vocab, hidden]`` and LM-head ``[vocab, hidden]`` matrices (the MTP fuses
    ``embed[token]`` and reads out through ``head``); ``prompt_ids``: the prompt token ids. ``eos_id``
    may be an ``int`` or a collection of stop ids. ``stats['mean_accept']`` is the mean number of
    tokens emitted per main-model forward (1 = the MTP never helped, 2 = every draft accepted)."""
    last = model.cfg.num_hidden_layers - 1   # capture the final main-layer hidden (the MTP feature)
    prompt_ids = list(prompt_ids)
    if not prompt_ids:
        raise ValueError("spec_generate needs a non-empty prompt")
    stop = _as_stop_set(eos_id)
    caches, ssm, conv = _capture_state(model)

    # --- prefill: verified token at position q = len(prompt)-1, plus its feature ---
    logits, caps = _forward(model, mx.array(prompt_ids), caches=caches, ssm=ssm, conv=conv,
                            offset=0, capture_layers=(last,))
    mx.eval(logits) if caps is None else mx.eval(logits, caps[last])
    q = len(prompt_ids) - 1
    prev_hidden = caps[last][-1][None, None]            # [1,1,hidden] main hidden at position q
    cur = int(mx.argmax(logits[0, -1]).item())          # verified token at q+1
    out = [cur]
    accept_lens: list[int] = []
    stopped = cur in stop

    while len(out) < max_new and not stopped:
        # --- draft 1 token: MTP(feat_q, embed[cur]) -> token_{q+2} ---
        token_emb = embed[cur][None, None].astype(prev_hidden.dtype)   # [1,1,hidden]
        dlog, _ = mtp(prev_hidden, token_emb, head)                    # [1,1,vocab]
        draft = int(mx.argmax(dlog[0, 0]).item())

        # --- verify both in ONE main forward over [cur, draft] at offset q+1 ---
        vlog, vcaps = _forward(model, mx.array([cur, draft]), caches=caches, ssm=ssm, conv=conv,
                               offset=q + 1, capture_layers=(last,))
        bpred = mx.argmax(vlog[0], axis=-1)             # [2]; bpred[0]=greedy@q+2, bpred[1]=greedy@q+3
        mx.eval(bpred) if vcaps is None else mx.eval(bpred, vcaps[last])
        b0, b1 = int(bpred[0].item()), int(bpred[1].item())

        j = 1 if draft == b0 else 0                     # accept the draft iff it matches main greedy
        bonus = b1 if j == 1 else b0                    # main greedy token at the first unverified pos
        if j == 1:
            out.append(draft)
        out.append(bonus)
        accept_lens.append(j + 1)

        # roll the decode state back to keep only [cur] + j accepted drafts; drop the rest so the
        # state is bit-identical to never having fed the rejected draft.
        _rollback(model, caches, (q + 1) + (j + 1))
        prev_hidden = vcaps[last][j][None, None]        # main hidden at the last accepted position
        q = q + 1 + j
        cur = bonus
        if bonus in stop or (j == 1 and draft in stop):
            stopped = True

    out = out[:max_new]
    if stop:                                            # terminate at the first eos (inclusive),
        for k, t in enumerate(out):                     # so the stream matches greedy's eos stop
            if t in stop:
                out = out[: k + 1]
                break
    return out, {
        "rounds": len(accept_lens),
        "tokens": len(out),
        "mean_accept": (sum(accept_lens) / len(accept_lens)) if accept_lens else 0.0,
        "max_accept": max(accept_lens) if accept_lens else 0,
        "k": 1,
    }


def _rollback(model, caches, length: int) -> None:
    """Roll the decode caches back to ``length`` consumed positions (drop a rejected draft) — and
    FAIL LOUD if it cannot (rule 6: never silently produce a lossy decode).

    Prefers a model-provided ``truncate(caches, length)`` (the hybrid rebuilds the Mamba recurrence +
    KV from the accepted window); else a single cache object's ``truncate`` (the stub); else a
    per-layer cache list. A ``None`` cache (a stateless re-run model) has nothing to roll back. Any
    non-``None`` cache that cannot ``truncate`` raises — it must NOT be left holding the rejected
    draft (that would silently diverge from greedy, breaking losslessness)."""
    truncate = getattr(model, "truncate", None)
    if truncate is not None:
        truncate(caches, length)
        return
    if caches is None:
        return
    if hasattr(caches, "truncate"):                     # a single cache object (the stub)
        caches.truncate(length)
        return
    try:
        layer_caches = list(caches)
    except TypeError as e:                               # not a model.truncate, not truncatable, not a list
        raise TypeError(
            f"cannot roll back caches of type {type(caches).__name__}: no model.truncate(), no "
            f"cache.truncate(), not iterable — refusing to silently leave a rejected draft resident "
            f"(rule 6 / losslessness)") from e
    for c in layer_caches:                              # a per-layer list
        if c is None:
            continue
        if not hasattr(c, "truncate"):
            raise TypeError(
                f"per-layer cache {type(c).__name__} has no truncate(); cannot drop a rejected draft "
                f"losslessly (rule 6 — wire a real-model rollback adapter before spec-decoding it)")
        c.truncate(length)


def spec_generate_k(model, mtp, embed: mx.array, head: mx.array, prompt_ids,
                    *, k: int, max_new: int, eos_id=None) -> tuple[list[int], dict]:
    """Multi-step lossless native-MTP self-speculation: chain ``k`` draft tokens per round.

    ``k == 1`` delegates to :func:`spec_generate` (the parity-tested single-step path) — bit-identical
    to it on the same inputs. ``k >= 2`` chains the single MTP head against its OWN post-block hidden:
    step 0 uses ``prev_hidden`` (the main model's last-layer hidden at the predicting position); step
    ``i in [1, k)`` feeds the MTP's own ``new_hidden`` from step ``i-1``. The main model verifies all
    ``k + 1`` tokens ``[cur, d_1, ..., d_k]`` in ONE forward; the accepted prefix is the longest
    sequence of drafts that match the main model's greedy next-token at each position. Bonus is the
    main model's greedy next-token at the first unverified position. Lossless by construction (the
    main model is the arbiter), so the output is bit-identical to greedy decode regardless of MTP
    quality. ``stats['mean_accept']`` is the mean tokens emitted **per round** (``1`` = no speedup
    on every round; ``k + 1`` = every chained draft accepted on every round); the wall-clock speedup
    on partial-reject rounds is HALF that ratio because the rollback path runs a second main forward
    (the re-run advances the recurrent state) — so for the wall-clock speedup ceiling, compare against
    a baseline of ``1`` token-per-forward, not ``mean_accept``.

    Rollback (the hybrid-aware part for ``k >= 2``):
    * ``j == k`` (all drafts accepted): no rollback; verify state already matches the committed offset.
      One main forward per round, ``k + 1`` tokens emitted.
    * ``j < k``: restore ``(ssm, conv)`` from the pre-verify snapshot, truncate the cache back to the
      pre-verify offset, then RE-RUN the main model on ``[cur, d_1, ..., d_j]`` so the recurrent
      state + KV are bit-exact at the committed offset. The re-run is bounded by ``j + 1 <= k`` tokens
      — one main forward, no inner Python loop. TWO main forwards per such round (verify + re-run),
      ``j + 1`` tokens emitted. ``eos_id`` (an ``int``, a collection, or ``None``) stops generation
      at the first emitted stop id (inclusive), matching greedy's eos stop."""
    if k < 1:
        raise ValueError(f"spec_generate_k k={k} must be >= 1")
    if k == 1:                                                  # parity-tested single-step path
        return spec_generate(model, mtp, embed, head, prompt_ids,
                             max_new=max_new, eos_id=eos_id)

    last = model.cfg.num_hidden_layers - 1
    prompt_ids = list(prompt_ids)
    if not prompt_ids:
        raise ValueError("spec_generate_k needs a non-empty prompt")
    stop = _as_stop_set(eos_id)
    caches, ssm, conv = _capture_state(model)

    # --- prefill: verified token at position q = len(prompt)-1, plus its feature ---
    logits, caps = _forward(model, mx.array(prompt_ids), caches=caches, ssm=ssm, conv=conv,
                            offset=0, capture_layers=(last,))
    mx.eval(logits) if caps is None else mx.eval(logits, caps[last])
    q = len(prompt_ids) - 1
    prev_hidden = caps[last][-1][None, None]            # [1,1,hidden] main hidden at position q
    cur = int(mx.argmax(logits[0, -1]).item())          # verified token at q+1
    out = [cur]
    accept_lens: list[int] = []
    stopped = cur in stop

    while len(out) < max_new and not stopped:
        # --- chain k drafts (step 0 uses main hidden; steps 1..k-1 feed MTP's own post-block hidden) ---
        drafts: list[int] = []
        h_mtp = prev_hidden                                          # rolling MTP feature
        tok = cur
        for _ in range(k):                                           # bounded chain (k <= max_rollback)
            token_emb = embed[tok][None, None].astype(h_mtp.dtype)   # [1,1,hidden]
            dlog, h_mtp = mtp(h_mtp, token_emb, head, return_hidden=True)
            tok = int(mx.argmax(dlog[0, 0]).item())
            drafts.append(tok)

        # --- snapshot recurrent state pre-verify (Mamba can't be sliced; restore on partial reject) ---
        ssm_pre = list(ssm) if ssm is not None else None
        conv_pre = list(conv) if conv is not None else None
        pre_offset = q + 1                                            # cache offset before verify

        # --- verify all k+1 tokens in ONE main forward over [cur, d_1, ..., d_k] at offset q+1 ---
        verify_in = [cur, *drafts]
        vlog, vcaps = _forward(model, mx.array(verify_in), caches=caches, ssm=ssm, conv=conv,
                               offset=pre_offset, capture_layers=(last,))
        bpred = mx.argmax(vlog[0], axis=-1)                          # [k+1]
        mx.eval(bpred) if vcaps is None else mx.eval(bpred, vcaps[last])
        gpreds = [int(bpred[i].item()) for i in range(k + 1)]

        # --- longest greedy-matching prefix: j drafts accepted, 0 <= j <= k ---
        j = 0
        for i in range(k):
            if drafts[i] == gpreds[i]:
                j += 1
            else:
                break
        bonus = gpreds[j]

        # --- commit: append accepted drafts (d_1..d_j) and the bonus (j+1 tokens this round) ---
        for i in range(j):
            out.append(drafts[i])
        out.append(bonus)
        accept_lens.append(j + 1)

        # --- rollback: keep [cur, d_1..d_j] (target offset pre_offset + j + 1); re-run on j < k ---
        if j < k:
            # restore recurrent state; truncate cache to pre-verify offset; re-run accepted prefix
            if ssm_pre is not None:
                for i in range(len(ssm)):
                    ssm[i] = ssm_pre[i]
            if conv_pre is not None:
                for i in range(len(conv)):
                    conv[i] = conv_pre[i]
            _rollback(model, caches, pre_offset)
            replay = [cur, *drafts[:j]]                              # j + 1 tokens (1 <= len <= k)
            rlog, rcaps = _forward(model, mx.array(replay), caches=caches, ssm=ssm, conv=conv,
                                   offset=pre_offset, capture_layers=(last,))
            mx.eval(rlog) if rcaps is None else mx.eval(rlog, rcaps[last])
            prev_hidden = rcaps[last][-1][None, None]                # hidden at the LAST committed pos
        else:                                                         # perfect chain — verify state OK
            prev_hidden = vcaps[last][j][None, None]                  # hidden at the j-th = k-th input pos

        q = q + j + 1                                                 # advance over j drafts + bonus
        cur = bonus
        if bonus in stop or any(drafts[i] in stop for i in range(j)):
            stopped = True

    out = out[:max_new]
    if stop:                                                          # terminate at first eos (inclusive)
        for i, t in enumerate(out):
            if t in stop:
                out = out[: i + 1]
                break
    return out, {
        "rounds": len(accept_lens),
        "tokens": len(out),
        "mean_accept": (sum(accept_lens) / len(accept_lens)) if accept_lens else 0.0,
        "max_accept": max(accept_lens) if accept_lens else 0,
        "k": k,
    }


def spec_generate_tree(model, mtp, embed: mx.array, head: mx.array, prompt_ids,
                       *, width: int, depth: int, max_new: int,
                       eos_id=None) -> tuple[list[int], dict]:
    """Tree drafting (EAGLE-2 style) over the native Nemotron-H MTP head — **structural stub** (#157).

    The per-round contract this entry point would implement, mirroring :func:`spec_generate_k` but
    with a tree-shaped draft instead of a linear chain:

      1. build a (``width``, ``depth``)-tree of drafts via :func:`quanta.spec.tree.build_tree`,
         taking the top-``width`` MTP children of each node (the trained Nemotron MTP from
         :mod:`quanta.nemotron.mtp`, used un-retrained);
      2. construct the tree-causal attention mask via :func:`quanta.spec.tree.tree_causal_mask`;
      3. verify the flat tree in **one** main forward at the current offset, with that mask applied
         in every **attention** layer's SDPA call;
      4. accept the longest greedy-matching root-to-leaf path via
         :func:`quanta.spec.tree.longest_accepted_path`; emit accepted drafts + bonus; roll the
         per-attention KV caches AND the (un-sliceable) Mamba SSM/conv state back to the committed
         offset via the snapshot-replay pattern already used by :func:`spec_generate_k`.

    EAGLE-2 reports ``mean_accept ≈ 3.5–4.5`` at ``width=4, depth=2``; lossless because the main
    model still arbitrates every emitted token (CLAUDE.md rule 4).

    **Status — structural piece only (this commit).** The tree-build + mask + acceptance primitives
    are in place and parity-tested model-free in ``parity/tree_spec_test.py``. The verify-side
    plumbing — passing a tree-causal mask through Nemotron-H's hybrid (8 GQA attention layers + 80
    Mamba-2 SSM layers) plus the snapshot-replay rollback for the Mamba state across an *accepted
    tree path* (not a chain) — is the per-model follow-on and is NOT included here. This entry
    point raises ``NotImplementedError`` with the follow-on named, so the contract is callable /
    importable but cannot silently produce wrong output (CLAUDE.md rule 6). Fall back to
    :func:`spec_generate_k` for now.
    """
    del model, mtp, embed, head, prompt_ids, width, depth, max_new, eos_id
    raise NotImplementedError(
        "nemotron.spec_generate_tree: structural piece in place (see quanta.spec.tree). "
        "Per-Nemotron attention-mask plumbing (GQA + Mamba snapshot-replay rollback across tree "
        "paths) is the follow-on task — fall back to spec_generate_k(k=width) until that lands."
    )
