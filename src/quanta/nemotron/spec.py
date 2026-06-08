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
``[cur]``. For ``k == 1`` :func:`spec_generate_k` delegates to :func:`spec_generate`, which applies
the **same** snapshot/restore + ``[cur]`` re-run on a rejected draft **when the model carries Mamba
state** (``ssm is not None``) — so the single-step path is lossless on the hybrid too; a stub /
pure-attention model (``ssm is None``) keeps the original KV-only single-step path verbatim.

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

from quanta.paged import manager_seq_of
from quanta.spec.tree import build_tree, enumerate_paths


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


def _resolve_state(model, make_state, *, max_rollback):
    """The decode state triple for a spec run: the caller's ``make_state`` factory when given (the
    tree-spec-over-paged hook, #158-160), else the model's own :func:`_capture_state`.

    ``make_state`` is a callable ``make_state(max_rollback) -> (caches, ssm, conv)`` the serving layer
    wires to build a **paged** triple (each attention layer's KV slot is a paged view into the shared
    :class:`~quanta.paged.PagedKVCacheManager`; the Mamba ``(ssm, conv)`` summary stays per-stream). The
    spec loop then prefills the prompt into it and drives the SAME replicate/truncate/offset contract,
    bracketing every forward with the paged manager lifecycle. ``make_state=None`` (default) is the
    discrete :func:`_capture_state` path, byte-identical to pre-#158 (rule 4).

    The returned state MUST be fresh (every attention KV at offset 0) — the spec loop owns the prefill;
    a pre-filled paged state would double-count positions (rule 6, loud)."""
    if make_state is None:
        return _capture_state(model)
    st = make_state(max_rollback=max_rollback)
    caches, ssm, conv = st if (isinstance(st, tuple) and len(st) == 3) else (st, None, None)
    for c in caches or ():
        if c is not None and getattr(c, "offset", 0) != 0:
            raise ValueError(
                f"nemotron spec make_state must return a FRESH state (every KV at offset 0); got "
                f"offset={c.offset}. The spec loop owns the prefill — pass an empty paged state.")
    return caches, ssm, conv


# --- paged forward lifecycle bracket (#158-160) ----------------------------------------------------
# Nemotron's decode state is a ``(caches, ssm, conv)`` triple, NOT a cache object, so there is no object
# to carry the lifecycle hooks DSV4's PagedDSV4Cache has. Instead the (manager, seq) is recovered from
# the paged KV views (:func:`quanta.paged.manager_seq_of`): ``advance`` opens a forward's positions
# BEFORE it writes them, ``commit`` content-hashes the just-filled blocks AFTER, and ``free`` releases a
# discarded verify replica's COW blocks (rule 6: no pool leak). ``advance`` does NOT move the write
# cursor, so a batched-verify replica's ``offset == q+1+t`` precondition still holds. A DISCRETE state
# has no paged views ⇒ ``manager_seq_of`` is ``None`` and all three are no-ops (rule 4: byte-unchanged).
def _begin_forward(caches, token_ids) -> None:
    h = manager_seq_of(caches)
    if h is not None:
        h[0].advance(h[1], [int(t) for t in token_ids])


def _end_forward(caches) -> None:
    h = manager_seq_of(caches)
    if h is not None:
        h[0].commit(h[1])


def _release(caches) -> None:
    h = manager_seq_of(caches)
    if h is not None:
        h[0].free(h[1])


def spec_generate(model, mtp, embed: mx.array, head: mx.array, prompt_ids,
                  *, max_new: int, eos_id=None, make_state=None) -> tuple[list[int], dict]:
    """Lossless native-MTP self-speculation (1 draft head). Returns ``(tokens, stats)``.

    ``model``: the resident Nemotron model; ``mtp``: the MTP draft head (called
    ``mtp(prev_hidden, token_emb, head) -> (logits, new_hidden)``); ``embed``/``head``: the shared
    embedding ``[vocab, hidden]`` and LM-head ``[vocab, hidden]`` matrices (the MTP fuses
    ``embed[token]`` and reads out through ``head``); ``prompt_ids``: the prompt token ids. ``eos_id``
    may be an ``int`` or a collection of stop ids. ``stats['mean_accept']`` is the mean number of
    tokens emitted per main-model forward (1 = the MTP never helped, 2 = every draft accepted).

    Hybrid-safe: when the model carries Mamba recurrence (``ssm is not None``, threaded via
    :meth:`make_caches`), a rejected draft is rolled out of the un-sliceable ``(ssm, conv)`` summary by
    restoring the pre-verify snapshot + re-running ``[cur]`` (the KV cache slices losslessly on its
    own). A stub / pure-attention model (``ssm is None``) takes the original KV-only path unchanged."""
    last = model.cfg.num_hidden_layers - 1   # capture the final main-layer hidden (the MTP feature)
    prompt_ids = list(prompt_ids)
    if not prompt_ids:
        raise ValueError("spec_generate needs a non-empty prompt")
    stop = _as_stop_set(eos_id)
    caches, ssm, conv = _resolve_state(model, make_state, max_rollback=1)

    # --- prefill: verified token at position q = len(prompt)-1, plus its feature ---
    _begin_forward(caches, prompt_ids)                  # paged: advance; discrete: no-op
    logits, caps = _forward(model, mx.array(prompt_ids), caches=caches, ssm=ssm, conv=conv,
                            offset=0, capture_layers=(last,))
    _end_forward(caches)                                # paged: commit; discrete: no-op
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

        # Snapshot the (un-sliceable) Mamba recurrence BEFORE the verify consumes [cur, draft], so a
        # REJECTED draft can be rolled out of the recurrent summary on a hybrid model. The KV cache
        # slices losslessly (``_rollback``); the Mamba ``(ssm, conv)`` state is a summary of every
        # consumed token and cannot be sliced — restore the snapshot + re-run [cur] (the same
        # snapshot/restore/re-run :func:`spec_generate_k` uses for k>=2, specialized to k=1). A stub
        # / pure-attention model (``ssm is None``) keeps the original KV-only single-step path verbatim.
        ssm_pre = list(ssm) if ssm is not None else None
        conv_pre = list(conv) if conv is not None else None
        pre_offset = q + 1

        # --- verify both in ONE main forward over [cur, draft] at offset q+1 ---
        _begin_forward(caches, [cur, draft])
        vlog, vcaps = _forward(model, mx.array([cur, draft]), caches=caches, ssm=ssm, conv=conv,
                               offset=pre_offset, capture_layers=(last,))
        _end_forward(caches)
        bpred = mx.argmax(vlog[0], axis=-1)             # [2]; bpred[0]=greedy@q+2, bpred[1]=greedy@q+3
        mx.eval(bpred) if vcaps is None else mx.eval(bpred, vcaps[last])
        b0, b1 = int(bpred[0].item()), int(bpred[1].item())

        j = 1 if draft == b0 else 0                     # accept the draft iff it matches main greedy
        bonus = b1 if j == 1 else b0                    # main greedy token at the first unverified pos
        if j == 1:
            out.append(draft)
        out.append(bonus)
        accept_lens.append(j + 1)

        if ssm_pre is not None and j == 0:
            # hybrid + rejected draft: restore the pre-verify recurrent state, roll the KV back to the
            # pre-verify offset, and re-run [cur] so the Mamba summary + KV advance to exactly the
            # committed position (the rejected draft never enters the un-sliceable recurrent state).
            for i in range(len(ssm)):
                ssm[i] = ssm_pre[i]
            for i in range(len(conv)):
                conv[i] = conv_pre[i]
            _rollback(model, caches, pre_offset)        # paged: manager.truncate (KV blocks)
            _begin_forward(caches, [cur])               # re-run [cur] to re-advance KV + Mamba state
            rlog, rcaps = _forward(model, mx.array([cur]), caches=caches, ssm=ssm, conv=conv,
                                   offset=pre_offset, capture_layers=(last,))
            _end_forward(caches)
            mx.eval(rlog) if rcaps is None else mx.eval(rlog, rcaps[last])
            prev_hidden = rcaps[last][-1][None, None]   # main hidden at the committed (cur) position
        else:
            # original single-step path — KV slices losslessly (stub / pure-attention model), or the
            # draft was accepted (j==1: both [cur, draft] are committed, the verify state already
            # matches the committed offset, so the truncate is a no-op).
            _rollback(model, caches, (q + 1) + (j + 1))
            prev_hidden = vcaps[last][j][None, None]    # main hidden at the last accepted position
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
                    *, k: int, max_new: int, eos_id=None, make_state=None) -> tuple[list[int], dict]:
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
                             max_new=max_new, eos_id=eos_id, make_state=make_state)

    last = model.cfg.num_hidden_layers - 1
    prompt_ids = list(prompt_ids)
    if not prompt_ids:
        raise ValueError("spec_generate_k needs a non-empty prompt")
    stop = _as_stop_set(eos_id)
    caches, ssm, conv = _resolve_state(model, make_state, max_rollback=k)

    # --- prefill: verified token at position q = len(prompt)-1, plus its feature ---
    _begin_forward(caches, prompt_ids)                  # paged: advance; discrete: no-op
    logits, caps = _forward(model, mx.array(prompt_ids), caches=caches, ssm=ssm, conv=conv,
                            offset=0, capture_layers=(last,))
    _end_forward(caches)                                # paged: commit; discrete: no-op
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
        _begin_forward(caches, verify_in)
        vlog, vcaps = _forward(model, mx.array(verify_in), caches=caches, ssm=ssm, conv=conv,
                               offset=pre_offset, capture_layers=(last,))
        _end_forward(caches)
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
            _rollback(model, caches, pre_offset)                    # paged: manager.truncate (KV blocks)
            replay = [cur, *drafts[:j]]                              # j + 1 tokens (1 <= len <= k)
            _begin_forward(caches, replay)
            rlog, rcaps = _forward(model, mx.array(replay), caches=caches, ssm=ssm, conv=conv,
                                   offset=pre_offset, capture_layers=(last,))
            _end_forward(caches)
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


def _mtp_top_w(logits_row: mx.array, width: int) -> list[int]:
    """Top-``width`` token ids from a ``[vocab]`` MTP logit row, most-likely first.

    Used by :func:`spec_generate_tree` to seed the BFS tree-build. For the canonical
    ``width ∈ {2, 4, 8}`` and Nemotron's vocab, a single ``argsort`` is trivially cheap (~µs)."""
    if width == 1:
        return [int(mx.argmax(logits_row).item())]
    idx = mx.argsort(-logits_row.astype(mx.float32))     # descending order
    return [int(idx[i].item()) for i in range(width)]


def batch_verify(model, state, cur: int, paths: list[list[int]],
                 *, depth: int, q: int, last_layer: int
                 ) -> tuple[int, int, list[int], mx.array]:
    """**B-wide batched verify** of ``B = len(paths)`` candidate draft paths against the main model
    over Nemotron's hybrid stack (88 layers = 80 Mamba SSM + 8 GQA + MoE).

    The :func:`spec_generate_tree`'s ``batched=True`` hot loop. Replicates the prefix state triple
    ``(caches, ssm, conv)`` ``B`` times via
    :func:`quanta.nemotron.batched_runtime.replicate_state` (zero-copy structural sharing under
    MLX's immutable arrays + per-replica list-spine ownership), then advances all ``B`` replicas
    through :meth:`quanta.nemotron.batched_runtime.NemotronBatchedResidentModel.batch_step` one
    tree-position at a time — every verify position is ONE batched-MoE call over all ``B``
    candidates, vs ``W ** D`` sequential per-path forwards in the non-batched form.

    Output is bit-identical to the sequential form (same paths considered, BFS-leftmost selection,
    same commit-replay); only the verify cost economy changes. See docs/batched_tree_verify.md.

    Inputs:

    * ``model`` — a :class:`NemotronBatchedResidentModel` (the only runtime with ``batch_step``).
    * ``state`` — the prefix state triple ``(caches, ssm, conv)``, un-replicated and read-only here.
    * ``cur`` — verified token at position ``q + 1`` (verify root).
    * ``paths`` — ``B = W ** D`` candidate draft paths, each of length ``depth``.
    * ``q`` — current consumed position (every attention-layer KV in ``state`` is at offset ``q+1``).
    * ``last_layer`` — layer index whose residual to capture for the next MTP build.

    Returns ``(best_j, best_bonus, best_path, best_hidden)``:

    * ``best_j`` — accepted-draft count for the best path, in ``[0, depth]``.
    * ``best_bonus`` — main greedy at the first un-accepted tree position (BFS-leftmost tie-break).
    * ``best_path`` — the candidate path that achieved ``best_j``.
    * ``best_hidden`` — main hidden ``[1, 1, hidden]`` at the last-accepted position along
      ``best_path`` — the ``prev_hidden`` for the next round's MTP tree-build.

    The replicas are discarded after pick; the commit-replay in :func:`spec_generate_tree` re-feeds
    ``[cur, *best_path[:best_j]]`` through the ORIGINAL (un-replicated) ``state``, exactly as the
    sequential form's final replay would — the Mamba ``(ssm, conv)`` and KV state advance to the
    committed offset.
    """
    if not hasattr(model, "batch_step"):
        raise TypeError(
            "batch_verify needs a model with .batch_step (NemotronBatchedResidentModel). The "
            "single-stream NemotronResidentModel does not implement it; use "
            "spec_generate_tree(..., batched=False) or wrap your single-stream model via "
            "NemotronBatchedResidentModel."
        )
    b = len(paths)
    if b < 1:
        raise ValueError(f"batch_verify needs >= 1 path (got {b})")
    if any(len(p) != depth for p in paths):
        raise ValueError(f"batch_verify: every path must have length depth={depth}; got "
                         f"lengths {[len(p) for p in paths]}")

    # Late import: nemotron/batched_runtime imports artifact code that touches the loader; spec.py
    # is consumed by tests that may not have the artifact reader path importable. The deferred
    # import keeps the module-load surface light.
    from quanta.nemotron.batched_runtime import replicate_state

    verify_seqs: list[list[int]] = [[cur, *p] for p in paths]
    replicas = replicate_state(state, b)

    per_pos_logits: list[mx.array] = []                   # depth+1 × [B, 1, vocab]
    per_pos_hidden: list[mx.array | None] = []            # depth+1 × [B, 1, hidden]
    for t in range(depth + 1):
        toks = [seq[t] for seq in verify_seqs]
        for rep, tk in zip(replicas, toks, strict=True):  # paged: advance each replica's seq; discrete: no-op
            _begin_forward(rep[0], [tk])
        logits_t, hidden_t = model.batch_step(
            tokens=toks, replicas=replicas, offset=q + 1 + t, capture_layer=last_layer,
        )
        for rep in replicas:                              # paged: commit each replica's seq; discrete: no-op
            _end_forward(rep[0])
        per_pos_logits.append(logits_t)
        per_pos_hidden.append(hidden_t)
    mx.eval(*per_pos_logits, *(h for h in per_pos_hidden if h is not None))

    bp: list[list[int]] = []
    for t in range(depth + 1):
        argmax_t = mx.argmax(per_pos_logits[t][:, 0], axis=-1)                  # [B]
        mx.eval(argmax_t)
        bp.append([int(argmax_t[k].item()) for k in range(b)])

    # BFS-leftmost strict-improvement so the tie-break + selection match the sequential form.
    best_j = -1
    best_bonus = cur
    best_path: list[int] = []
    best_hidden_b = 0
    for k in range(b):
        path_k = paths[k]
        j = 0
        while j < depth and path_k[j] == bp[j][k]:
            j += 1
        if j > best_j:
            best_j = j
            best_bonus = bp[j][k]
            best_path = path_k
            best_hidden_b = k

    if per_pos_hidden[best_j] is None:
        raise RuntimeError(
            f"batch_verify: capture_layer={last_layer} did not return a captured hidden "
            f"(every batch_step at depth t={best_j} returned None). Spec loop relies on the MTP "
            f"feature; refusing to silently drop it (rule 6).")
    best_hidden = per_pos_hidden[best_j][best_hidden_b:best_hidden_b + 1]       # [1, 1, hidden]
    mx.eval(best_hidden)                       # materialize BEFORE freeing the replicas' paged blocks
    for rep in replicas:                       # paged: free each replica's COW blocks (rule 6); discrete: no-op
        _release(rep[0])

    return best_j, best_bonus, best_path, best_hidden


def spec_generate_tree(model, mtp, embed: mx.array, head: mx.array, prompt_ids,
                       *, width: int, depth: int, max_new: int,
                       eos_id=None, batched: bool = True, make_state=None) -> tuple[list[int], dict]:
    """**W-parallel chain-verify** tree drafting over the native Nemotron-H MTP head — lossless,
    hybrid-safe (task #157).

    Each round drafts a ``(width, depth)`` tree from the MTP head, enumerates the ``W ** D``
    root-to-leaf draft paths, and verifies each as a chain ``[cur, *path_drafts]`` through the main
    model — with the per-attention-layer KV caches AND the (un-sliceable) Mamba ``(ssm, conv)``
    recurrent state rolled back losslessly between paths via the snapshot-replay pattern already
    used by :func:`spec_generate_k`. The longest greedy-matching prefix across paths is committed
    (accepted drafts + bonus); a final replay re-feeds the accepted prefix to advance state to the
    committed offset. Output is **bit-identical to plain greedy decode** for any MTP quality
    (CLAUDE.md rule 4) — the head only changes which candidate set is presented to verify, never
    which tokens commit.

    Why W-parallel chain verify and not the single-forward tree-causal-mask form (EAGLE-2)?
    Nemotron-H is an extreme hybrid: 8 GQA attention layers + **80 Mamba-2 SSM layers** (88 total).
    A tree-causal attention mask only makes sense for the 8 attention layers — the 80 Mamba layers
    process tokens via an O(1) recurrent state that has no causal-mask hook; feeding a BFS-
    linearized tree to a Mamba layer in one forward would corrupt the recurrent state by mixing
    alternative branches into the SSM summary. This entry point preserves losslessness for the
    hybrid stack by keeping each path's forward sequential (every layer always sees a contiguous
    chain) at the cost of more forwards per round.

    **Economics.** Per round: ``W ** D`` per-position batched verifies (one batched main-model
    forward in the default ``batched=True`` path) + 1 replay-forward to advance the committed
    state. Real-model parity is gated by ``parity/nemotron_batched_tree_verify_real.py``
    (commit 5 of ``docs/batched_tree_verify.md``): **bit-identical** 32-token streams between
    ``batched=True`` and ``batched=False`` at ``W=2, D=2`` on the int4-g64 resident Nemotron-H.

    The Nemotron baked artifact does **not** ship native MTP weights (0 ``mtp.*`` keys in the
    manifest), so the bench runs at the random-MTP accept floor (``mean_accept ≈ 1/W``) where the
    ``W ** D`` paths all reduce to a single accepted bonus per round — there's no W^D fan-out to
    amortize, so batched and sequential are within ~1% wall-clock at the floor. The lossless
    contract still holds, and once a native MTP head is trained / baked the W^D weight-amortization
    will kick in (mirrors the DSV4 result: 10.43× sequential tree at native-MTP accept rates).

    Returns ``(tokens, stats)`` where ``stats`` reports ``mean_accept``, ``rounds``, ``max_accept``,
    ``width``, ``depth``, ``paths_per_round`` (= ``W ** D``), and ``batched`` (the flag's value for
    this run, for stat collation). Mirrors :func:`spec_generate_k`'s eos / max_new semantics.
    ``width=1`` degenerates to a chain (= :func:`spec_generate_k` with ``k=depth``) and short-
    circuits to that proven path.

    ``batched=True`` (default since commit 5) replaces the per-path ``W ** D`` sequential verify
    forwards with ONE ``B = W ** D`` batched forward via :func:`batch_verify` — the routed-MoE
    weight reads amortize across all ``B`` candidate paths. Output is bit-identical to
    ``batched=False`` (real-parity gated as above); only verify cost changes. Requires ``model`` to
    be a :class:`quanta.nemotron.batched_runtime.NemotronBatchedResidentModel` (the only runtime
    with ``batch_step``); single-stream :class:`NemotronResidentModel` callers must pass
    ``batched=False``.
    """
    if width < 1 or depth < 1:
        raise ValueError(f"width must be >= 1, depth must be >= 1 (got width={width}, depth={depth})")
    if width == 1:
        # A width-1 tree is a length-``depth`` chain — defer to the proven chained spec_generate_k
        # rather than reinvent its accept/rollback logic here. ``batched`` has no effect on a chain
        # (no path-level fan-out to batch), so the short-circuit ignores the flag and the gate
        # ``test_width1_matches_spec_generate_k_both_flags`` covers it for both flag values.
        return spec_generate_k(model, mtp, embed, head, prompt_ids,
                               k=depth, max_new=max_new, eos_id=eos_id, make_state=make_state)

    last = model.cfg.num_hidden_layers - 1
    prompt_ids = list(prompt_ids)
    if not prompt_ids:
        raise ValueError("spec_generate_tree needs a non-empty prompt")
    stop = _as_stop_set(eos_id)
    caches, ssm, conv = _resolve_state(model, make_state, max_rollback=depth + 1)

    # --- prefill: verified token at q = len(prompt)-1, plus its feature ---
    _begin_forward(caches, prompt_ids)                   # paged: advance; discrete: no-op
    logits, caps = _forward(model, mx.array(prompt_ids), caches=caches, ssm=ssm, conv=conv,
                            offset=0, capture_layers=(last,))
    _end_forward(caches)                                 # paged: commit; discrete: no-op
    mx.eval(logits) if caps is None else mx.eval(logits, caps[last])
    q = len(prompt_ids) - 1
    prev_hidden = caps[last][-1][None, None]              # [1,1,hidden] main hidden at q
    cur = int(mx.argmax(logits[0, -1]).item())            # verified token at q+1
    out = [cur]
    accept_lens: list[int] = []
    stopped = cur in stop

    paths_per_round = width ** depth

    while len(out) < max_new and not stopped:
        # --- build the (W, D) MTP draft tree ---
        # The Nemotron MTP signature is ``mtp(prev_hidden, token_emb, head, return_hidden=True) ->
        # (logits, new_hidden)`` — note ``token_emb`` (a pre-embedded id), NOT ``next_ids``. All
        # ``width`` children of one parent share the SAME post-MTP hidden — the MTP head only sees
        # parent context, not the per-child token-id branching.
        def _expand(parent_hidden: mx.array, parent_token: int):
            token_emb = embed[int(parent_token)][None, None].astype(parent_hidden.dtype)
            d_log, d_h = mtp(parent_hidden, token_emb, head, return_hidden=True)
            top = _mtp_top_w(d_log[0, 0], width)
            return [(tok, d_h) for tok in top]

        draft = build_tree(_expand, root_hidden=prev_hidden, root_token=cur,
                           width=width, depth=depth)
        paths = enumerate_paths(draft.parents, draft.tokens)   # W^D paths of length D each

        pre_offset = q + 1

        if batched:
            # --- B-wide batched verify (one batched forward per tree position) ---
            # Replicas (B triples) advance lock-step under ``model.batch_step``; the original
            # ``(caches, ssm, conv)`` triple is NOT mutated (replicate_state shallow-copies KV via
            # ``_copy`` and clones the ssm/conv list spines per replica — all MLX arrays shared by
            # ref are immutable). The commit-replay below re-feeds the accepted prefix through the
            # ORIGINAL state, exactly as the sequential form's replay would.
            best_j, best_bonus, best_path, best_hidden = batch_verify(
                model, (caches, ssm, conv), cur, paths,
                depth=depth, q=q, last_layer=last,
            )
        else:
            # --- per-path verify: snapshot recurrent state ONCE at round start; restore between paths ---
            # The Mamba ``(ssm, conv)`` state is a summary of every consumed token — it cannot be
            # sliced like a KV cache. ``list(...)`` is O(layers); MLX arrays are immutable so the
            # captured references stay bit-exact across mutation of the live lists by intermediate
            # path verifies.
            ssm_round_start = list(ssm) if ssm is not None else None
            conv_round_start = list(conv) if conv is not None else None

            best_j = -1
            best_bonus = cur                              # safe sentinel; overwritten on first path
            best_path = []
            best_hidden = prev_hidden

            for path_drafts in paths:
                verify_in = [cur, *path_drafts]
                _begin_forward(caches, verify_in)
                vlog, vcaps = _forward(model, mx.array(verify_in), caches=caches, ssm=ssm,
                                       conv=conv, offset=pre_offset, capture_layers=(last,))
                _end_forward(caches)
                bpred = mx.argmax(vlog[0], axis=-1)        # [depth+1]
                mx.eval(bpred) if vcaps is None else mx.eval(bpred, vcaps[last])
                bp = [int(bpred[i].item()) for i in range(depth + 1)]

                # Longest matching prefix of ``path_drafts`` against ``bp[:depth]``.
                j = 0
                while j < depth and path_drafts[j] == bp[j]:
                    j += 1

                if j > best_j:
                    best_j = j
                    best_bonus = bp[j]
                    best_path = path_drafts
                    best_hidden = vcaps[last][j][None, None]   # hidden at last accepted draft / cur

                # Roll back to round-start for the next path verify.
                # KV cache: per-position slice via ``_rollback``. Mamba state: restore the
                # round-start snapshot in-place (each path-verify mutates the live ssm/conv lists).
                if ssm_round_start is not None:
                    for i in range(len(ssm)):
                        ssm[i] = ssm_round_start[i]
                if conv_round_start is not None:
                    for i in range(len(conv)):
                        conv[i] = conv_round_start[i]
                _rollback(model, caches, pre_offset)

        # --- commit: replay [cur, *best_path[:best_j]] to advance state to the committed offset ---
        # After all per-path rollbacks the cache + recurrent state are at the round-start (pre_offset).
        # One bounded replay forward of length ``best_j + 1 <= depth + 1`` brings them forward to the
        # committed offset. ``best_j == 0`` still replays ``[cur]`` (one token) — the single
        # post-cur prefill that the chained spec also runs to position state at the bonus.
        replay = [cur, *best_path[:best_j]]
        _begin_forward(caches, replay)                   # paged: advance the ORIGINAL seq; discrete: no-op
        rlog, rcaps = _forward(model, mx.array(replay), caches=caches, ssm=ssm, conv=conv,
                               offset=pre_offset, capture_layers=(last,))
        _end_forward(caches)
        mx.eval(rlog) if rcaps is None else mx.eval(rlog, rcaps[last])
        # The replay's final-position hidden is the bit-exact recompute of best_hidden (same input,
        # same starting state); we keep best_hidden as the value carried forward (avoid a needless
        # equality check, the result is the same MLX value graph).
        prev_hidden = best_hidden

        out.extend(best_path[:best_j])
        out.append(best_bonus)
        accept_lens.append(best_j + 1)

        q = q + 1 + best_j
        cur = best_bonus

        # Bounded eos check on the tokens emitted this round (inclusive of the bonus).
        if stop:
            tail = best_path[:best_j] + [best_bonus]
            for t in tail:
                if t in stop:
                    stopped = True
                    break

    out = out[:max_new]
    if stop:                                              # terminate at the first eos (inclusive)
        for i, t in enumerate(out):
            if t in stop:
                out = out[: i + 1]
                break
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
