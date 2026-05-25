"""Native MTP self-speculative decoding for Nemotron-H — lossless, 1 draft head.

Nemotron-H ships ONE native MTP head (``num_nextn_predict_layers == 1``; two stacked sub-blocks, see
:mod:`quanta.nemotron.mtp`), so each round drafts exactly **one** token. The structure mirrors
:mod:`quanta.dsv4.spec` / :mod:`quanta.glm.spec` (k == 1):

  1. draft 1 token with the MTP head from ``(main hidden at p, embed[token_{p+1}])`` → ``token_{p+2}``;
  2. verify both in **one** main-model forward over ``[cur, draft]`` at the current offset;
  3. accept the draft iff it equals the main model's greedy token there, and always emit the bonus
     (the main model's greedy token at the next position);
  4. roll the per-attention-layer KV caches and the Mamba recurrence state back to drop a rejected
     draft, so the decode state is bit-identical to never having fed it.

Every emitted token is the main model's own ``argmax`` (the draft is kept only when it *matches* that
argmax; the bonus is the argmax outright), so the output is **bit-identical to plain greedy decode**
(CLAUDE.md rule 4 / losslessness) regardless of MTP quality — the head changes only *speed*. ``stats``
reports ``mean_accept`` (mean tokens emitted per main forward; 1 = no speedup, 2 = every draft
accepted) and ``rounds``.

Consumed contracts (siblings — not reimplemented here):

* The resident Nemotron model — ``model(token_ids, *, caches, ssm, conv, offset, capture_layers)``
  returning ``logits`` or ``(logits, {layer: hidden})``; the hybrid threads per-layer KV caches
  (attention) + ``(ssm, conv)`` (mamba) state, and ``model.cfg`` / ``model.num_layers``. The Nemotron
  hybrid rolls back a rejected draft by **re-running** the accepted window from the verified offset
  (the recurrent Mamba/KV state is reconstructed by the model — the spec loop drives ``offset`` only).
* :class:`quanta.nemotron.mtp.NemotronMTP` — the MTP draft head, called as
  ``mtp(prev_hidden, token_emb, head) -> (logits, new_hidden)``.

Gated model-free (stub main model + stub MTP) in ``parity/nemotron_mtp_spec_test.py``.
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
