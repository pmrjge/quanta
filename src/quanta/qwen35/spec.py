"""Native MTP self-speculative decoding for Qwen3.5-397B-A17B — lossless, 1 draft head.

Qwen3.5 ships ONE native MTP head (``mtp_num_hidden_layers == 1``, ``qwen3_next_mtp``), so each round
drafts exactly **one** token. The structure mirrors :mod:`quanta.dsv4.spec` / :mod:`quanta.nemotron.spec`
(k == 1):

  1. draft 1 token with the MTP head from ``(main hidden at p, embed[token_{p+1}])`` → ``token_{p+2}``;
  2. verify both in **one** main-model forward over ``[cur, draft]`` at the current offset;
  3. accept the draft iff it equals the main model's greedy token there, and always emit the bonus
     (the main model's greedy token at the next position);
  4. ``truncate`` the decode cache to drop a rejected draft so BOTH the KV state (full layers) AND the
     Gated-DeltaNet recurrent state (linear layers) are bit-identical to never having fed it.

Because every emitted token is the main model's own ``argmax`` (the draft is kept only when it *matches*
that argmax, and the bonus is the argmax outright), the output is **bit-identical to plain greedy
decode** — the MTP head changes only *speed* (tokens per main forward), never correctness (CLAUDE.md
rule 4 / losslessness). The hybrid's lossless rollback is the hard part: the recurrent state cannot be
sliced, so :class:`quanta.qwen35.decode.Qwen35Cache` snapshots it per committed token and ``truncate``
restores the exact post-``length`` state (for ``k == 1`` the rollback is at most one trailing token);
a rollback it cannot perform losslessly RAISES rather than silently keeping a diverged state (rule 6).
``stats`` reports ``mean_accept`` (mean tokens emitted per main forward; 1 = no speedup, 2 = every draft
accepted), ``rounds``, ``max_accept`` and ``k``.

Consumed contracts (siblings — not reimplemented here):

* :class:`quanta.qwen35.runtime.Qwen35ResidentModel` — ``model(token_ids, *, caches, offset,
  capture_layers)`` returns ``logits`` or ``(logits, {layer: hidden})`` where the capture is the
  residual stream ``[T, hidden]`` after the layer; ``model.cfg`` / ``model.num_layers``.
* :class:`quanta.qwen35.decode.Qwen35Cache` — the per-layer decode cache; ``truncate(length)`` rolls
  back rejected drafts losslessly, ``offset`` is the consumed position. Constructed via
  ``model.make_caches()`` when the model provides it, else ``Qwen35Cache(model.num_layers, model.cfg)``.
* :func:`quanta.qwen35.mtp.mtp_forward` / :class:`quanta.qwen35.mtp.MTPHead` — the MTP draft forward,
  called as ``mtp(prev_hidden, next_ids, embed, head) -> logits``.

Gated model-free (stub main model + stub MTP + stub cache) in ``parity/qwen35_mtp_spec_test.py``.
"""

from __future__ import annotations

import mlx.core as mx


def _make_caches(model, *, max_rollback: int = 1):
    """A fresh decode cache for ``model`` — its own factory if it has one (called with
    ``max_rollback`` when supported, else without — so older factories keep working), else a
    default ``Qwen35Cache`` sized for ``max_rollback``.

    ``max_rollback`` = the maximum number of tokens a single ``truncate`` can drop in one call
    (= ``k`` for :func:`spec_generate_k`); the cache's per-linear-layer recurrent snapshot depth
    is sized to it so deeper rollbacks don't overrun the bounded snapshot ring.
    """
    factory = getattr(model, "make_caches", None)
    if factory is not None:
        try:
            return factory(max_rollback=max_rollback)
        except TypeError:
            return factory()
    from quanta.qwen35.decode import Qwen35Cache
    # int8 KV on full-attn layers by default — Kimi pattern (#47); recurrent snapshots sized for k.
    return Qwen35Cache(model.num_layers, model.cfg, quantized=True,
                       max_rollback=max_rollback)


def _normalize_eos(eos_id) -> set[int]:
    """Accept an ``int`` eos, a collection of stop ids, or ``None`` → a set of ints."""
    if eos_id is None:
        return set()
    if isinstance(eos_id, (set, frozenset, tuple, list)):
        return {int(s) for s in eos_id if s is not None}
    return {int(eos_id)}


def spec_generate(model, mtp, embed: mx.array, head: mx.array, prompt_ids,
                  *, max_new: int, eos_id=None) -> tuple[list[int], dict]:
    """Lossless native-MTP self-speculation (1 draft head). Returns ``(tokens, stats)``.

    ``model``: the resident Qwen3.5 model; ``mtp``: the MTP draft head (``mtp(prev_hidden, next_ids,
    embed, head) -> logits``); ``embed``/``head``: the shared embedding ``[vocab, hidden]`` and LM-head
    ``[vocab, hidden]`` matrices the MTP combine + readout need; ``prompt_ids``: the prompt token ids.
    ``eos_id`` may be an ``int`` or a collection of stop ids. ``stats['mean_accept']`` is the mean number
    of tokens emitted per main-model forward (1 = the MTP never helped, 2 = every draft was accepted)."""
    last = model.cfg.num_hidden_layers - 1   # capture the final-layer residual (the MTP feature)
    prompt_ids = list(prompt_ids)
    if not prompt_ids:
        raise ValueError("spec_generate needs a non-empty prompt")
    stop = _normalize_eos(eos_id)
    caches = _make_caches(model)

    # --- prefill: verified token at position q = len(prompt)-1, plus its feature ---
    logits, caps = model(mx.array(prompt_ids), caches=caches, offset=0, capture_layers=(last,))
    mx.eval(logits, caps[last])
    q = len(prompt_ids) - 1
    prev_hidden = caps[last][-1][None, None]            # [1,1,hidden] main hidden at position q
    cur = int(mx.argmax(logits[0, -1]).item())          # verified token at q+1
    out = [cur]
    accept_lens: list[int] = []
    stopped = cur in stop

    while len(out) < max_new and not stopped:
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

        # roll the decode state back to keep only [cur] + j accepted drafts; drop the rejected tail so
        # BOTH the KV (full layers) and the recurrent state (linear layers) are bit-identical to never
        # having fed it (lossless; raises if the recurrent rollback can't be done exactly — rule 6).
        caches.truncate((q + 1) + (j + 1))
        prev_hidden = vcaps[last][j][None, None]        # main hidden at the last accepted position
        q = q + 1 + j
        cur = bonus
        # stop iff an eos was emitted *this round* (the accepted draft or the bonus) — bounded check;
        # any earlier eos would already have stopped the loop, so this is exact.
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


def spec_generate_k(model, mtp, embed: mx.array, head: mx.array, prompt_ids,
                    *, k: int, max_new: int, eos_id=None
                    ) -> tuple[list[int], dict]:
    """Lossless self-speculation with ``k`` **chained** MTP drafts per round.

    For each round: draft 1 token with ``MTP(main_hidden, [cur])``, then chain ``k-1`` more drafts
    each consuming the *previous MTP step's own post-block hidden* as ``prev_hidden`` (the head was
    trained on main-model hidden — the chain runs off-distribution, so acceptance drops fast past
    step 1). Verify all ``k+1`` positions in **one** main-model forward over ``[cur, d_1, ..., d_k]``
    at the current offset; accept the longest greedy-matching prefix of ``[d_1, ..., d_k]`` and
    always emit the bonus (the main model's greedy at the first unverified position). The cache is
    truncated to drop the rejected tail so BOTH the KV state (full layers) AND the Gated-DeltaNet
    recurrent state (linear layers) are bit-identical to never having fed it (rule-4 / losslessness
    — output is identical to plain greedy).

    Why ``k≥2`` at all when chained drafts are off-distribution? On bandwidth-bound MoE decode the
    verify cost per token drops sub-linearly with verify length: even with low chained-accept rates
    a longer verify window can amortize expert weight loads enough to net a win — that's the bet
    this entry point makes measurable. ``k=1`` is equivalent to :func:`spec_generate` (kept as the
    parity-tested code path; this entry is the speed lever).

    Mirrors :func:`spec_generate` shape (same model/cache/MTP contracts, same eos stop semantics).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1 (got {k})")
    if k == 1:
        return spec_generate(model, mtp, embed, head, prompt_ids, max_new=max_new, eos_id=eos_id)

    last = model.cfg.num_hidden_layers - 1
    prompt_ids = list(prompt_ids)
    if not prompt_ids:
        raise ValueError("spec_generate_k needs a non-empty prompt")
    stop = _normalize_eos(eos_id)
    caches = _make_caches(model, max_rollback=k)    # recurrent snapshots must cover full-suffix rejection

    logits, caps = model(mx.array(prompt_ids), caches=caches, offset=0, capture_layers=(last,))
    mx.eval(logits, caps[last])
    q = len(prompt_ids) - 1
    prev_hidden = caps[last][-1][None, None]            # [1,1,hidden] main hidden at position q
    cur = int(mx.argmax(logits[0, -1]).item())
    out = [cur]
    accept_lens: list[int] = []
    stopped = cur in stop

    while len(out) < max_new and not stopped:
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

        if stop:
            tail = drafts[:j] + [bonus]
            for t in tail:
                if t in stop:
                    stopped = True
                    break

    out = out[:max_new]
    if stop:                                             # terminate at the first eos (inclusive)
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
