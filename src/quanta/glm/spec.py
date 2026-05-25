"""Native MTP self-speculative decoding for GLM-5.1 — lossless, 1 draft head.

GLM-5.1 ships ONE native MTP head (``num_nextn_predict_layers == 1``), so each round drafts exactly
**one** token. Identical in structure to :mod:`quanta.dsv4.spec` (k == 1):

  1. draft 1 token with the MTP head from ``(main hidden at p, embed[token_{p+1}])`` → ``token_{p+2}``;
  2. verify in **one** main-model forward over ``[cur, draft]`` at the current offset;
  3. accept the draft iff it equals the main model's greedy token there, and always emit the bonus
     (the main model's greedy token at the next position);
  4. ``truncate`` the decode cache to drop a rejected draft so the KV state is bit-identical to never
     having fed it.

Every emitted token is the main model's own ``argmax`` (the draft is kept only when it *matches* that
argmax; the bonus is the argmax outright), so the output is **bit-identical to plain greedy decode**
(rule 4 / losslessness) — the MTP head changes only *speed*. ``stats`` reports ``mean_accept`` (mean
tokens emitted per main forward; 1 = no speedup, 2 = every draft accepted) and ``rounds``.

Consumed contracts (siblings): the GLM resident model ``model(token_ids, *, caches, offset,
capture_layers)`` returning ``logits`` or ``(logits, {layer: hidden})``; the GLM decode cache with
``truncate(length)`` / ``offset`` (built lazily — the model-free gate passes a stub); and the MTP
forward ``mtp(prev_hidden, next_ids, embed, head) -> logits`` (real forward lands with the GLM block).

Gated model-free (stub main model + stub MTP + stub cache) in ``parity/glm_mtp_spec_test.py``.
"""

from __future__ import annotations

import mlx.core as mx


def _make_caches(model):
    """A fresh decode cache for ``model`` — its own factory if it has one, else a ``GLMCache`` (imported
    lazily so this module loads before the GLM decode unit lands)."""
    factory = getattr(model, "make_caches", None)
    if factory is not None:
        return factory()
    from quanta.glm.decode import GLMCache
    return GLMCache(model.num_layers)


def spec_generate(model, mtp, embed: mx.array, head: mx.array, prompt_ids,
                  *, max_new: int, eos_id=None) -> tuple[list[int], dict]:
    """Lossless native-MTP self-speculation (1 draft head). Returns ``(tokens, stats)``.

    ``model``: the resident GLM model; ``mtp``: the MTP draft head (``mtp(prev_hidden, next_ids, embed,
    head) -> logits``); ``embed``/``head``: the shared embedding ``[vocab, dim]`` and LM-head matrices
    the MTP combine + readout need; ``prompt_ids``: the prompt token ids. ``eos_id`` may be an ``int``
    or a collection of stop ids. ``stats['mean_accept']`` is the mean tokens emitted per main forward."""
    last = model.cfg.num_hidden_layers - 1   # capture the final main-layer hidden (the MTP feature)
    prompt_ids = list(prompt_ids)
    if not prompt_ids:
        raise ValueError("spec_generate needs a non-empty prompt")
    stop = {int(eos_id)} if isinstance(eos_id, int) else {int(s) for s in (eos_id or [])}
    caches = _make_caches(model)

    # --- prefill: verified token at position q = len(prompt)-1, plus its feature ---
    logits, caps = model(mx.array(prompt_ids), caches=caches, offset=0, capture_layers=(last,))
    mx.eval(logits, caps[last])
    q = len(prompt_ids) - 1
    prev_hidden = caps[last][-1][None, None]            # [1,1,...] main hidden at position q
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

        caches.truncate((q + 1) + (j + 1))              # keep cur + j accepted drafts; drop the rest
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
