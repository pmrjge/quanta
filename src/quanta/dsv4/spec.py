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


def _make_caches(model):
    """A fresh decode cache for ``model`` — its own factory if it has one, else a ``DSV4Cache``."""
    factory = getattr(model, "make_caches", None)
    if factory is not None:
        return factory()
    return DSV4Cache(model.num_layers)


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
