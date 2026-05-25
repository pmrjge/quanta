"""EAGLE-3 speculative decoding for Kimi-K2.6 — lossless self-speculation with verify.

Each round: the drafter proposes ``k`` tokens, the target verifies all of them in **one** forward,
and we accept the longest prefix whose drafted token equals the target's greedy token, plus one
bonus (the target's correct token at the first mismatch). Output is therefore **bit-identical to
plain greedy decode** regardless of drafter quality — the drafter only changes *speed* (tokens
accepted per target forward), never correctness.

Feature flow (matches training, :func:`quanta.eagle.train._ce_multistep`): each draft step feeds the
previous feature ``f`` (the real reduced target feature on step 0, the drafter's own **feature-space
recurrent** output ``recur`` thereafter) plus the *next* token's embedding; the head-space output maps
through the frozen head to the next draft token. After verify, the target's captured hidden at the last
accepted position becomes the next round's real feature. The MLA cache is rolled back (``truncate``) to
drop rejected drafts so it stays bit-exact.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.cache import MLACache
from quanta.eagle.drafter import DraftCache, EagleDrafter
from quanta.modeling.xattention import DEFAULT_SPARSE

LAYERS = (10, 30, 50)


def _feat3(caps: dict[int, mx.array], layers: tuple[int, ...], pos: int) -> mx.array:
    """Fused low/mid/high target feature at local position ``pos`` -> ``[1,1,3H]``."""
    return mx.concatenate([caps[L][pos] for L in layers])[None, None]


def spec_generate(
    model, drafter: EagleDrafter, embed: mx.array, head: mx.array, prompt_ids,
    *, max_new: int, k: int = 4, layers: tuple[int, ...] = LAYERS,
    quantized_kv: bool = True, sparse=DEFAULT_SPARSE, eos_id: int | None = None,
) -> tuple[list[int], dict]:
    """Lossless EAGLE spec-decode. Returns ``(tokens, stats)``; ``stats['mean_accept']`` = mean
    tokens emitted per target forward (1 = no speedup, k+1 = perfect)."""
    prompt_ids = list(prompt_ids)
    n = model.cfg.num_hidden_layers
    caches = [MLACache(quantized=quantized_kv) for _ in range(n)]
    head_t = head.T

    logits, caps = model(mx.array(prompt_ids), caches=caches, offset=0,
                         capture_layers=layers, absorbed=False, sparse=sparse)
    mx.eval(logits, [c.c_kv for c in caches], [c.k_pe for c in caches], [caps[L] for L in layers])

    q = len(prompt_ids) - 1                        # last position with a real target hidden
    feat3 = _feat3(caps, layers, -1)               # target feature at position q
    cur = int(mx.argmax(logits[0, -1]).item())     # verified token at position q+1
    out = [cur]
    accept_lens: list[int] = []
    stop = eos_id is not None and cur == eos_id

    while len(out) < max_new and not stop:
        # --- draft k tokens canonically: step(f_{q+j}, e_{q+1+j}) -> recur_{q+1+j} -> token_{q+2+j};
        #     self-feed the feature-space recurrent output `recur` (matches training) ---
        dc = DraftCache()
        feat = drafter.reduce_target_features(feat3)   # real reduced target feature f_q
        drafts: list[int] = []
        tok = cur
        for j in range(k):
            recur, normed = drafter.step(feat, embed[tok][None, None], offset=q + 1 + j, cache=dc, mask=None)
            tok = int(mx.argmax(normed[0, 0] @ head_t).item())
            drafts.append(tok)
            feat = recur                               # self-feed the feature-space recurrent output

        # --- verify: one target forward over [cur, d1..dk] at offset q+1 ---
        vin = [cur] + drafts
        vlog, vcaps = model(mx.array(vin), caches=caches, offset=q + 1,
                            capture_layers=layers, absorbed=False, sparse=None)
        bpred = mx.argmax(vlog[0], axis=-1)        # [k+1]; bpred[i] = target token at position q+2+i
        mx.eval(bpred, [c.c_kv for c in caches], [c.k_pe for c in caches], [vcaps[L] for L in layers])
        bpred_l = [int(x) for x in bpred.tolist()]

        j = 0
        while j < k and drafts[j] == bpred_l[j]:   # accept while drafts match the target's greedy
            j += 1
        bonus = bpred_l[j]                          # target's correct token at the first mismatch
        out.extend(drafts[:j])
        out.append(bonus)
        accept_lens.append(j + 1)

        for c in caches:                            # keep [cur] + j accepted drafts; drop rejected
            c.truncate((q + 1) + (j + 1))
        feat3 = _feat3(vcaps, layers, j)            # real target hidden at the last accepted position
        q = q + 1 + j
        cur = bonus
        if eos_id is not None and (bonus == eos_id or eos_id in drafts[:j]):
            stop = True

    out = out[:max_new]
    return out, {
        "rounds": len(accept_lens),
        "tokens": len(out),
        "mean_accept": (sum(accept_lens) / len(accept_lens)) if accept_lens else 0.0,
        "max_accept": max(accept_lens) if accept_lens else 0,
        "k": k,
    }
