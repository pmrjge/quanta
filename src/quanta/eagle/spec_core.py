"""Model-agnostic EAGLE-3 spec-decode core: lossless verify / accept / rollback over any resident
model whose runtime exposes ``(ids, caches, offset, capture_layers) -> (logits, {layer: hidden})``.

Refactored out of :mod:`quanta.eagle.spec` (the original Kimi-flavored entry point) so the same
lossless machinery now serves every model class. The caller supplies four hooks — a forward
callable, an opaque cache object, a truncate callable that rolls the cache back losslessly to drop a
rejected draft, and (optionally) a cache-state evaluator for explicit memory control under a heavy
resident base. Output is **bit-identical to plain greedy decode** regardless of drafter quality (the
drafter only changes *speed*, never correctness — that is exactly the EAGLE guarantee).

Feature flow (matches training, :func:`quanta.eagle.train._ce_multistep`): each draft step feeds the
previous feature ``f`` (the real reduced target feature on step 0, the drafter's own
**feature-space recurrent** output ``recur`` thereafter) plus the *next* token's embedding; the
head-space output maps through the frozen head to the next draft token. After verify, the target's
captured hidden at the last accepted position becomes the next round's real feature. The cache is
rolled back via ``truncate_fn`` to drop rejected drafts so its state stays bit-exact.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx

from quanta.eagle.drafter import DraftCache, EagleDrafter

ForwardFn = Callable[..., tuple[mx.array, dict[int, mx.array]]]
TruncateFn = Callable[[Any, int], None]
CacheEvalFn = Callable[[Any], list[mx.array]]


def _feat3(caps: dict[int, mx.array], layers: tuple[int, ...], pos: int) -> mx.array:
    """Fused capture-layer hidden states at local position ``pos`` -> ``[1, 1, n_layers * H]``."""
    return mx.concatenate([caps[L][pos] for L in layers])[None, None]


def spec_generate(
    forward_fn: ForwardFn,
    caches: Any,
    truncate_fn: TruncateFn,
    drafter: EagleDrafter,
    embed: mx.array,
    head: mx.array,
    prompt_ids,
    *,
    max_new: int,
    k: int = 4,
    layers: tuple[int, ...],
    eos_id: int | None = None,
    cache_eval_fn: CacheEvalFn | None = None,
) -> tuple[list[int], dict]:
    """Lossless EAGLE-3 spec-decode, model-agnostic.

    Args:
        forward_fn: ``(ids, caches, offset, capture_layers) -> (logits[1,T,V], caps[{layer:hidden}])``.
            Runs one target forward — prefill for the prompt, then verify for ``[cur] + drafts``.
        caches: opaque cache state, passed through to ``forward_fn`` and ``truncate_fn``. The
            caller built it with the right shape for their model (Kimi: ``list[MLACache]``;
            MiniMax: ``MiniMaxCache``; future models: their per-model cache class).
        truncate_fn: ``(caches, length) -> None``. Rolls the cache back to exactly the state after
            consuming ``length`` tokens (drops a rejected draft). Lossless or it raises (rule 6).
        drafter: trained :class:`EagleDrafter` — ``n_feature_layers`` must equal ``len(layers)``.
        embed: frozen target embedding ``[V, H]`` (bf16).
        head: frozen target LM head ``[V, H]`` (bf16).
        prompt_ids: input token ids.
        max_new: cap on emitted tokens.
        k: draft length per round (best-case speedup is ``k+1`` tokens per target forward).
        layers: capture layer indices — must match what the drafter was trained against.
        eos_id: optional early-stop token (stops on the bonus or any accepted draft equal to it).
        cache_eval_fn: optional ``(caches) -> list[mx.array]`` for explicit ``mx.eval`` of the cache
            state between rounds. Kimi uses this for memory control under 398 GiB resident; MiniMax's
            runtime self-evals each token's KV growth so ``None`` is correct there.

    Returns:
        ``(tokens, stats)`` — ``stats['mean_accept']`` is the mean number of tokens emitted per target
        forward (1 = no speedup, ``k+1`` = perfect). Output is bit-identical to plain greedy decode.
    """
    prompt_ids = list(prompt_ids)
    head_t = head.T

    logits, caps = forward_fn(mx.array(prompt_ids), caches, 0, layers)
    cache_evals = cache_eval_fn(caches) if cache_eval_fn is not None else []
    mx.eval(logits, *cache_evals, [caps[L] for L in layers])

    q = len(prompt_ids) - 1                        # last position with a real target hidden
    feat3 = _feat3(caps, layers, -1)               # target feature at position q
    cur = int(mx.argmax(logits[0, -1]).item())     # verified token at position q+1
    out = [cur]
    accept_lens: list[int] = []
    stop = eos_id is not None and cur == eos_id

    while len(out) < max_new and not stop:
        # --- draft k tokens: step(f_{q+j}, e_{q+1+j}) -> recur_{q+1+j} -> token_{q+2+j};
        #     self-feed the feature-space recurrent output `recur` (matches training) ---
        dc = DraftCache()
        feat = drafter.reduce_target_features(feat3)   # real reduced target feature f_q
        drafts: list[int] = []
        tok = cur
        for j in range(k):
            recur, normed = drafter.step(feat, embed[tok][None, None], offset=q + 1 + j,
                                         cache=dc, mask=None)
            tok = int(mx.argmax(normed[0, 0] @ head_t).item())
            drafts.append(tok)
            feat = recur                               # self-feed the feature-space recurrent output

        # --- verify: one target forward over [cur, d1..dk] at offset q+1 ---
        vin = [cur] + drafts
        vlog, vcaps = forward_fn(mx.array(vin), caches, q + 1, layers)
        bpred = mx.argmax(vlog[0], axis=-1)            # [k+1]; bpred[i] = target token at q+2+i
        cache_evals = cache_eval_fn(caches) if cache_eval_fn is not None else []
        mx.eval(bpred, *cache_evals, [vcaps[L] for L in layers])
        bpred_l = [int(x) for x in bpred.tolist()]

        j = 0
        while j < k and drafts[j] == bpred_l[j]:   # accept while drafts match the target's greedy
            j += 1
        bonus = bpred_l[j]                          # target's correct token at the first mismatch
        out.extend(drafts[:j])
        out.append(bonus)
        accept_lens.append(j + 1)

        truncate_fn(caches, (q + 1) + (j + 1))     # keep [cur] + j accepted drafts; drop rejected
        feat3 = _feat3(vcaps, layers, j)           # real target hidden at the last accepted position
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
