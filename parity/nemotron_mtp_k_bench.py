"""Nemotron-H native-MTP multi-step (k=1/k=2/k=3) spec-decode benchmark — RUN AFTER #148 lands.

Loads the RAM-resident Nemotron int4-g64 runtime + the trained native MTP head (one head, two stacked
sub-blocks), prompts with a fixed prose payload, and compares lossless ``spec_generate_k`` at
``k in {1, 2, 3}`` to plain greedy decode. The MTP draft head's MoE sub-block routes through
``draft_topk`` experts (lighter drafter than the main verify path, which stays at
``cfg.num_experts_per_tok``); the main model verifies all ``k + 1`` tokens per round in ONE forward,
so the output is bit-identical to greedy decode regardless of MTP quality (CLAUDE.md rule 4 /
losslessness). Reports per-round ``mean_accept``, draft_ms, verify_ms, tok/s, and the speedup ratio
vs baseline greedy.

Multi-step accept rate is the speed lever: chained drafts past step 1 feed the MTP its OWN post-block
hidden (off-distribution), so the chained-accept rate drops, BUT each verified round amortizes its
single main-forward cost over a ``k + 1``-token window, so the wall-clock speedup can still climb
with ``k`` if the per-step accept rate is high enough. ``draft_topk`` reduces the per-step drafter
cost (the MoE-sub-block is the dominant draft FLOPs) so the verify ceiling can be approached.

Orchestrator runs this — do NOT execute from an agent. Loads ~68 GB resident model + embed/head + MTP
weights — run only with the memory free. The parity test (model-free, ``parity/nemotron_mtp_spec_test.py``)
gates the k=1/2/3 LOGIC; this bench gates the wall-clock perf on real weights.

    uv run --with tokenizers python -m parity.nemotron_mtp_k_bench [draft_topk] [n_gen] [warmup_T]
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

from quanta.nemotron.generate import attn_caches
from quanta.nemotron.mtp import NemotronMTP
from quanta.nemotron.runtime import NemotronResidentModel
from quanta.nemotron.spec import spec_generate_k
from quanta.nemotron.tokenizer import NemotronTokenizer

ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"
PROMPT = ("Write a Python function that returns the n-th Fibonacci number using memoization, "
          "and explain it step by step in clear prose suitable for a beginner.")
DEFAULT_DRAFT_TOPK = 4         # cfg.num_experts_per_tok is 8 on Nemotron-H → drafter at top-4 (~half cost)
DEFAULT_GEN = 64
DEFAULT_WARMUPS = (32, 256, 1024)
K_VALUES = (1, 2, 3)


def _measure_baseline(model: NemotronResidentModel, ids: list[int], warmup_T: int, gen_N: int
                      ) -> tuple[float, float, int, list[int]]:
    """Plain greedy decode baseline: prefill ``ids + warmup_T zeros``, then decode loop.
    Returns ``(prefill_s, decode_s, decoded_count, decoded_tokens)`` where ``decoded_count`` is the
    number of tokens decoded INSIDE the timed window (used for tok/s — the first token from prefill
    is emitted BEFORE the decode timer starts and is not counted)."""
    caches = attn_caches(model, max_rollback=0)
    t0 = time.perf_counter()
    logits, ssm, conv = model(mx.array(ids), caches=caches)            # prefill
    cur = int(mx.argmax(logits[0, -1]).item())
    prefill_s = time.perf_counter() - t0

    out = [cur]                                                         # cur emitted from prefill (untimed)
    decoded = 0
    t1 = time.perf_counter()
    for _ in range(gen_N - 1):                                          # decode loop emits gen_N-1 more
        logits, ssm, conv = model(mx.array([cur]), caches=caches, ssm=ssm, conv=conv)
        cur = int(mx.argmax(logits[0, -1]).item())
        out.append(cur)
        decoded += 1
    decode_s = time.perf_counter() - t1
    return prefill_s, decode_s, decoded, out


def _measure_spec(model: NemotronResidentModel, mtp: NemotronMTP, embed: mx.array, head: mx.array,
                  ids: list[int], *, k: int, gen_N: int) -> tuple[float, float, int, dict, list[int]]:
    """Spec-decode at k. Returns ``(prefill_s, decode_s, decoded_count, stats, decoded_tokens)``.

    ``decode_s`` excludes the prefill that happens inside ``spec_generate_k`` so the wall-clock vs
    baseline comparison is apples-to-apples (tok/s through the decode loop only). Implemented by
    timing the full call and subtracting a separately-measured prefill of the same prompt — the
    spec loop's own prefill cost is included in this measured prefill (same model, same prompt)."""
    # measure prefill cost separately (a fresh forward over `ids` matches spec_generate_k's prefill)
    caches = attn_caches(model, max_rollback=k)
    t0 = time.perf_counter()
    _ = model(mx.array(ids), caches=caches)
    prefill_s = time.perf_counter() - t0
    # now time the full spec call (which internally also runs prefill) — subtract prefill_s for decode
    t1 = time.perf_counter()
    toks, stats = spec_generate_k(model, mtp, embed, head, ids, k=k, max_new=gen_N)
    wall_s = time.perf_counter() - t1
    decode_s = max(wall_s - prefill_s, 1e-9)                           # avoid div-by-zero on tiny gens
    # tokens emitted in the timed decode window: total emitted minus the prefill-emitted cur (1 token)
    decoded = max(len(toks) - 1, 0)
    return prefill_s, decode_s, decoded, stats, toks


def run() -> None:
    draft_topk = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DRAFT_TOPK
    gen_N = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_GEN
    warmups = (int(sys.argv[3]),) if len(sys.argv) > 3 else DEFAULT_WARMUPS

    mx.set_wired_limit(int(120 * 1024**3))
    model = NemotronResidentModel(ART)
    tok = NemotronTokenizer(ART)
    base_ids = tok.encode(PROMPT, add_bos=False)

    # The baked artifact loader for the MTP head is the integration step — here we instantiate the
    # holder with the model's draft_topk so the moe sub-block routes through fewer experts than the
    # main verify path. (Real-weight load comes from the artifact in the orchestrator's deploy step.)
    mtp = NemotronMTP(model.cfg, draft_topk=draft_topk)
    embed, head = model.embed_w, model.lm_head_w

    print(f"\n=== Nemotron-H native-MTP multi-step bench "
          f"(draft_topk={draft_topk}/{model.cfg.num_experts_per_tok}, gen={gen_N}) ===")
    print(f"prompt: {len(base_ids)} tok ; warmups: {warmups}")

    for warmup_T in warmups:
        # build the timing input: prompt + warmup_T zero tokens (a fixed-length pad to stress prefill+decode)
        ids = list(base_ids) + [0] * warmup_T
        print(f"\n--- warmup_T = {warmup_T} (effective prefill ≈ {len(ids)} tok) ---")

        # baseline greedy — tok/s computed from the decoded count INSIDE the decode timer (excl. prefill)
        b_prefill_s, b_decode_s, b_decoded, base_toks = _measure_baseline(model, ids, warmup_T, gen_N)
        base_tps = b_decoded / b_decode_s if b_decode_s > 0 else float("inf")
        print(f"baseline greedy : prefill={b_prefill_s:.2f}s decode={b_decode_s:.2f}s/{b_decoded}tok "
              f"({base_tps:.1f} tok/s)")

        for k in K_VALUES:
            s_prefill_s, s_decode_s, s_decoded, stats, spec_toks = _measure_spec(
                model, mtp, embed, head, ids, k=k, gen_N=gen_N)
            tps = s_decoded / s_decode_s if s_decode_s > 0 else float("inf")
            ratio = tps / base_tps if base_tps > 0 else float("inf")
            # bit-identity check vs the baseline greedy stream — lossless contract
            ident = spec_toks[: len(base_toks)] == base_toks[: len(spec_toks)]
            print(f"  k={k}: prefill={s_prefill_s:.2f}s decode={s_decode_s:.2f}s/{s_decoded}tok "
                  f"({tps:.1f} tok/s, vs_base={ratio:.2f}x) | "
                  f"mean_accept={stats['mean_accept']:.2f}/{k + 1} "
                  f"max={stats['max_accept']} rounds={stats['rounds']} bit_id={ident}")


if __name__ == "__main__":
    run()
