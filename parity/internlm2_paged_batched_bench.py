"""InternLM2.5 batched (B>1) PAGED decode throughput bench — the #153 KV loop-kill on the real model.

HEAVY: loads the resident int8-g64 InternLM2.5-7B-Chat-1M bake (~9 GiB). RUN SOLO — no other model
resident (one-model-at-a-time; an over-subscribed load OOM-reboots the host). Standalone:

    uv run python -m parity.internlm2_paged_batched_bench           # B in {1,32,48}
    uv run python -m parity.internlm2_paged_batched_bench 32         # only B=32
    uv run python -m parity.internlm2_paged_batched_bench 1,16,32

(Drives raw token-id lists straight into the runtime — no tokenizer, no extra deps.)

Drives the production batched session (:class:`quanta.shim.omlx._InternLM2BatchedSession`, paged
dense-GQA decode) over ``B`` concurrent streams with DISTINCT prompts (each stream keeps its own full
attention KV — distinct block-0 tokens defeat the prefix-dedup so we measure a real B-stream KV load,
not a shared prefix). For each B it times TWO decode paths on the SAME resident weights, flipping ONLY
``InternLM2BatchedResidentModel._paged_kv_batched`` (the #153 flag) between them:

  * **loop**     — ``_paged_kv_batched=False``: the per-stream ``PagedKVCacheView.update()`` loop inside
    :func:`~quanta.modeling.batched_attention.batched_decode_attention_kv` (the proven pre-#153 path);
  * **loopkill** — ``_paged_kv_batched=True``: that B-stream loop replaced by ONE ``write_batched``
    scatter + ONE ``gather_batched`` over the shared :class:`~quanta.paged.PagedKVCacheManager`, then
    the same padded SDPA (:func:`~quanta.modeling.batched_attention.batched_decode_attention_padded`).

Both paths run the IDENTICAL batched projections + stacked FFN + SDPA forward — only the attention KV
update differs — so ``loopkill/loop`` aggregate tok/s isolates the #153 win. InternLM2.5 is the CLEAN
dense case: 32 GQA layers, no MoE, no SSM, no recurrent state — so EVERY layer's KV loop is trimmed
(vs Nemotron's 8 of 88 hybrid layers). The expectation is therefore a BIGGER #153 win than Nemotron's
(+15% @ B=32 / +18% @ B=48), and it should grow with B and with context length (more per-stream gather
work removed). If it is a win it graduates the default like Nemotron did (an InternLM2-scoped flag).

This is ALSO the real-model correctness gate for the QUANTIZED int8-g64 k/v ``write_batched`` /
``gather_batched`` round-trip at the true ``head_dim=128`` (the model-free §C gate in
``internlm2_batched_attention_test`` used bf16 KV to stay head_dim-/codec-agnostic). ``run()`` asserts
the stream-0 greedy token trace is IDENTICAL loop-vs-loopkill — B=1 bit-exact (no padding) and B>=2
greedy-exact (the padded-SDPA reorder is argmax-stable ULP noise, the [[feedback-batched-rope-bf16]]
equivalence class) — so a divergence here is a real loop-kill bug, not fp noise, and fails loud (rule 6).

Memory is read with MLX's own counters (``get_active_memory`` / ``get_peak_memory``); ``clear_cache`` +
``reset_peak_memory`` run between every path/B so a peak is that configuration's true transient.

Geometry: prompt = 256 tok/stream (distinct per stream), GEN = 64 decoded tok/stream (steady state;
WARMUP steps JIT-warm and are not timed), no EOS stop (every stream decodes exactly GEN).
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.internlm2.batched_runtime import InternLM2BatchedResidentModel
from quanta.shim.omlx import _InternLM2BatchedSession

ART = "/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64"
PROMPT_LEN = 256                  # distinct prompt tokens per stream (full own KV, no prefix dedup)
GEN = 64                          # decoded tokens per stream (timed)
WARMUP_STEPS = 4                  # JIT warm (not timed)
BLOCK = 32                        # paged block size (prod default, omlx _BaseBatchedSession)
BATCH_SIZES = (1, 32, 48)         # 1 = bit-exact anchor; 32 = prod operating point; 48 = trend


def _gib(nbytes: int) -> float:
    return nbytes / (1024 ** 3)


def _distinct_prompt(b: int, vocab: int, bos: int, n: int) -> list[int]:
    """A length-``n`` prompt whose FIRST block differs across streams (so the paged prefix-match never
    dedups two streams' KV — each stream gets its own full attention cache, the real B-stream load).
    A large per-stream stride spreads the token ids; clamp into ``[1, vocab)`` (the token VALUES only
    need to be valid + stream-distinct; decode tok/s is matmul-bandwidth dominated, ~context-independent,
    and the bench never stops on EOS so an EOS landing in a prompt is just another token)."""
    stride = (b + 1) * 1009
    return [bos] + [1 + ((stride + j * 7) % (vocab - 2)) for j in range(n - 1)]


def _max_blocks(max_batch: int) -> int:
    """Per-layer block-pool size (the manager allocs one pool of ``max_blocks`` PER attention layer):
    one stream needs ``ceil((PROMPT+GEN)/BLOCK)`` blocks; size for ``max_batch`` independent streams +
    slack for partial/boundary blocks. Tight (vs the 4096 default) keeps the lazy pools lean."""
    per_seq = -(-(PROMPT_LEN + GEN) // BLOCK)
    return per_seq * max_batch + 2 * max_batch


def _time_path(rt: InternLM2BatchedResidentModel, B: int, prompts: list[list[int]], *,
               paged_batched: bool, max_batch: int) -> dict:
    """Time GEN steady-state decode steps at batch B on one paged path (``paged_batched`` False=loop /
    True=loopkill). Builds a FRESH session (fresh PagedKVCacheManager → fresh KV pool, no shared state),
    admits B distinct prompts, warms WARMUP_STEPS, times GEN. Returns aggregate/per-stream tok/s, the
    stream-0 greedy token trace, and arena-path active/peak GiB."""
    rt._paged_kv_batched = bool(paged_batched)          # the ONLY thing that differs between paths
    mx.clear_cache()
    mx.reset_peak_memory()
    sess = _InternLM2BatchedSession(runtime=rt, capacity=max_batch, paged_kv=True, block_size=BLOCK,
                                    max_blocks=_max_blocks(max_batch), model_name="internlm2-bench",
                                    native_decode=False)   # dense GQA: no recurrent state, no form-2
    try:
        cur: dict[int, int] = {}
        for b in range(B):
            row = sess.admit(b, prompts[b])             # paged prefill (own full KV); returns [vocab]
            cur[b] = int(mx.argmax(row).item())
        slots = list(range(B))
        toks0: list[int] = []

        for _ in range(WARMUP_STEPS):                   # JIT warm (not timed)
            out = sess.step_batch({s: cur[s] for s in slots})
            cur = {s: int(mx.argmax(out[s]).item()) for s in slots}
            toks0.append(cur[0])

        t0 = time.perf_counter()
        for _ in range(GEN):
            out = sess.step_batch({s: cur[s] for s in slots})
            mx.eval(list(out.values()))
            cur = {s: int(mx.argmax(out[s]).item()) for s in slots}
            toks0.append(cur[0])
        dt = time.perf_counter() - t0

        return {"per_stream": GEN / dt, "aggregate": B * GEN / dt, "toks": toks0,
                "active_gib": _gib(mx.get_active_memory()), "peak_gib": _gib(mx.get_peak_memory())}
    finally:
        for s in range(B):
            sess.release(s)                             # free each seq's blocks before the next path/B


def run(batch_sizes: tuple[int, ...] = BATCH_SIZES) -> None:
    max_batch = max(batch_sizes)
    # Pin the resident weight set (InternLM2.5-7B int8-g64 ≈ 9 GiB + KV pools + transients).
    mx.set_wired_limit(int(40 * 1024 ** 3))
    rt = InternLM2BatchedResidentModel(ART, max_batch=max_batch)

    bos = rt.cfg.bos_token_id
    vocab = int(rt.cfg.vocab_size)
    n_attn = rt.num_layers

    print(f"\n=== InternLM2.5-7B-Chat-1M int8-g64 PAGED batched decode (prompt {PROMPT_LEN} tok, "
          f"{GEN} gen/stream, {n_attn} attn layers — ALL dense): loop (per-stream KV update) vs loopkill "
          f"(#153: ONE scatter + ONE gather) ===")
    print("aggregate tok/s (per-stream = aggregate / B). loopkill/loop = the #153 KV-loop-kill win. "
          "GiB = loopkill-path active/peak. tok = loop==loopkill greedy-exact (real-model correctness).")
    print(f"{'B':>4}  {'loop':>9}  {'loopkill':>9}  {'loopkill/loop':>13}  "
          f"{'loopkill GiB a/p':>17}  {'tok':>4}")
    all_tok_ok = True
    ratio32: float | None = None
    for B in batch_sizes:
        prompts = [_distinct_prompt(b, vocab, bos, PROMPT_LEN) for b in range(B)]
        lp = _time_path(rt, B, prompts, paged_batched=False, max_batch=max_batch)  # per-stream loop
        lk = _time_path(rt, B, prompts, paged_batched=True, max_batch=max_batch)   # #153 loop-kill
        ratio = lk["aggregate"] / lp["aggregate"] if lp["aggregate"] else float("nan")
        tok_ok = lp["toks"] == lk["toks"]
        all_tok_ok = all_tok_ok and tok_ok
        if B == 32:
            ratio32 = ratio
        print(f"{B:>4}  {lp['aggregate']:>9.1f}  {lk['aggregate']:>9.1f}  {ratio:>12.2f}x  "
              f"{lk['active_gib']:>7.1f}/{lk['peak_gib']:>7.1f}  {'ok' if tok_ok else 'DIFF':>4}")
        if not tok_ok:
            # First divergence vs the per-stream loop reference — surface it loudly (rule 6).
            ref, got = lp["toks"], lk["toks"]
            d = next((k for k in range(min(len(ref), len(got))) if ref[k] != got[k]), None)
            if d is not None:
                print(f"    [DIFF] loopkill diverges from loop at step {d}: "
                      f"loop={ref[d]} loopkill={got[d]}")

    # Honest verdict (rule 6): the loop-kill MUST be token-identical to the per-stream loop (B=1
    # bit-exact, B>=2 greedy-exact — same padded SDPA, only the KV store differs). A divergence is a
    # real #153 bug; fail loud. Throughput is reported, not asserted (hardware-variable).
    if not all_tok_ok:
        print("FAIL — the #153 loop-kill diverged from the per-stream paged loop (see [DIFF] above)")
        raise SystemExit(1)
    verdict = "PASS — loop == loopkill greedy-exact on the real model"
    if ratio32 is not None:
        verdict += (f"; B=32 loopkill/loop = {ratio32:.2f}x "
                    f"({'a win — graduate the default' if ratio32 >= 1.0 else 'NOT a win — keep default OFF'})")
    print(verdict)


if __name__ == "__main__":
    import sys

    bs = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else BATCH_SIZES
    run(bs)
