"""DSV4-Flash batched (B>1) PAGED decode throughput bench — the #153 latent KV loop-kill on the real model.

HEAVY: loads the resident int4-g64 DeepSeek-V4-Flash bake (~180 GiB, packed_experts via gather_qmm).
RUN SOLO — no other model resident (one-model-at-a-time; an over-subscribed load OOM-reboots the host,
[[feedback-memory-safety]]). Orchestrator/standalone only:

    uv run python -m parity.dsv4_paged_batched_bench            # B in {1,32,48}
    uv run python -m parity.dsv4_paged_batched_bench 32          # only B=32 (the prod operating point)
    uv run python -m parity.dsv4_paged_batched_bench 1,32,48,64

(Drives raw token-id lists straight into the runtime — no tokenizer, no extra deps.)

This is the deferred solo-GPU **M4** of #153 (the paged sibling of #18 M5). Production serves DSV4 through
the PAGED path (``PAGED_KV_DEFAULT=True``), so ``admit``/``step_batch`` dispatch to ``_admit_paged`` /
``_step_paged`` and the #18 unpaged arena is never reached. Before #153 the paged decode still paid the
exact per-stream KV-update Python loop #18 killed: for each of B streams a per-stream
``PagedLatentCacheView.append`` (latent block write) + ``gather_one`` (latent block gather) + a
``_pad_stack`` readback, EVERY step, EVERY attention layer. #153 replaces that B-stream loop with ONE
``write_one_batched`` block-table scatter + ONE ``gather_one_batched`` block-table gather over the shared
:class:`~quanta.paged.PagedKVCacheManager` (the latent store only — the derived compressed-KV /
lightning-indexer / raw-hidden ring stay per-stream, the M2 design, so the boundary-snapshot lifecycle is
unchanged). M0–M3 proved that exchange bit-exact MODEL-FREE; this measures the throughput win on the real
43-layer forward (3 attention regimes + real fp4 experts + compressor + indexer).

Drives the production batched session (:class:`quanta.shim.omlx._DSV4BatchedSession`, paged latent decode)
over ``B`` concurrent streams with DISTINCT prompts (each stream's FIRST block differs so the paged
prefix-match never dedups two streams' latent KV — each gets its own full attention cache, the real
B-stream load, not a shared prefix). For each B it times TWO decode paths on the SAME resident weights,
flipping ONLY ``DSV4BatchedResidentModel._paged_kv_batched`` (the #153 flag) between them:

  * **loop**     — ``_paged_kv_batched=False``: the per-stream ``lcs`` latent loop inside
    ``_decode_batched_single`` (per-stream ``append_kv`` + ``gather_one`` + ``_pad_stack``; the proven
    pre-#153 paged path);
  * **loopkill** — ``_paged_kv_batched=True``: that B-stream latent loop replaced by ONE
    ``write_one_batched`` scatter + ONE ``gather_one_batched`` gather (via ``_PagedKVArena``), then the
    SAME batched windowed-sink SDPA.

Both paths run the IDENTICAL batched projections + fused SDPA + sparse MoE forward — only the latent KV
update differs — so ``loopkill/loop`` aggregate tok/s isolates the #153 win. Because the latent values
fed to SDPA are bit-identical between the two paths (M0: batched scatter == per-stream write, batched
gather == per-stream gather, both ``|Δ|=0``), the stream-0 greedy token trace is expected
**BIT-exact** loop-vs-loopkill at every B (NOT merely greedy-exact — unlike a batched-vs-single-stream
comparison, both paths here share the one batched SDPA; only the latent store materialization differs,
and that is bit-exact). ``run()`` asserts the trace is identical loop-vs-loopkill — a divergence is a real
#153 bug, not fp noise, and fails loud (rule 6). The throughput win is reported, not asserted
(hardware-variable). Expect roughly the #18 M5 magnitude (unpaged arena/bat +37% @ B=32), grown with B as
the per-stream loop stops scaling while the loop-kill holds.

Memory is read with MLX's own counters (``get_active_memory`` / ``get_peak_memory``), NOT ``ru_maxrss``;
``clear_cache`` + ``reset_peak_memory`` run between every path/B so a peak is that configuration's true
transient.

Geometry: prompt = 256 tok/stream (distinct per stream; decode tok/s is MoE-dominated, ~context-
independent, so a short seed barely moves throughput but keeps O(B) seeding cheap), GEN = 64 decoded
tok/stream (steady state; WARMUP_STEPS JIT-warm and are not timed), no EOS stop (every stream decodes
exactly GEN).
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.dsv4.batched_runtime import DSV4BatchedResidentModel
from quanta.shim.omlx import _DSV4BatchedSession

ART = "/Users/pmrj/models/DeepSeek-V4-Flash-quanta_int4g64"
PROMPT_LEN = 256                  # distinct prompt tokens per stream (full own latent KV, no prefix dedup)
GEN = 64                          # decoded tokens per stream (timed)
WARMUP_STEPS = 4                  # JIT warm (not timed)
BLOCK = 32                        # paged block size (prod default, omlx _BaseBatchedSession)
BATCH_SIZES = (1, 32, 48)         # 1 = bit-exact anchor; 32 = prod operating point; 48 = trend


def _gib(nbytes: int) -> float:
    return nbytes / (1024 ** 3)


def _distinct_prompt(b: int, vocab: int, bos: int, n: int) -> list[int]:
    """A length-``n`` prompt whose FIRST block differs across streams (so the paged prefix-match never
    dedups two streams' latent KV — each stream gets its own full attention cache, the real B-stream
    load). A large per-stream stride spreads the token ids; clamp into ``[1, vocab)`` (the token VALUES
    only need to be valid + stream-distinct; decode tok/s is matmul-bandwidth dominated, ~context-
    independent, and the bench never stops on EOS so an EOS landing in a prompt is just another token)."""
    stride = (b + 1) * 1009
    return [bos] + [1 + ((stride + j * 7) % (vocab - 2)) for j in range(n - 1)]


def _max_blocks(max_batch: int) -> int:
    """Per-layer latent block-pool size (the manager allocs one pool of ``max_blocks`` PER attention
    layer): one stream needs ``ceil((PROMPT+GEN)/BLOCK)`` blocks; size for ``max_batch`` independent
    streams + slack for partial/boundary blocks. The latent pool is tiny (single-stream codec), so this
    is comfortably within budget even at B=48."""
    per_seq = -(-(PROMPT_LEN + GEN) // BLOCK)
    return per_seq * max_batch + 2 * max_batch


def _time_path(rt: DSV4BatchedResidentModel, B: int, prompts: list[list[int]], *,
               paged_batched: bool, max_batch: int) -> dict:
    """Time GEN steady-state decode steps at batch B on one paged path (``paged_batched`` False=loop /
    True=loopkill). Builds a FRESH session (fresh PagedKVCacheManager + RecurrentPrefixCache → fresh
    latent pool, no shared state), admits B distinct prompts, warms WARMUP_STEPS, times GEN. Returns
    aggregate/per-stream tok/s, the stream-0 greedy token trace, and active/peak GiB."""
    rt._paged_kv_batched = bool(paged_batched)          # the ONLY thing that differs between paths
    mx.clear_cache()
    mx.reset_peak_memory()
    sess = _DSV4BatchedSession(runtime=rt, capacity=max_batch, paged_kv=True, block_size=BLOCK,
                               max_blocks=_max_blocks(max_batch), model_name="dsv4-paged-bench")
    try:
        cur: dict[int, int] = {}
        for b in range(B):
            row = sess.admit(b, prompts[b])             # paged prefill (own full latent KV); returns [vocab]
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
    # Pin the resident weight set (DSV4-Flash int4-g64 ≈ 180 GiB — keep MLX from paging it).
    mx.set_wired_limit(int(220 * 1024 ** 3))
    rt = DSV4BatchedResidentModel(ART, max_batch=max_batch, packed_experts=True)

    bos = rt.cfg.bos_token_id
    vocab = int(rt.cfg.vocab_size)
    n_attn = rt.num_layers

    print(f"\n=== DeepSeek-V4-Flash int4-g64 PAGED batched decode (prompt {PROMPT_LEN} tok, "
          f"{GEN} gen/stream, {n_attn} layers): loop (per-stream latent KV update) vs loopkill "
          f"(#153: ONE scatter + ONE gather) ===")
    print("aggregate tok/s (per-stream = aggregate / B). loopkill/loop = the #153 latent-KV-loop-kill "
          "win. GiB = loopkill-path active/peak. tok = loop==loopkill bit-exact (real-model correctness).")
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

    # Honest verdict (rule 6): the loop-kill MUST be token-identical to the per-stream paged loop —
    # bit-exact at every B (both paths run the one batched SDPA; only the latent store materialization
    # differs, and that is bit-exact per M0/M1). A divergence is a real #153 bug; fail loud. Throughput
    # is reported, not asserted (hardware-variable).
    if not all_tok_ok:
        print("FAIL — the #153 loop-kill diverged from the per-stream paged loop (see [DIFF] above)")
        raise SystemExit(1)
    verdict = "PASS — loop == loopkill bit-exact on the real model"
    if ratio32 is not None:
        verdict += (f"; B=32 loopkill/loop = {ratio32:.2f}x "
                    f"({'a win — #153 M4 confirmed' if ratio32 >= 1.0 else 'NOT a win'})")
    print(verdict)


if __name__ == "__main__":
    import sys

    bs = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else BATCH_SIZES
    run(bs)
