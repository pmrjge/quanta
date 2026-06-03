"""InternLM2.5-7B EAGLE-3 spec-decode real-model bench (M3): accept-rate + decode tok/s vs greedy.

M0–M2 proved the substrate (capture + lossless truncate), the adapter (spec==greedy, model-free), and
the drafter-training pipeline (capture→shards→train→reload, model-free). M3 is the FIRST real-weight
exercise: load the resident int8-g64 7B bake + the TRAINED drafter sidecar and measure, across a sweep
of draft length ``k``, the EAGLE operating point — mean accepted tokens per target forward and the
wall-clock decode tok/s vs plain greedy — while pinning the EAGLE guarantee on real weights: spec output
is **bit-identical** to greedy regardless of ``k`` or drafter quality (the drafter only changes *speed*).

PREREQ — produce the trained drafter sidecar FIRST (solo GPU, one model resident at a time):

    uv run --with sentencepiece python -m parity.eagle_capture_internlm2   # -> features_int8g64/ shards
    uv run python -m parity.eagle_train_internlm2                          # -> drafter_int8g64.safetensors

Then this bench (one model only — int8-g64 ≈ 8.3 GB on disk, ~6.2 GB resident + embed/head + drafter —
safe to run solo on the M3 Ultra):

    uv run python -m parity.internlm2_eagle_spec_bench [drafter.safetensors] [n_gen]

Decode-only timing mirrors :mod:`parity.nemotron_mtp_k_bench`: prefill is measured separately and
subtracted from the full ``spec_generate`` wall time, so the greedy-vs-spec tok/s comparison covers the
decode loop only (apples-to-apples — the prefill is identical work in both paths). The lossless contract
is asserted at the end across every ``k`` (CLAUDE.md rule 4 / rule 6).
"""

from __future__ import annotations

import os
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from quanta.eagle.capture import load_features
from quanta.eagle.drafter import EagleDrafter
from quanta.eagle.train import load_drafter
from quanta.internlm2.eagle import (
    DEFAULT_CAPTURE_LAYERS,
    INTERNLM2_DRAFTER_CFG,
)
from quanta.internlm2.eagle import spec_generate as il2_spec_generate
from quanta.internlm2.runtime import InternLM2ResidentModel

ART = "/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64"
DRAFTER = "/Users/pmrj/models/internlm2_eagle/drafter_int8g64.safetensors"
FEATURES = "/Users/pmrj/models/internlm2_eagle/features_int8g64"
PROMPT_LEN = 64                       # real corpus window (sourced from the captured feature shard)
DEFAULT_GEN = 128                     # timed decode tokens (long enough for a stable accept estimate)
K_VALUES = (2, 3, 4, 5, 6, 8)         # EAGLE draft length per round; best-case speedup ceiling is k+1


def _peak_rss_gib() -> float:
    """Peak resident set in GiB (macOS ``ru_maxrss`` is bytes; Linux KiB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2)


def _source_prompt(n: int) -> list[int]:
    """First ``n`` real corpus tokens (InternLM2 vocab) from the first captured feature shard — the same
    token stream the drafter trained on, so the measured accept-rate reflects in-distribution decode."""
    shards = sorted(Path(FEATURES).glob("feat_*.safetensors"))
    if not shards:
        raise FileNotFoundError(
            f"no feature shards under {FEATURES} — run `parity.eagle_capture_internlm2` first (the M3 "
            "PREREQ; it also feeds `parity.eagle_train_internlm2`).")
    return [int(x) for x in load_features(shards[0])["in_tokens"][:n].tolist()]


def _measure_baseline(model: InternLM2ResidentModel, ids: list[int], gen_N: int
                      ) -> tuple[float, float, int, list[int]]:
    """Plain greedy decode (the lossless reference): prefill timed separately, then a decode loop
    emitting ``gen_N-1`` more tokens inside the decode timer. Returns ``(prefill_s, decode_s, decoded,
    tokens)`` — the same per-token forwards as the M1 gate's ``_greedy``, so the emitted stream is the
    reference the spec output must match token-for-token."""
    cache = model.new_cache()
    t0 = time.perf_counter()
    logits = model(mx.array([list(ids)]), caches=cache)         # prefill [1, T, V]
    cur = int(mx.argmax(logits[0, -1]).item())                  # .item() forces the prefill eval
    prefill_s = time.perf_counter() - t0

    out = [cur]                                                 # emitted from prefill (untimed)
    decoded = 0
    t1 = time.perf_counter()
    for _ in range(gen_N - 1):
        logits = model(mx.array([[cur]]), caches=cache)         # decode at cache.offset
        cur = int(mx.argmax(logits[0, -1]).item())
        out.append(cur)
        decoded += 1
    decode_s = time.perf_counter() - t1
    return prefill_s, decode_s, decoded, out


def _measure_spec(model: InternLM2ResidentModel, drafter: EagleDrafter, embed: mx.array,
                  head: mx.array, ids: list[int], *, k: int, gen_N: int, head_bits: int | None = None
                  ) -> tuple[float, float, int, dict, list[int]]:
    """Spec-decode at ``k``. Returns ``(prefill_s, decode_s, decoded, stats, tokens)``. ``decode_s``
    subtracts a separately-measured prefill of the same prompt from the full ``spec_generate`` wall
    time, so the decode-loop tok/s is apples-to-apples with the greedy baseline (mirrors
    :func:`parity.nemotron_mtp_k_bench._measure_spec`). ``head_bits`` (opt-in) quantizes the
    per-draft-step vocab projection."""
    cache = model.new_cache()
    t0 = time.perf_counter()
    logits = model(mx.array([list(ids)]), caches=cache)         # prefill cost (throwaway cache)
    mx.eval(logits)
    prefill_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    toks, stats = il2_spec_generate(model, drafter, embed, head, ids, max_new=gen_N, k=k,
                                    layers=DEFAULT_CAPTURE_LAYERS, eos_id=None, head_bits=head_bits)
    wall_s = time.perf_counter() - t1
    decode_s = max(wall_s - prefill_s, 1e-9)                    # spec_generate prefills internally too
    decoded = max(len(toks) - 1, 0)                            # exclude the prefill-emitted first token
    return prefill_s, decode_s, decoded, stats, toks


def run() -> None:
    drafter_path = sys.argv[1] if len(sys.argv) > 1 else DRAFTER
    gen_N = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_GEN

    if not Path(drafter_path).exists():
        raise FileNotFoundError(
            f"no trained drafter at {drafter_path} — run the M3 PREREQ first (solo GPU, one model at a "
            "time):\n  uv run --with sentencepiece python -m parity.eagle_capture_internlm2\n"
            "  uv run python -m parity.eagle_train_internlm2")

    mx.set_wired_limit(int(32 * 1024 ** 3))                    # ~6.2 GB model + embed/head + drafter
    model = InternLM2ResidentModel(ART, packed=True)
    embed, head = model.embed_head()
    embed, head = embed.astype(mx.bfloat16), head.astype(mx.bfloat16)
    drafter = load_drafter(drafter_path, EagleDrafter(**asdict(INTERNLM2_DRAFTER_CFG)))
    qbits = int(os.environ.get("IL2_QUANT_BITS", "0"))   # 0 = bf16 drafter (default); 8 / 4 = PTQ int8 / int4
    if qbits:
        nn.quantize(drafter, group_size=64, bits=qbits)  # PTQ the drafter body: nn.Linear -> QuantizedLinear
    mx.eval(drafter.parameters(), embed, head)
    head_bits = qbits or None                            # also quantize the k×/round draft-step head proj
    ids = _source_prompt(PROMPT_LEN)

    # warm the JIT once (untimed) — the first spec/greedy call pays kernel-compile + KV-warm
    il2_spec_generate(model, drafter, embed, head, ids, max_new=8, k=K_VALUES[0],
                      layers=DEFAULT_CAPTURE_LAYERS, head_bits=head_bits)
    _measure_baseline(model, ids, 8)

    qmode = f"int{qbits}g64 body+head" if qbits else "bf16"
    print(f"\n=== InternLM2.5-7B EAGLE-3 spec-decode bench (drafter={Path(drafter_path).name}, "
          f"quant={qmode}, prompt {len(ids)} tok, gen {gen_N}, capture {DEFAULT_CAPTURE_LAYERS}) ===",
          flush=True)

    b_prefill_s, b_decode_s, b_decoded, base_toks = _measure_baseline(model, ids, gen_N)
    base_tps = b_decoded / b_decode_s if b_decode_s > 0 else float("inf")
    print(f"baseline greedy : prefill={b_prefill_s:.2f}s decode={b_decode_s:.2f}s/{b_decoded}tok "
          f"({base_tps:.1f} tok/s)", flush=True)
    print(f"{'k':>3}  {'mean_accept':>13}  {'max':>3}  {'rounds':>6}  {'tok/s':>7}  "
          f"{'vs_base':>7}  {'bit_id':>6}", flush=True)

    idents: list[bool] = []
    for k in K_VALUES:
        _, s_decode_s, s_decoded, stats, spec_toks = _measure_spec(
            model, drafter, embed, head, ids, k=k, gen_N=gen_N, head_bits=head_bits)
        tps = s_decoded / s_decode_s if s_decode_s > 0 else float("inf")
        ratio = tps / base_tps if base_tps > 0 else float("inf")
        ident = spec_toks[:len(base_toks)] == base_toks[:len(spec_toks)]
        idents.append(ident)
        ma = f"{stats['mean_accept']:.2f}/{k + 1}"
        print(f"{k:>3}  {ma:>13}  {stats['max_accept']:>3}  {stats['rounds']:>6}  {tps:>7.1f}  "
              f"{ratio:>6.2f}x  {str(ident):>6}", flush=True)

    print(f"\npeak RSS {_peak_rss_gib():.2f} GiB", flush=True)
    bad = [k for k, ok in zip(K_VALUES, idents) if not ok]
    assert not bad, f"EAGLE lossless VIOLATED on real weights — spec != greedy for k={bad} (a real bug)"
    print("PASS — EAGLE spec-decode lossless on real int8-g64 weights across all k (spec == greedy)",
          flush=True)


if __name__ == "__main__":
    run()
