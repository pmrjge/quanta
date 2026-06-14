"""MiniMax-M3-VL — teacher-forced ppl + top-1: the bf16-vs-int4 arbiter (+ the (1+w)-fold check).

Two **sequential** streamed forwards (rule 8: one decoder layer resident at a time; the MoE
128-expert bf16 stack ~14.5 GiB is the peak, NEVER the 809.5 GiB / 233.4 GiB whole model) over the
*same* held-out prose, the first freed before the second so only one model is ever resident:

  1. **bf16 M3 source**    (:class:`~quanta.minimax.loader_m3.MiniMaxM3SourceCheckpoint`)
     -> reference ppl + reference per-position argmax
  2. **int4-g64 artifact** (:class:`~quanta.minimax.artifact_m3.MiniMaxM3Artifact`, dequant-on-read)
     -> the served arm (the only shipped width; ``--artifact`` measures any other)

Both share :func:`streamed_logits` verbatim — the SAME forward (the proven
:class:`quanta.minimax.model_m3.MiniMaxM3Block` prefill path, built dense-bf16 and streamed one block
at a time; M1a/M1b already proved this block matches a numpy-fp64 reference to ~8e-7) — so the ppl Δ
isolates pure expert-coding (int4 RTN) error. Because the readers duck-type the same surface, this one
forward serves both arms. The quant mix measured here is the **SERVED mix**: int4 routed experts +
int8 dense/shared + bf16 control (so the Δ is the real deployment loss, not an expert-only proxy).

This is ALSO the **decisive end-to-end check for the pinned Gemma ``(1+w)`` fold** (CLAUDE.md: there is
NO HF/sglang M3 forward to diff against, so the M1a pin — one ``use_gemma_norm`` flag ⇒ uniform
``(1+w)`` on every non-gated RMSNorm — was validated only against transformers siblings + a numpy
oracle; a wrong fold degrades teacher-forced ppl UNIFORMLY). A *sane* bf16 ppl on coherent English
prose is the e2e confirmation the fold is right.

The corpus is original expository prose, **held out** from any calibration set (RTN is data-free, so
there is none regardless). M3's tokenizer ships ``add_bos_token`` absent (→ default ``False``), so the
prose is encoded raw (the Nex precedent); M3 is **natively 1M** (no YaRN) so both arms run the
identical RoPE — the only difference is the dequantized weights.

    uv run python -m parity.minimax_m3_ppl              # the real 2-arm arbiter (SOLO)
    uv run python -m parity.minimax_m3_ppl 8 160        # n_layers, n_tok (bounded code-path smoke)
    uv run python -m parity.minimax_m3_ppl --artifact /path/to/MiniMax-M3-quanta_int6g64  # any width

# parity-gate: real-weight
"""

from __future__ import annotations

import math
import os
import sys
import time

import mlx.core as mx

from quanta.minimax import model_m3 as M
from quanta.minimax.artifact_m3 import MiniMaxM3Artifact
from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.loader_m3 import MiniMaxM3SourceCheckpoint
from quanta.minimax.tokenizer import MiniMaxTokenizer

SRC = "/Users/pmrj/models/MiniMax-M3"
ART = "/Users/pmrj/models/MiniMax-M3-quanta_int4g64"   # int4-g64 — the shipped width (override --artifact)
N_TOK = 1024  # ~10x a 100-tok pilot → a de-noised, trustworthy ppl verdict

# the (1+w)-fold / forward e2e sanity ceiling: a healthy 397B model on clean prose lands well under
# this (~single digits); a broken (1+w) fold or forward bug degrades ppl to the hundreds/thousands
PPL_CEILING = 30.0
DPPL_CEILING = 5.0   # int4 RTN on a bf16 source is ~lossless; >5% would mean a real coding regression
AGREE_FLOOR = 0.90   # top-1 agreement floor (loose — bf16-ULP near-ties flip; the Nemotron-Ultra rule)


def _expert_bits(art) -> int:
    """The routed-expert width baked into ``art`` (read from the manifest, rule 6 — never assumed)."""
    for k, m in art.manifest.items():
        if k.endswith("experts.gate_up_proj") and m.get("format") == "affine_packed":
            return int(m["bits"])
    raise ValueError("artifact has no routed-expert stack (cannot determine the served width)")

# Original expository prose, held out from any calibration corpus (RTN is data-free regardless).
PROSE = """The history of mapmaking is in large part the history of how people have tried to compress \
the round, irregular, and only partly known surface of the earth onto a flat sheet that can be \
carried, copied, and consulted. The earliest maps were not drawn to scale and made no pretense of \
covering the whole world; they recorded the things that mattered to the people who made them, such as \
the course of a river, the location of a well, or the boundary between one field and the next. A clay \
tablet from Mesopotamia, a painted screen from central Mexico, and a stick chart from the islands of \
the Pacific each solve the same problem in a different visual language, and each assumes a reader who \
already knows a great deal about the territory and needs only to be reminded of its arrangement.

As trade and travel reached farther, the demands on a map grew. A merchant who would never see the \
far end of a route still had to plan for it, and a captain steering out of sight of land needed a way \
to reason about a place before arriving there. Greek geographers responded by treating the earth as a \
sphere and asking how its surface could be projected onto a plane, a question that has no perfect \
answer. Every projection sacrifices something: one preserves the shapes of small regions but stretches \
the poles into impossible widths, another keeps areas honest at the cost of bending every straight \
line, and a third keeps compass bearings true so a navigator can hold a single heading, even though \
the distances it implies are wrong. The mapmaker therefore chooses not the correct projection but the \
one whose particular distortions matter least for the task at hand.

The invention of printing changed maps as profoundly as it changed books. A hand-copied chart was \
unique, expensive, and slowly corrupted as each copyist introduced new errors; an engraved plate could \
print thousands of identical impressions and could be corrected in one place to improve them all. Maps \
became cheaper, more uniform, and more widely shared, and with that sharing came a new kind of \
authority, since a printed map looked settled and official even when its interior was filled with \
guesswork. Cartographers were rarely shy about the gaps. They drew elaborate sea monsters in the empty \
oceans and confident coastlines around lands no European had visited, and for a long time it was \
genuinely difficult to tell the carefully surveyed parts of a map from the invented ones.

What finally disciplined the map was measurement. The development of reliable instruments for fixing \
latitude, and much later longitude, meant that a position could be established by observation rather \
than by dead reckoning and hope. National surveys laid networks of triangles across whole countries, \
anchoring every village and hilltop to a common framework, and the slow, patient work of these surveys \
turned the map from a picture into an instrument. By the time aerial photography and then satellites \
arrived, the basic ambition was already fixed: to record where things actually are, precisely enough \
that a stranger can trust the record. The modern map, continuously updated from orbit and consulted on \
a screen in the palm of a hand, is the direct descendant of that ambition, and it carries forward the \
same old compromise, because the earth is still round and the screen is still flat."""

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _bf16(a: mx.array) -> mx.array:
    return a.astype(mx.bfloat16)


def _load_block(weights, cfg: MiniMaxM3Config, i: int) -> M.MiniMaxM3Block:
    """Build a :class:`MiniMaxM3Block` and fill it (bf16 weights, F32 router) from the duck-typed
    ``weights`` (source OR artifact), folding Gemma ``(1+w)`` on every RMSNorm. This is the bf16
    deployment precision (the served path): the block runs bf16 GEMMs with fp32 norms/activation/router
    internally. Load-time wiring — the coarse per-layer loop is rule-3-allowed; nothing on the hot
    path."""
    blk = M.MiniMaxM3Block(cfg, i)
    nm = weights.block_norms(i)
    blk.input_layernorm.weight = M.one_plus(_bf16(nm["input_layernorm"]))
    blk.post_attention_layernorm.weight = M.one_plus(_bf16(nm["post_attention_layernorm"]))

    at = weights.attention(i)
    blk.self_attn.q_proj.weight = _bf16(at["q_proj.weight"])
    blk.self_attn.k_proj.weight = _bf16(at["k_proj.weight"])
    blk.self_attn.v_proj.weight = _bf16(at["v_proj.weight"])
    blk.self_attn.o_proj.weight = _bf16(at["o_proj.weight"])
    blk.self_attn.q_norm = M.one_plus(_bf16(at["q_norm.weight"]))           # per-head q/k norm (1+w)
    blk.self_attn.k_norm = M.one_plus(_bf16(at["k_norm.weight"]))

    if cfg.is_moe_layer(i):
        mo = weights.moe(i)
        # router gate + bias stay F32 (routing precision — both readers return them at native F32)
        blk.mlp.gate = mo["gate"].astype(mx.float32)
        blk.mlp.e_score_correction_bias = mo["e_score_correction_bias"].astype(mx.float32)
        blk.mlp.set_experts(_bf16(mo["experts_gate_up"]), _bf16(mo["experts_down"]))
        blk.mlp.shared_gate_proj = _bf16(mo["shared_gate_proj"])
        blk.mlp.shared_up_proj = _bf16(mo["shared_up_proj"])
        blk.mlp.shared_down_proj = _bf16(mo["shared_down_proj"])
    else:
        dm = weights.dense_mlp(i)
        blk.mlp.gate_proj.weight = _bf16(dm["gate_proj"])
        blk.mlp.up_proj.weight = _bf16(dm["up_proj"])
        blk.mlp.down_proj.weight = _bf16(dm["down_proj"])
    return blk


def streamed_logits(weights, cfg: MiniMaxM3Config, ids: mx.array, *, n_layers: int | None = None
                    ) -> mx.array:
    """Teacher-forcing logits ``[1,T,vocab]`` via the streamed reference forward (rule 8).

    ``weights`` is a :class:`MiniMaxM3SourceCheckpoint` (bf16) OR a :class:`MiniMaxM3Artifact`
    (dequant-on-read) — they duck-type the same ``embed`` / ``block_norms`` / ``attention`` /
    ``dense_mlp`` / ``moe`` / ``final_norm`` / ``lm_head`` / ``release`` surface, so this one forward
    serves both arms. Each block is built dense-bf16 by :func:`_load_block`, its weights materialized
    off the shard (``mx.eval``) so the layer can be freed, run over the whole window with NO cache
    (prefill == ``MiniMaxM3Block.__call__(cache=None)``), then released before the next (one layer
    resident). At this length the trained block-sparse indexer is inert (top-16 blocks == all blocks)
    so dense full attention == the served forward."""
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    ids = ids.reshape(-1)
    h = weights.embed()[ids][None].astype(mx.bfloat16)          # [1,T,hidden]
    mx.eval(h)
    weights.release()
    for i in range(n):
        blk = _load_block(weights, cfg, i)
        mx.eval(blk.parameters())                              # materialize weights off the shard
        h = blk(h, cache=None, use_fast=True, sparse=True)
        mx.eval(h)
        del blk
        weights.release()
        mx.clear_cache()
    norm_w = M.one_plus(weights.final_norm())                  # Gemma (1+w) final RMSNorm
    hh = mx.fast.rms_norm(h, norm_w.astype(h.dtype), cfg.norm_eps)
    head = weights.lm_head()                                   # untied (tie_word_embeddings=False)
    logits = hh @ head.T.astype(hh.dtype)
    mx.eval(logits)
    return logits


def teacher_forced(logits: mx.array, ids: mx.array) -> tuple[float, float, mx.array]:
    """Teacher-forced perplexity + top-1 next-token accuracy + the per-position argmax.

    ``logits`` ``[1,T,vocab]`` predict positions ``1..T-1`` from ``0..T-2`` (causal teacher forcing)."""
    lp = logits[0, :-1].astype(mx.float32)                      # [T-1, vocab]
    tgt = ids.reshape(-1)[1:]                                   # [T-1]
    logp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(logp, tgt[:, None], axis=-1)[:, 0]
    ppl = float(mx.exp(mx.mean(nll)).item())
    argmax = mx.argmax(lp, axis=-1)
    acc = float(mx.mean((argmax == tgt).astype(mx.float32)).item())
    return ppl, acc, argmax


def _streamed_arm(label: str, weights, cfg: MiniMaxM3Config, ids: mx.array, n_layers: int | None,
                  ref_argmax: mx.array | None, ref_ppl: float | None) -> dict:
    """Run one streamed arm; report ppl/acc (+ Δppl% & top-1 agreement vs bf16 when given a ref); free
    the weights before returning so only one arm is ever resident (rule 8 / one-model-at-a-time)."""
    t0 = time.perf_counter()
    logits = streamed_logits(weights, cfg, ids, n_layers=n_layers)
    ppl, acc, argmax = teacher_forced(logits, ids)
    dt = time.perf_counter() - t0
    out = {"label": label, "ppl": ppl, "acc": acc, "argmax": argmax, "sec": dt}
    if ref_argmax is not None and ref_ppl is not None:
        out["dppl"] = 100.0 * (ppl / ref_ppl - 1.0)
        out["agree"] = float(mx.mean((argmax == ref_argmax).astype(mx.float32)).item())
        extra = f"  Δppl {out['dppl']:+.2f}%  agree {out['agree']:.4f}"
    else:
        extra = "  (reference)"
    weights.release()
    del logits
    mx.clear_cache()
    print(f"  [{label:9}] ppl {ppl:7.4f}  acc {acc:.4f}{extra}  ({dt:.0f}s)", flush=True)
    return out


def run(n_layers: int | None = None, n_tok: int = N_TOK, artifact: str = ART) -> None:
    full = n_layers is None
    mx.set_cache_limit(8 * 1024**3)
    cfg_src = MiniMaxM3Config.from_pretrained(SRC)
    # MiniMaxM3Config duck-types the token-id attrs the tokenizer reads (bos/eos/eos_token_ids); the
    # BPE is architecture-independent (reads only tokenizer.json), so construct it directly rather than
    # via from_pretrained (which parses the flat M2.7 config). add_bos_token absent → no BOS (raw).
    tok = MiniMaxTokenizer(os.path.join(SRC, "tokenizer.json"), cfg_src)
    ids_list = tok.encode(PROSE)[:n_tok]
    ids = mx.array(ids_list, dtype=mx.uint32)
    print(f"=== MiniMax-M3-VL ppl arbiter — {len(ids_list)} tok, "
          f"{'all' if full else n_layers} layers (streamed, SOLO) ===", flush=True)

    # 1) bf16 reference — its argmax + ppl anchor the quant Δ and confirm the (1+w) fold
    src = MiniMaxM3SourceCheckpoint(SRC, cfg_src)
    ref = _streamed_arm("bf16-src", src, cfg_src, ids, n_layers, None, None)
    del src
    mx.clear_cache()

    # 2) the quantized arm (default int4-g64) — streamed (dequant-on-read) vs the bf16 reference
    art = MiniMaxM3Artifact(artifact)
    bits = _expert_bits(art)
    label = f"int{bits}-g64"
    size = {4: "~233.4 GiB", 6: "~329.6 GiB"}.get(bits, "resident")
    qa = _streamed_arm(label, art, art.cfg, ids, n_layers, ref["argmax"], ref["ppl"])
    del art
    mx.clear_cache()

    # --- verdict: teacher-forced ppl is THE arbiter (CLAUDE.md methodology #4). The served width
    # (int4-g64 going forward) ships unless it regresses materially; top-1 agreement is a SECONDARY,
    # noisy signal (bf16-ULP near-ties flip at low-confidence positions — a settled finding) used as a
    # loose floor, not a tight gate. ---
    fold = ("CONFIRMED" if ref["ppl"] < PPL_CEILING else "SUSPECT") if full else "n/a (smoke)"
    print(f"\n(1+w) fold (bf16 ppl {ref['ppl']:.3f} vs {PPL_CEILING} ceiling): {fold}", flush=True)
    if full and ref["ppl"] < PPL_CEILING and qa["dppl"] < DPPL_CEILING and qa["agree"] > AGREE_FLOOR:
        verdict = (f"SHIP {label} ({size}) — Δppl {qa['dppl']:+.2f}% / acc {qa['acc']:.4f} vs "
                   f"bf16 {ref['acc']:.4f} / agree {qa['agree']:.4f} (~lossless; the served width is "
                   f"validated e2e).")
    elif full:
        verdict = (f"{label} Δppl {qa['dppl']:+.2f}% / acc {qa['acc']:.4f} / agree {qa['agree']:.4f} "
                   f"vs bf16 ppl {ref['ppl']:.3f} — INSPECT (regression or (1+w)-fold suspect).")
    else:
        verdict = (f"SMOKE ok — both arms ran ({n_layers} layers); bf16 ppl {ref['ppl']:.3f}, {label} "
                   f"Δppl {qa['dppl']:+.2f}% / agree {qa['agree']:.4f} (numbers not meaningful — "
                   f"partial model; run with no args for the real verdict).")
    print(f"VERDICT: {verdict}", flush=True)

    _ck(math.isfinite(ref["ppl"]), "bf16 arm produced a non-finite ppl")
    _ck(math.isfinite(qa["ppl"]), f"{label} arm produced a non-finite ppl")
    if full:
        _ck(ref["ppl"] < PPL_CEILING,
            f"bf16 ppl {ref['ppl']:.3f} >= {PPL_CEILING}: the (1+w) fold or forward is broken "
            f"(it degrades ppl uniformly — CLAUDE.md methodology #4)")
        _ck(qa["dppl"] < DPPL_CEILING and qa["agree"] > AGREE_FLOOR,
            f"{label} regresses materially: Δppl {qa['dppl']:+.2f}% (ceiling {DPPL_CEILING}%), "
            f"agree {qa['agree']:.4f} (floor {AGREE_FLOOR})")
    print(f"PARITY-CHECKS: {_N}", flush=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    art_path = ART
    if "--artifact" in argv:
        i = argv.index("--artifact")
        art_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    nl = int(argv[0]) if len(argv) > 0 else None
    nt = int(argv[1]) if len(argv) > 1 else N_TOK
    run(nl, nt, art_path)
