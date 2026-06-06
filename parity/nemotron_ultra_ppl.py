"""Nemotron-Ultra U3 — teacher-forced ppl + top-1, the int4 AWQ-vs-RTN e2e arbiter.

Three **sequential** streamed forwards (rule-8: one decoder layer resident at a time; the MoE
expert stack ~21.5 GiB bf16 is the peak, never the 1023 GiB / 336 GiB whole model) over the *same*
held-out prose, each freed before the next so only one is ever resident:

  1. **bf16 Ultra source**  (`NemotronSourceCheckpoint`) -> reference ppl + reference argmax
  2. **int4-AWQ g64 artifact** (`NemotronArtifact`, dequant-on-read incl. the AWQ `W·diag(1/s)` fold)
  3. **int4-RTN g64 artifact** (`NemotronArtifact`, s=1 → plain int4, one runtime path)

All three share `parity.nemotron_ppl.streamed_logits` verbatim, so the *forward* is identical and
each ppl Δ isolates pure expert-coding error (U1 already proved the forward matches the transformers
reference to ~7e-4). The arbiter ranks AWQ vs RTN vs bf16 head-to-head and the verdict picks the
lower-ppl arm (ship it if Δ < 5% and top-1 agreement > 0.90, else report for a decision).

Background: an earlier 109-token run measured the AWQ bake at **+11.2% ppl vs bf16** — the relu²
down-proj AWQ tax that finding #38 (codified in `nemotron.bake._bake_expert`) predicts and the U2
slice de-risk could not see (recon-only + L1-only; recon does NOT predict e2e — settled finding).
This harness uses a ~1024-token corpus (≈10× the prior sample) for a trustworthy verdict and adds
the RTN arm that #38 says should be ~lossless at the same 4-bit footprint. The corpus is original
expository prose, **held out** from the agentic calibration set (project code + INITIAL_PROMPT.md +
CLAUDE.md) that the AWQ bake fit its scales on.

AWQ retired (U3 verdict: ship int4-RTN; #38 e2e +24.3%) and its 336 GiB artifact removed, so the AWQ
arm auto-skips when absent and the harness degrades to the enduring bf16-vs-RTN losslessness check.

    uv run --with tokenizers python -m parity.nemotron_ultra_ppl
"""

from __future__ import annotations

import os
import time

import mlx.core as mx

from parity.nemotron_ppl import streamed_logits
from quanta.nemotron.artifact import NemotronArtifact
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.loader import NemotronSourceCheckpoint
from quanta.nemotron.tokenizer import NemotronTokenizer

BF16 = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
AWQ = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-quanta_int4awq_g64"
RTN = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-quanta_int4rtn_g64"
N_TOK = 1024  # ~10x the prior 109-tok sample → a de-noised, trustworthy ppl verdict

# Original expository prose, held out from the agentic calibration corpus (code + prompts).
LONG_PROSE = """The measurement of time has shaped human civilization in ways that are easy to \
overlook. For most of history, people marked the passage of days by the sun and the seasons by the \
stars, and these natural rhythms were sufficient for planting crops, holding markets, and observing \
festivals. The earliest devices for dividing the day were shadow clocks and sundials, which relied \
on the predictable arc of the sun across the sky. Water clocks, which measured time by the steady \
flow of liquid from one vessel to another, freed timekeeping from the need for sunlight and allowed \
the hours to be counted at night or beneath a clouded sky. These instruments were imprecise by \
modern standards, yet they were accurate enough to govern the routines of temples, courts, and armies.

The mechanical clock, which appeared in European towns during the late medieval period, introduced \
a new conception of time as something uniform and divisible into equal parts. Driven by falling \
weights and regulated by an oscillating escapement, these clocks struck the hours for entire \
communities and gradually imposed a shared schedule on daily life. The invention of the pendulum \
dramatically improved their accuracy, reducing the error of a good clock from many minutes a day to \
only a few seconds. With the pendulum came the possibility of comparing distant observations and \
coordinating events across great distances, a capacity that would prove essential to science and \
commerce alike.

Navigation at sea posed one of the most demanding problems in the history of timekeeping. A sailor \
could determine latitude by measuring the height of the sun or a known star above the horizon, but \
longitude required knowing the precise difference between local time and the time at a reference \
location. Because the earth turns through fifteen degrees of longitude every hour, an error of a few \
minutes in timekeeping could place a ship dozens of miles from its true position. The solution, \
after decades of effort, was the marine chronometer, a clock robust enough to keep accurate time \
despite the rolling of a ship, changes in temperature, and the corrosive salt air. Carrying reliable \
time across the ocean transformed exploration and trade, and it allowed mariners to chart coastlines \
with unprecedented confidence.

The natural world keeps its own clocks, often with remarkable precision. Many plants open and close \
their flowers at particular hours, and some continue to do so even when kept in constant darkness, \
revealing an internal rhythm rather than a simple response to light. Animals migrate, breed, and \
hibernate according to cycles tied to the length of the day, and certain birds navigate using the \
position of the sun and stars, compensating automatically for the passage of time. Within nearly \
every living cell, a network of molecules rises and falls over a roughly twenty-four-hour period, \
governing sleep, metabolism, and the repair of tissues. These biological clocks are not perfectly \
accurate on their own, but they are continually reset by the cues of the surrounding environment, \
chiefly the daily alternation of light and darkness.

The modern era replaced mechanical oscillation with the vibration of atoms. An atomic clock counts \
the extraordinarily steady transitions of electrons within an atom, and the best of these \
instruments would gain or lose less than a second over many millions of years. Such precision is \
not merely an academic achievement, for it underpins the satellite navigation systems that guide \
aircraft, ships, and ordinary travelers. A receiver on the ground determines its position by \
comparing the tiny differences in the arrival times of signals from several satellites, each \
carrying its own atomic clock. A timing error of a single millionth of a second would translate \
into a positional error of hundreds of meters, and so the management of time has become inseparable \
from the management of space.

The calendar, by contrast, is concerned not with the hours of a single day but with the reckoning \
of years, and it has proven equally troublesome to perfect. The interval between one spring equinox \
and the next is not a whole number of days, and this stubborn fraction has forced every \
civilization to choose between a calendar that drifts against the seasons and one that must be \
corrected by inserting extra days or months. The reform that produced the calendar in widest use \
today adjusted the rule for leap years so that the accumulated error would amount to only a few days \
over many thousands of years. Even so, the occasional insertion of a single second keeps official \
clocks aligned with the gradual and slightly irregular slowing of the earth's rotation.

The coming of the railway made the patchwork of local times intolerable. When every town set its \
clocks by its own noon, a journey of a few hundred miles could pass through dozens of slightly \
different times, and a printed timetable became nearly impossible to follow. The answer was to \
divide the world into broad zones, within each of which all clocks would agree, and to anchor them \
to a single agreed meridian. This standardization, once controversial, is now so familiar that \
travelers scarcely notice it, adjusting their watches by whole hours as they cross from one zone to \
the next and trusting that the time they keep will match the time kept by everyone around them.

It is striking how the pursuit of accurate time has repeatedly pushed forward the broader frontiers \
of knowledge. The study of pendulums advanced the understanding of gravity, the demands of \
navigation spurred improvements in metallurgy and astronomy, and the development of atomic clocks \
grew directly out of the new physics of the twentieth century. Each gain in precision opened \
questions that had previously been invisible, and each answer demanded instruments finer still. \
Time, which once seemed the most natural and unremarkable of things, turned out to be among the \
most difficult to measure and among the most rewarding to master."""


def _ppl_acc(logits: mx.array, targets: mx.array) -> tuple[float, float, mx.array]:
    """Teacher-forced ppl, top-1 next-token accuracy, and the argmax sequence (fp32 head)."""
    lg = logits[:-1].astype(mx.float32)
    ce = mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, targets[:, None], axis=-1)[:, 0]
    ppl = mx.exp(ce.mean()).item()
    pred = mx.argmax(lg, axis=-1)
    acc = (pred == targets).astype(mx.float32).mean().item()
    return ppl, acc, pred


def _arm(path: str, arr: mx.array, targets: mx.array, pred_ref: mx.array) -> tuple[float, float, float, float]:
    """Stream one int4 artifact, returning (ppl, top-1 acc, top-1 agree vs bf16, minutes)."""
    cfg = NemotronHConfig.from_pretrained(path)
    art = NemotronArtifact(path)
    t = time.perf_counter()
    ppl, acc, pred = _ppl_acc(streamed_logits(art, cfg, arr), targets)
    agree = (pred == pred_ref).astype(mx.float32).mean().item()
    mx.eval(pred)
    dt = (time.perf_counter() - t) / 60
    del art
    mx.clear_cache()
    return ppl, acc, agree, dt


def run() -> None:
    tok = NemotronTokenizer(RTN)  # AWQ retired (#38) and its artifact removed; the tokenizer is identical
    ids = tok.encode(LONG_PROSE, add_bos=False)[:N_TOK]
    arr = mx.array(ids)
    targets = arr[1:]

    # 1) bf16 reference — streamed from the source checkpoint, freed before the artifacts load.
    cfg_b = NemotronHConfig.from_pretrained(BF16)
    ck = NemotronSourceCheckpoint(BF16)
    t0 = time.perf_counter()
    ppl_b, acc_b, pred_b = _ppl_acc(streamed_logits(ck, cfg_b, arr), targets)
    mx.eval(pred_b)
    ck.release()
    del ck
    mx.clear_cache()
    t_b = (time.perf_counter() - t0) / 60

    # 2) int4-AWQ (retired #38 — only if the artifact still exists) then 3) int4-RTN; one resident at a time.
    awq = _arm(AWQ, arr, targets, pred_b) if os.path.isdir(AWQ) else None
    ppl_r, acc_r, agree_r, t_r = _arm(RTN, arr, targets, pred_b)
    dr = 100 * (ppl_r - ppl_b) / ppl_b

    print(f"\n=== Nemotron-Ultra U3  (streamed teacher-forced, tokens={len(ids)}) ===")
    print(f"bf16 reference   : ppl {ppl_b:7.3f} | top-1 acc {acc_b:.3f}                       [{t_b:.1f} min]")
    if awq is not None:
        ppl_a, acc_a, agree_a, t_a = awq
        da = 100 * (ppl_a - ppl_b) / ppl_b
        print(f"int4-AWQ g64     : ppl {ppl_a:7.3f} | top-1 acc {acc_a:.3f} | Δ {da:+5.1f}% | agree {agree_a:.3f}  [{t_a:.1f} min]")
    else:
        print("int4-AWQ g64     : retired (#38 e2e +24.3%); artifact removed — skipped")
    print(f"int4-RTN g64     : ppl {ppl_r:7.3f} | top-1 acc {acc_r:.3f} | Δ {dr:+5.1f}% | agree {agree_r:.3f}  [{t_r:.1f} min]")

    # Verdict — RTN ships (U3 settled); rank against AWQ only when its artifact is present.
    if awq is not None and ppl_a < ppl_r:
        win, wd, wagree = "AWQ", da, agree_a
    else:
        win, wd, wagree = "RTN", dr, agree_r
    ship = wd < 5.0 and wagree > 0.90
    print(f"  winner         : int4-{win}  (Δ {wd:+.1f}% vs bf16, agree {wagree:.3f})")
    print(f"VERDICT          : {'ship int4-' + win if ship else 'int4-' + win + ' best but >5% / <0.90 — decision needed'}")


if __name__ == "__main__":
    run()
