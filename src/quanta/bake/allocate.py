"""Mixed int3/int4 bit allocation for routed experts (sensitivity-driven knapsack).

Each projection carries its activation-weighted GPTQ loss at int3 and int4 (from
calibration). Start everything at int3 (cheapest, ~412 GB for all experts), then promote
projections to int4 in order of error-drop ``loss3 − loss4`` — which is exactly
error-reduction per byte, since the int3→int4 cost is ``params/8`` for every projection —
until the params-weighted total error falls below ``target`` (8%) or the byte budget is
spent. Greedy is optimal here (uniform cost-per-promoted-bit), and it's the "back-and-forward
auto-adjust": only the most sensitive experts spend the extra bit. The activation-weighted
loss is the allocation proxy; the final arbiter remains e2e teacher-forced ppl.
"""

from __future__ import annotations

from dataclasses import dataclass

BPP = {3: 3.25, 4: 4.25}  # affine group-128 effective bits/param


@dataclass(frozen=True)
class Projection:
    """One quantizable expert projection's calibration sensitivity."""

    key: str
    params: int
    loss3: float
    loss4: float


def allocate_bits(
    projs: list[Projection], byte_budget: float, target: float = 0.08
) -> tuple[dict[str, int], float, float]:
    """Assign int3/int4 per projection → ``(bits_by_key, total_error, total_bytes)``.

    ``total_error`` is the params-weighted mean projection loss; ``byte_budget`` caps the
    experts' resident bytes. Promotion stops as soon as ``target`` is met (no wasted bytes)
    or the next-best promotion won't fit.
    """
    bits = {p.key: 3 for p in projs}
    total_p = sum(p.params for p in projs)
    err_num = sum(p.params * p.loss3 for p in projs)  # all-int3 weighted error numerator
    used = sum(p.params for p in projs) * BPP[3] / 8.0

    for p in sorted(projs, key=lambda q: q.loss3 - q.loss4, reverse=True):
        if err_num / total_p <= target:
            break
        cost = p.params * (BPP[4] - BPP[3]) / 8.0  # = params/8
        if used + cost > byte_budget:
            continue  # won't fit; a smaller high-Δloss projection later might
        bits[p.key] = 4
        err_num += p.params * (p.loss4 - p.loss3)
        used += cost

    return bits, err_num / total_p, used
