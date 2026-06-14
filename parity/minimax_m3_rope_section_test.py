"""Model-free V3b-prep gate: the vision ``rope_section`` sweep knob is **live** + the candidate set.

The [PINNED-pending-e2e] vision ``rope_section`` (the ``(t,h,w)`` freq-pair split of the 3-D vision
RoPE) is the one knob no on-disk artifact fixes; the real-weight V3b arbiter
(``parity/minimax_m3_rope_section_real.py``) settles it by scoring
:func:`quanta.minimax.model_vision_m3.candidate_rope_sections` by downstream teacher-forced ppl on a
real image. Before spending the heavy ~235 GiB arbiter, this gate proves on tiny SYNTHETIC dims that
the machinery the arbiter drives is actually correct:

* **The knob is live.** Two *different* valid sections produce *materially different* merged ViT
  tokens (so the sweep can discriminate); two *identical* sections are **bit-exact** (the only thing
  changing between candidates is the section — the weights are reused, mirroring how the arbiter mutates
  ``vis.rope_section`` in place). If the section had no effect the whole arbiter would be meaningless.
* **The candidate set is valid.** :func:`candidate_rope_sections` returns only splits summing to
  ``head_dim//2``, keeps ``h == w`` symmetry, always includes the on-disk
  :func:`default_rope_section`, and matches the documented list for the real ``head_dim=80``.
* **rule 6.** An invalid section (one not summing to ``head_dim//2``) is refused, both at the rope
  primitive and through the full tower.

No real weights, no PIL — pure ``mlx`` on KB-size stubs; runs in the model-free sweep.

    uv run python -m parity.minimax_m3_rope_section_test
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from quanta.minimax import model_vision_m3 as V
from quanta.minimax.config_m3 import MiniMaxVisionConfig

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _np(x: mx.array) -> np.ndarray:
    return np.asarray(x.astype(mx.float32), dtype=np.float64)


def _rel(a: mx.array, b: mx.array) -> float:
    an, bn = _np(a), _np(b)
    return float(np.max(np.abs(an - bn)) / (np.max(np.abs(bn)) + 1e-9))


def _cfg() -> MiniMaxVisionConfig:
    # tiny but structurally faithful: head_dim = 32/4 = 8 ⇒ half = 4 (so sections sum to 4).
    return MiniMaxVisionConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
        patch_size=2, image_size=64, projection_dim=48, rope_theta=1e4, rope_mode="3d",
        layer_norm_eps=1e-5, hidden_act="gelu", num_channels=3,
        spatial_merge_size=2, temporal_patch_size=2,
    )


def run() -> None:
    rng = np.random.default_rng(0)
    cfg = _cfg()
    hd = cfg.hidden_size // cfg.num_attention_heads          # 8
    half = hd // 2                                            # 4
    in_dim = cfg.num_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size  # 24

    # --- (1) candidate set: valid, symmetric, includes the default, matches the real head_dim -----
    cands8 = V.candidate_rope_sections(hd)
    _ck(all(sum(s) == half for s in cands8), f"head_dim={hd} candidates not all summing to {half}: {cands8}")
    _ck(all(s[1] == s[2] for s in cands8), f"candidates not h==w symmetric: {cands8}")
    _ck(V.default_rope_section(hd) in cands8, f"default {V.default_rope_section(hd)} missing from {cands8}")
    _ck(len(cands8) == len(set(cands8)), f"candidate set has duplicates: {cands8}")
    cands80 = V.candidate_rope_sections(80)
    want80 = [(st, (40 - st) // 2, (40 - st) // 2) for st in range(0, 21, 2)]   # 11 splits, st 0..20
    _ck(cands80 == want80, f"candidate_rope_sections(80) {cands80} != {want80}")
    _ck((8, 16, 16) in cands80 and all(sum(s) == 40 for s in cands80), "head_dim=80 candidate set invalid")

    # --- a deterministic tiny tower (weights reused across sections — only the rope changes) -------
    mx.random.seed(0)
    model = V.MiniMaxM3VisionModel(cfg, rope_section=(0, 2, 2))
    mx.eval(model.parameters())
    grid = [(1, 4, 4)]                                        # one image, 16 patches → 4 merged tokens
    pv = mx.array(rng.standard_normal((16, in_dim)).astype(np.float32))

    def merged(section: tuple[int, int, int]) -> mx.array:
        model.rope_section = section                          # the arbiter mutates this in place
        out = model(pv, grid)
        mx.eval(out)
        return out

    # --- (2) knob is LIVE: two different valid sections ⇒ materially different merged tokens -------
    a = merged((0, 2, 2))                                     # all freq pairs on h/w
    b = merged((2, 1, 1))                                     # 2 pairs on the (image-inert) t-axis
    _ck(a.shape == (4, cfg.projection_dim), f"merged token shape {a.shape}")
    _ck(_rel(a, b) > 1e-3, f"rope_section knob inert: (0,2,2) vs (2,1,1) rel {_rel(a, b):.2e} (<=1e-3)")
    # an all-h vs all-w split must also differ (the section genuinely routes pairs to axes)
    c = merged((0, 4, 0))
    d = merged((0, 0, 4))
    _ck(_rel(c, d) > 1e-3, f"all-h vs all-w sections identical: rel {_rel(c, d):.2e}")

    # --- (3) identical section ⇒ BIT-EXACT (weights reused; only the section varies) ---------------
    a2 = merged((0, 2, 2))
    _ck(_rel(a, a2) == 0.0, f"same section not bit-exact across re-runs: rel {_rel(a, a2):.2e}")

    # --- (4) rule 6: an invalid section (sum != half) is refused, primitive AND full tower ---------
    pos = V.vision_position_ids(grid, cfg.spatial_merge_size)
    try:
        V.vision_rope_3d(pos, hd, cfg.rope_theta, (1, 1, 1))  # sums to 3 != 4
        _ck(False, "vision_rope_3d accepted a bad section")
    except ValueError:
        _ck(True, "vision_rope_3d refused a bad section")
    try:
        model.rope_section = (3, 3, 3)                        # sums to 9 != 4
        mx.eval(model(pv, grid))
        _ck(False, "tower accepted a bad section")
    except ValueError:
        _ck(True, "tower refused a bad section")

    print(f"PARITY-CHECKS: {_N}")
    print(f"PASS — MiniMax-M3-VL V3b-prep rope_section knob: LIVE (distinct sections ⇒ distinct merged "
          f"tokens; same section bit-exact), candidate set valid (default included, head_dim=80 == "
          f"{want80}), bad sections refused ({_N} checks). [the real split is settled by the V3b arbiter]")


if __name__ == "__main__":
    run()
