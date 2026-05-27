"""Model-free assertions for :func:`quanta.dsv4.routing_capture.capture_routing`.

Tiny synthetic config + a stub ``capture_fn`` that returns hand-crafted
``(x, idx)`` per layer. Validates:

* every *score* layer (``layer_id >= n_hash_layers``) yields a shard;
* hash layers are skipped (no file written);
* shapes inside the shard match contract (``[N, hidden]`` x, ``[N, topk]`` idx);
* idx values are in ``[0, n_routed_experts)``;
* shapes are rejected loudly when the stub emits wrong shapes;
* round-trip via :func:`load_routing_shard` recovers the captured data.

    uv run --with numpy python -m parity.dsv4_routing_capture_test
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.routing_capture import capture_routing, load_routing_shard

# A tiny but structurally valid DSV4 config: 4 layers, first 2 are hash; small
# everything. The capture_routing path only reads ``hidden_size``,
# ``num_experts_per_tok``, ``n_routed_experts``, ``num_hidden_layers``, and
# ``is_hash(layer_id)`` so the rest can carry the source defaults.
def _tiny_cfg() -> DeepSeekV4Config:
    # Build a real ``DeepSeekV4Config`` via ``replace`` from a minimal one;
    # construct one directly here to avoid a real ``config.json``.
    return DeepSeekV4Config(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=4,
        moe_intermediate_size=8,
        num_attention_heads=2,
        head_dim=8,
        rope_head_dim=4,
        q_lora_rank=4,
        o_lora_rank=4,
        o_groups=1,
        sliding_window=16,
        index_n_heads=1,
        index_head_dim=4,
        index_topk=2,
        compress_ratios=(0, 0, 0, 0),
        compress_rope_theta=10000.0,
        n_routed_experts=12,
        num_experts_per_tok=3,
        n_shared_experts=1,
        n_hash_layers=2,                # L0, L1 are hash; L2, L3 are score
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        swiglu_limit=0.0,
        hc_mult=1,
        hc_sinkhorn_iters=0,
        hc_eps=1e-6,
        n_mtp_layers=0,
        norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling={},
        max_position_embeddings=64,
        bos_token_id=0,
        eos_token_id=1,
        eos_token_ids=(1,),
        tie_word_embeddings=False,
    )


def _stub_capture(cfg: DeepSeekV4Config, n_tokens: int = 8, seed: int = 0):
    """Return a capture_fn that yields synthetic (x, idx) per layer."""
    rng = np.random.default_rng(seed)

    def _fn(ck, cfg_in, input_ids, *, n_layers=None):  # noqa: ARG001
        n = cfg_in.num_hidden_layers if n_layers is None else n_layers
        caps: dict[int, tuple[mx.array, mx.array]] = {}
        for i in range(n):
            x = (rng.standard_normal((n_tokens, cfg_in.hidden_size)) * 0.5).astype(np.float32)
            idx = rng.integers(
                0, cfg_in.n_routed_experts, size=(n_tokens, cfg_in.num_experts_per_tok)
            ).astype(np.int32)
            caps[i] = (mx.array(x).astype(mx.bfloat16), mx.array(idx).astype(mx.int32))
        return caps

    return _fn


def _stub_bad_shape(cfg: DeepSeekV4Config):
    """A capture_fn that returns a wrong-shape x to verify loud failure."""

    def _fn(ck, cfg_in, input_ids, *, n_layers=None):  # noqa: ARG001
        n = cfg_in.num_hidden_layers if n_layers is None else n_layers
        return {
            i: (
                mx.zeros((8, cfg_in.hidden_size + 1), dtype=mx.bfloat16),  # WRONG
                mx.zeros((8, cfg_in.num_experts_per_tok), dtype=mx.int32),
            )
            for i in range(n)
        }

    return _fn


def _stub_bad_idx(cfg: DeepSeekV4Config):
    """A capture_fn that returns an out-of-range idx to verify loud failure."""

    def _fn(ck, cfg_in, input_ids, *, n_layers=None):  # noqa: ARG001
        n = cfg_in.num_hidden_layers if n_layers is None else n_layers
        bad = mx.full((8, cfg_in.num_experts_per_tok), cfg_in.n_routed_experts + 5, dtype=mx.int32)
        return {
            i: (mx.zeros((8, cfg_in.hidden_size), dtype=mx.bfloat16), bad) for i in range(n)
        }

    return _fn


def run() -> None:
    cfg = _tiny_cfg()
    ok = True

    # 1) Happy-path: hash layers skipped, score layers written, shapes checked.
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        written = capture_routing(
            _stub_capture(cfg), ck=None, cfg=cfg, input_ids=mx.array([0]),
            output_dir=out_dir,
        )
        # Hash layers (0, 1) must NOT be written; score layers (2, 3) must be.
        wrote_only_score = set(written.keys()) == {2, 3}
        files = sorted(p.name for p in out_dir.glob("*.npz"))
        files_only_score = files == ["dsv4_routing_L002.npz", "dsv4_routing_L003.npz"]

        # Verify shard contents for L2
        x, idx, lid = load_routing_shard(out_dir / "dsv4_routing_L002.npz")
        shapes_ok = (
            x.shape == (8, cfg.hidden_size)
            and idx.shape == (8, cfg.num_experts_per_tok)
            and lid == 2
        )
        idx_ok = int(idx.min().item()) >= 0 and int(idx.max().item()) < cfg.n_routed_experts
        non_trivial = bool(mx.any(mx.abs(x) > 0).item())  # x not all zeros
        # idx must have at least 2 distinct values (variety) — random capture has many
        unique_idx = int(np.unique(np.asarray(idx)).size) > 1
        good = wrote_only_score and files_only_score and shapes_ok and idx_ok and non_trivial and unique_idx
        ok = ok and good
        print(
            f"  [{'OK' if good else 'FAIL'}] happy path: hash skipped={wrote_only_score}, "
            f"files={files_only_score}, shard shapes={shapes_ok}, idx range={idx_ok}, "
            f"x non-trivial={non_trivial}, idx non-trivial={unique_idx}"
        )

    # 2) Loud failure: wrong x shape -> ValueError
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        try:
            capture_routing(
                _stub_bad_shape(cfg), ck=None, cfg=cfg, input_ids=mx.array([0]),
                output_dir=out_dir,
            )
            raised = False
        except ValueError:
            raised = True
        ok = ok and raised
        print(f"  [{'OK' if raised else 'FAIL'}] loud failure on bad x shape (ValueError raised)")

    # 3) Loud failure: out-of-range idx -> ValueError
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        try:
            capture_routing(
                _stub_bad_idx(cfg), ck=None, cfg=cfg, input_ids=mx.array([0]),
                output_dir=out_dir,
            )
            raised = False
        except ValueError:
            raised = True
        ok = ok and raised
        print(f"  [{'OK' if raised else 'FAIL'}] loud failure on out-of-range idx (ValueError)")

    # 4) Subset selection: ``n_layers=3`` limits the capture range.
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        written = capture_routing(
            _stub_capture(cfg), ck=None, cfg=cfg, input_ids=mx.array([0]),
            output_dir=out_dir, n_layers=3,
        )
        # L0, L1 hash (skipped), L2 score (written); L3 NOT visited
        only_l2 = set(written.keys()) == {2}
        ok = ok and only_l2
        print(f"  [{'OK' if only_l2 else 'FAIL'}] n_layers=3 -> only L2 written: {only_l2}")

    # 5) n_layers out of range -> ValueError
    try:
        capture_routing(
            _stub_capture(cfg), ck=None, cfg=cfg, input_ids=mx.array([0]),
            output_dir="/tmp/quanta_should_not_exist_xyz", n_layers=99,
        )
        raised = False
    except ValueError:
        raised = True
    ok = ok and raised
    print(f"  [{'OK' if raised else 'FAIL'}] loud failure on out-of-range n_layers (ValueError)")

    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
