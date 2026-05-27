"""Model-free assertions for :func:`quanta.nemotron.routing_capture.capture_routing`.

Tiny synthetic ``NemotronHConfig`` + a stub ``capture_fn`` that returns
hand-crafted ``(x [N, hidden], idx [N, topk])`` per layer. Validates:

* every captured layer yields a shard;
* shapes match contract;
* idx values are in ``[0, n_routed_experts)``;
* loud failure when stub emits wrong x shape (LATENT not HIDDEN);
* loud failure when stub emits out-of-range idx;
* round-trip via :func:`load_routing_shard` recovers data.

    uv run --with numpy python -m parity.nemotron_routing_capture_test
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.routing_capture import capture_routing, load_routing_shard


def _tiny_cfg() -> NemotronHConfig:
    # Three layers: M=mamba, *=attention, E=moe; the routing capture only
    # consumes the captured tuples so the per-layer kind is moot in this test —
    # we just need a structurally valid config.
    return NemotronHConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=3,
        hybrid_override_pattern="M*E",
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        attention_bias=False,
        rope_theta=10000.0,
        partial_rotary_factor=1.0,
        mamba_num_heads=2,
        mamba_head_dim=8,
        mamba_n_groups=1,
        ssm_state_size=8,
        conv_kernel=4,
        expand=2,
        mamba_hidden_act="silu",
        chunk_size=64,
        use_conv_bias=True,
        n_routed_experts=8,
        num_experts_per_tok=3,
        n_shared_experts=1,
        moe_intermediate_size=8,
        moe_latent_size=8,
        moe_shared_expert_intermediate_size=8,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        norm_eps=1e-5,
        max_position_embeddings=64,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        num_nextn_predict_layers=0,
        tie_word_embeddings=False,
    )


def _stub_capture(cfg: NemotronHConfig, n_tokens: int = 8, seed: int = 0):
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


def _stub_latent_shape(cfg: NemotronHConfig, n_tokens: int = 8):
    """Stub that returns LATENT-shaped x (the meta-router needs HIDDEN)."""

    def _fn(ck, cfg_in, input_ids, *, n_layers=None):  # noqa: ARG001
        n = cfg_in.num_hidden_layers if n_layers is None else n_layers
        return {
            i: (
                mx.zeros((n_tokens, cfg_in.moe_latent_size), dtype=mx.bfloat16),  # WRONG
                mx.zeros((n_tokens, cfg_in.num_experts_per_tok), dtype=mx.int32),
            )
            for i in range(n)
        }

    return _fn


def _stub_bad_idx(cfg: NemotronHConfig, n_tokens: int = 8):
    def _fn(ck, cfg_in, input_ids, *, n_layers=None):  # noqa: ARG001
        n = cfg_in.num_hidden_layers if n_layers is None else n_layers
        bad = mx.full(
            (n_tokens, cfg_in.num_experts_per_tok),
            cfg_in.n_routed_experts + 5,
            dtype=mx.int32,
        )
        return {
            i: (mx.zeros((n_tokens, cfg_in.hidden_size), dtype=mx.bfloat16), bad)
            for i in range(n)
        }

    return _fn


def run() -> None:
    cfg = _tiny_cfg()
    ok = True

    # 1) Happy-path: every layer (no hash skip in Nemotron) -> shard.
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        written = capture_routing(
            _stub_capture(cfg), ck=None, cfg=cfg, input_ids=mx.array([0]),
            output_dir=out_dir,
        )
        wrote_all = set(written.keys()) == {0, 1, 2}
        files = sorted(p.name for p in out_dir.glob("*.npz"))
        files_all = files == [
            "nemotron_routing_L000.npz",
            "nemotron_routing_L001.npz",
            "nemotron_routing_L002.npz",
        ]

        x, idx, lid = load_routing_shard(out_dir / "nemotron_routing_L001.npz")
        shapes_ok = (
            x.shape == (8, cfg.hidden_size)
            and idx.shape == (8, cfg.num_experts_per_tok)
            and lid == 1
        )
        idx_ok = int(idx.min().item()) >= 0 and int(idx.max().item()) < cfg.n_routed_experts
        non_trivial = bool(mx.any(mx.abs(x) > 0).item())
        unique_idx = int(np.unique(np.asarray(idx)).size) > 1
        good = wrote_all and files_all and shapes_ok and idx_ok and non_trivial and unique_idx
        ok = ok and good
        print(
            f"  [{'OK' if good else 'FAIL'}] happy path: all layers={wrote_all}, "
            f"files={files_all}, shapes={shapes_ok}, idx range={idx_ok}, "
            f"x non-trivial={non_trivial}, idx non-trivial={unique_idx}"
        )

    # 2) Loud failure: x is LATENT-shaped (would silently mis-train the meta-router)
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        try:
            capture_routing(
                _stub_latent_shape(cfg), ck=None, cfg=cfg, input_ids=mx.array([0]),
                output_dir=out_dir,
            )
            raised = False
        except ValueError as e:
            raised = "HIDDEN" in str(e) or "hidden" in str(e) or "x shape" in str(e)
        ok = ok and raised
        print(f"  [{'OK' if raised else 'FAIL'}] loud failure on LATENT x (ValueError raised)")

    # 3) Loud failure: out-of-range idx
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

    # 4) n_layers out of range -> ValueError
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
