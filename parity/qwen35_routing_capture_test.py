"""Model-free assertions for :func:`quanta.qwen35.routing_capture.capture_routing`.

Tiny synthetic ``Qwen35Config`` + a stub ``capture_fn`` that returns
hand-crafted ``(x [N, hidden], idx [N, topk])`` per layer. Validates:

* every captured layer yields a shard (Qwen3.5 has MoE on every layer);
* shapes match contract;
* idx values are in ``[0, num_experts)``;
* loud failure when stub emits wrong x shape;
* loud failure when stub emits out-of-range idx;
* round-trip via :func:`load_routing_shard` recovers data.

    uv run --with numpy python -m parity.qwen35_routing_capture_test
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.routing_capture import capture_routing, load_routing_shard


def _tiny_cfg() -> Qwen35Config:
    # All-full-attention 3-layer Qwen3.5: enough to validate the capture path;
    # head_dim=8 so rotary_dim=2 and mrope_section=(1,) sums to rotary_dim//2=1.
    return Qwen35Config(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=3,
        layer_types=("full_attention", "full_attention", "full_attention"),
        full_attention_interval=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attn_output_gate=False,
        partial_rotary_factor=0.25,
        rope_theta=1e7,
        mrope_section=(1,),
        mrope_interleaved=False,
        use_qk_norm=True,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        mamba_ssm_dtype="float32",
        num_experts=8,
        num_experts_per_tok=3,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        scoring_func="softmax",
        norm_topk_prob=True,
        router_aux_loss_coef=0.001,
        num_mtp_modules=0,
        mtp_use_dedicated_embeddings=False,
        hidden_act="silu",
        norm_eps=1e-6,
        max_position_embeddings=64,
        eos_token_id=1,
        eos_token_ids=(1,),
        pad_token_id=2,
        tie_word_embeddings=False,
    )


def _stub_capture(cfg: Qwen35Config, n_tokens: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)

    def _fn(ck, cfg_in, input_ids, *, n_layers=None):  # noqa: ARG001
        n = cfg_in.num_hidden_layers if n_layers is None else n_layers
        caps: dict[int, tuple[mx.array, mx.array]] = {}
        for i in range(n):
            x = (rng.standard_normal((n_tokens, cfg_in.hidden_size)) * 0.5).astype(np.float32)
            idx = rng.integers(
                0, cfg_in.num_experts, size=(n_tokens, cfg_in.num_experts_per_tok)
            ).astype(np.int32)
            caps[i] = (mx.array(x).astype(mx.bfloat16), mx.array(idx).astype(mx.int32))
        return caps

    return _fn


def _stub_bad_shape(cfg: Qwen35Config, n_tokens: int = 8):
    def _fn(ck, cfg_in, input_ids, *, n_layers=None):  # noqa: ARG001
        n = cfg_in.num_hidden_layers if n_layers is None else n_layers
        return {
            i: (
                mx.zeros((n_tokens, cfg_in.hidden_size + 1), dtype=mx.bfloat16),  # WRONG
                mx.zeros((n_tokens, cfg_in.num_experts_per_tok), dtype=mx.int32),
            )
            for i in range(n)
        }

    return _fn


def _stub_bad_idx(cfg: Qwen35Config, n_tokens: int = 8):
    def _fn(ck, cfg_in, input_ids, *, n_layers=None):  # noqa: ARG001
        n = cfg_in.num_hidden_layers if n_layers is None else n_layers
        bad = mx.full(
            (n_tokens, cfg_in.num_experts_per_tok),
            cfg_in.num_experts + 5,
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

    # 1) Happy-path
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        written = capture_routing(
            _stub_capture(cfg), ck=None, cfg=cfg, input_ids=mx.array([0]),
            output_dir=out_dir,
        )
        wrote_all = set(written.keys()) == {0, 1, 2}
        files = sorted(p.name for p in out_dir.glob("*.npz"))
        files_all = files == [
            "qwen35_routing_L000.npz",
            "qwen35_routing_L001.npz",
            "qwen35_routing_L002.npz",
        ]
        x, idx, lid = load_routing_shard(out_dir / "qwen35_routing_L000.npz")
        shapes_ok = (
            x.shape == (8, cfg.hidden_size)
            and idx.shape == (8, cfg.num_experts_per_tok)
            and lid == 0
        )
        idx_ok = int(idx.min().item()) >= 0 and int(idx.max().item()) < cfg.num_experts
        non_trivial = bool(mx.any(mx.abs(x) > 0).item())
        unique_idx = int(np.unique(np.asarray(idx)).size) > 1
        good = wrote_all and files_all and shapes_ok and idx_ok and non_trivial and unique_idx
        ok = ok and good
        print(
            f"  [{'OK' if good else 'FAIL'}] happy path: all layers={wrote_all}, "
            f"files={files_all}, shapes={shapes_ok}, idx range={idx_ok}, "
            f"x non-trivial={non_trivial}, idx non-trivial={unique_idx}"
        )

    # 2) Loud failure: wrong x shape
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
        print(f"  [{'OK' if raised else 'FAIL'}] loud failure on bad x shape (ValueError)")

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
