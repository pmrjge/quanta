"""Model-free parity gate for Nemotron-H batched decode (#147): step_batch(B=4) == 4 × single-stream.

A tiny random-init NemotronModel (M*EM pattern, real per-tensor dims via direct dataclass
construction — NO checkpoint, NO real model directory) is run two ways:

  1. SINGLE-STREAM: one copy of the prompt fed token-by-token through the existing decode path
     (``NemotronModel.__call__`` with per-step caches + ssm/conv state). Captures per-position
     logits.
  2. BATCHED: ``B=4`` *identical* per-stream states + identical prompts, stepped through
     :func:`quanta.nemotron.batched_runtime.batched_decode_step` token by token. The MoE layer is
     the only data-mixing op (a single stacked ``[B,1,dim]`` call); per-stream Mamba/Attention
     are state-local. With identical inputs + identical states, EVERY per-stream logit row must
     equal the single-stream logit row at the same position.

A logic bug in the per-stream loop or in the stack-then-split MoE would be O(1) divergence
(the per-position log-probs would differ across streams or vs single-stream), not sub-1%. Same
gate signal as :mod:`parity.nemotron_decode_test` but for the batched dispatch.

Tests:
  * (1) step_batch logits at each decode position == single-stream logits, per stream, with
    rel ≤ 1e-2 (bf16 accumulation tolerance — matches :mod:`parity.nemotron_decode_test`'s
    prefill-vs-step tolerance; the MoE concatenate+slice is bit-exact at fp32 reduction
    granularity but stacked-vs-per-row matmul reordering in bf16 drifts ~1e-3).
  * (2) all B per-stream logits equal each other under identical inputs+states with rel ≤
    1e-6 (cross-stream consistency — proves no cross-stream contamination through the stacked
    MoE call; identical-input rows under identical state produce identical compute graphs).
  * (3) :func:`batched_generate` (greedy, identical prompts) returns identical token streams
    across B=4 streams AND matches single-stream :func:`generate` on the same prompt
    (argmax is robust to bf16 noise up to logit-margin / temperature).

    uv run --with numpy python -m parity.nemotron_batched_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.nemotron.batched_generate import batched_generate
from quanta.nemotron.batched_runtime import (
    NemotronBatchedResidentModel,
    batched_decode_step,
    make_step_state,
    make_stream_state,
)
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.generate import generate
from quanta.nemotron.mamba_mixer import MambaMixer
from quanta.nemotron.model import NemotronModel
from quanta.nemotron.moe import NemotronLatentMoE


def _rel(a: mx.array, b: mx.array) -> float:
    """Max-rel diff with a tiny absolute floor (matches the existing parity helpers)."""
    return float(mx.max(mx.abs(a - b)) / (mx.max(mx.abs(b)) + 1e-6))


def _tiny_cfg() -> NemotronHConfig:
    """Tiny config built directly (no checkpoint), with an M/*/E/M pattern so all three layer
    kinds are exercised in one batched step. Real-dim shape semantics, just much smaller.

    Dim invariants (must hold for the mamba mixer to reshape correctly):
      ``mamba_num_heads * mamba_head_dim == expand * hidden_size`` (Mamba-2 d_inner = H*P)."""
    return NemotronHConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        hybrid_override_pattern="M*EM",
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attention_bias=False,
        rope_theta=10000.0,
        partial_rotary_factor=1.0,
        mamba_num_heads=8,        # h * p = 8 * 8 = 64 = expand * hidden_size (2 * 32)
        mamba_head_dim=8,
        mamba_n_groups=2,
        ssm_state_size=8,
        conv_kernel=4,
        expand=2,
        mamba_hidden_act="silu",
        chunk_size=8,
        use_conv_bias=True,
        n_routed_experts=8,
        num_experts_per_tok=3,
        n_shared_experts=1,
        moe_intermediate_size=16,
        moe_latent_size=12,
        moe_shared_expert_intermediate_size=12,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        norm_eps=1e-5,
        max_position_embeddings=1024,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        num_nextn_predict_layers=0,
        tie_word_embeddings=False,
    )


def _randomize(model: NemotronModel) -> None:
    """Mamba SSM params + MoE expert stacks default to zeros/ones — give them real dynamics so
    the test is sensitive to ordering / routing bugs. Mirrors :mod:`parity.nemotron_decode_test`."""
    cfg = model.cfg
    for blk in model.layers:
        m = blk.mixer
        if isinstance(m, MambaMixer):
            m.conv_weight = mx.random.normal(m.conv_weight.shape) * 0.2
            m.A_log = mx.random.normal((cfg.mamba_num_heads,)) * 0.5
            m.dt_bias = mx.random.normal((cfg.mamba_num_heads,)) * 0.1
            m.D = mx.random.normal((cfg.mamba_num_heads,))
        elif isinstance(m, NemotronLatentMoE):
            e, lat, inter = cfg.n_routed_experts, cfg.moe_latent_size, cfg.moe_intermediate_size
            m.gate_weight = mx.random.normal((e, cfg.hidden_size))
            m.e_score_correction_bias = mx.random.normal((e,))
            m.up_stack = mx.random.normal((e, inter, lat)) * 0.1
            m.down_stack = mx.random.normal((e, lat, inter)) * 0.1


class _FakeBatchedRuntime:
    """Thin shim that ducks the :class:`NemotronBatchedResidentModel` surface against an
    in-memory :class:`NemotronModel` — bypasses the real-artifact loader so this test is fully
    model-free. Exposes exactly what :func:`batched_generate` consumes:
    ``cfg``/``max_batch``/``make_stream_state``/``prefill``/``step_batch``."""

    def __init__(self, model: NemotronModel, max_batch: int = 4) -> None:
        self._model = model
        self.max_batch = int(max_batch)

    @property
    def cfg(self) -> NemotronHConfig:
        return self._model.cfg

    def make_stream_state(self) -> tuple[list, list, list]:
        return make_stream_state(self.cfg)

    def prefill(self, prompt_ids: mx.array, state: tuple[list, list, list]) -> mx.array:
        caches, ssm, conv = state
        logits, _, _ = self._model(prompt_ids, caches=caches, ssm=ssm, conv=conv, use_fast=True)
        return logits

    def step_batch(
        self,
        stream_token_ids: list[mx.array],
        stream_caches: list[tuple[list, list, list]],
        offsets: list[int] | None = None,
    ) -> list[mx.array]:
        del offsets
        b = len(stream_token_ids)
        if b > self.max_batch:
            raise ValueError(f"B={b} exceeds max_batch={self.max_batch}")
        # Wrap each linear/embedding in an "embed_w-like" attribute pull — for the fake model
        # the embed/head live on nn.Linear/nn.Embedding objects rather than raw arrays.
        return batched_decode_step(
            layers=self._model.layers,
            embed_w=self._model.embed_tokens.weight,
            norm_f=self._model.norm_f.weight,
            lm_head_w=self._model.lm_head.weight,
            norm_eps=self.cfg.norm_eps,
            stream_token_ids=stream_token_ids,
            stream_caches=stream_caches,
        )


def _single_stream_step_seq(model: NemotronModel, ids: list[int]) -> mx.array:
    """Token-by-token single-stream decode through ``model``, returning ``[len(ids), vocab]`` logits.

    Uses the step-only state (zero conv-state so the O(1) step path engages from token 0,
    matching :mod:`parity.nemotron_decode_test`) so the comparison is apples-to-apples with the
    batched per-stream step. NOT the prefill path — that's covered by ``_test_batched_generate``."""
    caches, ssm, conv = make_step_state(model.cfg)
    rows: list[mx.array] = []
    for t in ids:
        logits, _, _ = model(mx.array([t]), caches=caches, ssm=ssm, conv=conv, use_fast=True)
        rows.append(logits[0, -1])
    return mx.stack(rows, axis=0)


def _batched_step_seq(rt: "_FakeBatchedRuntime", ids: list[int], b: int) -> list[mx.array]:
    """Step ``b`` identical streams through ``ids`` via ``step_batch`` (one token per step).
    Uses the step-only state per stream (zero conv-state from token 0) so this is apples-to-
    apples with the single-stream step sequence. Returns ``[per_stream_logits]`` shape
    ``[len(ids), vocab]`` each."""
    states = [make_step_state(rt.cfg) for _ in range(b)]
    per_stream: list[list[mx.array]] = [[] for _ in range(b)]
    for t in ids:
        token_ids = [mx.array([t]) for _ in range(b)]
        logits_list = rt.step_batch(token_ids, states)
        for s, logits in enumerate(logits_list):
            per_stream[s].append(logits[0, -1])
    return [mx.stack(rows, axis=0) for rows in per_stream]


def _test_step_batch_parity() -> None:
    """B=4 identical streams ⇒ each stream's per-position logits == single-stream logits."""
    mx.random.seed(0)
    cfg = _tiny_cfg()
    model = NemotronModel(cfg)
    _randomize(model)
    rt = _FakeBatchedRuntime(model, max_batch=4)

    ids = [3, 5, 7, 11, 13, 17, 19]  # arbitrary fixed token-ids (in-vocab for cfg.vocab_size=64)

    ref = _single_stream_step_seq(model, ids)
    per_stream = _batched_step_seq(rt, ids, b=4)

    # (1) every stream matches single-stream within bf16 accumulation tolerance — the
    # stacked-MoE path reorders matmul reductions slightly vs the per-row path, so we accept
    # ~1e-2 (same bf16 tolerance the existing decode-vs-prefill gate uses).
    rels = [_rel(s, ref) for s in per_stream]
    parity_ok = all(r < 1e-2 for r in rels)
    # (2) all streams equal each other (no cross-stream contamination — identical input rows
    # under identical state produce bit-identical compute graphs, so this is tight).
    cross_rels = [_rel(per_stream[i], per_stream[0]) for i in range(1, len(per_stream))]
    cross_ok = all(r < 1e-6 for r in cross_rels)

    print("\n=== Nemotron-H batched step_batch parity (B=4) ===")
    print(f"step_batch == single-stream (per-stream): {parity_ok}  rels={[f'{r:.2e}' for r in rels]}")
    print(f"cross-stream equality (B copies match)  : {cross_ok}")
    assert parity_ok, f"step_batch diverges from single-stream: rels={rels}"
    assert cross_ok, "streams diverge from each other under identical inputs"


def _test_batched_generate_parity() -> None:
    """batched_generate (greedy) on B=4 identical prompts == single-stream generate on the same prompt."""
    mx.random.seed(0)
    cfg = _tiny_cfg()
    model = NemotronModel(cfg)
    _randomize(model)
    rt = _FakeBatchedRuntime(model, max_batch=4)

    prompt = [3, 5, 7, 11]
    max_new = 5

    # single-stream reference (the existing decode path)
    ref_out = generate(model, prompt, max_new_tokens=max_new, temperature=0.0)

    # batched: 4 identical prompts, greedy
    bg_out = batched_generate(rt, [prompt] * 4, max_new=max_new, temperature=0.0)
    bg_match_ref = all(stream == ref_out for stream in bg_out)
    bg_cross_match = all(stream == bg_out[0] for stream in bg_out)

    print("\n=== Nemotron-H batched_generate parity (B=4, greedy) ===")
    print(f"single-stream ref tokens : {ref_out}")
    print(f"batched B=4 tokens       : {bg_out}")
    print(f"batched_generate == single-stream : {bg_match_ref}")
    print(f"all B streams equal each other     : {bg_cross_match}")
    assert bg_match_ref, f"batched_generate diverges from generate: bg={bg_out} ref={ref_out}"
    assert bg_cross_match, "batched_generate streams diverge from each other"


def _test_eos_stops_per_stream() -> None:
    """Per-stream eos stop: a stream that emits eos terminates without including it in output
    (matches single-stream :func:`generate`). Force eos as the next greedy token by replacing
    the head with a one-hot row at the eos id; max_new caps the unaffected streams."""
    mx.random.seed(0)
    cfg = _tiny_cfg()
    model = NemotronModel(cfg)
    _randomize(model)
    rt = _FakeBatchedRuntime(model, max_batch=4)

    # construct a forced-eos head: the lm_head returns logits where the eos position dominates.
    # NemotronModel.lm_head is an nn.Linear; we replace its weight with a row that makes eos win.
    eos = 1
    forced = mx.zeros(model.lm_head.weight.shape, dtype=model.lm_head.weight.dtype)
    # set the eos row to a large value across all hidden dims so dot(x, row) >> any other row
    forced[eos] = mx.ones((cfg.hidden_size,), dtype=forced.dtype) * 100.0
    model.lm_head.weight = forced

    bg_out = batched_generate(rt, [[3, 5, 7]] * 2, max_new=8, temperature=0.0, eos_ids=eos)
    eos_stops = all(eos not in stream and len(stream) == 0 for stream in bg_out)

    print("\n=== Nemotron-H batched_generate per-stream eos stop ===")
    print(f"eos stops with empty output (no eos in output)  : {eos_stops}  out={bg_out}")
    assert eos_stops, f"eos stop did not fire correctly: {bg_out}"


def _test_max_batch_admit() -> None:
    """N > max_batch prompts: continuous batching admits new prompts as slots free. With identical
    prompts + greedy decode + identical max_new, all streams produce the same tokens regardless of
    admit order. The slot-pool mechanics must NOT lose or reorder a prompt."""
    mx.random.seed(0)
    cfg = _tiny_cfg()
    model = NemotronModel(cfg)
    _randomize(model)
    rt = _FakeBatchedRuntime(model, max_batch=2)  # smaller than n_prompts to force admit cycling

    prompt = [3, 5, 7]
    n_prompts = 4
    max_new = 4
    ref_out = generate(model, prompt, max_new_tokens=max_new, temperature=0.0)
    bg_out = batched_generate(rt, [prompt] * n_prompts, max_new=max_new, temperature=0.0)
    all_match = (len(bg_out) == n_prompts) and all(stream == ref_out for stream in bg_out)

    print("\n=== Nemotron-H batched_generate continuous batching (N=4, max_batch=2) ===")
    print(f"all N prompts return == single-stream ref : {all_match}  n_out={len(bg_out)} ref={ref_out}")
    assert all_match, f"continuous batching lost/reordered a prompt: {bg_out} vs {ref_out}"


def _test_validation_errors() -> None:
    """Loud failures (rule-6): mismatched T_b across streams + B > max_batch must raise."""
    mx.random.seed(0)
    cfg = _tiny_cfg()
    model = NemotronModel(cfg)
    _randomize(model)
    rt = _FakeBatchedRuntime(model, max_batch=2)

    states = [rt.make_stream_state(), rt.make_stream_state()]

    # mixed T_b across streams: must raise (silent padding would change per-row counts)
    raised_mixed = False
    try:
        rt.step_batch([mx.array([3]), mx.array([3, 5])], states)
    except ValueError:
        raised_mixed = True

    # B > max_batch: must raise (the caller violated the contract)
    raised_oob = False
    try:
        rt.step_batch([mx.array([3])] * 4, [rt.make_stream_state()] * 4)
    except ValueError:
        raised_oob = True

    print("\n=== Nemotron-H batched runtime validation ===")
    print(f"mixed-T_b ValueError raised : {raised_mixed}")
    print(f"B > max_batch ValueError    : {raised_oob}")
    assert raised_mixed and raised_oob, "validation guards missing"


def _test_class_surface() -> None:
    """The class's documented surface (.cfg/.num_layers/.embed_w/.lm_head_w/.max_batch) is the
    contract :func:`batched_generate` consumes. The fake shim exposes it; verifying the real
    class compiles + has the right attrs would need a checkpoint, so verify the import + module
    surface here (the shim's parity is the functional gate)."""
    surface = (
        hasattr(NemotronBatchedResidentModel, "__init__")
        and hasattr(NemotronBatchedResidentModel, "step_batch")
        and hasattr(NemotronBatchedResidentModel, "prefill")
        and hasattr(NemotronBatchedResidentModel, "make_stream_state")
    )
    print("\n=== Nemotron-H NemotronBatchedResidentModel class surface ===")
    print(f"required methods present : {surface}")
    assert surface


def run() -> None:
    _test_step_batch_parity()
    _test_batched_generate_parity()
    _test_eos_stops_per_stream()
    _test_max_batch_admit()
    _test_validation_errors()
    _test_class_surface()
    print("\nNemotron-H batched runtime OK (step_batch B=4 == B=1×4; generate continuous-batches; "
          "per-stream eos + loud validation; class surface present)")


if __name__ == "__main__":
    run()
