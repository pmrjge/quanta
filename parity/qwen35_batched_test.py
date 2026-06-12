"""Gate: Qwen3.5 batched (B>1) decode step == single-stream decode step — model-free, tiny tensors.

MODEL-FREE per the project safety rule (a 398 GB capture may be GPU-resident). Builds a tiny random
:class:`quanta.qwen35.model.Qwen35Model` (small hidden, few heads/experts, 3 layers — linear +
full + linear so BOTH mixer regimes are exercised) and wraps its layers / embed / norm / lm_head
via :meth:`quanta.qwen35.batched_runtime.Qwen35BatchedResidentModel.from_inner` to drive the same
``step_batch`` machinery the real runtime uses. Few-KB of params; safe to run anytime.

The invariant under test (Design A — per-stream caches, stack-for-MoE):

  ``step_batch([tok]*B, [fresh_cache]*B, [off]*B)[b]  ==  single_stream_step(tok, fresh_cache, off)``
  for every stream b, at every step of a multi-step decode, for B in {1, 2, 4}.

Concretely:

* (a) B=4 identical prompts → per-stream prefill logits ``[1,1,vocab]`` all equal (and all equal
      a single-stream prefill of the same prompt). Verifies the per-stream prefill is independent.
* (b) B=4 identical prompts → multi-step decode: at every step, every stream's logits equal a
      reference single-stream decode at the same offset. Verifies that batching does not perturb
      any stream's forward (the per-stream mixer step + the stacked MoE call are output-equivalent
      to the single-stream forward — rule 4).
* (c) B=4 DIFFERENT prompts → per-stream output equals the corresponding single-stream output for
      each stream independently. Verifies that streams do not leak through the shared MoE call (a
      bug in which one stream's hidden bled into another's would surface as a non-zero diff).
* (d) Per-stream caches are independent: rolling stream 0 back losslessly via ``Qwen35Cache.truncate``
      does NOT touch streams 1..3's caches (a per-stream cache-isolation gate).
* (e) ``step_batch`` raises ``ValueError`` when ``cache.offset != declared offset`` (rule-6 fail-loud
      on an orchestrator desync — never silently advance a diverged stream).
* (f) ``prefill`` routes prompts >= ``QWEN35_CHUNKED_PREFILL_FROM`` through the N3-3 chunked path
      (observed via a spy, default-ON pinned, threshold > the real gates' 32-token seeds; ``None``
      restores per-token everywhere; below-threshold stays bit-identical per-token seeding).

    uv run --with numpy python -m parity.qwen35_batched_test

The real-model throughput sweep (B ∈ {1,2,4,8,16,32}) lives in ``parity/qwen35_batched_bench.py``;
this gate is the parity prerequisite for that bench.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.qwen35.attention import Qwen35Attention
from quanta.qwen35.batched_runtime import Qwen35BatchedResidentModel
from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.gated_deltanet import GatedDeltaNet
from quanta.qwen35.model import Qwen35Model

LAYER_TYPES = ("linear_attention", "full_attention", "linear_attention")


def _cfg() -> Qwen35Config:
    return Qwen35Config(
        vocab_size=64, hidden_size=32, num_hidden_layers=len(LAYER_TYPES),
        layer_types=LAYER_TYPES, full_attention_interval=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        attn_output_gate=True, partial_rotary_factor=0.25, rope_theta=1e7,
        mrope_section=(), mrope_interleaved=False, use_qk_norm=True,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=4, mamba_ssm_dtype="float32",
        num_experts=8, num_experts_per_tok=3, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, scoring_func="softmax", norm_topk_prob=True,
        router_aux_loss_coef=0.001, num_mtp_modules=1, mtp_use_dedicated_embeddings=False,
        hidden_act="silu", norm_eps=1e-6, max_position_embeddings=4096,
        eos_token_id=248046, eos_token_ids=(248046, 248044), pad_token_id=248044,
        tie_word_embeddings=False,
    )


def _rel(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))
                 / (mx.max(mx.abs(b.astype(mx.float32))) + 1e-6))


def _maxdiff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def _build_random_model(cfg: Qwen35Config, seed: int = 0) -> Qwen35Model:
    """A tiny Qwen35Model with random weights for both linear + full mixers + MoE — model-free."""
    mx.random.seed(seed)
    model = Qwen35Model(cfg)
    for blk in model.layers:
        m = blk.mixer
        blk.mlp.gate = mx.random.normal(blk.mlp.gate.shape)
        blk.mlp.experts_gate_up = mx.random.normal(blk.mlp.experts_gate_up.shape) * 0.1
        blk.mlp.experts_down = mx.random.normal(blk.mlp.experts_down.shape) * 0.1
        blk.mlp.shared_gate_proj = mx.random.normal(blk.mlp.shared_gate_proj.shape) * 0.1
        blk.mlp.shared_up_proj = mx.random.normal(blk.mlp.shared_up_proj.shape) * 0.1
        blk.mlp.shared_down_proj = mx.random.normal(blk.mlp.shared_down_proj.shape) * 0.1
        blk.mlp.shared_expert_gate = mx.random.normal(blk.mlp.shared_expert_gate.shape)
        if isinstance(m, GatedDeltaNet):
            m.in_proj_qkv.weight = mx.random.normal(m.in_proj_qkv.weight.shape) * 0.1
            m.in_proj_a.weight = mx.random.normal(m.in_proj_a.weight.shape) * 0.1
            m.in_proj_b.weight = mx.random.normal(m.in_proj_b.weight.shape) * 0.1
            m.in_proj_z.weight = mx.random.normal(m.in_proj_z.weight.shape) * 0.1
            m.out_proj.weight = mx.random.normal(m.out_proj.weight.shape) * 0.1
            m.conv_weight = mx.random.normal(m.conv_weight.shape) * 0.2
            m.A_log = mx.random.normal((cfg.linear_num_value_heads,)) * 0.5
            m.dt_bias = mx.random.normal((cfg.linear_num_value_heads,)) * 0.1
            m.norm = mx.random.uniform(0.5, 1.5, (cfg.linear_value_head_dim,))
            m.chunk = 4
        elif isinstance(m, Qwen35Attention):
            m.q_proj.weight = mx.random.normal(m.q_proj.weight.shape) * 0.1
            m.k_proj.weight = mx.random.normal(m.k_proj.weight.shape) * 0.1
            m.v_proj.weight = mx.random.normal(m.v_proj.weight.shape) * 0.1
            m.o_proj.weight = mx.random.normal(m.o_proj.weight.shape) * 0.1
            m.q_norm = mx.random.uniform(0.5, 1.5, (cfg.head_dim,))
            m.k_norm = mx.random.uniform(0.5, 1.5, (cfg.head_dim,))
    model.lm_head.weight = mx.random.normal(model.lm_head.weight.shape) * 0.1
    return model


def _wrap_batched(model: Qwen35Model, *, max_batch: int = 8, packed: bool = False,
                  loopkill: bool | None = None) -> Qwen35BatchedResidentModel:
    """Wrap a tiny model into the batched runtime contract WITHOUT loading any artifact (the
    test's whole point is to be model-free). Reuses ``from_inner`` so the batched step machinery
    is exactly the production code path.

    ``packed`` declares whether ``model``'s mixer projections are held packed (``nn.QuantizedLinear``)
    — it must be ``True`` to run the #153 loop-kill (``loopkill ⇒ packed``). ``loopkill`` overrides the
    instance flag (``None`` ⇒ the graduated default, now ON): a bf16 per-stream test passes
    ``loopkill=False`` so it can construct without packing; the loopkill gate packs its layers
    (``packed=True``) and lets the default stand."""
    return Qwen35BatchedResidentModel.from_inner(
        layers=model.layers,
        embed_w=model.embed_tokens.weight,
        norm_w=model.norm.weight,
        lm_head_w=model.lm_head.weight,
        cfg=model.cfg,
        max_batch=max_batch,
        packed=packed,
        loopkill=loopkill,
    )


def _single_stream_decode(bm: Qwen35BatchedResidentModel, prompt_ids: list[int],
                          n_decode: int) -> tuple[list[mx.array], list[mx.array]]:
    """Drive ``step_batch`` with B=1 over (prefill + ``n_decode`` decode steps), one token at a
    time. Returns (per_position_prefill_logits, per_step_decode_logits).

    This is the parity REFERENCE for the multi-stream tests — by construction it uses the SAME
    step machinery the batched path uses (just with B=1), so any divergence between B=1 and B>1
    isolates a batching bug (e.g. cross-stream MoE bleed) rather than a runtime/reference drift.
    """
    cache = bm.make_caches()
    pf_logits: list[mx.array] = []
    for pos, tid in enumerate(prompt_ids):
        lg = bm.step_batch([int(tid)], [cache], [pos])[0]
        mx.eval(lg)
        pf_logits.append(lg)
    # decode a few extra tokens by feeding the argmax forward (deterministic; no sampling — we're
    # checking forward parity, not the sampler).
    dec_logits: list[mx.array] = []
    tok = int(mx.argmax(pf_logits[-1][0, -1]).item())
    off = len(prompt_ids)
    for _ in range(n_decode):
        lg = bm.step_batch([tok], [cache], [off])[0]
        mx.eval(lg)
        dec_logits.append(lg)
        tok = int(mx.argmax(lg[0, -1]).item())
        off += 1
    return pf_logits, dec_logits


def _test_b_eq_1_prefill_match(bm: Qwen35BatchedResidentModel) -> bool:
    """Sanity: B=1 prefill via step_batch == B=1 prefill via the dedicated `prefill()` method
    (the per-position one-token-at-a-time seeding the two share)."""
    prompt = [3, 9, 15, 27, 42]
    cache_a = bm.make_caches()
    last = bm.prefill(prompt, cache_a)              # [1,1,vocab] at the last position
    pf, _ = _single_stream_decode(bm, prompt, n_decode=0)
    d = _maxdiff(last, pf[-1])
    ok = d < 1e-5 and cache_a.offset == len(prompt)
    print(f"  [{'OK' if ok else 'FAIL'}] B=1 prefill() == B=1 step_batch loop  |Δ|={d:.2e} "
          f"off={cache_a.offset}")
    return ok


def _test_chunked_prefill_routing(bm: Qwen35BatchedResidentModel) -> bool:
    """``prefill`` routes long prompts through the N3-3 chunked path (graduated ON) and keeps the
    bit-identical per-token seeding below the threshold. The routing is OBSERVED (a spy on the
    late-imported :func:`quanta.qwen35.runtime.chunked_prefill`), not inferred from numerics; the
    module constant + spy are restored on exit. Real-weight greedy-exactness of the chunked path
    itself is the N3-3 SOLO gate (``nex_n2_pro_prefill_chunked_real``)."""
    import quanta.qwen35.batched_runtime as qbr
    import quanta.qwen35.runtime as qrt

    thr = qbr.QWEN35_CHUNKED_PREFILL_FROM
    # default-ON pin (fail loud on a silent revert) + the threshold must stay ABOVE every existing
    # bit-exact gate's prompt length (the real Design-A B=1 gates seed 32-token prompts —
    # ``nex_n2_pro_batched_real.SEED_LEN``; lowering the default past 32 silently flips those
    # bit-exact assertions into the chunked ULP class).
    ok_pin = thr is not None and thr > 32
    print(f"  [{'OK' if ok_pin else 'FAIL'}] chunked-prefill routing default-ON pin "
          f"(thr={thr}, must be int > 32)")
    if not ok_pin:
        return False

    prompt = [3, 9, 15, 27, 42, 8, 33, 21]                 # 8 tokens
    calls: list[int] = []
    orig_cp = qrt.chunked_prefill

    def spy(layers, embed_w, norm_w, lm_head_w, cfg, token_ids, caches, **kw):
        calls.append(len(list(token_ids)))
        return orig_cp(layers, embed_w, norm_w, lm_head_w, cfg, token_ids, caches, **kw)

    try:
        qrt.chunked_prefill = spy
        # routed: threshold lowered to the prompt length -> prefill() delegates to the chunked path
        qbr.QWEN35_CHUNKED_PREFILL_FROM = len(prompt)
        c_r = bm.make_caches()
        lg_r = bm.prefill(prompt, c_r)
        routed = calls == [len(prompt)] and c_r.offset == len(prompt)
        # same path direct == routed, bit-exact (identical computation on a fresh cache)
        c_d = bm.make_caches()
        lg_d = bm.prefill_chunked(prompt, c_d)
        d_direct = _maxdiff(lg_r, lg_d)
        # one token below the threshold: the per-token loop, chunked NOT called
        calls.clear()
        short = prompt[:-1]
        c_s = bm.make_caches()
        lg_s = bm.prefill(short, c_s)
        pf, _ = _single_stream_decode(bm, short, n_decode=0)
        below = calls == [] and _maxdiff(lg_s, pf[-1]) < 1e-5
        # OFF (None): never routes, even for the at-threshold prompt
        qbr.QWEN35_CHUNKED_PREFILL_FROM = None
        calls.clear()
        c_o = bm.make_caches()
        lg_o = bm.prefill(prompt, c_o)
        off_never = calls == []
        # routed vs per-token: greedy-equal (the rule-4 arbiter for the chunked regime — the
        # chunked WY arm reorders the fp32 recurrence vs the per-token scan, ~5e-3 abs on this
        # tiny model; the bound is a sanity rail, greedy agreement is the gate)
        agree = (int(mx.argmax(lg_r[0, -1]).item()) == int(mx.argmax(lg_o[0, -1]).item())
                 and _maxdiff(lg_r, lg_o) < 2e-2)
    finally:
        qbr.QWEN35_CHUNKED_PREFILL_FROM = thr
        qrt.chunked_prefill = orig_cp

    ok = routed and d_direct == 0.0 and below and off_never and agree
    print(f"  [{'OK' if ok else 'FAIL'}] prefill() routing: >=thr -> chunked (observed, "
          f"|Δ|direct={d_direct:.1e}), <thr -> per-token, None -> never; routed greedy == per-token")
    return ok


def _test_identical_streams(bm: Qwen35BatchedResidentModel, B: int) -> bool:
    """B identical prompts -> per-stream prefill + decode logits all equal each other AND all equal
    the single-stream reference. This is the core parity gate (the per-stream mixer step + the
    stacked MoE call must be output-equivalent to single-stream — rule 4)."""
    prompt = [5, 19, 31, 11, 7]
    ref_pf, ref_dec = _single_stream_decode(bm, prompt, n_decode=4)

    # B-stream prefill (one token per stream per step, all feeding the SAME prompt id sequence)
    caches = bm.make_batch_caches(B)
    bs_pf: list[list[mx.array]] = [[] for _ in range(B)]
    for pos, tid in enumerate(prompt):
        per_stream = bm.step_batch([int(tid)] * B, caches, [pos] * B)
        mx.eval(per_stream)
        for s in range(B):
            bs_pf[s].append(per_stream[s])

    # B-stream decode: each stream feeds its own argmax (which is the same across streams given
    # identical prompts + identical caches, so they stay locked).
    bs_dec: list[list[mx.array]] = [[] for _ in range(B)]
    toks = [int(mx.argmax(bs_pf[s][-1][0, -1]).item()) for s in range(B)]
    off = len(prompt)
    for _ in range(4):
        per_stream = bm.step_batch(toks, caches, [off] * B)
        mx.eval(per_stream)
        for s in range(B):
            bs_dec[s].append(per_stream[s])
        toks = [int(mx.argmax(per_stream[s][0, -1]).item()) for s in range(B)]
        off += 1

    # invariant 1: every stream's prefill matches the single-stream prefill
    pf_diffs = [max(_maxdiff(bs_pf[s][p], ref_pf[p]) for p in range(len(prompt))) for s in range(B)]
    # invariant 2: every stream's decode matches the single-stream decode
    dec_diffs = [max(_maxdiff(bs_dec[s][p], ref_dec[p]) for p in range(4)) for s in range(B)]
    # invariant 3: streams agree with each other (no MoE cross-stream bleed). For B=1 there is no
    # cross-stream comparison to make (the test still gates invariants 1+2 — single==single).
    if B > 1:
        cross = max(_maxdiff(bs_dec[0][p], bs_dec[s][p]) for s in range(1, B) for p in range(4))
    else:
        cross = 0.0
    max_pf = max(pf_diffs)
    max_dec = max(dec_diffs)
    # rule-4 parity: same math, same weights — diffs must be at fp tolerance (a real bug would be
    # orders of magnitude larger; the linear-attention recurrence runs in fp32 so the bf16
    # numerical floor is well below 1e-3).
    ok = max_pf < 5e-3 and max_dec < 5e-3 and cross < 5e-3
    print(f"  [{'OK' if ok else 'FAIL'}] B={B} identical prompts == single-stream  "
          f"pf|Δ|={max_pf:.2e} dec|Δ|={max_dec:.2e} cross|Δ|={cross:.2e}")
    return ok


def _test_different_streams(bm: Qwen35BatchedResidentModel, B: int = 4) -> bool:
    """B DIFFERENT prompts -> each stream matches its own single-stream reference. Catches any
    cross-stream leak through the MoE / shared expert (a bug there would scramble different-prompt
    streams while leaving identical-prompt streams unaffected)."""
    mx.random.seed(7)
    cfg_vocab = bm.cfg.vocab_size
    L = 4
    n_dec = 3
    # construct B distinct prompts (different sequences so the per-stream forward truly differs)
    prompts = [
        [int(mx.random.randint(0, cfg_vocab, (1,)).item()) for _ in range(L)]
        for _ in range(B)
    ]
    # reference: B independent single-stream runs
    refs_pf: list[list[mx.array]] = []
    refs_dec: list[list[mx.array]] = []
    for p in prompts:
        rpf, rdc = _single_stream_decode(bm, p, n_decode=n_dec)
        refs_pf.append(rpf)
        refs_dec.append(rdc)

    # batched: B streams driven in lockstep with DIFFERENT prompts (every step feeds each stream
    # its own next token).
    caches = bm.make_batch_caches(B)
    bs_pf: list[list[mx.array]] = [[] for _ in range(B)]
    for pos in range(L):
        toks = [prompts[s][pos] for s in range(B)]
        per_stream = bm.step_batch(toks, caches, [pos] * B)
        mx.eval(per_stream)
        for s in range(B):
            bs_pf[s].append(per_stream[s])

    bs_dec: list[list[mx.array]] = [[] for _ in range(B)]
    toks = [int(mx.argmax(bs_pf[s][-1][0, -1]).item()) for s in range(B)]
    off = L
    for _ in range(n_dec):
        per_stream = bm.step_batch(toks, caches, [off] * B)
        mx.eval(per_stream)
        for s in range(B):
            bs_dec[s].append(per_stream[s])
        toks = [int(mx.argmax(per_stream[s][0, -1]).item()) for s in range(B)]
        off += 1

    # each stream's batched output must equal its own single-stream reference (no cross-leak)
    pf_diffs = [max(_maxdiff(bs_pf[s][p], refs_pf[s][p]) for p in range(L)) for s in range(B)]
    dec_diffs = [max(_maxdiff(bs_dec[s][p], refs_dec[s][p]) for p in range(n_dec)) for s in range(B)]
    max_pf, max_dec = max(pf_diffs), max(dec_diffs)
    ok = max_pf < 5e-3 and max_dec < 5e-3
    print(f"  [{'OK' if ok else 'FAIL'}] B={B} DIFFERENT prompts each == own single-stream  "
          f"pf|Δ|={max_pf:.2e} dec|Δ|={max_dec:.2e}")
    return ok


def _test_per_stream_isolation(bm: Qwen35BatchedResidentModel) -> bool:
    """Per-stream cache isolation: rollback on stream 0 must not perturb streams 1..3.

    Runs B=4 different prompts to a common offset, then ``truncate``s stream 0's cache (recurrent
    snapshot restore + KV slice — joint, lossless) and asserts streams 1..3 continue identically to
    a fresh batched run of just those 3. Catches a bug where ``Qwen35Cache.truncate`` shared state
    across streams (which it does not — every stream has its own cache instance — but the test
    keeps the invariant explicit)."""
    mx.random.seed(8)
    V = bm.cfg.vocab_size
    L = 4
    n_dec = 2
    prompts = [[int(mx.random.randint(0, V, (1,)).item()) for _ in range(L)] for _ in range(4)]
    caches = bm.make_batch_caches(4)
    # consume each prompt
    for pos in range(L):
        toks = [prompts[s][pos] for s in range(4)]
        per_stream = bm.step_batch(toks, caches, [pos] * 4)
        mx.eval(per_stream)
    # rollback stream 0 by 1 token (the k=1 spec rollback case — within the snapshot window)
    caches[0].truncate(L - 1)
    isolated_offs_ok = all(c.offset == L for c in caches[1:]) and caches[0].offset == L - 1

    # now continue streams 1..3 in lockstep (skip stream 0 — it's mid-rollback) and check the
    # other streams' outputs equal a fresh batched run from the same prompts.
    sub_caches = [caches[1], caches[2], caches[3]]
    sub_dec: list[list[mx.array]] = [[], [], []]
    toks = [prompts[s][-1] for s in (1, 2, 3)]   # dummy continuation (test cares about parity)
    off = L
    for _ in range(n_dec):
        per_stream = bm.step_batch(toks, sub_caches, [off] * 3)
        mx.eval(per_stream)
        for s in range(3):
            sub_dec[s].append(per_stream[s])
        toks = [int(mx.argmax(per_stream[s][0, -1]).item()) for s in range(3)]
        off += 1

    # fresh batched run of just those 3 prompts — should match
    fresh_caches = bm.make_batch_caches(3)
    for pos in range(L):
        fresh_toks = [prompts[s][pos] for s in (1, 2, 3)]
        per_stream = bm.step_batch(fresh_toks, fresh_caches, [pos] * 3)
        mx.eval(per_stream)
    ref_dec: list[list[mx.array]] = [[], [], []]
    toks = [prompts[s][-1] for s in (1, 2, 3)]
    off = L
    for _ in range(n_dec):
        per_stream = bm.step_batch(toks, fresh_caches, [off] * 3)
        mx.eval(per_stream)
        for s in range(3):
            ref_dec[s].append(per_stream[s])
        toks = [int(mx.argmax(per_stream[s][0, -1]).item()) for s in range(3)]
        off += 1

    # parity of streams 1..3 continuation
    d = max(_maxdiff(sub_dec[s][p], ref_dec[s][p]) for s in range(3) for p in range(n_dec))
    # check stream 0 also rolled cleanly: feeding [prompts[0][L-1], ...] resumes from offset L-1
    # equivalent to a fresh stream consuming the full prompt then truncating to L-1.
    fresh0 = bm.make_caches()
    for pos in range(L):
        bm.step_batch([prompts[0][pos]], [fresh0], [pos])
    fresh0.truncate(L - 1)
    isolated_off_ok = fresh0.offset == L - 1 == caches[0].offset
    ok = d < 5e-3 and isolated_offs_ok and isolated_off_ok
    print(f"  [{'OK' if ok else 'FAIL'}] per-stream cache isolation (rollback stream 0 only)  "
          f"|Δ|={d:.2e} offs_ok={isolated_offs_ok} truncate_ok={isolated_off_ok}")
    return ok


def _test_fail_loud_desync(bm: Qwen35BatchedResidentModel) -> bool:
    """``step_batch`` must RAISE ValueError on a per-stream offset desync (rule 6, no silent advance)."""
    caches = bm.make_batch_caches(2)
    # advance one stream by one token via a single step — caches[0].offset becomes 1
    bm.step_batch([3, 5], caches, [0, 0])
    # now try to step with offsets that LIE about the cache state -> must raise
    raised = False
    try:
        bm.step_batch([7, 7], caches, [0, 0])     # caches[0].offset is 1, declared is 0 -> bad
    except ValueError:
        raised = True
    # also: a too-large batch must raise
    raised_batch = False
    try:
        bm.step_batch([1] * (bm.max_batch + 1),
                      [bm.make_caches() for _ in range(bm.max_batch + 1)],
                      [0] * (bm.max_batch + 1))
    except ValueError:
        raised_batch = True
    # mismatched lengths
    raised_len = False
    try:
        bm.step_batch([1, 2], [bm.make_caches()], [0, 0])
    except ValueError:
        raised_len = True
    ok = raised and raised_batch and raised_len
    print(f"  [{'OK' if ok else 'FAIL'}] fail-loud rule-6: desync={raised} oversized={raised_batch} "
          f"mismatched_len={raised_len}")
    return ok


def _test_generate_batched_matches_single(bm: Qwen35BatchedResidentModel) -> bool:
    """End-to-end: :func:`quanta.qwen35.batched_generate.generate_batched` over B=4 identical
    prompts emits the same tokens as :func:`quanta.qwen35.generate.generate` on one prompt.

    Greedy (temperature=0) so the comparison is exact — argmax at every step. Verifies the
    continuous-batching loop + per-stream sampling re-uses :func:`sample_logits` identically.

    Also gates the **trailing cache-advance**: after both generators return, the per-stream
    batched cache offsets must equal the single-stream cache offset (== ``len(prompt) + len(out)``).
    Single-stream :func:`generate` is feed-then-sample, so the last emitted token is committed
    to its cache; ``generate_batched`` is sample-then-feed in the inner loop, so it must do ONE
    trailing ``step_batch`` to align. Without that pass the orchestrator's next batched call on
    the same caches would desync (caught loudly by :meth:`step_batch`'s offset check — rule 6 —
    but still a behavioral divergence).
    """
    from quanta.qwen35.batched_generate import generate_batched
    from quanta.qwen35.generate import generate

    prompt = [2, 11, 23, 17]
    max_new = 6
    # single-stream greedy — capture the cache so we can read its end offset.
    single_cache = bm.make_caches()
    single = generate(bm, prompt, max_new_tokens=max_new, temperature=0.0, cache=single_cache)
    # batched: 4 identical prompts; per-stream output should equal `single`. Pass external caches
    # so we can inspect their end offsets after the call.
    stream_caches = [bm.make_caches() for _ in range(4)]
    batched_outs = generate_batched(bm, [prompt] * 4, max_new_tokens=max_new, temperature=0.0,
                                    caches=stream_caches)
    all_equal = all(o == single for o in batched_outs)
    single_off = single_cache.offset
    stream_offs = [c.offset for c in stream_caches]
    offsets_match = all(off == single_off for off in stream_offs)
    expected_off = len(prompt) + len(single)
    offsets_correct = single_off == expected_off
    ok = all_equal and offsets_match and offsets_correct
    print(f"  [{'OK' if ok else 'FAIL'}] generate_batched B=4 == generate B=1 (greedy)  "
          f"single={single}  per_stream_match={all_equal}  "
          f"single_off={single_off} stream_offs={stream_offs} "
          f"(expected {expected_off} = len(prompt)+len(out))")
    return ok


def run() -> None:
    cfg = _cfg()
    model = _build_random_model(cfg, seed=1)
    # This gate validates the per-stream Design-A path (per-stream caches + batched MoE == single-
    # stream) on bf16 layers; pin loopkill=False so it constructs without packing now that the
    # loop-kill default is ON (the loop-kill path is gated separately in qwen35_batched_loopkill_test).
    bm = _wrap_batched(model, max_batch=8, loopkill=False)
    ok = True
    print("\n=== Qwen3.5 batched (B>1) decode parity (tiny, model-free) ===")
    ok &= _test_b_eq_1_prefill_match(bm)
    ok &= _test_chunked_prefill_routing(bm)
    ok &= _test_identical_streams(bm, B=1)
    ok &= _test_identical_streams(bm, B=2)
    ok &= _test_identical_streams(bm, B=4)
    ok &= _test_different_streams(bm, B=4)
    ok &= _test_per_stream_isolation(bm)
    ok &= _test_fail_loud_desync(bm)
    ok &= _test_generate_batched_matches_single(bm)
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    run()
