"""Lossless gate for Qwen3.5 native-MTP spec-decode: spec_generate reproduces greedy decode.

MODEL-FREE — a STUB main model + a STUB MTP head over a tiny vocab (like the EAGLE / DSV4 fake
runtimes), with a stub decode cache supporting ``truncate`` / ``offset``. No checkpoint, no GPU, a few
KB of tensors — safe to run while a 398 GB capture is GPU-resident. The verify step makes losslessness
hold for ANY MTP quality (the head only changes *speed*), so this validates the draft → verify →
accept-or-bonus → rollback LOGIC, not the real weights.

The stub main model is a deterministic next-token chain ``next = t + STEP`` with a spike on that token,
so greedy decode is well-defined; per-position over a verify window ``[cur, draft]`` it returns
``greedy(cur)`` then ``greedy(draft)`` (exactly what the rollback logic consumes). Asserts:
  (1) ``spec_generate`` output is BIT-IDENTICAL to a plain greedy reference decode on the same stub main
      model (losslessness — the core invariant), for a perfect MTP AND a wrong MTP;
  (2) a correct-drafting MTP raises ``mean_accept`` (>1, →2 here), and an always-wrong MTP still
      reproduces greedy with ``mean_accept`` ≈ 1;
  (3) eos stops generation (the chain hits eos; spec terminates there, matching greedy), for an int eos
      AND an eos SET;
  (4) the stub cache's ``truncate`` was driven exactly as the rollback expects (≤1-token rollbacks for
      k==1) — proving the spec loop rolls the (recurrent-bearing) cache back losslessly;
plus a structural :func:`quanta.qwen35.mtp.mtp_forward` check on tiny real params (combine fuses
embed+hidden through ``fc``; the inherited block + readout produce finite, correctly-shaped logits).

    uv run --with numpy python -m parity.qwen35_mtp_spec_test

    # deferred (needs the resident Qwen3.5 model — do NOT run while another large job is resident):
    #   real MTP accept-rate / decode-speedup benchmark against Qwen35ResidentModel + the baked MTP
    #   head, asserting spec == greedy on real prose and reporting mean_accept.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.mtp import mtp_forward
from quanta.qwen35.model import Qwen35Block
from quanta.qwen35.spec import spec_generate

VOCAB = 64
HIDDEN = 8
NL = 3          # stub "decoder layers" — only cfg.num_hidden_layers matters to spec_generate
STEP = 3        # deterministic chain: greedy(t) = t + STEP
EOS = 40
MAXN = 16


def _greedy_next(t: int) -> int:
    return t + STEP


def _row(tok: int) -> mx.array:
    """A logit row over VOCAB with a clear argmax on ``tok`` (everything else far below)."""
    return mx.where(mx.arange(VOCAB) == tok, 30.0, -30.0)


class _StubCache:
    """Minimal stand-in for ``Qwen35Cache``: tracks a length, supports ``truncate`` / ``offset``.

    The stub main model ignores cache *contents* (its logits depend only on the input tokens), so the
    cache need only honor the rollback surface the spec loop drives. ``append`` advances the length;
    ``truncate`` rolls it back (and must be exact — the losslessness proof depends on it). Records each
    truncate so the test can confirm the spec loop's rollback pattern (≤1-token for k==1)."""

    def __init__(self) -> None:
        self._len = 0
        self.truncations: list[tuple[int, int]] = []   # (from_len, to_len)

    @property
    def offset(self) -> int:
        return self._len

    def append(self, n: int) -> None:
        self._len += n

    def truncate(self, length: int) -> None:
        if length < 0:
            raise ValueError(f"truncate length {length} < 0")
        if length < self._len:
            self.truncations.append((self._len, length))
            self._len = length


class _StubMainModel:
    """Deterministic stub of ``Qwen35ResidentModel``: greedy(t) = t + STEP, with the MTP-feature capture.

    ``__call__`` matches the consumed contract — ``(token_ids, *, caches, offset, capture_layers)`` ->
    ``(logits [1,T,vocab], {last: hidden [T,hidden]})``. It advances the stub cache by the input length
    (so ``offset`` stays consistent after rollbacks) and returns a deterministic per-position capture so
    the spec loop's feature plumbing is exercised. Records each call for the test to inspect."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(num_hidden_layers=NL)
        self.num_layers = NL
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def make_caches(self) -> _StubCache:
        return _StubCache()

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        ids = [int(x) for x in (token_ids.tolist() if isinstance(token_ids, mx.array) else token_ids)]
        self.calls.append((tuple(ids), offset))
        if caches is not None:
            caches.append(len(ids))
        t = len(ids)
        logits = mx.stack([_row(_greedy_next(tok)) for tok in ids])[None]   # [1,T,vocab]
        if not capture_layers:
            return logits
        last = max(capture_layers)
        feat = mx.broadcast_to(mx.arange(t, dtype=mx.float32)[:, None], (t, HIDDEN))  # [T,hidden]
        return logits, {last: feat}


class _PerfectMTP:
    """A drafter that always predicts the main model's greedy token from ``next_ids`` (= ``cur``) →
    every draft is accepted → mean_accept rises to 2. Mirrors ``mtp(prev_hidden, next_ids, embed,
    head)``; ignores ``prev_hidden`` content (only the *next* token determines the draft here)."""

    def __call__(self, prev_hidden, next_ids, embed, head):
        cur = int(next_ids[0, 0].item())
        return _row(_greedy_next(cur))[None, None]            # [1,1,vocab]


class _WrongMTP:
    """A drafter that always proposes a token the main model would NOT pick → every draft is rejected
    → mean_accept ≈ 1, yet the output is still bit-identical to greedy (the verify guarantees it)."""

    def __call__(self, prev_hidden, next_ids, embed, head):
        cur = int(next_ids[0, 0].item())
        wrong = (_greedy_next(cur) + 1) % VOCAB               # != greedy(cur)
        return _row(wrong)[None, None]


def _greedy_reference(model: _StubMainModel, prompt, max_new: int, eos_id) -> list[int]:
    """Plain greedy decode on the SAME stub main model — one token per forward, argmax each step,
    terminate at the first eos (inclusive). The bit-identity target for spec_generate."""
    stop = set() if eos_id is None else ({int(eos_id)} if isinstance(eos_id, int)
                                         else {int(s) for s in eos_id})
    caches = model.make_caches()
    logits = model(mx.array(prompt), caches=caches, offset=0)
    cur = int(mx.argmax(logits[0, -1]).item())
    out = [cur]
    q = len(prompt) - 1
    while len(out) < max_new and cur not in stop:
        logits = model(mx.array([cur]), caches=caches, offset=q + 1)
        q += 1
        cur = int(mx.argmax(logits[0, -1]).item())
        out.append(cur)
    out = out[:max_new]
    if stop:
        for k, t in enumerate(out):
            if t in stop:
                out = out[: k + 1]
                break
    return out


# --- structural mtp_forward check on tiny real params --------------------------------------------
def _tiny_cfg() -> Qwen35Config:
    return Qwen35Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=2,
        layer_types=("linear_attention", "full_attention"), full_attention_interval=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        attn_output_gate=True, partial_rotary_factor=0.25, rope_theta=1e7,
        mrope_section=(), mrope_interleaved=False, use_qk_norm=True,
        linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=4, mamba_ssm_dtype="float32",
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=8,
        shared_expert_intermediate_size=8, scoring_func="softmax", norm_topk_prob=True,
        router_aux_loss_coef=0.001, num_mtp_modules=1, mtp_use_dedicated_embeddings=False,
        hidden_act="silu", norm_eps=1e-6, max_position_embeddings=4096,
        eos_token_id=248046, eos_token_ids=(248046, 248044), pad_token_id=248044,
        tie_word_embeddings=False,
    )


def _structural_mtp_forward() -> bool:
    """mtp_forward over tiny real params: combine (fc fuses embed+hidden) + inherited block + readout
    produces finite, correctly-shaped logits ``[B,T,vocab]``."""
    mx.random.seed(7)
    cfg = _tiny_cfg()
    full_id = 1                                              # a full-attention layer for the MTP block
    blk = Qwen35Block(cfg, full_id)
    m = blk.mixer
    m.q_proj.weight = mx.random.normal(m.q_proj.weight.shape) * 0.1
    m.k_proj.weight = mx.random.normal(m.k_proj.weight.shape) * 0.1
    m.v_proj.weight = mx.random.normal(m.v_proj.weight.shape) * 0.1
    m.o_proj.weight = mx.random.normal(m.o_proj.weight.shape) * 0.1
    m.q_norm = mx.random.uniform(0.5, 1.5, (cfg.head_dim,))
    m.k_norm = mx.random.uniform(0.5, 1.5, (cfg.head_dim,))
    blk.mlp.gate = mx.random.normal(blk.mlp.gate.shape)
    blk.mlp.experts_gate_up = mx.random.normal(blk.mlp.experts_gate_up.shape) * 0.1
    blk.mlp.experts_down = mx.random.normal(blk.mlp.experts_down.shape) * 0.1
    blk.mlp.shared_gate_proj = mx.random.normal(blk.mlp.shared_gate_proj.shape) * 0.1
    blk.mlp.shared_up_proj = mx.random.normal(blk.mlp.shared_up_proj.shape) * 0.1
    blk.mlp.shared_down_proj = mx.random.normal(blk.mlp.shared_down_proj.shape) * 0.1
    blk.mlp.shared_expert_gate = mx.random.normal(blk.mlp.shared_expert_gate.shape)

    p = {
        "fc": mx.random.normal((HIDDEN, 2 * HIDDEN)) * 0.1,   # fuses [.,2*hidden] -> [.,hidden]
        "pre_fc_norm_embedding": mx.random.uniform(0.5, 1.5, (HIDDEN,)),
        "pre_fc_norm_hidden": mx.random.uniform(0.5, 1.5, (HIDDEN,)),
        "norm": mx.random.uniform(0.5, 1.5, (HIDDEN,)),
    }
    embed = mx.random.normal((VOCAB, HIDDEN)) * 0.1
    head = mx.random.normal((VOCAB, HIDDEN)) * 0.1

    T = 4
    prev_hidden = mx.random.normal((1, T, HIDDEN))
    next_ids = mx.random.randint(0, VOCAB, (1, T))
    logits = mtp_forward(prev_hidden, next_ids, embed, head, p, cfg, blk)
    shape_ok = logits.shape == (1, T, VOCAB)
    finite_ok = bool(mx.all(mx.isfinite(logits)).item())
    good = shape_ok and finite_ok
    print(f"  [{'OK' if good else 'FAIL'}] mtp_forward structural: shape={logits.shape} "
          f"(expect (1,{T},{VOCAB})) finite={finite_ok}")
    return good


def run() -> None:
    ok = True
    embed = mx.zeros((VOCAB, HIDDEN))   # unused by the stub MTP, but the real signature passes them
    head = mx.zeros((VOCAB, HIDDEN))
    prompt = [2, 5, 7]                  # last token 7 → chain 10,13,16,19,22,25,28,31,34,37,40(eos)

    greedy = _greedy_reference(_StubMainModel(), prompt, MAXN, eos_id=None)

    # (1)+(2a) perfect MTP: bit-identical to greedy AND mean_accept rises to 2
    m = _StubMainModel()
    spec_p, st_p = spec_generate(m, _PerfectMTP(), embed, head, prompt, max_new=MAXN, eos_id=None)
    good = spec_p == greedy and st_p["mean_accept"] > 1.0
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] perfect MTP: spec==greedy={spec_p == greedy} "
          f"mean_accept={st_p['mean_accept']:.2f} rounds={st_p['rounds']} k={st_p['k']}")
    print(f"             greedy[:10]={greedy[:10]}")
    print(f"             spec  [:10]={spec_p[:10]}")
    n_main = len(m.calls)
    good = n_main < len(spec_p)
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] perfect MTP fewer main forwards than tokens: "
          f"forwards={n_main} tokens={len(spec_p)}")

    # (1)+(2b) wrong MTP: still bit-identical to greedy, mean_accept ≈ 1
    mw = _StubMainModel()
    spec_w, st_w = spec_generate(mw, _WrongMTP(), embed, head, prompt, max_new=MAXN, eos_id=None)
    good = spec_w == greedy and abs(st_w["mean_accept"] - 1.0) < 1e-9
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] wrong MTP: spec==greedy={spec_w == greedy} "
          f"mean_accept={st_w['mean_accept']:.2f} (≈1)")

    # (3) eos stops generation — greedy and spec both terminate at the first eos (inclusive)
    greedy_e = _greedy_reference(_StubMainModel(), prompt, MAXN, eos_id=EOS)
    spec_e, st_e = spec_generate(_StubMainModel(), _PerfectMTP(), embed, head, prompt,
                                 max_new=MAXN, eos_id=EOS)
    good = (spec_e == greedy_e and len(spec_e) > 0 and spec_e[-1] == EOS and EOS not in spec_e[:-1])
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] eos stops: spec==greedy={spec_e == greedy_e} "
          f"ends_with_eos={spec_e[-1] == EOS if spec_e else False} len={len(spec_e)}")

    # eos as a SET is honored too, and the wrong MTP + eos still matches greedy's stop
    spec_set, _ = spec_generate(_StubMainModel(), _PerfectMTP(), embed, head, prompt,
                                max_new=MAXN, eos_id={EOS, 99})
    spec_we, _ = spec_generate(_StubMainModel(), _WrongMTP(), embed, head, prompt,
                               max_new=MAXN, eos_id=EOS)
    good = spec_set == greedy_e and spec_we == greedy_e
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] eos set + wrong MTP both match greedy stop: "
          f"set={spec_set == greedy_e} wrong={spec_we == greedy_e}")

    # (4 real) cache truncate pattern: drive a fresh perfect-MTP run with a recording cache and assert
    #     every rollback drops at most one token (the k==1 lossless-rollback contract).
    rec_model = _StubMainModel()
    rec_cache = _StubCache()
    spec_generate(_WrapCache(rec_model, rec_cache), _WrongMTP(), embed, head, prompt,
                  max_new=MAXN, eos_id=None)
    rb_ok = bool(rec_cache.truncations) and all(0 <= frm - to <= 1 for frm, to in rec_cache.truncations)
    ok = ok and rb_ok
    print(f"  [{'OK' if rb_ok else 'FAIL'}] rollback ≤1 token each round (k==1 lossless): "
          f"truncations={rec_cache.truncations}")

    # structural mtp_forward on tiny real params
    ok = ok and _structural_mtp_forward()

    print("PASS" if ok else "FAIL")
    assert ok


class _WrapCache:
    """A stub main model that uses a SHARED, test-provided cache (so the test can inspect its truncate
    pattern). Same deterministic chain as ``_StubMainModel``; ``make_caches`` returns the shared cache."""

    def __init__(self, base: _StubMainModel, cache: _StubCache) -> None:
        self.cfg = base.cfg
        self.num_layers = base.num_layers
        self._cache = cache

    def make_caches(self) -> _StubCache:
        return self._cache

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        ids = [int(x) for x in (token_ids.tolist() if isinstance(token_ids, mx.array) else token_ids)]
        if caches is not None:
            caches.append(len(ids))
        t = len(ids)
        logits = mx.stack([_row(_greedy_next(tok)) for tok in ids])[None]
        if not capture_layers:
            return logits
        last = max(capture_layers)
        feat = mx.broadcast_to(mx.arange(t, dtype=mx.float32)[:, None], (t, HIDDEN))
        return logits, {last: feat}


if __name__ == "__main__":
    run()
