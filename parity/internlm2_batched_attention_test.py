"""Model-free parity gate for InternLM2.5 batched-decode attention (Approach-1, the default decode path).

Proves the fused :meth:`quanta.internlm2.model.InternLM2Model.decode_batched` (one batched SDPA +
batched matmuls across ``B`` streams) is equivalent to the per-stream ``step_batch`` loop it replaces —
on a tiny random-init bf16 :class:`InternLM2Model` (NO checkpoint, NO GPU). Streams are seeded at
**different prefill lengths** (heterogeneous RoPE offsets — the case that forced the batched explicit
RoPE) and decoded for several steps:

A. **core** — ``decode_batched`` vs looping single-stream ``__call__`` per stream, bf16 + int8-g32 KV.
B. **runtime dispatch** — ``InternLM2BatchedResidentModel.step_batch`` (fused default) vs the retained
   ``_step_batch_looped`` reference, through the real batched runtime.

The arbiter is **greedy-token agreement** (the decode that actually ships): every stream must emit the
identical next-token sequence as the loop. Logits match to fp ULPs (batched explicit RoPE + padded-SDPA
tiling) — argmax-stable, the same equivalence class the project accepts for tiled/batched paths.

    uv run --with numpy python -m parity.internlm2_batched_attention_test
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from quanta.internlm2.attention import KVCache
from quanta.internlm2.batched_runtime import InternLM2BatchedResidentModel
from quanta.internlm2.config import InternLM2Config
from quanta.internlm2.decode import InternLM2Cache
from quanta.internlm2.model import InternLM2Model

STEPS = 6
LOGIT_TOL = 5e-3  # bf16: RoPE + padded-SDPA tiling reorder; the hard gate is greedy-token agreement


def _tiny_cfg() -> InternLM2Config:
    return InternLM2Config(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, attention_bias=False,
        rope_theta=1.0e4, rope_scaling_type="dynamic", rope_scaling_factor=2.5,
        max_position_embeddings=4096, hidden_act="silu", norm_eps=1e-5, tie_word_embeddings=False,
        eos_token_id=2, eos_token_ids=(2,), pad_token_id=2, bos_token_id=1, add_bos_token=True,
    )


def _bf16_model(cfg: InternLM2Config) -> InternLM2Model:
    from mlx.utils import tree_map

    model = InternLM2Model(cfg)
    model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
    mx.eval(model.parameters())
    return model


def _prompt(cfg: InternLM2Config, n: int) -> mx.array:
    return mx.random.randint(0, cfg.vocab_size, (n,))


def _layer_caches(cfg: InternLM2Config, quantized: bool, bits: int, gs: int) -> list[KVCache]:
    return [KVCache(quantized=quantized, group_size=gs, bits=bits)
            for _ in range(cfg.num_hidden_layers)]


def _seed(model: InternLM2Model, cfg: InternLM2Config, prompt: mx.array, *,
          quantized: bool, bits: int, gs: int) -> tuple[list[KVCache], int, int]:
    """Prefill ``prompt`` into a fresh per-layer cache list; return (caches, offset, first decode token)."""
    caches = _layer_caches(cfg, quantized, bits, gs)
    logits = model(prompt[None], caches=caches, use_fast=True, abs_pos_start=0, last_only=True)
    tok = int(mx.argmax(logits[0, -1]).item())
    return caches, int(prompt.shape[0]), tok


def _core(quantized: bool, bits: int) -> None:
    """A: decode_batched == per-stream __call__ loop over STEPS decode steps (ragged offsets)."""
    mx.random.seed(0)
    cfg = _tiny_cfg()
    model = _bf16_model(cfg)
    gs = 32 if quantized else 64
    lengths = [13, 7, 10]                       # heterogeneous prefill lengths -> different offsets
    prompts = [_prompt(cfg, n) for n in lengths]

    # two independent cache sets (prefill twice, identically) so fused + loop don't share mutated state
    loop = [_seed(model, cfg, p, quantized=quantized, bits=bits, gs=gs) for p in prompts]
    fuse = [_seed(model, cfg, p, quantized=quantized, bits=bits, gs=gs) for p in prompts]
    l_caches = [c for c, _, _ in loop]
    l_off = [o for _, o, _ in loop]
    l_tok = [t for _, _, t in loop]
    f_caches = [c for c, _, _ in fuse]
    f_off = [o for _, o, _ in fuse]
    f_tok = [t for _, _, t in fuse]

    worst = 0.0
    tokens_match = True
    for _ in range(STEPS):
        # looped reference: one single-stream forward per stream
        l_logits = []
        for s in range(len(prompts)):
            out = model(mx.array([[l_tok[s]]]), caches=l_caches[s], use_fast=True,
                        abs_pos_start=l_off[s], last_only=True)[0, -1]
            l_logits.append(out)
            l_tok[s] = int(mx.argmax(out).item())
            l_off[s] += 1
        # fused: one batched forward across all streams
        fb = model.decode_batched([mx.array([t]) for t in f_tok], f_caches, f_off)  # [B,1,vocab]
        mx.eval(fb)
        for s in range(len(prompts)):
            fout = fb[s, -1]
            worst = max(worst, float(mx.max(mx.abs(fout - l_logits[s]))))
            ftok = int(mx.argmax(fout).item())
            tokens_match = tokens_match and (ftok == l_tok[s])
            f_tok[s] = ftok
            f_off[s] += 1

    mode = f"int{bits} g{gs}" if quantized else "bf16"
    ok = tokens_match and worst < LOGIT_TOL
    print(f"  [{'OK' if ok else 'XX'}] {mode:<9} B={len(prompts)} offsets={lengths} steps={STEPS} "
          f"greedy_match={tokens_match} |Δlogit|={worst:.2e}")
    assert tokens_match, f"{mode}: fused greedy tokens diverged from the per-stream loop"
    assert worst < LOGIT_TOL, f"{mode}: |Δlogit|={worst:.2e} >= {LOGIT_TOL:.0e}"


class _FakeInner:
    """Ducks InternLM2ResidentModel around a bf16 InternLM2Model, exposing ``decode_batched`` so the
    batched runtime's fused step_batch engages (mirrors internlm2_paged_test's fake + delegates the
    batched path)."""

    def __init__(self, model: InternLM2Model, cfg: InternLM2Config) -> None:
        self._m = model
        self.cfg = cfg
        self.quantized_kv = True
        self.kv_group_size = 32
        self.kv_bits = 8

    @property
    def num_layers(self) -> int:
        return self._m.n_layers

    def new_cache(self) -> InternLM2Cache:
        return InternLM2Cache(self.cfg, quantized=True, group_size=32, bits=8)

    def __call__(self, token_ids: mx.array, *, cache: Any = None, caches: Any = None,
                 offset: int | None = None, last_only: bool = False, **_: Any) -> mx.array:
        obj = cache if cache is not None else caches
        clist = obj.as_list() if (obj is not None and not isinstance(obj, list)) else obj
        pos = int(offset) if offset is not None else (self.cfg and 0)
        if offset is None and obj is not None:
            pos = obj[0].offset if isinstance(obj, list) else obj.offset
        ids = token_ids if token_ids.ndim == 2 else token_ids[None]
        return self._m(ids, caches=clist, use_fast=True, abs_pos_start=pos, last_only=last_only)

    def decode_batched(self, stream_tokens: list, caches: list, offsets: list[int]) -> mx.array:
        return self._m.decode_batched(stream_tokens, caches, offsets)


def _dispatch() -> None:
    """B: runtime step_batch (fused default) == _step_batch_looped reference (one step, ragged offsets)."""
    mx.random.seed(0)
    cfg = _tiny_cfg()
    model = _bf16_model(cfg)
    batched = InternLM2BatchedResidentModel.from_inner(_FakeInner(model, cfg), max_batch=4)
    assert batched._fused, "batched runtime must default to the fused decode path"

    lengths = [9, 5, 12]
    prompts = [_prompt(cfg, n) for n in lengths]

    def _fresh():
        caches, toks = [], []
        for p in prompts:
            c = batched.new_cache()
            logits = batched.prefill(p, c)
            caches.append(c)
            toks.append(mx.array([int(mx.argmax(logits[0, -1]).item())]))
        return caches, toks

    f_caches, f_toks = _fresh()
    l_caches, l_toks = _fresh()
    offs = list(lengths)

    fused = batched.step_batch(f_toks, f_caches, offsets=list(offs))         # fused default
    loope = batched._step_batch_looped(l_toks, l_caches, offsets=list(offs))  # retained reference
    mx.eval(fused, loope)

    worst = 0.0
    match = True
    for s in range(len(prompts)):
        worst = max(worst, float(mx.max(mx.abs(fused[s][0, -1] - loope[s][0, -1]))))
        match = match and (int(mx.argmax(fused[s][0, -1]).item()) == int(mx.argmax(loope[s][0, -1]).item()))
    ok = match and worst < LOGIT_TOL
    print(f"  [{'OK' if ok else 'XX'}] runtime step_batch fused==looped B={len(prompts)} "
          f"greedy_match={match} |Δlogit|={worst:.2e}")
    assert match, "runtime fused step_batch greedy tokens diverged from _step_batch_looped"
    assert worst < LOGIT_TOL, f"runtime fused != looped: |Δlogit|={worst:.2e}"


def run() -> None:
    print("A. decode_batched == per-stream __call__ loop (ragged offsets, multi-step):")
    _core(quantized=False, bits=8)
    _core(quantized=True, bits=8)
    print("B. runtime dispatch: step_batch(fused default) == _step_batch_looped:")
    _dispatch()
    print("PASS — InternLM2.5 batched-decode attention is per-stream-equivalent (greedy-exact)")


if __name__ == "__main__":
    run()
