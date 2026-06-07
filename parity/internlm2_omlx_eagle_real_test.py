"""Real-weight gate: InternLM2.5 EAGLE-3 spec-decode served END-TO-END through the oMLX shim.

The model-free gate (:mod:`parity.internlm2_omlx_eagle_test`) proved the shim *plumbing* on a toy model;
the standalone bench (:mod:`parity.internlm2_eagle_spec_bench`) proved real-weight losslessness driving
the adapter directly. THIS gate closes the loop: it drives the production serving path — build a
:class:`~quanta.shim.omlx.QuantaOmlxEngine` over the real int8-g64 artifact (model_type ``internlm2``),
let :meth:`~quanta.shim.omlx.QuantaOmlxEngine._ensure_eagle` load the **embedded** ``eagle/`` drafter
sidecar (``load_eagle``) and PTQ it to the int4-g64 serving operating point, then route ``spec_k>1``
through :meth:`_dispatch_spec_k` and assert the emitted stream is **bit-identical to plain greedy** (the
EAGLE guarantee, now through the shim, on real weights), reporting the mean accept-rate.

PREREQ — embed the drafter once (one-shot, ~seconds): ``uv run python -m parity.internlm2_embed_eagle``.

Heavy (loads the ~6.2 GB resident 7B bake + the ~1 GB drafter). Run ALONE on a free GPU:

    uv run --with sentencepiece python -m parity.internlm2_omlx_eagle_real_test [n_gen]
"""

from __future__ import annotations

import sys

import mlx.core as mx

from quanta.internlm2.eagle import spec_generate as il2_spec_generate
from quanta.internlm2.runtime import InternLM2ResidentModel
from quanta.internlm2.tokenizer import InternLM2Tokenizer
from quanta.shim.omlx import QuantaOmlxEngine
from parity.internlm2_eagle_spec_test import _greedy

ART = "/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64"
K_VALUES = (2, 3)
DEFAULT_GEN = 48

# Original expository prose (self-authored, no copyright) — a real in-distribution decode prompt; the
# removed training-feature shards are no longer needed (losslessness holds for any prompt regardless).
PROSE = (
    "A reliable clock at sea solved the longitude problem because comparing local noon against a fixed "
    "reference meridian gives the distance travelled around the globe; the difference between the two "
    "clocks, converted at fifteen degrees to the hour, is the answer a navigator needs."
)


def run(n_gen: int = DEFAULT_GEN) -> None:
    mx.set_wired_limit(int(32 * 1024 ** 3))                 # ~6.2 GB model + embed/head + drafter
    rm = InternLM2ResidentModel(ART, packed=True)
    tok = InternLM2Tokenizer.from_pretrained(ART)
    ids = tok.encode(PROSE, add_bos=True)[:64]

    # production serving path: engine over the real artifact, model_type=internlm2 (real detect),
    # real resident injected (skip re-detecting/loading the runtime twice), embedded EAGLE auto-loaded.
    eng = QuantaOmlxEngine(ART, runtime=rm, tokenizer=tok)
    drafter, embed, head, layers, head_bits = eng._ensure_eagle()   # load_eagle sidecar + int4 PTQ
    mx.eval(embed, head)
    print(f"=== InternLM2.5 EAGLE-3 via oMLX shim (real int8-g64; capture={layers}, "
          f"head_bits={head_bits} int4-PTQ serving op; prompt {len(ids)} tok, gen {n_gen}) ===",
          flush=True)

    greedy = _greedy(rm, ids, n_gen)                        # the lossless reference
    bad: list[int] = []
    for k in K_VALUES:
        disp = eng._dispatch_spec_k(prompt_ids=list(ids), spec_k=k, max_new=n_gen, eos_id=None)
        toks, stats = il2_spec_generate(rm, drafter, embed, head, ids, max_new=n_gen, k=k,
                                        layers=layers, eos_id=None, head_bits=head_bits)
        ok = disp == greedy and toks == greedy
        if not ok:
            bad.append(k)
        print(f"k={k}: dispatch==greedy {disp == greedy}, adapter==greedy {toks == greedy}  "
              f"mean_accept {stats['mean_accept']:.2f}/{k + 1}  (max {stats['max_accept']}, "
              f"{stats['rounds']} rounds)", flush=True)

    assert not bad, f"EAGLE via shim VIOLATED lossless — spec != greedy for k={bad} (a real bug)"
    print("PASS — InternLM2.5 EAGLE-3 spec-decode is lossless end-to-end through the oMLX shim "
          "(spec == greedy on real int8-g64 weights, embedded drafter, int4-PTQ serving op)", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GEN)
