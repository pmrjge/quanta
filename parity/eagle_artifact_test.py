"""Gate: ``embed_eagle`` + ``load_eagle`` — bake-time embed of an EAGLE drafter into a quanta
artifact + the inference-time loader. Model-free (~ms): tiny ``EagleDrafter`` (hidden=8, 2×4 heads,
intermediate=16), synthetic artifact dir (stub ``manifest.json``). Verifies:

  (1) embed writes ``<art>/eagle/{eagle.json, drafter.safetensors}``; the artifact's main
      ``manifest.json`` is **NOT** mutated (the canonical bake manifest stays byte-exact);
      ``eagle.json`` records ``format=quanta-eagle``, ``capture_layers``, and a **relative**
      drafter-weights ref (``drafter.safetensors``, no path);
  (2) ``load_eagle`` returns a drafter with bit-identical params and the exact ``capture_layers`` we
      embedded;
  (3) PORTABILITY: rename the artifact dir, ``load_eagle`` still works (proves there's no absolute
      / source / symlink ref baked in — the artifact is movable as a whole);
  (4) FAIL-LOUD (CLAUDE.md rule 6): missing artifact dir, no ``manifest.json``, missing weights file,
      existing ``eagle/`` (refuse to overwrite the canonical bake), absolute weights ref in
      ``eagle.json``, wrong ``format``/``version``, and ``capture_layers`` length ≠
      ``n_feature_layers`` — every wrong shape raises loud.

    uv run python -m parity.eagle_artifact_test
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from quanta.eagle.artifact import (
    DRAFTER_WEIGHTS,
    EAGLE_DIR,
    EAGLE_FORMAT,
    EAGLE_MANIFEST,
    EAGLE_VERSION,
    DrafterConfig,
    embed_eagle,
    load_eagle,
)
from quanta.eagle.drafter import EagleDrafter

CFG = DrafterConfig(hidden=8, n_heads=2, head_dim=4, intermediate=16, eps=1e-6, rope_base=1e4,
                    n_feature_layers=3, layerscale_init=1e-4)
LAYERS = (1, 2, 3)


def _stub_artifact(art: Path) -> Path:
    art.mkdir(parents=True)
    (art / "manifest.json").write_text(json.dumps({"format": "quanta", "tensors": {}}))
    return art


def _make_drafter() -> EagleDrafter:
    mx.random.seed(0)
    d = EagleDrafter(**asdict(CFG))
    mx.eval(d.parameters())
    return d


def _save_weights(path: Path, drafter: EagleDrafter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), dict(tree_flatten(drafter.parameters())))


def _params_equal(a: EagleDrafter, b: EagleDrafter) -> bool:
    pa, pb = dict(tree_flatten(a.parameters())), dict(tree_flatten(b.parameters()))
    if pa.keys() != pb.keys():
        return False
    return all(np.array_equal(np.array(pa[k]), np.array(pb[k])) for k in pa)


def _raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def _write_tampered(art: Path, override: dict) -> None:
    """Build a tampered ``eagle/eagle.json`` for the FAIL-LOUD load tests."""
    td = art / EAGLE_DIR
    if td.exists():
        shutil.rmtree(td)
    td.mkdir()
    base = {"format": EAGLE_FORMAT, "version": EAGLE_VERSION, "capture_layers": list(LAYERS),
            "drafter": {"weights": DRAFTER_WEIGHTS, **asdict(CFG)}, "training": {}}
    base.update(override)
    (td / EAGLE_MANIFEST).write_text(json.dumps(base))


def run() -> None:
    ok = True
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        art = _stub_artifact(tmp / "art")
        drafter = _make_drafter()
        weights = tmp / "src" / "drafter.safetensors"
        _save_weights(weights, drafter)

        # (1) embed sidecar; main manifest untouched; relative weights ref
        main_before = (art / "manifest.json").read_text()
        out = embed_eagle(art, weights, capture_layers=LAYERS, drafter_cfg=CFG,
                          training_meta={"corpus_tokens": 999})
        e_dir = (art / EAGLE_DIR).resolve()
        files_ok = (out.resolve() == (e_dir / EAGLE_MANIFEST).resolve() and out.is_file()
                    and (e_dir / DRAFTER_WEIGHTS).is_file()
                    and (art / "manifest.json").read_text() == main_before)
        man = json.loads(out.read_text())
        meta_ok = (man["format"] == EAGLE_FORMAT and man["version"] == EAGLE_VERSION
                   and man["drafter"]["weights"] == DRAFTER_WEIGHTS
                   and tuple(man["capture_layers"]) == LAYERS
                   and man["training"]["corpus_tokens"] == 999)
        good = files_ok and meta_ok
        ok &= good
        print(f"  [{'OK' if good else 'FAIL'}] embed sidecar: files={files_ok} meta={meta_ok} "
              f"weights_ref={man['drafter']['weights']!r}")

        # (2) load round-trip — bit-identical params + capture_layers
        d2, layers2 = load_eagle(art)
        rt = _params_equal(drafter, d2) and layers2 == LAYERS
        ok &= rt
        print(f"  [{'OK' if rt else 'FAIL'}] load_eagle round-trip: layers={layers2}")

        # (3) portability — rename the artifact dir, load still works (relative refs)
        moved = tmp / "moved"
        shutil.move(str(art), str(moved))
        d3, layers3 = load_eagle(moved)
        portable = _params_equal(drafter, d3) and layers3 == LAYERS
        ok &= portable
        print(f"  [{'OK' if portable else 'FAIL'}] portable after rename")

        # (4) FAIL-LOUD cases
        # 4a re-embed into existing eagle/ refuses
        r1 = _raises(lambda: embed_eagle(moved, weights, capture_layers=LAYERS, drafter_cfg=CFG),
                     FileExistsError)
        # 4b embed into non-artifact dir refuses
        bad_dir = _stub_artifact(tmp / "no_manifest")
        (bad_dir / "manifest.json").unlink()
        r2 = _raises(lambda: embed_eagle(bad_dir, weights, capture_layers=LAYERS, drafter_cfg=CFG),
                     ValueError)
        # 4c missing weights
        clean = _stub_artifact(tmp / "clean")
        r3 = _raises(lambda: embed_eagle(clean, tmp / "nope.safetensors",
                                         capture_layers=LAYERS, drafter_cfg=CFG),
                     FileNotFoundError)
        # 4d load with no eagle/
        r4 = _raises(lambda: load_eagle(clean), FileNotFoundError)
        # 4e load with absolute weights ref
        tamp = _stub_artifact(tmp / "tamp")
        _write_tampered(tamp, {"drafter": {"weights": "/etc/passwd", **asdict(CFG)}})
        r5 = _raises(lambda: load_eagle(tamp), ValueError)
        # 4f load with .. traversal in weights ref
        _write_tampered(tamp, {"drafter": {"weights": "../../etc/passwd", **asdict(CFG)}})
        r6 = _raises(lambda: load_eagle(tamp), ValueError)
        # 4g load with bad format
        _write_tampered(tamp, {"format": "evil"})
        r7 = _raises(lambda: load_eagle(tamp), ValueError)
        # 4h load with unsupported version
        _write_tampered(tamp, {"version": "999"})
        r8 = _raises(lambda: load_eagle(tamp), ValueError)
        # 4i capture_layers length must equal n_feature_layers
        bad_cfg = _stub_artifact(tmp / "bad_cfg")
        r9 = _raises(lambda: embed_eagle(bad_cfg, weights, capture_layers=(1, 2),
                                         drafter_cfg=CFG),
                     ValueError)
        # 4j empty capture_layers
        empty_cfg = _stub_artifact(tmp / "empty_cfg")
        r10 = _raises(lambda: embed_eagle(empty_cfg, weights, capture_layers=(),
                                          drafter_cfg=DrafterConfig(**{**asdict(CFG), "n_feature_layers": 0})),
                      ValueError)

        loud = all([r1, r2, r3, r4, r5, r6, r7, r8, r9, r10])
        ok &= loud
        print(f"  [{'OK' if loud else 'FAIL'}] fail-loud: reembed={r1} no_manifest={r2} "
              f"missing_weights={r3} no_eagle={r4} abs_ref={r5} traversal={r6} bad_format={r7} "
              f"bad_version={r8} layers_mismatch={r9} empty_layers={r10}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
