"""Embed and load the EAGLE-3 drafter as a self-contained sidecar inside a baked quanta artifact.

oMLX serves the target model from a baked artifact dir; the EAGLE drafter needed for lossless
speculative decode must live with it so the bundle stays portable (move the dir, EAGLE moves with
it). Layout — sibling subdir under the artifact root, **relative refs only** (no absolute / source /
symlink / cache paths, per the artifact rule in CLAUDE.md):

    <art_dir>/
      manifest.json          # untouched — we do not mutate the canonical bake manifest
      <target tensors...>
      eagle/
        eagle.json           # NEW: format + version + capture_layers + drafter cfg + relative weights ref
        drafter.safetensors  # NEW: drafter weights, re-serialized canonically through mx.save_safetensors

``embed_eagle`` is a one-shot bake-time op: it instantiates ``EagleDrafter(**cfg)`` and loads the
caller's weights (so wrong shapes raise loud from ``tree_unflatten``), then re-serializes them as
``<art>/eagle/drafter.safetensors`` and writes ``eagle.json``. It refuses to overwrite an existing
``eagle/`` — re-embed is destructive (the artifact is canonical) so the caller must remove the dir
first. ``load_eagle`` is the inference-time counterpart: it reads ``eagle.json``, rejects anything
that escapes ``<art>/eagle/`` (absolute paths or ``..`` traversal), reconstructs the drafter, and
returns ``(drafter, capture_layers)``. Combined with ``load_frozen_embed_head(art_dir)`` (in
``quanta.eagle.train``) the only input ``spec_generate`` then needs is the artifact root.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from quanta.eagle.drafter import EagleDrafter

__all__ = ["DRAFTER_WEIGHTS", "DrafterConfig", "EAGLE_DIR", "EAGLE_FORMAT", "EAGLE_MANIFEST",
           "EAGLE_VERSION", "embed_eagle", "load_eagle"]

EAGLE_DIR = "eagle"
EAGLE_MANIFEST = "eagle.json"
DRAFTER_WEIGHTS = "drafter.safetensors"
EAGLE_FORMAT = "quanta-eagle"
EAGLE_VERSION = "1"


@dataclass
class DrafterConfig:
    """``EagleDrafter`` constructor args. Defaults match Kimi-K2.6 (H=7168, 56 heads × 128 head_dim,
    14336 SwiGLU, RoPE base 50000)."""

    hidden: int = 7168
    n_heads: int = 56
    head_dim: int = 128
    intermediate: int = 14336
    eps: float = 1e-6
    rope_base: float = 50000.0
    n_feature_layers: int = 3
    layerscale_init: float = 1e-4


# Inlined here (instead of importing ``save_drafter``/``load_drafter`` from ``quanta.eagle.train``)
# to keep this module decoupled from the runtime — ``train.py`` transitively pulls in the Kimi
# resident runtime via its ``ResidentArtifact`` dependency, which is unnecessary baggage for a
# bake-time embed and inference-time loader.
def _save(path: Path, drafter: EagleDrafter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), dict(tree_flatten(drafter.parameters())))


def _load(path: Path, drafter: EagleDrafter) -> None:
    drafter.update(tree_unflatten(list(mx.load(str(path)).items())))
    mx.eval(drafter.parameters())


def embed_eagle(
    art_dir: str | Path,
    drafter_weights: str | Path,
    *,
    capture_layers: tuple[int, ...] | list[int],
    drafter_cfg: DrafterConfig | dict | None = None,
    training_meta: dict | None = None,
) -> Path:
    """Embed a trained drafter into the artifact as a sidecar bundle and return the ``eagle.json``
    path. Validates the weights against the declared config (instantiating ``EagleDrafter(**cfg)``
    and loading through ``tree_unflatten`` — wrong shapes raise loud), then re-serializes them as
    ``<art>/eagle/drafter.safetensors`` and writes ``eagle.json`` with a *relative* weights ref.

    Refuses to overwrite an existing ``eagle/`` subdir — re-embedding is a destructive op (the
    artifact is canonical) so the caller must delete ``eagle/`` first to opt in."""
    art = Path(art_dir).resolve()
    src = Path(drafter_weights).resolve()
    if not art.is_dir():
        raise FileNotFoundError(f"artifact dir not found: {art}")
    if not (art / "manifest.json").is_file():
        raise ValueError(f"not a quanta artifact (no manifest.json): {art}")
    if not src.is_file():
        raise FileNotFoundError(f"drafter weights not found: {src}")

    e_dir = art / EAGLE_DIR
    if e_dir.exists():
        raise FileExistsError(f"EAGLE already embedded; remove {e_dir} to re-embed")

    cfg = drafter_cfg if isinstance(drafter_cfg, DrafterConfig) else DrafterConfig(**(drafter_cfg or {}))
    layers = tuple(int(L) for L in capture_layers)
    if not layers:
        raise ValueError("capture_layers must be non-empty")
    if len(layers) != cfg.n_feature_layers:
        raise ValueError(f"capture_layers has {len(layers)} entries but drafter_cfg.n_feature_layers="
                         f"{cfg.n_feature_layers} — they must agree (feat3 = concat over these layers)")

    # Validate by reconstructing + loading: shape mismatch raises loud from MLX's tree_unflatten.
    drafter = EagleDrafter(**asdict(cfg))
    _load(src, drafter)

    e_dir.mkdir(exist_ok=False)
    _save(e_dir / DRAFTER_WEIGHTS, drafter)

    manifest = {
        "format": EAGLE_FORMAT,
        "version": EAGLE_VERSION,
        "capture_layers": list(layers),
        "drafter": {"weights": DRAFTER_WEIGHTS, **asdict(cfg)},
        "training": dict(training_meta or {}),
    }
    out = e_dir / EAGLE_MANIFEST
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def load_eagle(art_dir: str | Path) -> tuple[EagleDrafter, tuple[int, ...]]:
    """Reconstruct the embedded drafter + capture layers from a baked artifact. The drafter weights
    ref in ``eagle.json`` is resolved RELATIVE to ``<art_dir>/eagle/``; absolute paths or ``..``
    traversal are rejected (so the bundle stays portable + tamper-safe). Returns ``(drafter,
    capture_layers)`` — combine with ``load_frozen_embed_head(art_dir)`` to feed
    :func:`quanta.eagle.spec.spec_generate`."""
    art = Path(art_dir).resolve()
    e_dir = art / EAGLE_DIR
    mpath = e_dir / EAGLE_MANIFEST
    if not mpath.is_file():
        raise FileNotFoundError(f"no embedded EAGLE in artifact: {mpath}")
    m = json.loads(mpath.read_text())
    if m.get("format") != EAGLE_FORMAT:
        raise ValueError(f"unexpected eagle.json format={m.get('format')!r} (want {EAGLE_FORMAT!r})")
    if m.get("version") != EAGLE_VERSION:
        raise ValueError(f"unsupported EAGLE version={m.get('version')!r} (want {EAGLE_VERSION!r})")

    dcfg = dict(m["drafter"])
    weights_name = dcfg.pop("weights")
    wp = Path(weights_name)
    if wp.is_absolute() or ".." in wp.parts:
        raise ValueError(f"drafter weights ref must be relative inside eagle/: {weights_name!r}")
    weights_path = e_dir / wp
    if not weights_path.is_file():
        raise FileNotFoundError(f"drafter weights missing: {weights_path}")

    drafter = EagleDrafter(**dcfg)
    _load(weights_path, drafter)
    return drafter, tuple(int(L) for L in m["capture_layers"])
