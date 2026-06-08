"""Pytest configuration for the quanta suite.

``test_parity_modelfree.py`` imports ``parity._modelfree`` to discover + run the model-free parity
gates. ``parity`` is a top-level package in the repo root (NOT installed like ``quanta``, which
resolves via the editable ``src/`` install), so the repo root must be on ``sys.path`` for the
import to resolve at collection time. Insert it here — conftest is loaded before any test module.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
