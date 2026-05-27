"""Auto-arm the quanta oMLX import-hook patch at Python startup.

Imported via the sibling ``_quanta_omlx_autopatch.pth`` file (which is shipped to ``site-packages/``
by the wheel — see ``pyproject.toml`` ``[tool.uv.build-backend].data.purelib = "pth_data"``).

Calling :func:`quanta.omlx_patch.install` adds an import-hook to ``sys.meta_path``; the hook stays
dormant until ``omlx.model_discovery`` / ``omlx.engine_pool`` / ``omlx.api.tool_calling`` actually
import, at which point each gets the quanta engine registered + the artifact-detection patches
applied. This module never raises — a failure (``quanta`` not yet on ``sys.path``, env-var disabled,
oMLX not installed) silently leaves the process running vanilla oMLX (rule-6: no silent wrong
behavior, but Python startup MUST stay alive).

Opt out at process level: ``QUANTA_OMLX_AUTOPATCH=0 omlx serve`` runs vanilla oMLX. The env var is
checked inside :func:`quanta.omlx_patch.install`.
"""

try:
    from quanta.omlx_patch import install
    install()
except Exception:
    # Never crash interpreter startup. The user runs vanilla oMLX if the autopatch can't arm.
    pass
