"""Side-effect: put ``services/orion-cortex-exec`` first and drop a foreign ``app`` package."""

from __future__ import annotations

import sys
from pathlib import Path

_EXEC_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _EXEC_ROOT.parents[1]

_OTHER_APP_SERVICES = ("orion-mind", "orion-cortex-orch")


def _purge_app_tree() -> None:
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]


def _strip_other_service_paths() -> None:
    sys.path[:] = [
        p
        for p in sys.path
        if not any(svc in str(p).replace("\\", "/") for svc in _OTHER_APP_SERVICES)
    ]


def ensure_orion_cortex_exec_app() -> None:
    _purge_app_tree()
    _strip_other_service_paths()
    for p in (_REPO_ROOT, _EXEC_ROOT):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_EXEC_ROOT))
