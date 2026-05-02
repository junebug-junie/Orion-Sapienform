"""Ensure Hub ``scripts`` package resolves to ``services/orion-hub/scripts`` (not repo ``scripts/``)."""
from __future__ import annotations

import sys
from pathlib import Path

_HUB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _HUB_ROOT.parents[1]


def pytest_configure() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(_HUB_ROOT), str(_REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
