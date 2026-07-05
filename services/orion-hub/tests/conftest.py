"""Ensure Hub ``scripts`` package resolves to ``services/orion-hub/scripts`` (not repo ``scripts/``)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HUB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _HUB_ROOT.parents[1]
_OTHER_SERVICE_ROOTS = tuple(
    p
    for p in _REPO_ROOT.glob("services/orion-*")
    if p.is_dir() and p.resolve() != _HUB_ROOT.resolve()
)


def _ensure_hub_paths() -> None:
    # Drop cached imports so the corrected sys.path wins (repo-root `scripts/` namespace vs Hub package).
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    # Remove stale entries so we can prepend Hub before cwd (`''`) and duplicate PYTHONPATH entries.
    for p in (str(_REPO_ROOT), str(_HUB_ROOT), *(str(p) for p in _OTHER_SERVICE_ROOTS)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_HUB_ROOT))


def pytest_configure() -> None:
    _ensure_hub_paths()


@pytest.fixture(autouse=True)
def _hub_service_isolation() -> None:
    _ensure_hub_paths()
    yield
