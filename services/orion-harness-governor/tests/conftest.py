"""Ensure orion-harness-governor ``app`` resolves to this service during tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_GOVERNOR_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _GOVERNOR_ROOT.parents[1]


def _ensure_governor_paths() -> None:
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    for p in (str(_REPO_ROOT), str(_GOVERNOR_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_GOVERNOR_ROOT))


@pytest.fixture(autouse=True)
def _governor_service_isolation() -> None:
    _ensure_governor_paths()
    yield
