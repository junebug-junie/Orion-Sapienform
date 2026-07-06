"""Ensure orion-dream ``app`` resolves to this service during tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_DREAM_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _DREAM_ROOT.parents[1]


def _ensure_dream_paths() -> None:
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    for p in (str(_REPO_ROOT), str(_DREAM_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_DREAM_ROOT))


@pytest.fixture(autouse=True)
def _dream_service_isolation() -> None:
    _ensure_dream_paths()
    yield
