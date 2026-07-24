"""Ensure orion-heartbeat's ``app`` resolves to this service during tests
(mirrors services/orion-substrate-runtime/tests/conftest.py -- every service
uses the generic package name ``app``, so cross-service test collisions are
possible without this isolation)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SERVICE_ROOT.parents[1]


def _ensure_service_paths() -> None:
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    for p in (str(_REPO_ROOT), str(_SERVICE_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_SERVICE_ROOT))


@pytest.fixture(autouse=True)
def _heartbeat_service_isolation() -> None:
    _ensure_service_paths()
    yield
