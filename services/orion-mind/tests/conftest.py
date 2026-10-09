"""Re-bind ``app`` to ``services/orion-mind`` before each test (other suites overwrite ``app``)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_guard_path = Path(__file__).resolve().parent / "_mind_import_guard.py"
_spec = importlib.util.spec_from_file_location("_mind_guard_mod", _guard_path)
assert _spec and _spec.loader
_guard_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard_mod)
_guard_mod.ensure_orion_mind_app()

import pytest


@pytest.fixture(autouse=True)
def _mind_app_before_each_test() -> None:
    _guard_mod.ensure_orion_mind_app()
    yield



@pytest.fixture(autouse=True)
def _no_live_transport_threshold_state(monkeypatch):
    """Unit tests must never read or write the live transport-threshold Redis
    state. Tests that exercise it patch orion.field.transport_thresholds._client."""
    monkeypatch.setenv("TRANSPORT_THRESHOLDS_DERIVED_ENABLED", "false")
