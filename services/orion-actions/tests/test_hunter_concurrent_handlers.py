"""The Hunter bus consumer's `concurrent_handlers` flag governs whether a
long-running handler (e.g. the daily_pulse_v1/journal-trigger startup catch-up
jobs, each RPC-bound with a 420s timeout) blocks every other incoming message
on this service for its full duration (sequential, the class default) or runs
alongside them (concurrent). This service's Hunter instantiation
(app/main.py) was the only one of six sibling services' Hunter/Rabbit
consumers not explicitly setting this -- confirmed live 2026-07-13: a chat
message queued behind two startup catch-up jobs for ~4 minutes before being
processed at all. See services/orion-recall, orion-llm-gateway,
orion-cortex-exec, orion-spark-introspector, orion-cortex-orch for the
already-established `concurrent_handlers=True` pattern this mirrors.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.settings import settings
from orion.core.bus.bus_service_chassis import Hunter


def test_actions_concurrent_handlers_defaults_to_true():
    assert settings.actions_concurrent_handlers is True


def test_hunter_call_site_passes_configured_concurrent_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors the exact call shape at app/main.py's Hunter instantiation:
    constructing a real Hunter with `concurrent_handlers=settings.actions_concurrent_handlers`
    and asserting the attribute lands correctly, for both settings values."""
    monkeypatch.setattr(settings, "actions_concurrent_handlers", True)
    hunter = Hunter(
        MagicMock(),
        patterns=["orion:actions:manage:workflow.v1"],
        handler=MagicMock(),
        concurrent_handlers=settings.actions_concurrent_handlers,
    )
    assert hunter.concurrent_handlers is True

    monkeypatch.setattr(settings, "actions_concurrent_handlers", False)
    hunter = Hunter(
        MagicMock(),
        patterns=["orion:actions:manage:workflow.v1"],
        handler=MagicMock(),
        concurrent_handlers=settings.actions_concurrent_handlers,
    )
    assert hunter.concurrent_handlers is False


def test_main_hunter_instantiation_wires_concurrent_handlers_setting():
    """Guards against the wiring silently regressing: the actual call site in
    app/main.py must reference settings.actions_concurrent_handlers, not a
    bare literal or an omitted parameter (which would silently fall back to
    the Hunter class default of False -- the exact regression this test
    exists to catch)."""
    import inspect

    from app import main as actions_main

    source = inspect.getsource(actions_main)
    assert "concurrent_handlers=settings.actions_concurrent_handlers" in source
