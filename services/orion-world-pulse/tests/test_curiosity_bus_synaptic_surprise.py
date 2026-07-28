"""Regression: curiosity.py's default bus_synaptic surprise wiring.

ORION_ACTION_OUTCOME_DB_URL is optional for this service (it had no other Postgres
dependency before 2026-07-28) -- unconfigured must degrade to None on every call, not
raise or spin up a real connection attempt. Configured, it must actually resolve a real
value without any caller needing to pass surprise_source explicitly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import app.services.curiosity as curiosity


@pytest.fixture(autouse=True)
def _reset_surprise_engine():
    """Guarantees curiosity._surprise_engine is reset before AND after every test in this
    file, regardless of test order or a mid-test failure -- a manual reset at the end of
    each test body is order-dependent and doesn't run if the test raises (found in review:
    one test was missing its manual reset, relying on execution order to mask it)."""
    curiosity._surprise_engine = None
    yield
    curiosity._surprise_engine = None


def test_default_source_none_when_db_url_unconfigured(monkeypatch):
    monkeypatch.setattr("app.settings.settings.action_outcome_db_url", "")
    assert curiosity._get_surprise_engine() is None
    assert curiosity._default_bus_synaptic_surprise_source() is None


def test_default_source_returns_real_value_when_configured(monkeypatch):
    monkeypatch.setattr(
        "app.settings.settings.action_outcome_db_url",
        "postgresql://test:test@localhost/test",
    )
    with patch.object(curiosity, "latest_bus_synaptic_prediction_error", return_value=0.1675):
        assert curiosity._default_bus_synaptic_surprise_source() == 0.1675


def test_default_source_engine_cached_across_calls(monkeypatch):
    monkeypatch.setattr(
        "app.settings.settings.action_outcome_db_url",
        "postgresql://test:test@localhost/test",
    )
    with patch("sqlalchemy.create_engine") as mock_create_engine:
        mock_create_engine.return_value = MagicMock()
        curiosity._get_surprise_engine()
        curiosity._get_surprise_engine()
    mock_create_engine.assert_called_once()


def test_default_source_fails_open_on_query_error(monkeypatch):
    monkeypatch.setattr(
        "app.settings.settings.action_outcome_db_url",
        "postgresql://test:test@localhost/test",
    )
    with patch.object(
        curiosity, "latest_bus_synaptic_prediction_error", side_effect=RuntimeError("db down")
    ):
        assert curiosity._default_bus_synaptic_surprise_source() is None


def test_resolve_domain_surprise_falls_back_to_default_when_no_explicit_source(monkeypatch):
    with patch.object(
        curiosity, "_default_bus_synaptic_surprise_source", return_value=0.037
    ):
        assert curiosity._resolve_domain_surprise(None) == 0.037


def test_resolve_domain_surprise_prefers_explicit_source_over_default(monkeypatch):
    with patch.object(
        curiosity, "_default_bus_synaptic_surprise_source", return_value=0.037
    ):
        assert curiosity._resolve_domain_surprise(lambda: 0.9) == 0.9
