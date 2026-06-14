"""Unit tests for substrate cursor tail-seed semantics."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.cursor_gaps import clear_tail_seeds_for_tests, tail_seed_snapshot
from app.publish import publish_accepted_events


@pytest.fixture(autouse=True)
def _clear_gaps() -> None:
    clear_tail_seeds_for_tests()


@pytest.mark.asyncio
async def test_publish_accepted_events_uses_separate_channel_not_canonical() -> None:
    bus = MagicMock()
    event = MagicMock()
    event.correlation_id = None

    with patch("app.publish.publish_grammar_event", new_callable=AsyncMock) as mock_publish:
        await publish_accepted_events(
            bus,
            [event],
            channel="orion:grammar:accepted-pressure",
        )
        mock_publish.assert_awaited_once()
        _args, kwargs = mock_publish.await_args
        assert kwargs["source_name"] == "orion-substrate-runtime"
        assert kwargs["channel"] == "orion:grammar:accepted-pressure"
        assert kwargs["channel"] != "orion:grammar:event"


def test_lag_resync_does_not_tail_seed_when_disabled(monkeypatch) -> None:
    from app.store import BiometricsSubstrateStore

    store = BiometricsSubstrateStore.__new__(BiometricsSubstrateStore)
    conn = MagicMock()
    stale_at = datetime.now(timezone.utc) - timedelta(hours=12)
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "last_event_created_at": stale_at,
        "last_event_id": "evt-old",
    }

    before = tail_seed_snapshot()["count"]
    store._ensure_grammar_cursor_at_tail(
        conn,
        cursor_name="test-cursor",
        source_service="orion-biometrics",
        trace_prefix="biometrics.node:",
        max_lag_sec=6 * 3600,
        tail_seed_on_lag=False,
    )
    after = tail_seed_snapshot()["count"]
    assert after == before
    conn.execute.assert_called_once()


def test_lag_resync_enabled_records_data_gap(monkeypatch) -> None:
    from app.store import BiometricsSubstrateStore

    store = BiometricsSubstrateStore.__new__(BiometricsSubstrateStore)
    stale_at = datetime.now(timezone.utc) - timedelta(hours=12)

    select_cursor = MagicMock()
    select_cursor.mappings.return_value.first.return_value = {
        "last_event_created_at": stale_at,
        "last_event_id": "evt-old",
    }
    select_tail = MagicMock()
    select_tail.mappings.return_value.first.return_value = {
        "created_at": datetime.now(timezone.utc),
        "event_id": "evt-tail",
    }
    conn = MagicMock()
    conn.execute.side_effect = [select_cursor, select_tail, MagicMock()]

    store._ensure_grammar_cursor_at_tail(
        conn,
        cursor_name="test-cursor",
        source_service="orion-biometrics",
        trace_prefix="biometrics.node:",
        max_lag_sec=6 * 3600,
        tail_seed_on_lag=True,
    )
    snap = tail_seed_snapshot()
    assert snap["count"] == 1
    assert snap["latest"]["reason"] == "lag_exceeded"


def test_cold_start_tail_seed_records_data_gap() -> None:
    from app.store import BiometricsSubstrateStore

    store = BiometricsSubstrateStore.__new__(BiometricsSubstrateStore)
    select_missing = MagicMock()
    select_missing.mappings.return_value.first.return_value = None
    select_tail = MagicMock()
    select_tail.mappings.return_value.first.return_value = {
        "created_at": datetime.now(timezone.utc),
        "event_id": "evt-tail",
    }
    conn = MagicMock()
    conn.execute.side_effect = [select_missing, select_tail, MagicMock()]

    store._ensure_grammar_cursor_at_tail(
        conn,
        cursor_name="test-cursor",
        source_service="orion-biometrics",
        trace_prefix="biometrics.node:",
        max_lag_sec=6 * 3600,
        tail_seed_on_lag=False,
    )
    snap = tail_seed_snapshot()
    assert snap["count"] == 1
    assert snap["latest"]["reason"] == "cold_start"


def test_substrate_truth_reflects_operator_reset_skip_history(monkeypatch) -> None:
    from app.cursor_reset import clear_cursor_resets_for_tests, record_cursor_reset
    from app.grammar_truth import build_substrate_grammar_truth

    clear_cursor_resets_for_tests()
    record_cursor_reset(
        cursor_name="biometrics_grammar_consumer",
        mode="tail",
        requested_timestamp=None,
        prior_created_at=None,
        prior_event_id=None,
        new_created_at="2026-06-13T00:00:00+00:00",
        new_event_id="evt-tail",
        actor="operator",
        history_may_be_skipped=True,
    )

    settings = MagicMock(
        orion_bus_enabled=True,
        grammar_poll_interval_sec=5.0,
        publish_accepted_pressure_grammar=True,
        accepted_pressure_grammar_channel="orion:grammar:accepted-pressure",
        grammar_event_channel="orion:grammar:event",
        substrate_cursor_tail_seed_on_lag=False,
        substrate_cursor_lag_resync_hours=6.0,
        substrate_cursor_reset_operator_token="secret",
        enable_biometrics_node_reducer=True,
        enable_biometrics_pressure_organ=True,
        enable_node_pressure_reducer=True,
        enable_execution_trajectory_reducer=False,
        enable_transport_bus_reducer=False,
    )
    monkeypatch.setattr("app.grammar_truth.get_settings", lambda: settings)

    store = MagicMock()
    store.cursor_positions.return_value = []
    snap = build_substrate_grammar_truth(store)
    assert snap["operator_cursor_reset"]["count"] == 1
    assert "operator_cursor_reset_skipped_history" in snap["degraded_reasons"]
