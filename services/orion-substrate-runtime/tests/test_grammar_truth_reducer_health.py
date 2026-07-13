"""Unit tests for substrate grammar truth reducer health payload."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.grammar_truth import build_substrate_grammar_truth
from app.reducer_health import clear_health_for_tests, record_tick


def setup_function() -> None:
    clear_health_for_tests()


def test_truth_includes_reducer_health_and_backlog(monkeypatch) -> None:
    store = MagicMock()
    store.cursor_positions.return_value = [
        {
            "cursor_name": "transport_grammar_reducer",
            "lag_sec": 7 * 3600,
            "last_event_created_at": "2026-06-15T00:00:00+00:00",
            "last_event_id": "gev_old",
            "updated_at": "2026-06-16T00:00:00+00:00",
        }
    ]
    store.grammar_cursor_metrics.return_value = {
        "cursor_name": "transport_grammar_reducer",
        "pending_backlog": 500,
        "stream_lag_sec": 7 * 3600,
        "cursor_wall_lag_sec": 7 * 3600,
        "head_event_created_at": "2026-06-16T00:00:00+00:00",
        "head_event_id": "gev_new",
    }
    store.grammar_cursor_metrics.side_effect = lambda name: {
        "cursor_name": name,
        "pending_backlog": 500 if name == "transport_grammar_reducer" else 0,
        "stream_lag_sec": 7 * 3600 if name == "transport_grammar_reducer" else 0,
        "cursor_wall_lag_sec": 7 * 3600 if name == "transport_grammar_reducer" else 0,
        "head_event_created_at": "2026-06-16T00:00:00+00:00",
        "head_event_id": "gev_new",
    }
    store.quarantine_summary.return_value = {
        "unacknowledged_quarantine_count_by_reducer": {},
        "unacknowledged_quarantine_count_by_cursor": {},
        "quarantine_by_reducer": {},
    }

    monkeypatch.setattr(
        "app.grammar_truth.get_settings",
        lambda: MagicMock(
            orion_bus_enabled=True,
            enable_execution_trajectory_reducer=True,
            enable_transport_bus_reducer=True,
            enable_biometrics_node_reducer=True,
            enable_biometrics_pressure_organ=True,
            enable_node_pressure_reducer=True,
            substrate_cursor_lag_resync_hours=6.0,
            reducer_heartbeat_stale_sec=120.0,
            grammar_poll_interval_sec=5.0,
            substrate_cursor_tail_seed_on_lag=False,
            substrate_cursor_reset_operator_token="tok",
            biometrics_grammar_batch_limit=50,
            execution_grammar_batch_limit=100,
            transport_grammar_batch_limit=500,
            accepted_pressure_grammar_channel="orion:grammar:accepted-pressure",
            grammar_event_channel="orion:grammar:event",
            publish_accepted_pressure_grammar=True,
        ),
    )
    monkeypatch.setattr("app.grammar_truth.has_recent_tail_seed", lambda: False)
    monkeypatch.setattr("app.grammar_truth.has_cold_start_tail_seed", lambda: False)
    monkeypatch.setattr("app.grammar_truth.last_reset_skipped_history", lambda: False)
    monkeypatch.setattr(
        "app.grammar_truth.tail_seed_snapshot",
        lambda: {"count": 0, "latest": None, "recent": []},
    )
    monkeypatch.setattr(
        "app.grammar_truth.cursor_reset_snapshot",
        lambda: {"count": 0, "last": None, "recent": []},
    )

    payload = build_substrate_grammar_truth(store)
    assert payload["degraded"] is True
    assert "cursor_lag:transport_grammar_reducer" in payload["degraded_reasons"]
    assert payload["pending_backlog_by_reducer"]["transport_grammar_reducer"] == 500
    assert "transport_bus" in payload["reducer_health_by_name"]


def test_idle_traffic_gated_lane_with_stale_lag_and_no_backlog_is_not_cursor_lag_degraded(
    monkeypatch,
) -> None:
    """chat_grammar only advances on real chat traffic; a long-idle cursor with
    zero pending backlog is healthy, not degraded -- regression test for a false
    positive that fired in production (cursor_lag:chat_grammar_consumer with
    pending_backlog=0 after ~8h of no chat activity)."""
    store = MagicMock()
    store.cursor_positions.return_value = [
        {
            "cursor_name": "chat_grammar_consumer",
            "lag_sec": 28308.0,  # ~7.9h, well past the 6h resync threshold
            "last_event_created_at": "2026-07-13T07:52:47+00:00",
            "last_event_id": "gev_old",
            "updated_at": "2026-07-13T07:52:48+00:00",
        }
    ]
    store.grammar_cursor_metrics.side_effect = lambda name: {
        "cursor_name": name,
        "pending_backlog": 0,
        "stream_lag_sec": 0.0,
        "cursor_wall_lag_sec": 28308.0 if name == "chat_grammar_consumer" else 0.0,
        "head_event_created_at": "2026-07-13T07:52:47+00:00",
        "head_event_id": "gev_old",
    }
    store.quarantine_summary.return_value = {
        "unacknowledged_quarantine_count_by_reducer": {},
        "unacknowledged_quarantine_count_by_cursor": {},
        "quarantine_by_reducer": {},
    }

    # Simulate the reducer loop ticking normally (worker.py polls every reducer
    # on every cycle regardless of whether it finds new events) so the
    # heartbeat-based classification stays "healthy" and only the cursor_lag
    # gating under test is exercised.
    for reducer_key, cursor_name in [
        ("biometrics", "biometrics_grammar_consumer"),
        ("chat_grammar", "chat_grammar_consumer"),
        ("execution_trajectory", "execution_grammar_reducer"),
        ("transport_bus", "transport_grammar_reducer"),
        ("route_grammar", "route_grammar_consumer"),
    ]:
        record_tick(reducer_key, cursor_name=cursor_name, enabled=True)

    monkeypatch.setattr(
        "app.grammar_truth.get_settings",
        lambda: MagicMock(
            orion_bus_enabled=True,
            enable_execution_trajectory_reducer=True,
            enable_transport_bus_reducer=True,
            enable_biometrics_node_reducer=True,
            enable_biometrics_pressure_organ=True,
            enable_node_pressure_reducer=True,
            enable_chat_grammar_reducer=True,
            enable_route_grammar_reducer=True,
            substrate_cursor_lag_resync_hours=6.0,
            reducer_heartbeat_stale_sec=120.0,
            grammar_poll_interval_sec=5.0,
            substrate_cursor_tail_seed_on_lag=False,
            substrate_cursor_reset_operator_token="tok",
            biometrics_grammar_batch_limit=50,
            execution_grammar_batch_limit=100,
            transport_grammar_batch_limit=500,
            accepted_pressure_grammar_channel="orion:grammar:accepted-pressure",
            grammar_event_channel="orion:grammar:event",
            publish_accepted_pressure_grammar=True,
        ),
    )
    monkeypatch.setattr("app.grammar_truth.has_recent_tail_seed", lambda: False)
    monkeypatch.setattr("app.grammar_truth.has_cold_start_tail_seed", lambda: False)
    monkeypatch.setattr("app.grammar_truth.last_reset_skipped_history", lambda: False)
    monkeypatch.setattr(
        "app.grammar_truth.tail_seed_snapshot",
        lambda: {"count": 0, "latest": None, "recent": []},
    )
    monkeypatch.setattr(
        "app.grammar_truth.cursor_reset_snapshot",
        lambda: {"count": 0, "last": None, "recent": []},
    )

    payload = build_substrate_grammar_truth(store)
    assert payload["degraded"] is False
    assert "cursor_lag:chat_grammar_consumer" not in payload["degraded_reasons"]
    assert payload["pending_backlog_by_reducer"]["chat_grammar_consumer"] == 0
