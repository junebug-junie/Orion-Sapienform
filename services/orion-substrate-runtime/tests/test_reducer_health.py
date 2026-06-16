"""Unit tests for reducer health classification."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.reducer_health import (
    ReducerHealthSnapshot,
    clear_health_for_tests,
    record_cursor_advance,
    record_error,
    record_success,
    record_tick,
)


def setup_function() -> None:
    clear_health_for_tests()


def test_classify_alive_behind_on_stream_lag() -> None:
    now = datetime.now(timezone.utc)
    snap = ReducerHealthSnapshot(
        reducer_key="transport_bus",
        cursor_name="transport_grammar_reducer",
        enabled=True,
        last_tick_at=now,
        stream_lag_sec=8 * 3600,
        pending_backlog=1000,
    )
    assert (
        snap.classify(heartbeat_stale_sec=120, stream_lag_degraded_sec=6 * 3600)
        == "alive_behind"
    )


def test_classify_dead_without_heartbeat() -> None:
    snap = ReducerHealthSnapshot(
        reducer_key="execution_trajectory",
        cursor_name="execution_grammar_reducer",
        enabled=True,
    )
    assert (
        snap.classify(heartbeat_stale_sec=120, stream_lag_degraded_sec=6 * 3600)
        == "dead_no_heartbeat"
    )


def test_classify_blocked_on_event() -> None:
    now = datetime.now(timezone.utc)
    snap = ReducerHealthSnapshot(
        reducer_key="transport_bus",
        cursor_name="transport_grammar_reducer",
        enabled=True,
        last_tick_at=now,
        blocked_event_id="gev_bad",
        blocked_failures=2,
    )
    assert (
        snap.classify(heartbeat_stale_sec=120, stream_lag_degraded_sec=6 * 3600)
        == "blocked_on_event"
    )


def test_record_success_clears_blocked_state() -> None:
    record_error(
        "transport_bus",
        cursor_name="transport_grammar_reducer",
        enabled=True,
        event_id="gev_bad",
        reason="boom",
    )
    record_success(
        "transport_bus",
        cursor_name="transport_grammar_reducer",
        enabled=True,
        batch_events=10,
    )
    record_tick("transport_bus", cursor_name="transport_grammar_reducer", enabled=True)
    record_cursor_advance(
        "transport_bus",
        cursor_name="transport_grammar_reducer",
        enabled=True,
    )
    from app.reducer_health import health_snapshots

    snap = health_snapshots()["transport_bus"]
    assert snap.blocked_event_id is None
    assert snap.last_cursor_advance_at is not None
