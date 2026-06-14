"""Unit tests for grammar production truth helpers."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

from app.grammar_truth import (
    apply_grammar_events_retention,
    build_grammar_truth_snapshot,
    reset_retention_state_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_retention_state() -> None:
    reset_retention_state_for_tests()


def _mock_settings(**overrides):
    mock = MagicMock(
        orion_bus_enabled=True,
        sql_writer_enable_grammar_channel=True,
        effective_subscribe_channels=["orion:grammar:event"],
        sql_writer_grammar_workers=4,
        grammar_events_retention_days=30,
        grammar_events_retention_batch_size=5000,
        grammar_events_retention_max_batches_per_startup=20,
        grammar_events_retention_max_elapsed_sec=120.0,
        sql_writer_allow_accepted_pressure_ingest=False,
    )
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


def _patch_truth_deps(monkeypatch, settings) -> None:
    monkeypatch.setattr("app.grammar_truth.get_settings", lambda: settings)
    monkeypatch.setattr(
        "app.worker.grammar_queue_snapshot",
        lambda: {"workers": 4, "total_depth": 0, "shards": []},
    )
    monkeypatch.setattr(
        "app.grammar_truth._fallback_counts",
        lambda: {"total": 0, "last_5m": 0, "last_30m": 0, "last_60m": 0},
    )
    monkeypatch.setattr("app.grammar_truth._latest_events_by_source", lambda: [])
    monkeypatch.setattr(
        "app.grammar_truth._grammar_index_valid",
        lambda: {"idx_grammar_events_source_created": True, "indexdef": "CREATE INDEX ..."},
    )


def test_build_grammar_truth_snapshot_flags_degraded_when_grammar_disabled(monkeypatch) -> None:
    settings = _mock_settings(sql_writer_enable_grammar_channel=False)
    _patch_truth_deps(monkeypatch, settings)
    reset_retention_state_for_tests()
    from app import grammar_truth as gt

    gt._retention_state.last_run_at = datetime.now(timezone.utc)

    snap = build_grammar_truth_snapshot()
    assert snap["degraded"] is True
    assert "grammar_channel_disabled" in snap["degraded_reasons"]


def test_retention_failure_marks_truth_degraded(monkeypatch) -> None:
    settings = _mock_settings()
    _patch_truth_deps(monkeypatch, settings)

    from app import grammar_truth as gt

    gt._retention_state.last_run_at = datetime.now(timezone.utc)
    gt._retention_state.failure_reason = "timeout"

    snap = build_grammar_truth_snapshot()
    assert snap["degraded"] is True
    assert "grammar_retention_failed" in snap["degraded_reasons"]
    assert snap["grammar_retention"]["failure_reason"] == "timeout"


def test_accepted_pressure_not_in_default_subscribe_channels() -> None:
    from app.settings import settings

    assert "orion:grammar:event" in settings.effective_subscribe_channels
    assert "orion:grammar:accepted-pressure" not in settings.effective_subscribe_channels


def test_apply_grammar_events_retention_skips_non_positive_days(monkeypatch) -> None:
    monkeypatch.setattr("app.grammar_truth.get_settings", lambda: _mock_settings())
    assert apply_grammar_events_retention(0).rows_pruned_last_run == 0


def test_retention_uses_bounded_batches_not_single_unbounded_delete(monkeypatch) -> None:
    settings = _mock_settings(
        grammar_events_retention_batch_size=100,
        grammar_events_retention_max_batches_per_startup=3,
    )
    monkeypatch.setattr("app.grammar_truth.get_settings", lambda: settings)

    delete_results = [MagicMock(rowcount=100), MagicMock(rowcount=100), MagicMock(rowcount=25)]

    conn = MagicMock()
    conn.execute.side_effect = [
        MagicMock(scalar_one=lambda: 0),
        MagicMock(scalar_one=lambda: 0),
    ]
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    begin_conn = MagicMock()
    begin_conn.execute.side_effect = delete_results
    begin_conn.__enter__ = MagicMock(return_value=begin_conn)
    begin_conn.__exit__ = MagicMock(return_value=False)

    engine = MagicMock()
    engine.connect.return_value = conn
    engine.begin.return_value = begin_conn
    monkeypatch.setattr("app.grammar_truth.grammar_engine", engine)

    result = apply_grammar_events_retention(30)
    assert result.rows_pruned_last_run == 225
    assert result.batches_attempted == 3
    assert begin_conn.execute.call_count == 3
    sql_texts = [str(c.args[0]) for c in begin_conn.execute.call_args_list]
    assert all("LIMIT :batch_size" in sql for sql in sql_texts)
    assert all("DELETE FROM grammar_events" in sql for sql in sql_texts)


def test_retention_stops_at_max_batch_cap_and_reports_debt(monkeypatch) -> None:
    settings = _mock_settings(
        grammar_events_retention_batch_size=10,
        grammar_events_retention_max_batches_per_startup=2,
        grammar_events_retention_max_elapsed_sec=120.0,
    )
    monkeypatch.setattr("app.grammar_truth.get_settings", lambda: settings)

    conn = MagicMock()
    conn.execute.side_effect = [
        MagicMock(scalar_one=lambda: 0),
        MagicMock(scalar_one=lambda: 5),
    ]
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    begin_conn = MagicMock()
    begin_conn.execute.side_effect = [MagicMock(rowcount=10), MagicMock(rowcount=10)]
    begin_conn.__enter__ = MagicMock(return_value=begin_conn)
    begin_conn.__exit__ = MagicMock(return_value=False)

    engine = MagicMock()
    engine.connect.return_value = conn
    engine.begin.return_value = begin_conn
    monkeypatch.setattr("app.grammar_truth.grammar_engine", engine)

    result = apply_grammar_events_retention(30)
    assert result.batches_attempted == 2
    assert result.rows_pruned_last_run == 20
    assert result.remaining_debt == 5
    assert result.capped_by_startup_limit is True


def test_fk_unsafe_state_prevents_prune_and_marks_degraded(monkeypatch) -> None:
    settings = _mock_settings()
    monkeypatch.setattr("app.grammar_truth.get_settings", lambda: settings)

    conn = MagicMock()
    conn.execute.return_value = MagicMock(scalar_one=lambda: 2)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    engine = MagicMock()
    engine.connect.return_value = conn
    monkeypatch.setattr("app.grammar_truth.grammar_engine", engine)

    result = apply_grammar_events_retention(30)
    assert result.rows_pruned_last_run == 0
    assert result.failure_reason is not None
    assert result.fk_delete_verified is False

    _patch_truth_deps(monkeypatch, settings)
    snap = build_grammar_truth_snapshot()
    assert "grammar_retention_failed" in snap["degraded_reasons"]


def test_retention_debt_marks_degraded(monkeypatch) -> None:
    settings = _mock_settings()
    _patch_truth_deps(monkeypatch, settings)
    from app import grammar_truth as gt

    gt._retention_state.last_run_at = datetime.now(timezone.utc)
    gt._retention_state.remaining_debt = 42
    snap = build_grammar_truth_snapshot()
    assert "grammar_retention_debt_remaining" in snap["degraded_reasons"]


def test_accepted_pressure_subscribed_without_allow_flag_degraded(monkeypatch) -> None:
    settings = _mock_settings(
        effective_subscribe_channels=["orion:grammar:event", "orion:grammar:accepted-pressure"],
        sql_writer_allow_accepted_pressure_ingest=False,
    )
    _patch_truth_deps(monkeypatch, settings)
    from app import grammar_truth as gt

    gt._retention_state.last_run_at = datetime.now(timezone.utc)
    snap = build_grammar_truth_snapshot()
    assert "accepted_pressure_subscribed_without_explicit_allow" in snap["degraded_reasons"]


def test_default_route_map_does_not_route_accepted_pressure() -> None:
    from app.settings import DEFAULT_ROUTE_MAP

    assert "grammar.event.v1" in DEFAULT_ROUTE_MAP
    assert DEFAULT_ROUTE_MAP["grammar.event.v1"] == "GrammarEventSQL"
    assert "orion:grammar:accepted-pressure" not in DEFAULT_ROUTE_MAP.values()
