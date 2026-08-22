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

# Imported as a MODULE OBJECT, and every monkeypatch below targets it
# directly rather than through the "app.grammar_truth.<name>" string form.
#
# pytest re-resolves the string form at call time by walking
# sys.modules["app"] and getattr-ing each segment. Mid-suite that walk raised
#
#   AttributeError: module 'app' has no attribute 'grammar_truth'
#
# so monkeypatch.setattr never applied, the real grammar_engine stayed bound,
# and every assertion here failed against a live Postgres connection attempt
# ("could not translate host name orion-athena-sql-db") instead of the mock.
# All 7 failures in this file were that, not a defect in app/grammar_truth.py
# -- confirmed by the file passing in isolation, where nothing has disturbed
# the walk.
#
# A module reference captured at import time cannot be re-resolved and so
# cannot break this way, whatever else the suite does to sys.modules.
import app.grammar_truth as grammar_truth_module  # noqa: E402
from app.grammar_truth import (
    apply_grammar_atoms_retention,
    apply_grammar_edges_retention,
    apply_grammar_events_retention,
    apply_substrate_organ_emissions_retention,
    build_grammar_truth_snapshot,
    extra_retention_state,
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
        grammar_events_retention_days=15,
        grammar_events_retention_batch_size=5000,
        grammar_events_retention_max_batches_per_startup=20,
        grammar_events_retention_max_elapsed_sec=120.0,
        sql_writer_allow_accepted_pressure_ingest=False,
    )
    # Derived, not hand-listed. A MagicMock answers hasattr() for ANY name, so a table
    # added to _EXTRA_RETENTION_TABLES without a matching attribute here does not fail
    # cleanly -- it hands _retention_block a MagicMock and blows up on `configured_days > 0`
    # with a TypeError several frames away. Deriving the set removes the drift instead.
    for table in grammar_truth_module._EXTRA_RETENTION_TABLES:
        setattr(mock, f"{table}_retention_days", 15)
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


def _patch_truth_deps(monkeypatch, settings) -> None:
    # All four of these target app.grammar_truth (or its sibling app.worker) by
    # MODULE OBJECT, not by string path -- see the file-level comment above about
    # why: monkeypatch's string form re-resolves "app.grammar_truth.<name>" by
    # walking sys.modules at call time, and something elsewhere in a full-suite
    # run can leave that walk raising AttributeError mid-suite, silently
    # skipping the patch and leaving the real DB-backed function bound (which
    # then fails with a DNS/connection error instead of the intended
    # assertion). Confirmed live 2026-08-19: `_grammar_index_valid` and
    # `_fallback_counts` were still using the string form and both failed this
    # way under `pytest services/orion-sql-writer/tests -q` (the documented
    # gate command) despite passing every time when this file ran alone.
    monkeypatch.setattr(grammar_truth_module, "get_settings", lambda: settings)
    from app import worker as worker_mod

    monkeypatch.setattr(
        worker_mod,
        "grammar_queue_snapshot",
        lambda: {"workers": 4, "total_depth": 0, "shards": []},
    )
    monkeypatch.setattr(
        grammar_truth_module,
        "_fallback_counts",
        lambda: {"total": 0, "last_5m": 0, "last_30m": 0, "last_60m": 0},
    )
    monkeypatch.setattr(grammar_truth_module, "_latest_events_by_source", lambda: [])
    monkeypatch.setattr(
        grammar_truth_module,
        "_grammar_index_valid",
        lambda: {
            "idx_grammar_events_source_created": True,
            "idx_grammar_events_created_at": True,
            "idx_grammar_edges_created_at": True,
            "idx_grammar_atoms_created_at": True,
            "indexdef": "CREATE INDEX ...",
        },
    )



def _live_cursor_floor():
    """A reduction cursor that is caught up, so it never binds the retention cutoff.

    Added 2026-08-20: grammar_events retention now reads substrate_reduction_cursor FIRST
    and refuses to prune if it cannot resolve a floor (a cursor-consumed table must not be
    pruned past what its reducers have actually consumed). Every grammar_events retention
    test therefore has to supply this call. Tests that want the floor to actually BIND live
    in test_grammar_retention_periodic.py.
    """
    return MagicMock(scalar=lambda: datetime.now(timezone.utc))

def test_build_grammar_truth_snapshot_flags_degraded_when_grammar_disabled(monkeypatch) -> None:
    settings = _mock_settings(sql_writer_enable_grammar_channel=False)
    _patch_truth_deps(monkeypatch, settings)
    reset_retention_state_for_tests()
    # Use the module captured at import, NOT a fresh `from app import ...`:
    # re-resolving here can hand back a DIFFERENT module object than the
    # one build_grammar_truth_snapshot() closes over, so the state set
    # below would be written to one copy and read from the other.
    gt = grammar_truth_module

    gt._retention_state.last_run_at = datetime.now(timezone.utc)

    snap = build_grammar_truth_snapshot()
    assert snap["degraded"] is True
    assert "grammar_channel_disabled" in snap["degraded_reasons"]


def test_retention_failure_marks_truth_degraded(monkeypatch) -> None:
    settings = _mock_settings()
    _patch_truth_deps(monkeypatch, settings)

    # Use the module captured at import, NOT a fresh `from app import ...`:
    # re-resolving here can hand back a DIFFERENT module object than the
    # one build_grammar_truth_snapshot() closes over, so the state set
    # below would be written to one copy and read from the other.
    gt = grammar_truth_module

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
    monkeypatch.setattr(grammar_truth_module, "get_settings", lambda: _mock_settings())
    assert apply_grammar_events_retention(0).rows_pruned_last_run == 0


def test_retention_uses_bounded_batches_not_single_unbounded_delete(monkeypatch) -> None:
    settings = _mock_settings(
        grammar_events_retention_batch_size=100,
        grammar_events_retention_max_batches_per_startup=3,
    )
    monkeypatch.setattr(grammar_truth_module, "get_settings", lambda: settings)

    delete_results = [MagicMock(rowcount=100), MagicMock(rowcount=100), MagicMock(rowcount=25)]

    conn = MagicMock()
    conn.execute.side_effect = [
        _live_cursor_floor(),
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
    monkeypatch.setattr(grammar_truth_module, "grammar_engine", engine)

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
    monkeypatch.setattr(grammar_truth_module, "get_settings", lambda: settings)

    conn = MagicMock()
    conn.execute.side_effect = [
        _live_cursor_floor(),
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
    monkeypatch.setattr(grammar_truth_module, "grammar_engine", engine)

    result = apply_grammar_events_retention(30)
    assert result.batches_attempted == 2
    assert result.rows_pruned_last_run == 20
    assert result.remaining_debt == 5
    assert result.capped_by_startup_limit is True


def _make_fake_engine() -> MagicMock:
    conn = MagicMock()
    conn.execute.side_effect = [
        MagicMock(scalar_one=lambda: 0),  # FK check
        MagicMock(scalar_one=lambda: 0),  # remaining debt
    ]
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    begin_conn = MagicMock()
    begin_conn.execute.side_effect = [MagicMock(rowcount=7)]
    begin_conn.__enter__ = MagicMock(return_value=begin_conn)
    begin_conn.__exit__ = MagicMock(return_value=False)

    engine = MagicMock()
    engine.connect.return_value = conn
    engine.begin.return_value = begin_conn
    return engine


@pytest.mark.parametrize(
    "apply_fn, table, id_column, extra_state_key, expected_engine_attr",
    [
        (apply_grammar_edges_retention, "grammar_edges", "edge_id", "grammar_edges", "grammar_engine"),
        (apply_grammar_atoms_retention, "grammar_atoms", "atom_id", "grammar_atoms", "grammar_engine"),
        (
            apply_substrate_organ_emissions_retention,
            "substrate_organ_emissions",
            "emission_id",
            "substrate_organ_emissions",
            "default_engine",
        ),
    ],
)
def test_new_table_retention_deletes_and_records_extra_state(
    monkeypatch, apply_fn, table, id_column, extra_state_key, expected_engine_attr
) -> None:
    # Regression coverage for the three tables that had NO retention at all before
    # this patch (confirmed live 2026-08-19: unbounded growth, zero deletes ever).
    #
    # Two DISTINCT mock engines, not one shared mock -- substrate_organ_emissions
    # must use `default_engine`, the other two must use `grammar_engine` (real
    # reason: different statement_timeout per pool, see db.py). A single shared
    # mock would let a wrong-engine regression pass silently since both names
    # would resolve to the same object either way.
    settings = _mock_settings(
        grammar_events_retention_batch_size=10,
        grammar_events_retention_max_batches_per_startup=5,
    )
    monkeypatch.setattr(grammar_truth_module, "get_settings", lambda: settings)

    expected_engine = _make_fake_engine()
    other_attr = "default_engine" if expected_engine_attr == "grammar_engine" else "grammar_engine"
    other_engine = _make_fake_engine()
    monkeypatch.setattr(grammar_truth_module, expected_engine_attr, expected_engine)
    monkeypatch.setattr(grammar_truth_module, other_attr, other_engine)

    result = apply_fn(15)
    assert result.rows_pruned_last_run == 7
    assert result.failure_reason is None
    sql_texts = [str(c.args[0]) for c in expected_engine.begin.return_value.execute.call_args_list]
    assert all(f"DELETE FROM {table}" in sql for sql in sql_texts)
    assert all(f"{id_column} IN" in sql for sql in sql_texts)

    # The un-expected engine must never have been touched.
    other_engine.connect.assert_not_called()
    other_engine.begin.assert_not_called()

    # Recorded into the per-table slot, NOT the legacy single grammar_events global.
    assert extra_retention_state(extra_state_key) is result
    assert grammar_truth_module._retention_state.rows_pruned_last_run == 0


def test_other_table_retention_failure_flags_its_own_degraded_reason(monkeypatch) -> None:
    settings = _mock_settings()
    _patch_truth_deps(monkeypatch, settings)

    gt = grammar_truth_module
    from app.grammar_truth import GrammarRetentionState

    failed = GrammarRetentionState()
    failed.last_run_at = datetime.now(timezone.utc)
    failed.failure_reason = "timeout"
    gt._extra_retention_state["grammar_edges"] = failed

    snap = build_grammar_truth_snapshot()
    assert "grammar_edges_retention_failed" in snap["degraded_reasons"]
    assert snap["other_table_retention"]["grammar_edges"]["failure_reason"] == "timeout"
    # grammar_atoms/substrate_organ_emissions weren't touched -- shouldn't be flagged
    # just because a sibling table's retention failed.
    assert "grammar_atoms_retention_failed" not in snap["degraded_reasons"]


def test_missing_new_retention_index_flags_degraded(monkeypatch) -> None:
    settings = _mock_settings()
    _patch_truth_deps(monkeypatch, settings)
    monkeypatch.setattr(
        grammar_truth_module,
        "_grammar_index_valid",
        lambda: {
            "idx_grammar_events_source_created": True,
            "idx_grammar_events_created_at": False,
            "idx_grammar_edges_created_at": True,
            "idx_grammar_atoms_created_at": True,
            "indexdef": "CREATE INDEX ...",
        },
    )

    snap = build_grammar_truth_snapshot()
    assert "grammar_events_created_at_index_missing" in snap["degraded_reasons"]


def test_fk_unsafe_state_prevents_prune_and_marks_degraded(monkeypatch) -> None:
    settings = _mock_settings()
    monkeypatch.setattr(grammar_truth_module, "get_settings", lambda: settings)

    conn = MagicMock()
    conn.execute.return_value = MagicMock(scalar_one=lambda: 2)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    engine = MagicMock()
    engine.connect.return_value = conn
    monkeypatch.setattr(grammar_truth_module, "grammar_engine", engine)

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
    # Use the module captured at import, NOT a fresh `from app import ...`:
    # re-resolving here can hand back a DIFFERENT module object than the
    # one build_grammar_truth_snapshot() closes over, so the state set
    # below would be written to one copy and read from the other.
    gt = grammar_truth_module

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
    # Use the module captured at import, NOT a fresh `from app import ...`:
    # re-resolving here can hand back a DIFFERENT module object than the
    # one build_grammar_truth_snapshot() closes over, so the state set
    # below would be written to one copy and read from the other.
    gt = grammar_truth_module

    gt._retention_state.last_run_at = datetime.now(timezone.utc)
    snap = build_grammar_truth_snapshot()
    assert "accepted_pressure_subscribed_without_explicit_allow" in snap["degraded_reasons"]


def test_default_route_map_does_not_route_accepted_pressure() -> None:
    from app.settings import DEFAULT_ROUTE_MAP

    assert "grammar.event.v1" in DEFAULT_ROUTE_MAP
    assert DEFAULT_ROUTE_MAP["grammar.event.v1"] == "GrammarEventSQL"
    assert "orion:grammar:accepted-pressure" not in DEFAULT_ROUTE_MAP.values()
