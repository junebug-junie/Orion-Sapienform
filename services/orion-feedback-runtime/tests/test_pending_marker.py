"""ROADMAP D2: the `feedback_pending` marker that replaced the unbounded anti-join.

The performance claim (O(pending) instead of O(history)) is verified live with EXPLAIN, not
here. What these tests pin is the SAFETY of the marker, because every failure mode of a
work-queue flag is silent:

  * the marker must be cleared in the SAME transaction as the feedback insert, or a crash
    between them loses the work,
  * the reconciler must only ever ADD work back, never remove it,
  * a stale-true marker must not spin the FIFO forever on one row.

A time-bounded scan was tried first and reverted for exactly this class of failure -- it
stranded the backlog silently. These tests exist so the replacement cannot do the same.
"""
from __future__ import annotations

import json

import pytest

from app.store import FeedbackRuntimeStore


class _Result:
    def __init__(self, row=None, rowcount=0):
        self._row, self.rowcount = row, rowcount

    def mappings(self):
        return self

    def first(self):
        return self._row


class _Conn:
    """Records statements, and which transaction/connection scope issued them."""

    def __init__(self, owner, scope, script=None):
        self.owner, self.scope, self.script = owner, scope, list(script or [])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split())
        self.owner.calls.append((self.scope, sql, params))
        return self.script.pop(0) if self.script else _Result(rowcount=0)


class _Engine:
    def __init__(self, connect_script=None, begin_script=None):
        self.calls = []
        self._connect_script, self._begin_script = connect_script, begin_script

    def connect(self):
        return _Conn(self, "connect", self._connect_script)

    def begin(self):
        return _Conn(self, "begin", self._begin_script)


def _store(engine, **kw):
    s = FeedbackRuntimeStore.__new__(FeedbackRuntimeStore)
    s._engine = engine
    s._reconcile_interval_sec = kw.get("reconcile_interval_sec", 900.0)
    s._last_reconcile_mono = kw.get("last_reconcile_mono")
    return s


def _frame():
    from orion.schemas.feedback_frame import FeedbackFrameV1
    from datetime import datetime, timezone

    return FeedbackFrameV1(
        frame_id="feedback.frame:abc",
        generated_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
        source_execution_dispatch_frame_id="dispatch-1",
        outcome_status="unknown",
        outcome_score=0.0,
        confidence_score=0.0,
    )


class TestTheLookupUsesTheMarker:
    def test_it_is_not_an_anti_join_any_more(self):
        """The whole point: no LEFT JOIN over the downstream table, no time window."""
        sql = " ".join(str(FeedbackRuntimeStore._PENDING_SQL).split())
        assert "d.feedback_pending" in sql
        assert "LEFT JOIN" not in sql.upper()
        assert "make_interval" not in sql, "no time bound -- that approach strands the backlog"
        assert "ORDER BY d.generated_at ASC" in sql and "LIMIT 1" in sql

    def test_it_reads_without_opening_a_write_transaction(self):
        eng = _Engine(connect_script=[_Result(row=None)])
        assert _store(eng).load_latest_dispatch_frame_without_feedback() is None
        assert [c[0] for c in eng.calls] == ["connect"]


class TestTheMarkerIsClearedAtomically:
    def test_insert_and_clear_share_one_transaction(self):
        """Separate transactions would mean a crash between them either loses the work or
        duplicates it. Only one of those is recoverable."""
        eng = _Engine(begin_script=[_Result(), _Result()])
        _store(eng).save_feedback_frame(_frame())
        scopes = {c[0] for c in eng.calls}
        assert scopes == {"begin"}, "both statements must run inside engine.begin()"
        assert len(eng.calls) == 2
        assert "INSERT INTO substrate_feedback_frames" in eng.calls[0][1]
        assert "SET feedback_pending = false" in eng.calls[1][1]

    def test_it_clears_the_marker_on_the_SOURCE_dispatch_row(self):
        """Not the feedback frame's own id -- an easy and silent off-by-one."""
        eng = _Engine(begin_script=[_Result(), _Result()])
        _store(eng).save_feedback_frame(_frame())
        assert eng.calls[1][2] == {"frame_id": "dispatch-1"}


class TestTheReconcilerCanOnlyAddWork:
    def test_it_sets_the_marker_true_never_false(self):
        eng = _Engine(begin_script=[_Result(rowcount=0)])
        _store(eng).reconcile_feedback_pending(force=True)
        sql = eng.calls[0][1]
        assert "SET feedback_pending = true" in sql
        assert "false" not in sql.lower(), "a reconciler that can clear markers can lose work"
        assert "NOT EXISTS" in sql, "it must only re-queue rows with no feedback frame"

    def test_it_is_rate_limited(self):
        """It IS the expensive anti-join. Running it per poll reinstates the original problem."""
        eng = _Engine(begin_script=[_Result(rowcount=0)] * 4)
        store = _store(eng, reconcile_interval_sec=900.0)
        store.reconcile_feedback_pending(force=True)
        store.reconcile_feedback_pending()
        store.reconcile_feedback_pending()
        assert len(eng.calls) == 1

    def test_force_bypasses_the_rate_limit(self):
        eng = _Engine(begin_script=[_Result(rowcount=0)] * 3)
        store = _store(eng)
        store.reconcile_feedback_pending(force=True)
        store.reconcile_feedback_pending(force=True)
        assert len(eng.calls) == 2

    def test_a_requeue_is_logged_loudly(self, caplog):
        """Re-queuing means work WOULD have been lost. That is not a debug line."""
        eng = _Engine(begin_script=[_Result(rowcount=7)])
        with caplog.at_level("WARNING"):
            assert _store(eng).reconcile_feedback_pending(force=True) == 7
        assert any("feedback_pending_reconciled" in r.getMessage() for r in caplog.records)

    def test_finding_nothing_is_silent(self, caplog):
        eng = _Engine(begin_script=[_Result(rowcount=0)])
        with caplog.at_level("WARNING"):
            assert _store(eng).reconcile_feedback_pending(force=True) == 0
        assert not any("feedback_pending_reconciled" in r.getMessage() for r in caplog.records)


class TestStaleMarkersCannotSpinTheFifo:
    def test_it_clears_the_specific_row(self):
        """The marker defaults to TRUE, so every pre-migration row starts pending even though
        most already have feedback. Without this the worker's already-have-feedback guard
        returns early without advancing, and the FIFO re-selects the same row forever."""
        eng = _Engine(begin_script=[_Result(), _Result(rowcount=0)])
        _store(eng).clear_feedback_pending("dispatch-9")
        first = eng.calls[0]
        assert "SET feedback_pending = false" in first[1]
        assert "WHERE frame_id = :frame_id" in first[1]
        assert first[2] == {"frame_id": "dispatch-9"}

    def test_it_also_drains_a_batch_of_already_done_rows(self):
        """One row per poll is not enough. `ORDER BY generated_at ASC` puts every stale row
        ahead of genuinely new work, so a skipped backfill would drain 423k rows at 1 per 2s --
        about 9.8 days producing nothing, with no error and no log line to show it."""
        eng = _Engine(begin_script=[_Result(), _Result(rowcount=4999)])
        drained = _store(eng).clear_feedback_pending("dispatch-9")
        assert len(eng.calls) == 2
        bulk = eng.calls[1]
        assert "LIMIT :batch" in bulk[1] and bulk[2] == {"batch": 5000}
        assert "JOIN substrate_feedback_frames" in bulk[1], "only rows that ARE done"
        assert drained == 5000

    def test_the_batch_only_clears_rows_that_actually_have_feedback(self):
        """A bulk clear that could touch un-processed rows would lose work silently -- the same
        class of failure as the reverted time bound, arriving through the drain path."""
        eng = _Engine(begin_script=[_Result(), _Result(rowcount=0)])
        _store(eng).clear_feedback_pending("dispatch-9")
        bulk = eng.calls[1][1]
        assert "JOIN substrate_feedback_frames y" in bulk
        assert "WHERE x.feedback_pending" in bulk
