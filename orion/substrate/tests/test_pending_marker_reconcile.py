"""Bounded + chunked-full pending-marker reconciler (2026-09-25).

The fake engine (pending_marker_fake.py) does not just record SQL: it APPLIES each requeue statement to an
in-memory table using the statement's own generated_at bounds. A statement with no
`generated_at >= :lo` bound (the pre-2026-09-25 unbounded anti-join) re-queues every row,
so the "old rows are left alone by the frequent sweep" tests fail against that shape.
"""
from __future__ import annotations

import logging
import pytest

from orion.substrate.tests.pending_marker_fake import DAY, NOW, Clock, FakeDb, Row, updates as _updates
from orion.substrate import pending_marker_reconcile as pmr
from orion.substrate.pending_marker_reconcile import (
    PendingMarkerReconciler,
    PendingMarkerSpec,
    bounded_requeue_sql,
    full_candidates_sql,
    full_requeue_batch_sql,
)

SPEC = PendingMarkerSpec(
    parent_table="parent_frames",
    marker_column="child_pending",
    child_table="child_frames",
    child_fk_column="source_parent_frame_id",
    log_prefix="child_pending",
    parent_label="parent frames",
    child_label="child frame",
)


def _reconciler(clock=None, *, hour=-1, full_interval=DAY, wall=NOW, **kw):
    return PendingMarkerReconciler(
        SPEC,
        interval_sec=kw.get("interval", 900.0),
        window_sec=kw.get("window", DAY),  # 1 day: Row ages are in days
        full_sweep_interval_sec=full_interval,
        full_sweep_hour_utc=hour,
        monotonic=clock or Clock(),
        utcnow=lambda: wall,
    )


class TestBoundedSweep:
    def test_only_recent_rows_are_requeued(self):
        recent = Row("recent", 0.5)
        old = Row("old", 30)
        db = FakeDb([recent, old])
        assert _reconciler().run(db, force=True) == 1
        assert recent.pending is True
        assert old.pending is False, "the frequent sweep must not walk the whole history"

    def test_requeue_behaviour_unchanged_for_recent_rows(self):
        lost = Row("lost", 0.1)  # marker cleared, no child: must come back
        done = Row("done", 0.1, has_child=True)  # processed: must stay cleared
        waiting = Row("waiting", 0.1, pending=True)  # already queued: untouched
        db = FakeDb([lost, done, waiting])
        assert _reconciler().run(db, force=True) == 1
        assert (lost.pending, done.pending, waiting.pending) == (True, False, True)

    def test_it_is_one_short_statement_bounded_by_generated_at(self):
        db = FakeDb([Row("r", 0.1)])
        _reconciler(window=7200.0).run(db, force=True)
        assert len(db.statements) == 1
        scope, sql, params = db.statements[0]
        assert scope == "begin"
        assert "p.generated_at >= now() - make_interval(secs => :window_sec)" in sql
        assert params == {"window_sec": 7200.0}

    def test_rate_limited_and_seeded_to_now(self):
        clock = Clock()
        rec = _reconciler(clock)
        db = FakeDb([Row("r", 0.1)])
        assert rec.run(db) == 0, "no sweep on the first tick after boot"
        clock.t += 899
        rec.run(db)
        assert _updates(db) == []
        clock.t += 1
        rec.run(db)
        assert len(_updates(db)) == 1


class TestFullSweep:
    def test_full_sweep_reaches_old_rows(self):
        rows = [Row(f"r{i}", age) for i, age in enumerate((0.2, 1.5, 7.3, 40.0))]
        rows.append(Row("done", 40.0, has_child=True))
        db = FakeDb(rows)
        assert _reconciler().run(db, force=True, full=True) == 4
        assert [r.pending for r in rows] == [True, True, True, True, False]

    def test_full_sweep_scan_is_read_only_and_writes_are_short_batches(self, monkeypatch):
        monkeypatch.setattr(pmr, "FULL_SWEEP_UPDATE_BATCH", 2)
        rows = [Row(f"r{i}", 10 + i) for i in range(5)]
        db = FakeDb(rows)
        assert _reconciler().run(db, force=True, full=True) == 5
        scan, *writes = db.statements
        assert scan[0] == "connect" and scan[1].startswith("SELECT p.frame_id")
        assert "generated_at" not in scan[1], "the full sweep covers ALL history"
        assert [w[0] for w in writes] == ["begin"] * 3, "one short transaction per batch"
        assert [len(w[2]["ids"]) for w in writes] == [2, 2, 1]

    def test_full_sweep_rechecks_each_candidate_before_requeueing(self):
        """A child written between the scan and the UPDATE must not re-queue its parent."""
        late = Row("late", 10)
        lost = Row("lost", 10)
        db = FakeDb([late, lost])

        def child_arrives():
            late.has_child = True

        db.after_candidate_scan = child_arrives
        assert _reconciler().run(db, force=True, full=True) == 1
        assert late.pending is False and lost.pending is True

    def test_full_sweep_with_nothing_to_do_writes_nothing(self):
        db = FakeDb([Row("done", 3, has_child=True)])
        assert _reconciler().run(db, force=True, full=True) == 0
        assert _updates(db) == []

    def test_ungated_full_sweep_runs_on_its_interval(self):
        clock = Clock()
        rec = _reconciler(clock, hour=-1, full_interval=DAY)
        old = Row("old", 10)
        db = FakeDb([old])
        clock.t += 900
        rec.run(db)
        assert old.pending is False, "not due yet: only the bounded sweep ran"
        clock.t += DAY
        rec.run(db)
        assert old.pending is True, "a day later the full sweep must reach it"

    def test_hour_gated_full_sweep_only_in_its_hour_and_once_per_day(self):
        clock = Clock()
        wall = {"now": NOW.replace(hour=8)}
        rec = PendingMarkerReconciler(
            SPEC, interval_sec=900, window_sec=DAY, full_sweep_interval_sec=DAY,
            full_sweep_hour_utc=9, monotonic=clock, utcnow=lambda: wall["now"],
        )
        old = Row("old", 10)
        db = FakeDb([old])
        clock.t += 900
        rec.run(db)
        assert old.pending is False, "08:xx UTC is not the sweep hour"
        wall["now"] = NOW.replace(hour=9)
        clock.t += 900
        rec.run(db)
        assert old.pending is True
        old.pending = False
        clock.t += 900
        rec.run(db)
        assert old.pending is False, "second tick in the same hour must not re-run it"
        clock.t += DAY - 1800  # next day, same hour, within the one-hour slack
        rec.run(db)
        assert old.pending is True

    def test_zero_interval_disables_full_sweep(self):
        clock = Clock()
        rec = _reconciler(clock, hour=-1, full_interval=0)
        old = Row("old", 10)
        db = FakeDb([old])
        for _ in range(5):
            clock.t += 10 * DAY
            rec.run(db)
        assert old.pending is False


class TestSafety:
    @pytest.mark.parametrize(
        "build", [bounded_requeue_sql, full_requeue_batch_sql], ids=["bounded", "batch"]
    )
    def test_it_only_sets_the_marker_true(self, build):
        sql = " ".join(build(SPEC).split())
        assert "SET child_pending = true" in sql
        assert "false" not in sql.lower()
        assert "NOT p.child_pending" in sql
        assert "NOT EXISTS ( SELECT 1 FROM child_frames c WHERE c.source_parent_frame_id = p.frame_id )" in sql

    def test_candidate_scan_is_a_select(self):
        sql = " ".join(full_candidates_sql(SPEC).split())
        assert sql.startswith("SELECT p.frame_id FROM parent_frames p WHERE NOT p.child_pending")
        assert "UPDATE" not in sql.upper()

    def test_a_requeue_is_logged_loudly_and_nothing_is_silent(self, caplog):
        with caplog.at_level(logging.WARNING):
            _reconciler().run(FakeDb([Row("r", 0.1, has_child=True)]), force=True)
        assert not any("child_pending_reconciled" in r.getMessage() for r in caplog.records)
        with caplog.at_level(logging.WARNING):
            _reconciler().run(FakeDb([Row("r", 0.1)]), force=True)
        assert any("child_pending_reconciled requeued=1" in r.getMessage() for r in caplog.records)

    def test_full_sweep_completion_is_observable(self, caplog):
        with caplog.at_level(logging.INFO):
            _reconciler().run(FakeDb([Row("r", 3)]), force=True, full=True)
        assert any("child_pending_full_sweep_done candidates=1" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("kw", [{"window_sec": 0}, {"full_sweep_hour_utc": 24}])
    def test_bad_config_is_refused(self, kw):
        with pytest.raises(ValueError):
            PendingMarkerReconciler(SPEC, **kw)
