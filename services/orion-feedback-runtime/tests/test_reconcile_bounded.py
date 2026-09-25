"""orion-feedback-runtime: the `feedback_pending` reconciler is bounded, with a rare read-only full sweep (2026-09-25).

The unbounded version (`UPDATE substrate_execution_dispatch_frames ... WHERE NOT feedback_pending AND NOT EXISTS (...)`
over all history, every 15 min) was one of the top-3 I/O statements on athena's Postgres and
found nothing in 72 h. These tests pin that the frequent sweep only touches recent rows, that
the full sweep still reaches old rows on its schedule, and that recent-row requeue behaviour
is unchanged. The fake DB applies each statement by its own generated_at bounds, so the old
unbounded statement fails the "old row untouched" test.
"""
from __future__ import annotations

from app.settings import Settings
from app.store import FeedbackRuntimeStore
from orion.substrate.tests.pending_marker_fake import DAY, NOW, Clock, FakeDb, Row, updates


def _store(**kw):
    store = FeedbackRuntimeStore("postgresql://u:p@127.0.0.1:1/none", **kw)
    return store


def _run(store, db, **kw):
    store._engine = db
    return store.reconcile_feedback_pending(**kw)


def _pin_clock(store, clock, *, wall=NOW):
    r = store._reconciler
    r._monotonic = clock
    r._utcnow = lambda: wall
    r.last_sweep_mono = clock()
    if r.full_sweep_hour_utc < 0:
        r.last_full_sweep_mono = clock()


class TestItGuardsTheRightTables:
    def test_spec_names_the_real_marker_and_child_fk(self):
        """A wrong column here is a silent no-op safety net."""
        db = FakeDb([Row("r", 0.02)])
        _run(_store(), db, force=True)
        (_, sql, _), = updates(db)
        assert sql.startswith("UPDATE substrate_execution_dispatch_frames p SET feedback_pending = true")
        assert "NOT p.feedback_pending" in sql
        assert "SELECT 1 FROM substrate_feedback_frames c WHERE c.source_execution_dispatch_frame_id = p.frame_id" in sql


class TestBoundedSweep:
    def test_frequent_sweep_leaves_old_rows_to_the_full_sweep(self):
        recent, old = Row("recent", 0.02), Row("old", 30)
        assert _run(_store(), FakeDb([recent, old]), force=True) == 1
        assert recent.pending and not old.pending

    def test_recent_requeue_behaviour_unchanged(self):
        lost, done = Row("lost", 0.02), Row("done", 0.02, has_child=True)
        assert _run(_store(), FakeDb([lost, done]), force=True) == 1
        assert lost.pending is True and done.pending is False

    def test_window_setting_reaches_the_query(self):
        db = FakeDb([Row("r", 0.02)])
        _run(_store(reconcile_window_sec=3600.0), db, force=True)
        assert updates(db)[0][2] == {"window_sec": 3600.0}

    def test_default_window_is_two_hours(self):
        """Each checked row costs a random read on athena's spinning disk: keep it small."""
        three_hours = Row("three_hours", 3 / 24)
        _run(_store(), FakeDb([three_hours]), force=True)
        assert three_hours.pending is False

    def test_still_rate_limited(self):
        clock = Clock()
        store = _store()
        _pin_clock(store, clock)
        db = FakeDb([Row("r", 0.02)])
        _run(store, db)
        clock.t += 899
        _run(store, db)
        assert updates(db) == []


class TestFullSweepSchedule:
    def test_full_sweep_runs_in_its_hour_and_reaches_old_rows(self):
        clock = Clock()
        store = _store(reconcile_full_sweep_hour_utc=9)
        _pin_clock(store, clock, wall=NOW.replace(hour=8))
        old = Row("old", 30)
        db = FakeDb([old])
        clock.t += 900
        _run(store, db)
        assert old.pending is False, "not the sweep hour"
        store._reconciler._utcnow = lambda: NOW.replace(hour=9)
        clock.t += 900
        _run(store, db)
        assert old.pending is True
        scan = db.statements[-2]
        assert scan[0] == "connect" and scan[1].startswith("SELECT p.frame_id"), "read-only scan"
        assert updates(db)[-1][0] == "begin" and "= ANY(:ids)" in updates(db)[-1][1]

    def test_explicit_full_flag(self):
        old = Row("old", 30)
        assert _run(_store(), FakeDb([old]), force=True, full=True) == 1


class TestSettings:
    def test_defaults_and_env_keys(self, monkeypatch):
        for k in ("WINDOW_SEC", "FULL_SWEEP_INTERVAL_SEC", "FULL_SWEEP_HOUR_UTC"):
            monkeypatch.delenv(f"FEEDBACK_RECONCILE_{k}", raising=False)
        s = Settings(POSTGRES_URI="postgresql://x/y")
        assert s.feedback_reconcile_window_sec == 7200.0
        assert s.feedback_reconcile_full_sweep_interval_sec == DAY
        assert s.feedback_reconcile_full_sweep_hour_utc == 9
        monkeypatch.setenv("FEEDBACK_RECONCILE_WINDOW_SEC", "1800")
        monkeypatch.setenv("FEEDBACK_RECONCILE_FULL_SWEEP_HOUR_UTC", "-1")
        s = Settings(POSTGRES_URI="postgresql://x/y")
        assert s.feedback_reconcile_window_sec == 1800
        assert s.feedback_reconcile_full_sweep_hour_utc == -1
