"""ROADMAP D2: this service's half of the bounded pending-work scan.

The scan LOGIC (rate limit, backstop, tripwire, rollback) is tested once in
`orion/tests/test_pending_scan.py`. Duplicating it here would just pin the same behaviour twice.
What is service-specific, and what these tests cover, is the wiring: that the store actually
delegates to the helper, that its two SQL shapes differ only by the window predicate, and that
the settings reach it -- the ways this could be broken without the shared tests noticing.
"""
from __future__ import annotations

import json

import pytest

from app.store import FeedbackRuntimeStore
from orion.db.pending_scan import BoundedPendingScan


def _store(**kw):
    s = FeedbackRuntimeStore.__new__(FeedbackRuntimeStore)
    s._scan = BoundedPendingScan(label="dispatch->feedback", **kw)
    return s


def _sql(stmt) -> str:
    return " ".join(str(stmt).split())


class TestTheTwoQueriesDifferOnlyByTheBound:
    """If they diverge in any other way, the backstop stops being a faithful superset of the
    fast path -- and the bound would then change WHICH row is chosen, not just how fast."""

    def test_only_the_window_predicate_differs(self):
        fast = _sql(FeedbackRuntimeStore._FAST_PATH_SQL)
        back = _sql(FeedbackRuntimeStore._BACKSTOP_SQL)
        assert "make_interval(secs => :window_sec)" in fast
        assert "make_interval" not in back
        stripped = fast.replace("AND d.generated_at > now() - make_interval(secs => :window_sec) ", "")
        assert stripped == back

    def test_both_order_oldest_first(self):
        for stmt in (FeedbackRuntimeStore._FAST_PATH_SQL, FeedbackRuntimeStore._BACKSTOP_SQL):
            assert "ORDER BY d.generated_at ASC" in _sql(stmt)
            assert "LIMIT 1" in _sql(stmt)

    def test_both_select_generated_at_for_the_tripwire(self):
        """The backstop's warning names the row's age; selecting it in both keeps the two
        result shapes identical."""
        for stmt in (FeedbackRuntimeStore._FAST_PATH_SQL, FeedbackRuntimeStore._BACKSTOP_SQL):
            assert "d.generated_at" in _sql(stmt)


class TestTheStoreDelegates:
    def test_load_uses_the_scan_helper(self, monkeypatch):
        store = _store()
        seen = {}

        def _fake_fetch(conn, *, bounded_sql, unbounded_sql):
            seen["bounded"] = _sql(bounded_sql)
            seen["unbounded"] = _sql(unbounded_sql)
            return {"dispatch_frame_json": json.dumps({
                "frame_id": "11111111-1111-1111-1111-111111111111",
                "generated_at": "2026-08-19T03:00:00+00:00",
                "source_policy_frame_id": "22222222-2222-2222-2222-222222222222",
                "source_proposal_frame_id": "33333333-3333-3333-3333-333333333333",
            })}

        monkeypatch.setattr(store._scan, "fetch", _fake_fetch)
        conn = type("C", (), {"__enter__": lambda s: s, "__exit__": lambda *a: False})()
        store._engine = type("E", (), {"connect": staticmethod(lambda: conn)})()
        frame = store.load_latest_dispatch_frame_without_feedback()
        assert frame is not None
        assert "make_interval" in seen["bounded"] and "make_interval" not in seen["unbounded"]

    def test_none_result_returns_none(self, monkeypatch):
        store = _store()
        monkeypatch.setattr(store._scan, "fetch", lambda *a, **k: None)
        conn = type("C", (), {"__enter__": lambda s: s, "__exit__": lambda *a: False})()
        store._engine = type("E", (), {"connect": staticmethod(lambda: conn)})()
        assert store.load_latest_dispatch_frame_without_feedback() is None


class TestSettingsReachTheScan:
    @pytest.mark.parametrize("window,backstop", [(3600.0, 300.0), (0.0, 0.0), (900.0, 60.0)])
    def test_constructor_passes_them_through(self, monkeypatch, window, backstop):
        captured = {}

        class _FakeEngine:
            pass

        monkeypatch.setattr("app.store.create_engine", lambda *a, **k: _FakeEngine())
        store = FeedbackRuntimeStore(
            "postgresql://x/y", scan_window_sec=window, backstop_interval_sec=backstop
        )
        assert store._scan.window_sec == window
        assert store._scan.backstop_interval_sec == backstop
        assert store._scan.label == "dispatch->feedback"
