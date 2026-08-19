"""ROADMAP D2: the bounded pending-work scan and the backstop that keeps it correct.

The bound is the performance fix. The backstop is the correctness guarantee, and it is what
these tests are mostly about: a pipeline stage that silently drops its own backlog is a far
worse failure than a slow one.
"""
from __future__ import annotations

import logging

import pytest

from orion.db.pending_scan import BoundedPendingScan


class _Result:
    def __init__(self, row): self._row = row
    def mappings(self): return self
    def first(self): return self._row


class _Conn:
    def __init__(self, script): self.script, self.calls = list(script), []
    def execute(self, sql, params=None):
        self.calls.append((str(sql), params))
        return _Result(self.script.pop(0) if self.script else None)


BOUNDED, UNBOUNDED = "SELECT ... bounded", "SELECT ... unbounded"
ROW = {"generated_at": "2026-08-19T03:00:00+00:00", "payload": "x"}


def _scan(**kw):
    kw.setdefault("label", "a->b")
    return BoundedPendingScan(**kw)


def test_the_fast_path_alone_answers_the_normal_case():
    scan, conn = _scan(), _Conn([ROW])
    assert scan.fetch(conn, bounded_sql=BOUNDED, unbounded_sql=UNBOUNDED) is ROW
    assert [c[0] for c in conn.calls] == [BOUNDED]
    assert conn.calls[0][1] == {"window_sec": 3600.0}


def test_a_straggler_outside_the_window_is_still_found():
    """Without this the bound would silently drop work the first time a stage fell behind."""
    scan, conn = _scan(), _Conn([None, ROW])
    assert scan.fetch(conn, bounded_sql=BOUNDED, unbounded_sql=UNBOUNDED) is ROW
    assert [c[0] for c in conn.calls] == [BOUNDED, UNBOUNDED]


def test_the_backstop_is_rate_limited():
    """Otherwise an IDLE system runs the expensive query every poll -- reintroducing exactly the
    behaviour this exists to remove, at precisely the time there is no work to justify it."""
    scan, conn = _scan(backstop_interval_sec=300.0), _Conn([None] * 6)
    for _ in range(3):
        scan.fetch(conn, bounded_sql=BOUNDED, unbounded_sql=UNBOUNDED)
    assert [c[0] for c in conn.calls].count(UNBOUNDED) == 1


def test_the_backstop_runs_again_after_its_interval():
    clock = {"t": 100.0}
    scan = _scan(backstop_interval_sec=300.0, monotonic=lambda: clock["t"])
    conn = _Conn([None] * 8)
    scan.fetch(conn, bounded_sql=BOUNDED, unbounded_sql=UNBOUNDED)
    clock["t"] += 299.0
    scan.fetch(conn, bounded_sql=BOUNDED, unbounded_sql=UNBOUNDED)
    assert [c[0] for c in conn.calls].count(UNBOUNDED) == 1, "299s < 300s must not re-run it"
    clock["t"] += 2.0
    scan.fetch(conn, bounded_sql=BOUNDED, unbounded_sql=UNBOUNDED)
    assert [c[0] for c in conn.calls].count(UNBOUNDED) == 2


def test_a_backstop_hit_logs_a_tripwire(caplog):
    scan, conn = _scan(), _Conn([None, ROW])
    with caplog.at_level(logging.WARNING):
        scan.fetch(conn, bounded_sql=BOUNDED, unbounded_sql=UNBOUNDED)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("pending_scan_backstop_hit" in m for m in msgs)
    assert any("a->b" in m for m in msgs), "the stage must be named, or two stages are one alarm"


def test_a_backstop_that_finds_nothing_is_silent(caplog):
    """An idle pipeline is not an alarm. Only a straggler is."""
    scan, conn = _scan(), _Conn([None, None])
    with caplog.at_level(logging.WARNING):
        assert scan.fetch(conn, bounded_sql=BOUNDED, unbounded_sql=UNBOUNDED) is None
    assert not any("pending_scan_backstop_hit" in r.getMessage() for r in caplog.records)


def test_a_row_without_the_diagnostic_column_does_not_break_the_tripwire():
    scan, conn = _scan(), _Conn([None, {"payload": "x"}])
    assert scan.fetch(conn, bounded_sql=BOUNDED, unbounded_sql=UNBOUNDED) is not None


class TestRollback:
    @pytest.mark.parametrize("window", [0.0, -1.0])
    def test_window_zero_runs_only_the_unbounded_query(self, window):
        scan, conn = _scan(window_sec=window), _Conn([ROW])
        assert scan.fetch(conn, bounded_sql=BOUNDED, unbounded_sql=UNBOUNDED) is ROW
        assert [c[0] for c in conn.calls] == [UNBOUNDED]

    def test_window_zero_is_not_rate_limited(self):
        """With the bound off, the rate limit must not silently become a 5-minute poll."""
        scan, conn = _scan(window_sec=0.0), _Conn([ROW, ROW, ROW])
        for _ in range(3):
            scan.fetch(conn, bounded_sql=BOUNDED, unbounded_sql=UNBOUNDED)
        assert [c[0] for c in conn.calls] == [UNBOUNDED] * 3

    def test_window_zero_does_not_log_a_tripwire(self, caplog):
        """Unbounded IS the behaviour, so nothing is out of window by definition."""
        scan, conn = _scan(window_sec=0.0), _Conn([ROW])
        with caplog.at_level(logging.WARNING):
            scan.fetch(conn, bounded_sql=BOUNDED, unbounded_sql=UNBOUNDED)
        assert not any("pending_scan_backstop_hit" in r.getMessage() for r in caplog.records)


def test_backstop_interval_zero_means_always():
    scan, conn = _scan(backstop_interval_sec=0.0), _Conn([None] * 4)
    for _ in range(2):
        scan.fetch(conn, bounded_sql=BOUNDED, unbounded_sql=UNBOUNDED)
    assert [c[0] for c in conn.calls].count(UNBOUNDED) == 2
