"""In-memory stand-in for Postgres that APPLIES pending-marker reconcile statements.

Shared by orion/substrate/tests and the policy / execution-dispatch / feedback runtime tests.
It honours only the restrictions the statement itself names:

* ``generated_at >= now() - make_interval(secs => :window_sec)`` -> only rows inside the window,
* ``frame_id = ANY(:ids)`` -> only those ids,
* ``NOT p.<marker>`` / ``NOT EXISTS`` -> skip already-pending rows / rows with a child,
* neither (the pre-2026-09-25 unbounded anti-join UPDATE) -> every row in history.

So the "old rows are left to the full sweep" tests fail against the old statement shape.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

NOW = datetime(2026, 9, 25, 9, 10, tzinfo=timezone.utc)
DAY = 86400.0


class Row:
    def __init__(self, frame_id, age_days, *, pending=False, has_child=False):
        self.frame_id = frame_id
        self.generated_at = NOW - timedelta(days=age_days)
        self.pending = pending
        self.has_child = has_child


class _Result:
    def __init__(self, rowcount=0, values=()):
        self.rowcount, self._values = rowcount, list(values)

    def partitions(self, size):
        vals = self._values
        for i in range(0, len(vals), size):
            yield [(v,) for v in vals[i : i + size]]


class FakeDb:
    def __init__(self, rows):
        self.rows = rows
        self.statements: list[tuple[str, str, dict]] = []  # (scope, sql, params)
        # Hook run after the candidate SELECT, before the batched UPDATEs (race tests).
        self.after_candidate_scan = None

    def connect(self):
        return _Conn(self, "connect")

    def begin(self):
        return _Conn(self, "begin")

    def matching(self, sql, params):
        out = []
        for r in self.rows:
            if "make_interval(secs => :window_sec)" in sql:
                if r.generated_at < NOW - timedelta(seconds=float(params["window_sec"])):
                    continue
            if "= ANY(:ids)" in sql and r.frame_id not in params["ids"]:
                continue
            # Apply each guard ONLY if the statement actually contains it, so a statement that
            # drops its re-check really does re-queue the wrong rows here.
            if "NOT p." in sql and r.pending:
                continue
            if "NOT EXISTS" in sql and r.has_child:
                continue
            out.append(r)
        return out


class _Conn:
    def __init__(self, db, scope):
        self.db, self.scope = db, scope

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execution_options(self, **_kw):  # stream_results/yield_per: no-op in the fake
        return self

    def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split())
        params = dict(params or {})
        self.db.statements.append((self.scope, sql, params))
        rows = self.db.matching(sql, params)
        if sql.startswith("SELECT"):
            values = [r.frame_id for r in rows]
            if self.db.after_candidate_scan:
                self.db.after_candidate_scan()
            return _Result(values=values)
        assert sql.startswith("UPDATE"), sql
        for r in rows:
            r.pending = True
        return _Result(rowcount=len(rows))


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def updates(db):
    return [s for s in db.statements if s[1].startswith("UPDATE")]
