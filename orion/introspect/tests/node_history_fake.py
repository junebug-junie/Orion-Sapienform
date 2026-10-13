"""In-memory stand-in for asyncpg answering orion/introspect/node_history.py's SQL constants.

Each query is read straight from its SQL: (start, end] window, last bucket clamped to
count - 1, lag over the window ordered by observed_at, changes ordered by |delta| desc then
most recent first. test_node_history.py pins the SQL text to those same rules, so the fake
and the SQL cannot drift apart silently.
"""
from __future__ import annotations

import statistics

from orion.introspect import node_history as nh


class FakeConn:
    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda r: (r[0], r[1]))  # (node_id, observed_at, value)
        self.calls: list[tuple[str, tuple]] = []

    def _window(self, node, start, end):
        return [(t, v) for n, t, v in self.rows if (node is None or n == node) and start < t <= end]

    async def fetchrow(self, sql, *args):
        self.calls.append((sql, args))
        assert sql == nh.SUMMARY_SQL
        node, start, end = args
        pts = self._window(node, start, end)
        if not pts:
            return {"n": 0, "min_v": None, "max_v": None, "median_v": None, "first_v": None,
                    "first_at": None, "last_v": None, "last_at": None}
        values = [v for _, v in pts]
        return {"n": len(pts), "min_v": min(values), "max_v": max(values),
                "median_v": statistics.median(values), "first_v": pts[0][1], "first_at": pts[0][0],
                "last_v": pts[-1][1], "last_at": pts[-1][0]}

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        if sql == nh.SERIES_SQL:
            node, start, end, width, count = args
            buckets: dict[int, list[float]] = {}
            for t, v in self._window(node, start, end):
                b = min(int((t - start).total_seconds() // width), count - 1)
                buckets.setdefault(b, []).append(v)
            return [{"b": b, "n": len(vs), "mean_v": sum(vs) / len(vs), "min_v": min(vs), "max_v": max(vs)}
                    for b, vs in sorted(buckets.items())]
        if sql == nh.CHANGES_SQL:
            node, start, end, limit = args
            pts = self._window(node, start, end)
            pairs = [{"observed_at": b[0], "value": b[1], "prev_at": a[0], "prev_value": a[1]}
                     for a, b in zip(pts, pts[1:])]
            pairs.sort(key=lambda r: (-abs(r["value"] - r["prev_value"]), -r["observed_at"].timestamp()))
            return pairs[:limit]
        if sql == nh.NODES_SQL:
            start, end = args
            out: dict[str, list] = {}
            for n, t, _ in self.rows:
                if start < t <= end:
                    out.setdefault(n, []).append(t)
            return [{"node_id": n, "n": len(ts), "last_at": max(ts)} for n, ts in sorted(out.items())]
        raise AssertionError(sql)

    async def fetchval(self, sql, *args):
        self.calls.append((sql, args))
        assert sql == nh.EARLIEST_SQL
        ts = [t for n, t, _ in self.rows if n == args[0]]
        return min(ts) if ts else None
