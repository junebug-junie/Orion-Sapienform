"""node_history: argument bounds, empty node, summary shape, parameterized SQL.

The fake connection answers orion/introspect/node_history.py's own SQL constants from an
in-memory row list with a straightforward Python reading of each query, so the summary
assembly (rounding, bucket timestamps, change ordering, budget) runs for real. The SQL itself
is checked against live Postgres by services/orion-hub/evals/run_node_history_live_eval.py.
"""
from __future__ import annotations

import asyncio
import json
import statistics
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from orion.introspect import node_history as nh
from orion.schemas.introspect import (
    NODE_HISTORY_MAX_HOURS,
    NODE_HISTORY_MAX_POINTS,
    IntrospectResultV1,
    NodeHistoryArguments,
)

NOW = datetime(2026, 10, 13, 18, 0, tzinfo=timezone.utc)
NODE = "node:substrate.bus_synaptic"
MCP_BUDGET = 12000


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
            pairs.sort(key=lambda r: (-abs(r["value"] - r["prev_value"]), r["observed_at"]))
            return pairs[:limit]
        if sql == nh.NODES_SQL:
            start, end = args
            out: dict[str, list] = {}
            for n, t, _ in self.rows:
                if start < t <= end:
                    out.setdefault(n, []).append(t)
            return [{"node_id": n, "n": len(ts), "last_at": max(ts)} for n, ts in sorted(out.items())]
        raise AssertionError(sql)


def _rows(node=NODE, hours=72, every_sec=40, base=0.02):
    start = NOW - timedelta(hours=hours)
    rows, t, i = [], start + timedelta(seconds=every_sec), 0
    while t <= NOW:
        rows.append((node, t, base + 0.001 * (i % 7)))
        t += timedelta(seconds=every_sec)
        i += 1
    return rows


def run(coro):
    return asyncio.run(coro)


# --- argument bounds --------------------------------------------------------

def test_defaults_and_window():
    args = NodeHistoryArguments(node_id=NODE)
    assert args.hours == 24
    start, end = args.window(NOW)
    assert end == NOW and start == NOW - timedelta(hours=24)


@pytest.mark.parametrize("hours", [0, -1, NODE_HISTORY_MAX_HOURS + 0.5, 10_000])
def test_hours_out_of_bounds_refused(hours):
    with pytest.raises(ValidationError):
        NodeHistoryArguments(node_id=NODE, hours=hours)


def test_max_hours_accepted():
    assert NodeHistoryArguments(node_id=NODE, hours=NODE_HISTORY_MAX_HOURS).hours == NODE_HISTORY_MAX_HOURS


def test_until_must_keep_window_inside_retention():
    now = datetime.now(timezone.utc)
    NodeHistoryArguments(node_id=NODE, hours=24, until=now - timedelta(hours=100))
    with pytest.raises(ValidationError, match="168 hours"):
        NodeHistoryArguments(node_id=NODE, hours=24, until=now - timedelta(hours=150))


def test_until_needs_timezone():
    with pytest.raises(ValidationError, match="timezone"):
        NodeHistoryArguments(node_id=NODE, until=datetime(2026, 10, 13, 12, 0))


def test_future_until_is_clamped_to_now():
    args = NodeHistoryArguments(node_id=NODE, hours=2, until=datetime.now(timezone.utc) + timedelta(days=3))
    start, end = args.window(NOW)
    assert end == NOW and start == NOW - timedelta(hours=2)


@pytest.mark.parametrize("bad", ["bus_synaptic", "node:", "node:x'; drop table t;--", "node:a b", "x" * 300])
def test_bad_node_ids_refused(bad):
    with pytest.raises(ValidationError):
        NodeHistoryArguments(node_id=bad)


def test_extra_fields_refused():
    with pytest.raises(ValidationError):
        NodeHistoryArguments(node_id=NODE, sql="select 1")


def test_queries_are_parameterized_and_bounded():
    for sql in (nh.SUMMARY_SQL, nh.SERIES_SQL, nh.CHANGES_SQL):
        assert "node_id = $1" in sql and "observed_at > $2" in sql and "observed_at <= $3" in sql
    assert "limit $4" in nh.CHANGES_SQL
    assert "observed_at > $1 and observed_at <= $2" in nh.NODES_SQL


# --- empty node ---------------------------------------------------------------

def test_unknown_node_returns_none_and_skips_follow_up_queries():
    conn = FakeConn(_rows())
    assert run(nh.read_node_history(conn, "node:substrate.nope", NOW - timedelta(hours=24), NOW)) is None
    assert [sql for sql, _ in conn.calls] == [nh.SUMMARY_SQL]


def test_node_with_no_reading_in_window_is_empty():
    old = [(NODE, NOW - timedelta(hours=100), 0.5)]
    assert run(nh.read_node_history(FakeConn(old), NODE, NOW - timedelta(hours=24), NOW)) is None


def test_no_nodes_in_window_lists_nothing():
    assert run(nh.list_history_nodes(FakeConn([]), NOW - timedelta(hours=1), NOW)) is None


# --- summary --------------------------------------------------------------

def test_summary_shape_and_budget_for_a_full_week():
    rows = _rows(hours=NODE_HISTORY_MAX_HOURS, every_sec=40)
    start = NOW - timedelta(hours=NODE_HISTORY_MAX_HOURS)
    conn = FakeConn(rows)
    item = run(nh.read_node_history(conn, NODE, start, NOW))
    assert item is not None and item.kind == "node_prediction_error_history" and item.id == NODE
    extra = item.extra
    assert extra["count"] == len(rows)
    assert len(extra["series"]) == NODE_HISTORY_MAX_POINTS and extra["empty_buckets"] == 0
    assert extra["series"][0][0] == start.isoformat().replace("+00:00", "Z")
    assert sum(p[1] for p in extra["series"]) == len(rows)
    assert len(extra["largest_changes"]) == 5
    assert extra["min"] == 0.02 and extra["max"] == 0.026
    result = IntrospectResultV1(ok=True, operation="node_history", as_of=NOW, total_available=1, items=[item])
    assert len(json.dumps(result.model_dump(mode="json"), ensure_ascii=False)) < MCP_BUDGET
    # every call passed the node, window bounds as parameters, never in the SQL text
    for sql, args in conn.calls:
        assert NODE not in sql and args[:3] == (NODE, start, NOW)


def test_gap_buckets_are_left_out_not_zeroed():
    rows = [r for r in _rows(hours=24) if not (NOW - timedelta(hours=12) <= r[1] < NOW - timedelta(hours=6))]
    item = run(nh.read_node_history(FakeConn(rows), NODE, NOW - timedelta(hours=24), NOW))
    assert item.extra["empty_buckets"] == 12  # 6h of 24h at 30-minute buckets
    assert all(p[1] > 0 and p[2] is not None for p in item.extra["series"])


def test_largest_change_is_dated_and_signed():
    rows = _rows(hours=6, every_sec=600)
    spike_at = rows[10][1]
    rows[10] = (NODE, spike_at, 0.9)
    item = run(nh.read_node_history(FakeConn(rows), NODE, NOW - timedelta(hours=6), NOW))
    top = item.extra["largest_changes"][0]
    assert top["delta"] > 0.8 and top["to"] == 0.9 and top["since_previous_sec"] == 600.0
    assert top["at"] in {spike_at.isoformat().replace("+00:00", "Z"),
                         rows[11][1].isoformat().replace("+00:00", "Z")}
    assert "Largest single change" in item.text


def test_tiny_values_keep_significant_figures():
    rows = [(NODE, NOW - timedelta(minutes=10), 3.2e-9), (NODE, NOW - timedelta(minutes=5), 1.23456e-7)]
    item = run(nh.read_node_history(FakeConn(rows), NODE, NOW - timedelta(hours=1), NOW))
    assert item.extra["min"] == 3.2e-9 and item.extra["max"] == 1.235e-7


def test_index_lists_every_node_with_counts():
    rows = _rows(hours=2) + _rows(node="node:substrate.cabinet", hours=2, every_sec=600)
    item = run(nh.list_history_nodes(FakeConn(rows), NOW - timedelta(hours=2), NOW))
    names = [n["node_id"] for n in item.extra["nodes"]]
    assert names == ["node:substrate.bus_synaptic", "node:substrate.cabinet"]
    assert "node_id=" in item.text
