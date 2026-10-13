"""Read a substrate node's prediction-error history for the orion-introspect ``node_history`` tool.

Source: ``substrate_node_prediction_error_history`` (one row per real ``observed_at`` advance,
written by orion-substrate-runtime, pruned at SUBSTRATE_PE_HISTORY_RETENTION_HOURS). Every
query is parameterized, bounded to one window and one node, and aggregated in Postgres, so a
7-day window of a busy node (~14k rows) still returns a few kilobytes.

A stale node writes no rows, so a series bucket with no readings is "no new reading", never
zero: those buckets are left out and counted, not filled.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from orion.schemas.introspect import NODE_HISTORY_MAX_POINTS, NODE_HISTORY_TOP_CHANGES, IntrospectItemV1

TABLE = "substrate_node_prediction_error_history"

SUMMARY_SQL = f"""
select count(*)::int as n,
       min(value) as min_v,
       max(value) as max_v,
       percentile_cont(0.5) within group (order by value) as median_v,
       (array_agg(value order by observed_at asc))[1] as first_v,
       min(observed_at) as first_at,
       (array_agg(value order by observed_at desc))[1] as last_v,
       max(observed_at) as last_at
from {TABLE}
where node_id = $1 and observed_at > $2 and observed_at <= $3
"""

# $4 = bucket width in seconds, $5 = bucket count. The last bucket is closed on the right.
SERIES_SQL = f"""
select least(floor(extract(epoch from (observed_at - $2)) / $4)::int, $5 - 1) as b,
       count(*)::int as n, avg(value) as mean_v, min(value) as min_v, max(value) as max_v
from {TABLE}
where node_id = $1 and observed_at > $2 and observed_at <= $3
group by 1
order by 1
"""

CHANGES_SQL = f"""
select observed_at, value, prev_at, prev_value
from (
    select observed_at, value,
           lag(observed_at) over w as prev_at,
           lag(value) over w as prev_value
    from {TABLE}
    where node_id = $1 and observed_at > $2 and observed_at <= $3
    window w as (order by observed_at)
) d
where prev_value is not null
order by abs(value - prev_value) desc, observed_at
limit $4
"""

NODES_SQL = f"""
select node_id, count(*)::int as n, max(observed_at) as last_at
from {TABLE}
where observed_at > $1 and observed_at <= $2
group by node_id
order by node_id
"""


def sig(value: Any) -> float | None:
    """Four significant figures: readings span ~1e-6 to ~1, so fixed decimals would zero the small ones."""
    if value is None:
        return None
    return float(f"{float(value):.4g}")


def _iso(value: datetime) -> str:
    aware = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _hours_label(start: datetime, end: datetime) -> str:
    hours = (end - start).total_seconds() / 3600.0
    return f"{hours:g}h"


async def read_node_history(conn: Any, node_id: str, start: datetime, end: datetime) -> IntrospectItemV1 | None:
    """One summary item for ``node_id`` over (start, end], or None when it has no reading there."""
    summary = await conn.fetchrow(SUMMARY_SQL, node_id, start, end)
    if summary is None or not summary["n"]:
        return None
    width = (end - start).total_seconds() / NODE_HISTORY_MAX_POINTS
    buckets = await conn.fetch(SERIES_SQL, node_id, start, end, width, NODE_HISTORY_MAX_POINTS)
    changes = await conn.fetch(CHANGES_SQL, node_id, start, end, NODE_HISTORY_TOP_CHANGES)
    series = [
        [_iso(start + timedelta(seconds=width * int(r["b"]))), int(r["n"]),
         sig(r["mean_v"]), sig(r["min_v"]), sig(r["max_v"])]
        for r in buckets
    ]
    largest = [
        {
            "at": _iso(r["observed_at"]), "from": sig(r["prev_value"]), "to": sig(r["value"]),
            "delta": sig(float(r["value"]) - float(r["prev_value"])),
            "since_previous_sec": round((r["observed_at"] - r["prev_at"]).total_seconds(), 1),
        }
        for r in changes
    ]
    n = int(summary["n"])
    label = _hours_label(start, end)
    text = (
        f"{node_id} prediction error over the {label} ending {_iso(end)}: {n} readings, "
        f"min {sig(summary['min_v'])}, median {sig(summary['median_v'])}, max {sig(summary['max_v'])}; "
        f"first {sig(summary['first_v'])} at {_iso(summary['first_at'])}, "
        f"last {sig(summary['last_v'])} at {_iso(summary['last_at'])}."
    )
    if largest:
        top = largest[0]
        text += f" Largest single change {top['delta']:+g} at {top['at']}."
    return IntrospectItemV1(
        id=node_id,
        occurred_at=summary["last_at"],
        kind="node_prediction_error_history",
        epistemic_status="record",
        text=text,
        extra={
            "window_start": _iso(start),
            "window_end": _iso(end),
            "count": n,
            "min": sig(summary["min_v"]),
            "median": sig(summary["median_v"]),
            "max": sig(summary["max_v"]),
            "first": {"at": _iso(summary["first_at"]), "value": sig(summary["first_v"])},
            "last": {"at": _iso(summary["last_at"]), "value": sig(summary["last_v"])},
            "bucket_sec": round(width, 1),
            "series_columns": ["bucket_start", "n", "mean", "min", "max"],
            "series": series,
            "empty_buckets": NODE_HISTORY_MAX_POINTS - len(series),
            "largest_changes": largest,
        },
    )


async def list_history_nodes(conn: Any, start: datetime, end: datetime) -> IntrospectItemV1 | None:
    """One item naming every node with readings in (start, end], or None when there are none."""
    rows = await conn.fetch(NODES_SQL, start, end)
    if not rows:
        return None
    nodes = [{"node_id": r["node_id"], "count": int(r["n"]), "last_at": _iso(r["last_at"])} for r in rows]
    label = _hours_label(start, end)
    text = (
        f"{len(nodes)} nodes have prediction-error readings in the {label} ending {_iso(end)}: "
        + ", ".join(f"{n['node_id']} ({n['count']})" for n in nodes)
        + ". Pass node_id=<one of these> for its history."
    )
    return IntrospectItemV1(
        id="node_history_index",
        occurred_at=max(r["last_at"] for r in rows),
        kind="node_history_index",
        epistemic_status="record",
        text=text,
        extra={"window_start": _iso(start), "window_end": _iso(end), "nodes": nodes},
    )
