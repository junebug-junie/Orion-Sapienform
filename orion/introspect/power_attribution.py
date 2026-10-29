"""Read the rows heat archaeology needs, for the orion-introspect ``power_attribution`` tool.

Three parameterized, window-bounded SELECTs, aggregated to hours in Postgres ($1 = window
start, $2 = window end). The analysis itself is pure: ``orion.physiology.heat_archaeology``.
Run inside a READ ONLY transaction by orion-hub's responder; nothing here writes.

* ``energy_hourly``: per-node wall watts from sql-writer's ``orion_energy_hourly`` rollup
  (kWh over the seconds the hour was covered; an hour under half covered is left out, never
  read as low).
* ``biometrics_hourly``: per-node hourly means of GPU watts, CPU watts and ``load_15m`` from
  ``orion_biometrics_cluster.measurements_by_node`` (raw rows are kept ~30 days).
* ``pool_busy_hours``: GPU-pool busy-hours per holder per hour from ``gpu_pool_events`` end
  rows (released / aborted / expired), each ``held_ms`` spread over the hours it covered (ends
  up to 7 days after the window are read, so a lease that outlives the window still counts
  inside it), plus leases still open at the window's end (``granted`` with no end row, granted
  in the last 6 h, counted up to the end); ``durable_run_workflow`` names a
  ``durable-runs:<run_id>`` holder's workflow.

Hour buckets are UTC hours: the reader sets ``timezone = 'UTC'`` for its own transaction.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Sequence

from orion.physiology.heat_archaeology import (
    POOL_NODE,
    HourReading,
    PoolBusy,
    analyze,
    holder_group,
)
from orion.schemas.heat_archaeology import NodeHeatArchaeologyV1
from orion.schemas.introspect import IntrospectItemV1

# An hour covered for less than this many seconds is a gap, not a low reading.
MIN_COVERED_SEC = 1800.0

ENERGY_SQL = """
select hour_start, source, kwh, covered_sec
from orion_energy_hourly
where hour_start >= $1 and hour_start < $2 and covered_sec >= $3
order by hour_start
"""

BIOMETRICS_SQL = """
select date_trunc('hour', c.observed_at) as hour, n.key as node,
       avg((n.value->>'gpu_watts_total')::float) filter (where jsonb_typeof(n.value->'gpu_watts_total') = 'number') as gpu_w,
       avg((n.value->>'cpu_watts_total')::float) filter (where jsonb_typeof(n.value->'cpu_watts_total') = 'number') as cpu_w,
       avg((n.value->>'load_15m')::float) filter (where jsonb_typeof(n.value->'load_15m') = 'number') as load_15m
from orion_biometrics_cluster c, jsonb_each(c.measurements_by_node) n
where c.observed_at >= $1 and c.observed_at < $2 and jsonb_typeof(c.measurements_by_node) = 'object'
group by 1, 2
"""

POOL_BUSY_SQL = """
with ends as (
  select e.holder, w.workflow,
         e.generated_at as t1,
         e.generated_at - make_interval(secs => e.held_ms / 1000.0) as t0
  from gpu_pool_events e
  left join durable_run_workflow w on e.holder = 'durable-runs:' || w.run_id
  where e.event in ('released', 'aborted', 'expired') and e.held_ms > 0
    and e.generated_at >= $1::timestamptz and e.generated_at < $2::timestamptz + interval '7 days'
), opens as (
  -- Still held at the window's end (no end row at all): counted up to the end, but only when
  -- granted in the last OPEN_LEASE_MAX_H hours, since the pool sometimes never records an end
  -- and an unbounded ghost would read as a days-long hold.
  select g.holder, w.workflow, g.generated_at as t0,
         least($2::timestamptz, g.generated_at + interval '6 hours') as t1
  from gpu_pool_events g
  left join durable_run_workflow w on g.holder = 'durable-runs:' || w.run_id
  where g.event = 'granted'
    and g.generated_at >= $2::timestamptz - interval '6 hours' and g.generated_at < $2::timestamptz
    and not exists (
      select 1 from gpu_pool_events x
      where x.lease_id = g.lease_id and x.generated_at > g.generated_at
        and x.event in ('released', 'aborted', 'expired', 'cancelled'))
), spans as (
  select * from ends union all select * from opens
)
select h as hour, holder, workflow,
       sum(extract(epoch from least(t1, h + interval '1 hour') - greatest(t0, h))) / 3600.0 as busy_h
from spans, generate_series(date_trunc('hour', t0), date_trunc('hour', t1), interval '1 hour') h
where h >= date_trunc('hour', $1::timestamptz) and h < $2::timestamptz
group by 1, 2, 3
"""

QUERIES = {"energy_hourly": ENERGY_SQL, "biometrics_hourly": BIOMETRICS_SQL, "pool_busy_hours": POOL_BUSY_SQL}

ITEM_KIND = "power_attribution"
CLIP_MARKER = " [clipped]"
QUERIES_KIND = "power_attribution_queries"

# The whole result must fit the harness proxy's per-tool budget (ORION_FCC_MCP_TOOL_RESULT_MAX_CHARS;
# see orion/schemas/introspect.py MAX_ITEMS), with room for the envelope around the items.
RESULT_BUDGET_CHARS = 11500
# Dropped in this order until the items fit; each drop is named in the item's extra.omitted.
# The trend line and evidence refs go first (re-readable with the SQL item), then the structure
# the summary sentence already states. Summaries are clipped only if even that does not fit.
_SHED_ORDER = (
    "daily_wall_w", "evidence", "reservations", "regime_shifts", "attributions", "gpu_fit", "fit", "steps",
)


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value).astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def merge_rows(
    energy: Sequence[Any], biometrics: Sequence[Any],
) -> list[HourReading]:
    """One HourReading per (node, hour) from the energy and biometrics rows."""
    by_key: dict[tuple[str, datetime], dict[str, Any]] = {}
    for r in energy:
        covered = float(r["covered_sec"])
        if covered <= 0:
            continue
        key = (str(r["source"]), _utc(r["hour_start"]))
        by_key.setdefault(key, {})["wall_w"] = float(r["kwh"]) * 3_600_000.0 / covered
    for r in biometrics:
        key = (str(r["node"]), _utc(r["hour"]))
        slot = by_key.setdefault(key, {})
        for col in ("gpu_w", "cpu_w", "load_15m"):
            if r[col] is not None:
                slot[col] = float(r[col])
    return [HourReading(hour=h, node=n, **vals) for (n, h), vals in sorted(by_key.items(), key=lambda kv: kv[0][1])]


def busy_rows(rows: Sequence[Any]) -> list[PoolBusy]:
    out = []
    for r in rows:
        group, reservation = holder_group(str(r["holder"] or "?"), r["workflow"])
        out.append(PoolBusy(hour=_utc(r["hour"]), group=group, busy_h=float(r["busy_h"]), reservation=reservation))
    return out


async def fetch_power_rows(
    conn: Any, start: datetime, end: datetime, node: str | None = None,
) -> tuple[list[HourReading], list[PoolBusy], list[str]]:
    """(hourly readings, pool busy-hours, nodes to analyze). Three SELECTs, nothing else.

    ``set local timezone`` lasts only for the caller's (READ ONLY) transaction: date_trunc on a
    timestamptz buckets in the session zone, and a half-hour zone would shift every hour.
    """
    await conn.execute("set local timezone = 'UTC'")
    energy = await conn.fetch(ENERGY_SQL, start, end, MIN_COVERED_SEC)
    biometrics = await conn.fetch(BIOMETRICS_SQL, start, end)
    busy = await conn.fetch(POOL_BUSY_SQL, start, end) if node in (None, POOL_NODE) else []
    readings = merge_rows(energy, biometrics)
    # A node is analyzed when it has wall watts; biometrics-only hosts have nothing to split.
    nodes = [node] if node is not None else sorted({r.node for r in readings if r.wall_w is not None})
    return readings, busy_rows(busy), nodes


def analyze_rows(
    readings: list[HourReading], busy: list[PoolBusy], nodes: list[str], start: datetime, end: datetime,
) -> list[NodeHeatArchaeologyV1]:
    """CPU-bound (plain-Python least squares): callers on an event loop run it in a thread."""
    return analyze(readings, busy, window_start=start, window_end=end, nodes=nodes)


async def read_power_attribution(
    conn: Any, start: datetime, end: datetime, node: str | None = None,
) -> list[NodeHeatArchaeologyV1]:
    readings, busy, nodes = await fetch_power_rows(conn, start, end, node)
    return await asyncio.to_thread(analyze_rows, readings, busy, nodes, start, end)


def node_item(result: NodeHeatArchaeologyV1) -> IntrospectItemV1:
    return IntrospectItemV1(
        id=f"power:{result.node}:{_iso(result.window_end)}",
        occurred_at=_utc(result.window_end),
        kind=ITEM_KIND,
        epistemic_status="unsettled",
        text=result.summary,
        extra=result.model_dump(mode="json", exclude={"summary", "caveats"}, exclude_none=True),
    )


def _one_line(sql: str) -> str:
    """The SQL as one runnable line: comment lines dropped (they are for code readers)."""
    return " ".join(" ".join(ln for ln in sql.splitlines() if not ln.strip().startswith("--")).split())


def queries_item(start: datetime, end: datetime) -> IntrospectItemV1:
    """The exact SQL behind every claim, so a curiosity run can re-run it with psql."""
    return IntrospectItemV1(
        id=f"power:queries:{_iso(end)}",
        occurred_at=_utc(end),
        kind=QUERIES_KIND,
        epistemic_status="record",
        text=(
            "Every claim's evidence names one of these queries; $1 = window start, $2 = window "
            f"end ({_iso(start)} .. {_iso(end)}), energy_hourly's $3 = {MIN_COVERED_SEC:.0f} s. "
            "Caveats: correlational fit, UNVERIFIED until an intervention (stop the suspect, watch "
            "the watts); an always-on marker is confounded with anything starting the same hour; "
            "durable-run holds are reservations, not load; a CPU-load shift points at host "
            "processes and container restart counts, which GPU leases cannot see."
        ),
        extra={"queries": {k: _one_line(v) for k, v in QUERIES.items()},
               "window_start": _iso(start), "window_end": _iso(end)},
    )


def _size(items: Sequence[IntrospectItemV1]) -> int:
    return sum(len(json.dumps(i.model_dump(mode="json"), ensure_ascii=False)) for i in items)


def build_items(
    results: Sequence[NodeHeatArchaeologyV1], start: datetime, end: datetime,
    *, budget: int = RESULT_BUDGET_CHARS,
) -> list[IntrospectItemV1]:
    """One item per node plus the SQL item, shed to fit ``budget``. Summaries are clipped only as
    the last resort, longest first, and marked truncated."""
    items = [node_item(r) for r in results] + [queries_item(start, end)]
    for field in _SHED_ORDER:
        if _size(items) <= budget:
            break
        shed = []
        for item in items:
            if item.kind == ITEM_KIND and field in item.extra:
                extra = {k: v for k, v in item.extra.items() if k != field}
                extra["omitted"] = [*item.extra.get("omitted", []), field]
                item = item.model_copy(update={"extra": extra})
            shed.append(item)
        items = shed
    while (over := _size(items) - budget) > 0:
        # Last resort: clip the longest node text by the remaining overage (plus room for the
        # marker), then re-measure. The SQL item's text is the caveat, so it stays.
        nodes = [i for i, it in enumerate(items) if it.kind == ITEM_KIND and len(it.text) > 40]
        if not nodes:
            break
        i = max(nodes, key=lambda k: len(items[k].text))
        text = items[i].text
        keep = max(len(text) - over - len(CLIP_MARKER) - 8, 0)
        items[i] = items[i].model_copy(update={"text": text[:keep] + CLIP_MARKER, "truncated": True})
    return items
