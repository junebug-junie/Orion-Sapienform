"""The physiology node's reads (body economy Patch A3, Organ 1 Body), on the service's psycopg pool.

Read-only, bounded, each read failing on its own: a failed read turns only its own constraint into
``no_reading``, never 0. The pure reducer is ``orion.physiology.body``; this module only fetches.

* energy  -- ``orion_energy_hourly`` (sql-writer's rollup) summed per hour since the local month
  start, and ``energy_cost_accrued`` next-kWh rates (newest ``computed_at`` per interval; the
  table keeps one row per tariff version, so a plain sum triples).
* lanes   -- ``gpu_pool_events``: every lease close in the window (``held_ms``), each joined to the
  pool's own lease projection ``gpu_pool_leases`` for ``hold_lease_id`` (a child call runs in
  its hold's slot); open leases from ``gpu_pool_leases`` itself (status granted/recalling: the
  pool's truth, so a lost close event never reads as a lease held for hours); the newest event
  of any kind (feed freshness). Capacity per role comes from the pool's own state RPC
  (``orion.gpu_pool.placement.fetch_pool_state``). Read-only, like field-digester's and
  proposal-runtime's reads of the same table.
* backlog -- ``gpu_pool_state_history.deferrable_depth`` (#2873): the newest snapshot per pool host,
  and per (host, work class) waiting there, the newest snapshot inside the lookback where that
  class's queue was empty (``idx_gpu_pool_state_generated`` (host, generated_at) bounds both).
  Hunger beside each lane, not strain.
* juniper -- ``endogenous_outreach_decisions``: sends today (local date), and the newest decision
  of any outcome, so a stopped decision log reads stale instead of "0 sent".
* claude  -- not a table: the latest ``ClaudeLimitObservationV1`` per kind, held in memory by
  ``ClaudeLimitCache`` from the bus (``orion:substrate:claude_limit``).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from orion.physiology.body import (
    LEASE_CLOSE_EVENTS,
    BacklogInput,
    BacklogSnapshot,
    ClaudeObservation,
    EnergyHour,
    EnergyInput,
    EnergyPrice,
    JuniperInput,
    LaneInput,
    LeaseEvent,
)

logger = logging.getLogger("orion-durable-runs.physiology_store")

# Rates from before the month start seed the carry-forward for the month's first hours.
PRICE_SEED_LOOKBACK = timedelta(days=14)
# Outreach sources the Hub's daily cap does not count (scripts/endogenous_outreach_decisions.py
# count_sent_on). They still reach Juniper, so ``sent_today`` counts them.
CAP_EXEMPT_SOURCES = ("curiosity_outreach", "reading_finding_outreach")

ENERGY_HOURS_SQL = (
    "SELECT hour_start, SUM(kwh) AS kwh FROM orion_energy_hourly "
    "WHERE hour_start >= %(since)s GROUP BY hour_start ORDER BY hour_start"
)
ENERGY_PRICES_SQL = """
SELECT date_trunc('hour', interval_start) AS hour_start,
       SUM(marginal_usd_per_kwh * energy_kwh) / NULLIF(SUM(energy_kwh), 0) AS marginal_usd_per_kwh,
       AVG(marginal_usd_per_kwh) AS marginal_unweighted
FROM (
  SELECT DISTINCT ON (usage_point_id, interval_start) interval_start, energy_kwh, marginal_usd_per_kwh
  FROM energy_cost_accrued
  WHERE interval_start >= %(since)s
  ORDER BY usage_point_id, interval_start, computed_at DESC
) latest
GROUP BY 1 ORDER BY 1
"""
LANE_CLOSES_SQL = """
SELECT e.lease_id, e.role, e.event, e.generated_at, e.held_ms, l.hold_lease_id
FROM gpu_pool_events e LEFT JOIN gpu_pool_leases l ON l.lease_id = e.lease_id
WHERE e.generated_at >= %(since)s AND e.event = ANY(%(events)s)
"""
# gpu_pool_leases_live_idx (status <> 'released') serves it.
LANE_OPEN_SQL = """
SELECT lease_id, role, 'granted' AS event, granted_at AS generated_at, NULL::float8 AS held_ms, hold_lease_id
FROM gpu_pool_leases
WHERE status IN ('granted', 'recalling') AND status <> 'released' AND granted_at IS NOT NULL
"""
LANE_NEWEST_SQL = "SELECT max(generated_at) AS newest FROM gpu_pool_events"
BACKLOG_LOOKBACK_SEC = 6 * 3600.0
BACKLOG_NEWEST_SQL = """
SELECT DISTINCT ON (host) host, generated_at, deferrable_depth
FROM gpu_pool_state_history
WHERE generated_at >= %(since)s
ORDER BY host, generated_at DESC
"""
# deferrable_depth is a json column: a class absent from it has nothing waiting; NULL (a row from
# before #2873) is unknown, so it ends the waiting run too (the age never claims more than it saw).
BACKLOG_LAST_EMPTY_SQL = """
SELECT max(generated_at) AS last_empty
FROM gpu_pool_state_history
WHERE host = %(host)s AND generated_at >= %(since)s AND generated_at <= %(until)s
  AND (deferrable_depth IS NULL OR coalesce((deferrable_depth->>%(wc)s)::int, 0) = 0)
"""
JUNIPER_SENT_SQL = """
SELECT count(*) FILTER (WHERE reason = 'sent') AS sent,
       count(*) FILTER (WHERE reason = 'sent'
                        AND coalesce(result_json->>'source', '') <> ALL(%(exempt)s)) AS cap_counted
FROM endogenous_outreach_decisions
WHERE (decided_at AT TIME ZONE %(tz)s)::date = %(day)s::date AND decided_at >= %(since)s
"""
JUNIPER_NEWEST_SQL = "SELECT max(decided_at) AS newest FROM endogenous_outreach_decisions"


def _utc(v: datetime) -> datetime:
    return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)


def month_start_utc(now: datetime, tz: ZoneInfo) -> datetime:
    local = now.astimezone(tz)
    return datetime(local.year, local.month, 1, tzinfo=tz).astimezone(timezone.utc)


async def read_energy(pool: Any, now: datetime, tz: ZoneInfo) -> EnergyInput:
    start = month_start_utc(now, tz)
    try:
        async with pool.connection() as conn:
            hours = await (await conn.execute(ENERGY_HOURS_SQL, {"since": start})).fetchall()
    except Exception:  # noqa: BLE001
        logger.warning("physiology_energy_hours_read_failed", exc_info=True)
        return EnergyInput(read_ok=False, failed_source="orion_energy_hourly")
    try:
        async with pool.connection() as conn:
            prices = await (await conn.execute(ENERGY_PRICES_SQL, {"since": start - PRICE_SEED_LOOKBACK})).fetchall()
    except Exception:  # noqa: BLE001
        logger.warning("physiology_energy_prices_read_failed", exc_info=True)
        return EnergyInput(read_ok=False, failed_source="energy_cost_accrued")
    out_prices = []
    for r in prices:
        rate = r.get("marginal_usd_per_kwh")
        if rate is None:
            rate = r.get("marginal_unweighted")
        if rate is not None:
            out_prices.append(EnergyPrice(_utc(r["hour_start"]), float(rate)))
    return EnergyInput(
        read_ok=True, month_start=start,
        hours=[EnergyHour(_utc(r["hour_start"]), float(r["kwh"])) for r in hours if r.get("kwh") is not None],
        prices=out_prices)


async def read_lane_events(pool: Any, now: datetime, window_sec: float
                           ) -> tuple[bool, list[LeaseEvent], Optional[datetime]]:
    try:
        async with pool.connection() as conn:
            closes = await (await conn.execute(LANE_CLOSES_SQL, {
                "since": now - timedelta(seconds=window_sec), "events": list(LEASE_CLOSE_EVENTS)})).fetchall()
            opens = await (await conn.execute(LANE_OPEN_SQL)).fetchall()
            newest = await (await conn.execute(LANE_NEWEST_SQL)).fetchone()
    except Exception:  # noqa: BLE001
        logger.warning("physiology_lane_events_read_failed", exc_info=True)
        return False, [], None
    newest_at = (newest or {}).get("newest")
    events = [LeaseEvent(lease_id=r.get("lease_id"), role=r.get("role"), event=r["event"],
                         at=_utc(r["generated_at"]),
                         held_ms=None if r.get("held_ms") is None else float(r["held_ms"]),
                         hold_lease_id=r.get("hold_lease_id"))
              for r in list(closes) + list(opens) if r.get("generated_at") is not None]
    return True, events, (_utc(newest_at) if newest_at is not None else None)


def capacity_from_pool_state(payload: Optional[dict]) -> Optional[dict[str, int]]:
    """role -> discovered slots, from a ``GpuPoolStateV1`` payload. None: no state (RPC failed).
    A role the pool cannot serve right now (down, silent, unloaded) has 0 slots of capacity."""
    if not isinstance(payload, dict):
        return None
    out: dict[str, int] = {}
    for role in payload.get("roles") or []:
        name = role.get("role") if isinstance(role, dict) else None
        if not name:
            continue
        ok = role.get("status") in ("confirmed", "static")
        out[str(name)] = int(role.get("slots") or 0) if ok else 0
    return out


async def read_lanes(pool: Any, now: datetime, window_sec: float, pool_state: Optional[dict]) -> LaneInput:
    ok, events, newest = await read_lane_events(pool, now, window_sec)
    return LaneInput(events_read_ok=ok, events=events, capacity=capacity_from_pool_state(pool_state),
                     newest_event_at=newest)


async def read_backlog(pool: Any, now: datetime, *, lookback_sec: float = BACKLOG_LOOKBACK_SEC
                       ) -> BacklogInput:
    since = now - timedelta(seconds=lookback_sec)
    try:
        async with pool.connection() as conn:
            newest = await (await conn.execute(BACKLOG_NEWEST_SQL, {"since": since})).fetchall()
            snaps: list[BacklogSnapshot] = []
            last_empty: dict[tuple[str, str], Optional[datetime]] = {}
            for r in newest:
                depth = _json_field(r.get("deferrable_depth"))
                if depth is not None and not isinstance(depth, dict):
                    depth = None
                at = _utc(r["generated_at"])
                snap = BacklogSnapshot(str(r["host"]), at,
                                       None if depth is None else {str(k): int(v or 0) for k, v in depth.items()})
                snaps.append(snap)
                for wc, n in (snap.deferrable or {}).items():
                    if n <= 0:
                        continue
                    row = await (await conn.execute(BACKLOG_LAST_EMPTY_SQL, {
                        "host": snap.host, "since": since, "until": r["generated_at"], "wc": wc})).fetchone()
                    le = (row or {}).get("last_empty")
                    last_empty[(snap.host, wc)] = _utc(le) if le is not None else None
    except Exception:  # noqa: BLE001
        logger.warning("physiology_backlog_read_failed", exc_info=True)
        return BacklogInput(read_ok=False, lookback_sec=lookback_sec)
    return BacklogInput(read_ok=True, snapshots=snaps, last_empty_at=last_empty, lookback_sec=lookback_sec)


def _json_field(value: Any) -> Any:
    """A psycopg ``json`` column arrives decoded; a text one (or a test double) as a string."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


async def read_juniper(pool: Any, now: datetime, tz: ZoneInfo) -> JuniperInput:
    local = now.astimezone(tz)
    day_start = datetime(local.year, local.month, local.day, tzinfo=tz).astimezone(timezone.utc)
    try:
        async with pool.connection() as conn:
            row = await (await conn.execute(JUNIPER_SENT_SQL, {
                "tz": str(tz), "day": local.date().isoformat(), "since": day_start - timedelta(hours=2),
                "exempt": list(CAP_EXEMPT_SOURCES)})).fetchone()
            newest = await (await conn.execute(JUNIPER_NEWEST_SQL)).fetchone()
    except Exception:  # noqa: BLE001
        logger.warning("physiology_juniper_read_failed", exc_info=True)
        return JuniperInput(read_ok=False)
    newest_at = (newest or {}).get("newest")
    return JuniperInput(read_ok=True, sent_today=int((row or {}).get("sent") or 0),
                        cap_counted_today=int((row or {}).get("cap_counted") or 0),
                        newest_decision_at=_utc(newest_at) if newest_at is not None else None)


class ClaudeLimitCache:
    """The latest ``ClaudeLimitObservationV1`` per kind, from the bus, with its receive time."""

    def __init__(self) -> None:
        self._latest: dict[str, ClaudeObservation] = {}

    def offer(self, payload: Any, received_at: datetime) -> bool:
        from orion.schemas.claude_limit import ClaudeLimitObservationV1

        try:
            obs = ClaudeLimitObservationV1.model_validate(payload)
        except Exception:  # noqa: BLE001 -- a malformed observation is no observation
            logger.warning("physiology_claude_limit_invalid", exc_info=True)
            return False
        self._latest[obs.kind] = ClaudeObservation(
            kind=obs.kind, state=obs.state, observed=obs.observed, event_count=int(obs.event_count),
            received_at=_utc(received_at),
            resets_at=_utc(obs.resets_at) if obs.resets_at is not None else None)
        return True

    def snapshot(self) -> tuple[ClaudeObservation, ...]:
        return tuple(self._latest[k] for k in sorted(self._latest))
