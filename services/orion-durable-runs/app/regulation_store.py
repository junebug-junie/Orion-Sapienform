"""The regulate node's reads and its one write, on the service's psycopg pool (autocommit, dict rows).

Reads (read-only, bounded):
* E1 -- minutes since Juniper's last turn (``orion.regulation.juniper_turns``, the same rule the
  dream's idle gate uses);
* S1 -- athena's cabinet readings for the last hour (``orion.autonomy.cabinet_heat``'s own query,
  folded by its own pure verdict);
* S2 -- saved GPU pool snapshots (``gpu_pool_state_history``, written by sql-writer from
  ``orion:gpu_pool:state`` in ~10 ms) for the sustain window plus a margin, and the pool's own
  controller refusals/answers (``gpu_pool_events``, last ``GPU_FAULT_LOOKBACK_SEC``) so a queue
  jammed by a broken lane controller is not read as load. Either read failing makes S2 stale.

Each read fails on its own: a failed read makes only that input stale (-> ``unknown``), never 0.

Write: one ``arousal_transition`` row in ``temporal_self_event`` per level change, idempotent on a
deterministic event id (manual migration ``manual_migration_temporal_self_event_v1.sql``).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from orion.autonomy.cabinet_heat import CABINET_NODE, CABINET_POINTS_SQL, _ts_key, read_cabinet_heat
from orion.hardware_watch.rules import TempPoint
from orion.regulation.arousal import (
    GPU_FAULT_EVENTS,
    GPU_FAULT_LOOKBACK_SEC,
    GPU_STATE_STALE_SEC,
    gpu_fault_from_events,
    gpu_queue_evidence,
)
from orion.regulation.juniper_turns import JUNIPER_IDLE_MINUTES_SQL
from orion.schemas.regulation import ArousalInputsV1, ArousalReadingV1

logger = logging.getLogger("orion-durable-runs.regulation_store")

CABINET_LOOKBACK = timedelta(hours=1)
GPU_MARGIN_SEC = 60.0

# The cabinet module's SQLAlchemy query, in psycopg's parameter style (same text otherwise).
CABINET_POINTS_PG = (CABINET_POINTS_SQL.replace(":node", "%(node)s").replace(":since", "%(since)s")
                     .replace(":until", "%(until)s"))

GPU_SNAPSHOTS_SQL = (
    "SELECT host, generated_at, queue_depth FROM gpu_pool_state_history "
    "WHERE generated_at >= %(since)s ORDER BY host, generated_at"
)

# Index idx_gpu_pool_events_generated bounds it (0.2 ms live, 10-13). ``role`` is the seat.
GPU_FAULT_EVENTS_SQL = (
    "SELECT generated_at, event, reason, role FROM gpu_pool_events "
    "WHERE generated_at >= %(since)s AND event = ANY(%(events)s) ORDER BY generated_at, event_id"
)

INSERT_EVENT_SQL = """
INSERT INTO temporal_self_event (event_id, day_id, occurred_at, source_kind, source_table, source_ref,
                                 subject_ref, label, payload_json)
VALUES (%(event_id)s, %(day_id)s, %(occurred_at)s, 'arousal_transition', 'regulation',
        %(source_ref)s, 'orion:arousal', %(label)s, %(payload)s::jsonb)
ON CONFLICT (event_id) DO NOTHING
"""


async def _juniper_minutes(pool: Any) -> tuple[bool, Optional[float]]:
    try:
        async with pool.connection() as conn:
            row = await (await conn.execute(JUNIPER_IDLE_MINUTES_SQL)).fetchone()
    except Exception:  # noqa: BLE001
        logger.warning("regulation_juniper_turn_read_failed", exc_info=True)
        return False, None
    idle = (row or {}).get("idle")
    return True, (None if idle is None else max(0.0, float(idle)))


async def _cabinet(pool: Any, now: datetime, prev: Optional[ArousalInputsV1]) -> dict:
    try:
        async with pool.connection() as conn:
            rows = await (await conn.execute(CABINET_POINTS_PG, {
                "node": CABINET_NODE, "since": _ts_key(now - CABINET_LOOKBACK),
                "until": _ts_key(now) + "~"})).fetchall()
    except Exception:  # noqa: BLE001
        logger.warning("regulation_cabinet_read_failed", exc_info=True)
        return {"cabinet_read_ok": False}
    points = [TempPoint(ts=r["ts"], value=float(r["v"])) for r in rows if r.get("v") is not None]
    # Seed the reflex's two hysteresis folds from the last step, so a trip older than the hour
    # re-read still holds until its re-arm point (cabinet_heat.read_cabinet_heat's `previous`).
    seed = None
    if prev is not None and prev.cabinet_thermal_state not in (None, "unknown"):
        seed = dataclasses.replace(read_cabinet_heat([], now), thermal_state=prev.cabinet_thermal_state,
                                   critical=prev.cabinet_critical)
    verdict = read_cabinet_heat(points, now, previous=seed)
    return {"cabinet_read_ok": True, "cabinet_reflex": verdict.reflex,
            "cabinet_thermal_state": verdict.thermal_state, "cabinet_critical": verdict.critical,
            "cabinet_temp_c": verdict.temp_c}


async def _gpu(pool: Any, now: datetime, *, floor: int, sustain_sec: float) -> dict:
    try:
        async with pool.connection() as conn:
            rows = await (await conn.execute(GPU_SNAPSHOTS_SQL, {
                "since": now - timedelta(seconds=sustain_sec + GPU_MARGIN_SEC)})).fetchall()
            events = await (await conn.execute(GPU_FAULT_EVENTS_SQL, {
                "since": now - timedelta(seconds=GPU_FAULT_LOOKBACK_SEC),
                "events": list(GPU_FAULT_EVENTS)})).fetchall()
    except Exception:  # noqa: BLE001
        # Without the fault read a jammed queue cannot be told from load: the whole input is stale.
        logger.warning("regulation_gpu_state_read_failed", exc_info=True)
        return {}
    fault = gpu_fault_from_events(
        ((r["generated_at"], r["event"], r["reason"], r["role"]) for r in events), now)
    by_host: dict[str, list] = {}
    for r in rows:
        depth = r["queue_depth"]
        if isinstance(depth, str):
            try:
                depth = json.loads(depth)
            except ValueError:
                continue
        by_host.setdefault(r["host"], []).append((r["generated_at"], depth))
    # One evidence run per pool host; the most strained FRESH host speaks for the fleet. A host
    # that stopped publishing must not outrank a fresh one and turn S2 stale (review finding).
    best: Optional[tuple] = None
    for snaps in by_host.values():
        age, depth, sustained = gpu_queue_evidence(snaps, now, floor=floor)
        if age is None:
            continue
        key = (age <= GPU_STATE_STALE_SEC, sustained or 0.0, -age)
        if best is None or key > best[0]:
            best = (key, age, depth, sustained)
    if best is None:
        return {}
    _, age, depth, sustained = best
    return {"gpu_state_age_sec": age, "gpu_queue_depth": depth, "gpu_queue_sustained_sec": sustained,
            "gpu_fault": fault}


async def read_arousal_inputs(
    pool: Any,
    now: datetime,
    prev: Optional[ArousalInputsV1],
    *,
    last_turn_event_at: Optional[datetime] = None,
    gpu_queue_floor: int,
    gpu_sustain_sec: float,
) -> ArousalInputsV1:
    """One step's evidence. ``last_turn_event_at``: a Juniper turn seen on the bus, which wakes the
    step before sql-writer may have landed its row, so E1 takes the more recent of the two."""
    ok, minutes = await _juniper_minutes(pool)
    if last_turn_event_at is not None:
        event_min = max(0.0, (now - last_turn_event_at).total_seconds() / 60.0)
        if ok:
            minutes = event_min if minutes is None else min(minutes, event_min)
    fields: dict = {"observed_at": now, "juniper_turn_read_ok": ok, "minutes_since_juniper_turn": minutes}
    fields.update(await _cabinet(pool, now, prev))
    fields.update(await _gpu(pool, now, floor=gpu_queue_floor, sustain_sec=gpu_sustain_sec))
    return ArousalInputsV1(**fields)


def transition_row(prev: Optional[ArousalReadingV1], new: ArousalReadingV1, day_id: str) -> dict:
    since = new.since.astimezone(timezone.utc)
    return {
        "event_id": f"arousal_transition:{since.isoformat()}:{new.arousal_level}",
        "day_id": day_id,
        "occurred_at": since,
        "source_ref": f"regulation.state.v1:{new.observed_at.astimezone(timezone.utc).isoformat()}",
        "label": new.arousal_level,
        "payload": json.dumps({"from": prev.arousal_level if prev is not None else None,
                               "to": new.arousal_level, "reading": new.model_dump(mode="json")}),
    }


def cause_row(new: ArousalReadingV1, day_id: str, joined: list[str], seed: bool) -> dict:
    """Strain loop part 1: a row for a strain episode whose level did not change. ``joined``: the
    causes new to the episode (empty for a keepalive); ``seed``: nothing was recorded for this
    episode yet (the chronology has no onset row for it). Same table, kind and label as a level row, so every reader of levels
    (personality situation) still reads ``strained`` at that time. Time: when it was observed."""
    since = new.since.astimezone(timezone.utc)
    observed = new.observed_at.astimezone(timezone.utc)
    return {
        "event_id": f"arousal_transition:{since.isoformat()}:{new.arousal_level}:at:{observed.isoformat()}",
        "day_id": day_id,
        "occurred_at": observed,
        "source_ref": f"regulation.state.v1:{observed.isoformat()}",
        "label": new.arousal_level,
        "payload": json.dumps({"from": new.arousal_level, "to": new.arousal_level, "joined": list(joined),
                               "seed": bool(seed), "reading": new.model_dump(mode="json")}),
    }


async def record_transition(pool: Any, row: dict) -> bool:
    try:
        async with pool.connection() as conn:
            await conn.execute(INSERT_EVENT_SQL, row)
        return True
    except Exception:  # noqa: BLE001 - history is best-effort; Redis still carries the state
        logger.warning("regulation_transition_write_failed event=%s", row.get("event_id"), exc_info=True)
        return False


# --- R4 runtime immune sweep: one read of the semantic field channels, one write per finding --------

# One sample per minute over the window, every node's value for the named channels plus the
# per-node/channel ``node_vector_updated_at`` stamp (the replay check's "fresh re-stamp"). The
# generated_at index bounds the scan (live 2026-10-13: 41k rows/24 h -> 1,441 minutes, ~3 s; the
# field_json TOAST read dominates). Read-only transaction with its own statement timeout.
IMMUNE_SERIES_SQL = """
WITH m AS (
  SELECT DISTINCT ON (date_trunc('minute', generated_at)) tick_id
  FROM substrate_field_state
  WHERE generated_at >= %(since)s AND generated_at < %(until)s
  ORDER BY date_trunc('minute', generated_at), generated_at DESC
)
SELECT s.generated_at AS t, n.key AS node, c.key AS channel, c.value #>> '{}' AS value,
       s.field_json->'node_vector_updated_at'->n.key->>c.key AS updated_at
FROM m JOIN substrate_field_state s USING (tick_id),
     jsonb_each(s.field_json->'node_vectors') n, jsonb_each(n.value) c
WHERE c.key = ANY(%(channels)s)
ORDER BY 2, 3, 1
"""
IMMUNE_STATEMENT_TIMEOUT = "45s"

INSERT_IMMUNE_SQL = """
INSERT INTO temporal_self_event (event_id, day_id, occurred_at, source_kind, source_table, source_ref,
                                 subject_ref, label, verdict, payload_json)
VALUES (%(event_id)s, %(day_id)s, %(occurred_at)s, 'immune_finding', 'substrate_field_state',
        %(source_ref)s, %(subject_ref)s, %(label)s, %(verdict)s, %(payload)s::jsonb)
ON CONFLICT (event_id) DO UPDATE SET payload_json = EXCLUDED.payload_json, source_ref = EXCLUDED.source_ref
"""


async def read_immune_series(pool: Any, now: datetime, channels: list[str], *, window_sec: float = 86400.0) -> dict:
    """``(node, channel) -> [(t, value, updated_at)]`` over the last ``window_sec``. Raises on a
    failed read (the graph's immune node catches it; a failed read is never an empty day)."""
    from orion.regulation.immune import ImmuneSample

    params = {"since": now - timedelta(seconds=window_sec), "until": now, "channels": list(channels)}
    async with pool.connection() as conn:
        async with conn.transaction():
            await conn.execute("SET TRANSACTION READ ONLY")
            await conn.execute(f"SET LOCAL statement_timeout = '{IMMUNE_STATEMENT_TIMEOUT}'")
            rows = await (await conn.execute(IMMUNE_SERIES_SQL, params)).fetchall()
    out: dict = {}
    for r in rows:
        try:
            value = float(r["value"])
        except (TypeError, ValueError):
            continue
        upd = None
        if r.get("updated_at"):
            try:
                upd = datetime.fromisoformat(str(r["updated_at"]).replace("Z", "+00:00"))
            except ValueError:
                upd = None
        out.setdefault((r["node"], r["channel"]), []).append(ImmuneSample(r["t"], value, updated_at=upd))
    return out


def immune_finding_row(finding: Any, day_id: str, observed_at: datetime) -> dict:
    """One finding -> one ``temporal_self_event`` row. The id is deterministic on (metric, kind,
    day): a later sweep the same day refreshes the evidence window instead of adding a row."""
    observed = observed_at.astimezone(timezone.utc)
    payload = finding.to_dict()
    payload["observed_at"] = observed.isoformat()
    return {
        "event_id": f"immune_finding:{finding.metric_id}:{finding.kind}:{day_id}",
        "day_id": day_id,
        "occurred_at": observed,
        "source_ref": f"substrate_field_state:{finding.window_start.isoformat()}..{finding.window_end.isoformat()}",
        "subject_ref": f"field:{finding.metric_id}",
        "label": finding.kind,
        "verdict": "open",
        "payload": json.dumps(payload, default=str),
    }


async def record_immune_findings(pool: Any, rows: list[dict]) -> bool:
    if not rows:
        return True
    try:
        async with pool.connection() as conn:
            async with conn.transaction():
                for row in rows:
                    await conn.execute(INSERT_IMMUNE_SQL, row)
        return True
    except Exception:  # noqa: BLE001 - the trace is best-effort; regulation never depends on it
        logger.warning("regulation_immune_write_failed n=%s", len(rows), exc_info=True)
        return False
