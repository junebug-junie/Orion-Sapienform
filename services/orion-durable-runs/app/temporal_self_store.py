"""Persistence for the Temporal Self chronicle (``manual_migration_temporal_self_v1.sql``).

One transaction per window (``commit_window``) writes new events, changed arcs, closed days, the
current-day frame, the cursors and the reducer state, in that order. The reducer state is the
restart point: it is written in the same transaction as everything derived from it, so after a
crash the chronicle resumes from the last committed window and re-reads exactly what it had not
yet committed.

Loads are tolerant: a stored state or frame that no longer validates (a schema change, a reducer
version bump) is reported, never raised, so a stale row cannot crash-loop the writer.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

from orion.schemas.temporal_self import (
    TemporalSelfArcV1,
    TemporalSelfDayV1,
    TemporalSelfEventV1,
    TemporalSelfFrameV1,
    TemporalSelfStateV1,
)

logger = logging.getLogger("orion-durable-runs.temporal_self_store")

STATE_ID = "orion"
PROJECTION_ID = "current_day"
READ_WATERMARK = "read_watermark"


@dataclass
class WindowWrite:
    """Everything one window commits."""

    state: TemporalSelfStateV1
    watermark: datetime
    frame: TemporalSelfFrameV1
    events: list[TemporalSelfEventV1] = field(default_factory=list)
    late_events: list[TemporalSelfEventV1] = field(default_factory=list)
    arcs: list[TemporalSelfArcV1] = field(default_factory=list)
    days: list[TemporalSelfDayV1] = field(default_factory=list)
    now: Optional[datetime] = None
    origin: Optional[datetime] = None
    # The watermark this window started from (None on the first window after a fresh start).
    # The commit refuses if the stored watermark moved: a second writer, or a stale cache.
    expected_prev: Optional[datetime] = None


@dataclass
class LoadedState:
    state: Optional[TemporalSelfStateV1]
    watermark: Optional[datetime]
    origin: Optional[datetime] = None
    error: Optional[str] = None


class StaleWriterError(RuntimeError):
    """The stored watermark is not the one this window started from."""


INSERT_EVENT_SQL = """
INSERT INTO temporal_self_event (event_id, day_id, occurred_at, source_kind, source_table, source_ref,
  correlation_id, subject_ref, related_refs, label, verdict, payload_json, late_unfolded)
VALUES (%(event_id)s, %(day_id)s, %(occurred_at)s, %(source_kind)s, %(source_table)s, %(source_ref)s,
  %(correlation_id)s, %(subject_ref)s, %(related_refs)s, %(label)s, %(verdict)s, %(payload)s::jsonb, %(late)s)
ON CONFLICT (event_id) DO NOTHING
"""
UPSERT_ARC_SQL = """
INSERT INTO temporal_self_arc (arc_id, day_id, kind, subject_ref, began_at, ended_at, status,
  attention_returns, arc_json, updated_at)
VALUES (%(arc_id)s, %(day_id)s, %(kind)s, %(subject_ref)s, %(began_at)s, %(ended_at)s, %(status)s,
  %(attention_returns)s, %(arc_json)s::jsonb, %(now)s)
ON CONFLICT (arc_id) DO UPDATE SET day_id = EXCLUDED.day_id, kind = EXCLUDED.kind,
  subject_ref = EXCLUDED.subject_ref, began_at = EXCLUDED.began_at, ended_at = EXCLUDED.ended_at,
  status = EXCLUDED.status, attention_returns = EXCLUDED.attention_returns, arc_json = EXCLUDED.arc_json,
  updated_at = EXCLUDED.updated_at
"""
UPSERT_DAY_SQL = """
INSERT INTO temporal_self_day (day_id, closed_at, day_json, stored_at)
VALUES (%(day_id)s, %(closed_at)s, %(day_json)s::jsonb, %(now)s)
ON CONFLICT (day_id) DO UPDATE SET closed_at = EXCLUDED.closed_at, day_json = EXCLUDED.day_json,
  stored_at = EXCLUDED.stored_at
"""
UPSERT_PROJECTION_SQL = """
INSERT INTO temporal_self_projection (projection_id, generated_at, projection_json, created_at)
VALUES (%(id)s, %(generated_at)s, %(json)s::jsonb, %(now)s)
ON CONFLICT (projection_id) DO UPDATE SET generated_at = EXCLUDED.generated_at,
  projection_json = EXCLUDED.projection_json, created_at = EXCLUDED.created_at
"""
UPSERT_CURSOR_SQL = """
INSERT INTO temporal_self_cursor (source_kind, last_occurred_at, last_source_ref, updated_at)
VALUES (%(kind)s, %(at)s, %(ref)s, %(now)s)
ON CONFLICT (source_kind) DO UPDATE SET last_occurred_at = EXCLUDED.last_occurred_at,
  last_source_ref = EXCLUDED.last_source_ref, updated_at = EXCLUDED.updated_at
"""
UPSERT_STATE_SQL = """
INSERT INTO temporal_self_state (state_id, watermark, origin, reducer_version, state_gz, updated_at)
VALUES (%(id)s, %(watermark)s, %(origin)s, %(version)s, %(gz)s, %(now)s)
ON CONFLICT (state_id) DO UPDATE SET watermark = EXCLUDED.watermark, origin = EXCLUDED.origin,
  reducer_version = EXCLUDED.reducer_version, state_gz = EXCLUDED.state_gz, updated_at = EXCLUDED.updated_at
"""


def event_row(e: TemporalSelfEventV1, *, late: bool = False) -> dict:
    return {
        "event_id": e.event_id, "day_id": e.day_id, "occurred_at": e.occurred_at,
        "source_kind": e.source_kind, "source_table": e.source_table, "source_ref": e.source_ref,
        "correlation_id": e.correlation_id, "subject_ref": e.subject_ref, "related_refs": list(e.related_refs),
        "label": e.label, "verdict": e.verdict,
        "payload": json.dumps(dict(e.payload, ended_at=e.ended_at.isoformat() if e.ended_at else None,
                                   privacy_class=e.privacy_class), default=str),
        "late": late,
    }


def arc_row(a: TemporalSelfArcV1, now: datetime) -> dict:
    return {
        "arc_id": a.arc_id, "day_id": a.day_id, "kind": a.kind, "subject_ref": a.subject_ref,
        "began_at": a.began_at, "ended_at": a.ended_at, "status": a.status,
        "attention_returns": a.attention_returns, "arc_json": a.model_dump_json(), "now": now,
    }


def cursor_rows(state: TemporalSelfStateV1, watermark: datetime, now: datetime) -> list[dict]:
    """``state.cursors`` holds ``<iso>|<ref>`` per source kind (``arcs.fold``)."""
    rows = [{"kind": READ_WATERMARK, "at": watermark, "ref": None, "now": now}]
    for kind, raw in sorted(state.cursors.items()):
        at, _, ref = raw.partition("|")
        rows.append({"kind": kind, "at": datetime.fromisoformat(at), "ref": ref or None, "now": now})
    return rows


def pack_state(state: TemporalSelfStateV1) -> bytes:
    return gzip.compress(state.model_dump_json().encode(), compresslevel=6, mtime=0)


def unpack_state(raw: bytes) -> TemporalSelfStateV1:
    return TemporalSelfStateV1.model_validate_json(gzip.decompress(bytes(raw)))


class ChronicleStore:
    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def load_state(self) -> LoadedState:
        async with self._pool.connection() as conn:
            row = await (await conn.execute(
                "SELECT watermark, origin, state_gz FROM temporal_self_state WHERE state_id = %(id)s",
                {"id": STATE_ID})).fetchone()
        if row is None:
            return LoadedState(None, None)
        try:
            return LoadedState(unpack_state(row["state_gz"]), row["watermark"], row["origin"])
        except Exception as exc:  # noqa: BLE001 - tolerant loader: a stale row must not crash-loop
            logger.warning("temporal_self_state_invalid watermark=%s", row["watermark"], exc_info=True)
            return LoadedState(None, row["watermark"], row["origin"], error=f"{type(exc).__name__}: {str(exc)[:200]}")

    async def unseen(self, event_ids: Iterable[str]) -> set[str]:
        ids = sorted(set(event_ids))
        if not ids:
            return set()
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(
                "SELECT event_id FROM temporal_self_event WHERE event_id = ANY(%(ids)s)", {"ids": ids})).fetchall()
        return set(ids) - {r["event_id"] for r in rows}

    async def deferrals(self, lo: datetime, hi: datetime) -> list[TemporalSelfEventV1]:
        """Stored visual deferrals in ``[lo, hi]`` (the body summary's ``thermal_refusals``)."""
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(
                "SELECT event_id, day_id, occurred_at, source_table, source_ref, verdict, payload_json "
                "FROM temporal_self_event WHERE source_kind = 'visual_deferral' AND occurred_at >= %(lo)s "
                "AND occurred_at <= %(hi)s", {"lo": lo, "hi": hi})).fetchall()
        out = []
        for r in rows:
            payload = {k: v for k, v in (r["payload_json"] or {}).items() if k not in ("ended_at", "privacy_class")}
            out.append(TemporalSelfEventV1(
                event_id=r["event_id"], day_id=r["day_id"], occurred_at=r["occurred_at"], source_kind="visual_deferral",
                source_table=r["source_table"] or "reverie_visual_attempt", source_ref=r["source_ref"] or "",
                verdict=r["verdict"], payload=payload))
        return out

    async def commit_window(self, w: WindowWrite) -> None:
        now = w.now or w.watermark
        async with self._pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute("SELECT watermark FROM temporal_self_state WHERE state_id = %(id)s FOR UPDATE",
                                      {"id": STATE_ID})
                    held = await cur.fetchone()
                    stored = held["watermark"] if held else None
                    if stored != w.expected_prev:
                        raise StaleWriterError(f"stored watermark {stored} != expected {w.expected_prev}")
                    rows = [event_row(e) for e in w.events] + [event_row(e, late=True) for e in w.late_events]
                    if rows:
                        await cur.executemany(INSERT_EVENT_SQL, rows)
                    if w.arcs:
                        await cur.executemany(UPSERT_ARC_SQL, [arc_row(a, now) for a in w.arcs])
                    if w.days:
                        await cur.executemany(UPSERT_DAY_SQL, [
                            {"day_id": d.day_id, "closed_at": d.closed_at, "day_json": d.model_dump_json(), "now": now}
                            for d in w.days])
                    await cur.execute(UPSERT_PROJECTION_SQL, {
                        "id": PROJECTION_ID, "generated_at": w.frame.as_of, "json": w.frame.model_dump_json(), "now": now})
                    await cur.executemany(UPSERT_CURSOR_SQL, cursor_rows(w.state, w.watermark, now))
                    await cur.execute(UPSERT_STATE_SQL, {
                        "id": STATE_ID, "watermark": w.watermark, "origin": w.origin or w.watermark,
                        "version": w.state.schema_version,
                        "gz": pack_state(w.state), "now": now})

    async def retention(self, now: datetime, *, event_days: int, arc_days: int, day_days: int) -> dict[str, int]:
        """Delete past each retention horizon. Covers every row in temporal_self_event, including
        the regulate node's arousal transitions (which had no retention job before this)."""
        out: dict[str, int] = {}
        async with self._pool.connection() as conn:
            for name, sql, days in (
                ("event", "DELETE FROM temporal_self_event WHERE occurred_at < %(cut)s", event_days),
                ("arc", "DELETE FROM temporal_self_arc WHERE status = 'closed' AND coalesce(ended_at, began_at) < %(cut)s", arc_days),
                ("day", "DELETE FROM temporal_self_day WHERE closed_at < %(cut)s", day_days),
            ):
                cur = await conn.execute(sql, {"cut": now - timedelta(days=days)})
                out[name] = cur.rowcount
        return out

    # --- reads for the routes ------------------------------------------------------------------

    async def frame(self) -> Optional[dict]:
        async with self._pool.connection() as conn:
            row = await (await conn.execute(
                "SELECT projection_json FROM temporal_self_projection WHERE projection_id = %(id)s",
                {"id": PROJECTION_ID})).fetchone()
        return row["projection_json"] if row else None

    async def day(self, day_id: str) -> Optional[dict]:
        async with self._pool.connection() as conn:
            row = await (await conn.execute(
                "SELECT day_json FROM temporal_self_day WHERE day_id = %(d)s", {"d": day_id})).fetchone()
        return row["day_json"] if row else None

    async def arcs(self, day_id: str, kind: Optional[str] = None, limit: int = 500) -> list[dict]:
        sql = "SELECT arc_json FROM temporal_self_arc WHERE day_id = %(d)s"
        params: dict = {"d": day_id, "limit": limit}
        if kind:
            sql += " AND kind = %(k)s"
            params["k"] = kind
        sql += " ORDER BY began_at, arc_id LIMIT %(limit)s"
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(sql, params)).fetchall()
        return [r["arc_json"] for r in rows]

    async def cursors(self) -> list[dict]:
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(
                "SELECT source_kind, last_occurred_at, last_source_ref, updated_at FROM temporal_self_cursor "
                "ORDER BY source_kind")).fetchall()
        return [dict(r) for r in rows]

    async def late_counts(self, since: datetime) -> dict[str, int]:
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(
                "SELECT source_kind, count(*) AS n FROM temporal_self_event WHERE late_unfolded "
                "AND ingested_at >= %(since)s GROUP BY source_kind ORDER BY source_kind", {"since": since})).fetchall()
        return {r["source_kind"]: int(r["n"]) for r in rows}
