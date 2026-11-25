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
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional

from orion.schemas.temporal_self import (
    TemporalSelfArcV1,
    TemporalSelfDayV1,
    TemporalSelfEventV1,
    TemporalSelfFrameV1,
    TemporalSelfStateV1,
)
from orion.temporal_self.arcs import AVAILABLE_AT_KINDS
from orion.integration.coverage import RATE_CITES_MADE_SQL, RATE_KINDS_SQL, RATE_USES_SQL
from orion.temporal_self.now import stored_payload_surprise

logger = logging.getLogger("orion-durable-runs.temporal_self_store")

STATE_ID = "orion"
# Every chronicle transaction is bounded: a stalled statement fails the window (retried next step)
# instead of holding the regulate thread.
STATEMENT_TIMEOUT = "30s"
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
    state_gz: Optional[bytes] = None   # pack_state(state), precomputed off the event loop
    # event_id -> salience (Orion's now, integration patch 1); absent ids are stored NULL.
    salience: dict[str, Optional[float]] = field(default_factory=dict)


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
  correlation_id, subject_ref, related_refs, label, verdict, payload_json, late_unfolded, salience, surprise_nats)
VALUES (%(event_id)s, %(day_id)s, %(occurred_at)s, %(source_kind)s, %(source_table)s, %(source_ref)s,
  %(correlation_id)s, %(subject_ref)s, %(related_refs)s, %(label)s, %(verdict)s, %(payload)s::jsonb, %(late)s,
  %(salience)s, %(surprise_nats)s)
ON CONFLICT (event_id) DO UPDATE SET late_unfolded = false
  WHERE temporal_self_event.late_unfolded AND NOT EXCLUDED.late_unfolded
"""
# A re-fold after a reset folds a row that was stored late before: it is no longer unfolded.
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


def stored_payload(e: TemporalSelfEventV1) -> dict:
    """``payload_json`` as stored: the event's payload plus ``ended_at`` and ``privacy_class``."""
    return dict(e.payload, ended_at=e.ended_at.isoformat() if e.ended_at else None, privacy_class=e.privacy_class)


def stored_shape(e: TemporalSelfEventV1) -> dict:
    """The row shape the magnitude rules (orion/temporal_self/now.py) read, for a new event."""
    payload = stored_payload(e)
    return {"occurred_at": e.occurred_at, "verdict": e.verdict, "payload": payload,
            "surprise_nats": stored_payload_surprise(payload)}


def event_row(e: TemporalSelfEventV1, *, late: bool = False, salience: Optional[float] = None) -> dict:
    payload = stored_payload(e)
    return {
        "event_id": e.event_id, "day_id": e.day_id, "occurred_at": e.occurred_at,
        "source_kind": e.source_kind, "source_table": e.source_table, "source_ref": e.source_ref,
        "correlation_id": e.correlation_id, "subject_ref": e.subject_ref, "related_refs": list(e.related_refs),
        "label": e.label, "verdict": e.verdict,
        "payload": json.dumps(payload, default=str),
        "late": late,
        # Orion's now (integration patch 1): NULL unless the kind declares a magnitude and has 20 past
        # events (salience), or the source carried a real prediction's surprisal (surprise_nats).
        "salience": salience,
        "surprise_nats": stored_payload_surprise(payload),
    }


def window_event_rows(w: WindowWrite) -> list[dict]:
    """The window's event rows in event_id order, which is the order they are stored in.

    Never in source order (#3458): a dream cycle's hypotheses come back dream arm first, then control, and rows of
    one cycle share occurred_at, so source-order inserts let anyone who can read the table recover the blinded arm
    from physical row order. event_id carries the random hypothesis id, so this order says nothing about the arm."""
    rows = ([event_row(e, salience=w.salience.get(e.event_id)) for e in w.events]
            + [event_row(e, late=True, salience=w.salience.get(e.event_id)) for e in w.late_events])
    return sorted(rows, key=lambda r: r["event_id"])


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

    async def stored_available(self, event_ids: Iterable[str]) -> dict[str, datetime]:
        """Already-stored events -> the available time they were stored with (ended_at, else
        occurred_at; a street event's or office session's ``available_at`` when later, the same rule as
        ``arcs.event_available_at``). Absent ids were never stored (the late probe's test)."""
        ids = sorted(set(event_ids))
        if not ids:
            return {}
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(
                "SELECT event_id, occurred_at, source_kind, payload_json->>'ended_at' AS ended_at, "
                "payload_json->>'available_at' AS written FROM temporal_self_event "
                "WHERE event_id = ANY(%(ids)s)", {"ids": ids})).fetchall()
        out: dict[str, datetime] = {}
        for r in rows:
            t = datetime.fromisoformat(r["ended_at"]) if r["ended_at"] else r["occurred_at"]
            if r["source_kind"] in AVAILABLE_AT_KINDS and r["written"]:
                t = max(t, datetime.fromisoformat(r["written"]))
            out[r["event_id"]] = t
        return out

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
                    await cur.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                    await cur.execute("SELECT watermark FROM temporal_self_state WHERE state_id = %(id)s FOR UPDATE",
                                      {"id": STATE_ID})
                    held = await cur.fetchone()
                    stored = held["watermark"] if held else None
                    if stored != w.expected_prev:
                        raise StaleWriterError(f"stored watermark {stored} != expected {w.expected_prev}")
                    rows = window_event_rows(w)
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
                        "gz": w.state_gz if w.state_gz is not None else pack_state(w.state), "now": now})

    async def retention(self, now: datetime, *, event_days: int, arc_days: int, day_days: int) -> dict[str, int]:
        """Delete past each retention horizon. Covers every row in temporal_self_event, including
        the regulate node's arousal transitions (which had no retention job before this)."""
        out: dict[str, int] = {}
        async with self._pool.connection() as conn:
            for name, sql, days in (
                ("event", "DELETE FROM temporal_self_event WHERE occurred_at < %(cut)s", event_days),
                ("arc", "DELETE FROM temporal_self_arc WHERE status = 'closed' AND coalesce(ended_at, began_at) < %(cut)s", arc_days),
                ("day", "DELETE FROM temporal_self_day WHERE closed_at < %(cut)s", day_days),
                # Orion's now: reads and uses live as long as the experiences they name.
                ("experience_shown", "DELETE FROM experience_shown WHERE shown_at < %(cut)s", event_days),
                ("experience_use", "DELETE FROM experience_use WHERE cited_at < %(cut)s", event_days),
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

    async def arc(self, arc_id: str) -> Optional[dict]:
        """One arc by id, any day still retained (orion/refs.py kind ``arc``)."""
        async with self._pool.connection() as conn:
            row = await (await conn.execute(
                "SELECT arc_json FROM temporal_self_arc WHERE arc_id = %(a)s", {"a": arc_id})).fetchone()
        return row["arc_json"] if row else None

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

    # --- Orion's now (integration patch 1) -------------------------------------------------------

    async def magnitude_history(self, kind: str, since: datetime) -> list[dict]:
        """Stored-shape rows of one kind since ``since`` (the salience history, read once per kind)."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                rows = await (await conn.execute(
                    "SELECT event_id, occurred_at, verdict, payload_json, surprise_nats FROM temporal_self_event "
                    "WHERE source_kind = %(k)s AND occurred_at >= %(since)s", {"k": kind, "since": since})).fetchall()
        return [{"event_id": r["event_id"], "occurred_at": r["occurred_at"], "verdict": r["verdict"],
                 "payload": r["payload_json"] or {}, "surprise_nats": r["surprise_nats"]} for r in rows]


NOW_CANDIDATES_SQL = """
SELECT event_id, source_kind, occurred_at, label, verdict, related_refs, payload_json, salience, surprise_nats,
  source_ref, correlation_id, subject_ref, kind_total
FROM (
  SELECT e.event_id, e.source_kind, e.occurred_at, e.label, e.verdict, e.related_refs, e.payload_json,
    e.salience, e.surprise_nats, e.source_ref, e.correlation_id, e.subject_ref,
    row_number() OVER (PARTITION BY e.source_kind
                       ORDER BY e.salience DESC NULLS LAST, e.occurred_at DESC, e.event_id) AS rk,
    count(*) OVER (PARTITION BY e.source_kind) AS kind_total
  FROM temporal_self_event e
  WHERE e.ingested_at > %(since)s AND e.ingested_at <= %(as_of)s AND e.source_kind <> ALL(%(own)s)
) x WHERE rk <= %(per_kind)s
"""
# Held-concern rows (#3783): other organs' rows in (lo, as_of] whose own ids equal one of the reader's held keys
# (orion.temporal_self.now.held_lookup), newest first. Exact equality only; build_now re-checks which ref matched.
NOW_HELD_SQL = """
SELECT e.event_id, e.source_kind, e.occurred_at, e.label, e.verdict, e.related_refs, e.payload_json,
  e.salience, e.surprise_nats, e.source_ref, e.correlation_id, e.subject_ref,
  e.ingested_at <= %(window_lo)s AS before_window
FROM temporal_self_event e
WHERE e.ingested_at > %(lo)s AND e.ingested_at <= %(as_of)s AND e.source_kind <> ALL(%(own)s)
  AND (e.event_id = ANY(%(keys)s) OR e.source_ref = ANY(%(keys)s) OR e.correlation_id = ANY(%(keys)s)
       OR e.subject_ref = ANY(%(keys)s) OR e.related_refs && %(keys)s::text[])
ORDER BY e.occurred_at DESC, e.event_id
LIMIT %(limit)s
"""
NOW_ARC_SUBJECTS_SQL = "SELECT arc_id, subject_ref FROM temporal_self_arc WHERE arc_id = ANY(%(ids)s)"
# Pinned now rows (Prime hunger, integration Part B): by id, other organs' kinds only.
NOW_BY_ID_SQL = """
SELECT e.event_id, e.source_kind, e.occurred_at, e.label, e.verdict, e.related_refs, e.payload_json,
  e.salience, e.surprise_nats, e.source_ref, e.correlation_id, e.subject_ref
FROM temporal_self_event e
WHERE e.event_id = ANY(%(ids)s) AND e.source_kind <> ALL(%(own)s)
"""
# Prime hunger's surprise need: other organs' experiences in (lo, as_of] whose surprisal is strictly above their
# own kind's 90th percentile over (hist_lo, as_of], and that no organ has cited.
HUNGER_SURPRISES_SQL = """
WITH hist AS (
  SELECT source_kind, percentile_cont(0.9) WITHIN GROUP (ORDER BY surprise_nats) AS p90, count(*) AS n
  FROM temporal_self_event
  WHERE surprise_nats IS NOT NULL AND occurred_at > %(hist_lo)s AND occurred_at <= %(as_of)s
  GROUP BY source_kind
)
SELECT e.event_id, e.source_kind, e.occurred_at, e.label, e.verdict, e.payload_json, e.surprise_nats,
  h.p90, h.n
FROM temporal_self_event e JOIN hist h USING (source_kind)
WHERE e.occurred_at > %(lo)s AND e.occurred_at <= %(as_of)s AND e.surprise_nats > h.p90
  AND e.source_kind <> ALL(%(own)s)
  AND NOT EXISTS (SELECT 1 FROM experience_use u WHERE u.event_id = e.event_id)
ORDER BY e.surprise_nats DESC, e.occurred_at DESC
LIMIT %(cap)s
"""
# Prime hunger's connection need: Juniper's own messages to Orion in Hub (chat_history_log.created_at is UTC
# without a zone, as the chronicle's chat_turn source reads it).
HUNGER_JUNIPER_SQL = """
SELECT id, created_at AT TIME ZONE 'UTC' AS at, prompt
FROM chat_history_log
WHERE source = 'hub_orion' AND btrim(coalesce(prompt, '')) <> ''
  AND created_at >= (%(lo)s::timestamptz AT TIME ZONE 'UTC')
ORDER BY created_at
"""
HUNGER_CHAT_EVENT_SQL = """
SELECT event_id FROM temporal_self_event WHERE source_kind = 'chat_turn' AND source_ref = %(ref)s LIMIT 1
"""
# The organ's cursor (#3498): the cut-off of its newest CLOSED read that advances the cursor. A read is closed by
# POST /temporal-self/now/use (zero or more cites); an unclosed read (the model never saw it) never moves it, so the
# next read offers the same rows again. max() over closed rows: closing an older read never moves it backwards.
LAST_SHOWN_SQL = ("SELECT max(shown_at) AS at FROM experience_shown"
                  " WHERE reader_organ = %(o)s AND advances_cursor AND closed_at IS NOT NULL")
INSERT_SHOWN_SQL = """
INSERT INTO experience_shown (shown_id, reader_organ, reader_run_ref, event_ids, since, shown_at, advances_cursor)
VALUES (%(shown_id)s, %(reader_organ)s, %(reader_run_ref)s, %(event_ids)s, %(since)s, %(shown_at)s, %(advances_cursor)s)
"""
SHOWN_BY_ID_SQL = "SELECT shown_id, reader_organ, event_ids, shown_at FROM experience_shown WHERE shown_id = %(s)s"
# The same row, locked: concurrent cite calls on one shown row are serialized, so the 3-cite cap holds.
SHOWN_FOR_UPDATE_SQL = SHOWN_BY_ID_SQL + " FOR UPDATE"
USED_FOR_SHOWN_SQL = "SELECT event_id FROM experience_use WHERE shown_id = %(s)s"
# Closing a read (#3498): the first close stamps it; a later close of the same row keeps the first time.
CLOSE_SHOWN_SQL = "UPDATE experience_shown SET closed_at = COALESCE(closed_at, %(at)s) WHERE shown_id = %(s)s"
INSERT_USE_SQL = """
INSERT INTO experience_use (use_id, event_id, citing_organ, citing_ref, shown_id, cited_at)
VALUES (%(use_id)s, %(event_id)s, %(citing_organ)s, %(citing_ref)s, %(shown_id)s, %(cited_at)s)
ON CONFLICT (use_id) DO NOTHING
"""


def _now_row(r: Any) -> dict:
    """One temporal_self_event row in build_now's candidate shape (orion.temporal_self.now.build_now)."""
    return {"event_id": r["event_id"], "source_kind": r["source_kind"], "occurred_at": r["occurred_at"],
            "label": r["label"], "verdict": r["verdict"], "related_refs": list(r["related_refs"] or []),
            "payload": r["payload_json"] or {}, "salience": r["salience"], "surprise_nats": r["surprise_nats"],
            "source_ref": r.get("source_ref"), "correlation_id": r.get("correlation_id"),
            "subject_ref": r.get("subject_ref")}


class ExperienceStore:
    """``experience_shown`` / ``experience_use`` and the now's reads (manual_migration_experience_now_v1.sql)."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def last_shown(self, organ: str) -> Optional[datetime]:
        async with self._pool.connection() as conn:
            row = await (await conn.execute(LAST_SHOWN_SQL, {"o": organ})).fetchone()
        return row["at"] if row else None

    async def candidates(self, *, since: datetime, as_of: datetime, own_kinds: Iterable[str],
                         per_kind: int) -> tuple[list[dict], int]:
        """At most ``per_kind`` rows per source kind (by salience, then newest), ingested in
        ``(since, as_of]``, of kinds other than ``own_kinds``; and how many rows matched in all."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                rows = await (await conn.execute(NOW_CANDIDATES_SQL, {
                    "since": since, "as_of": as_of, "own": sorted(own_kinds), "per_kind": per_kind})).fetchall()
        totals: dict[str, int] = {}
        out = []
        for r in rows:
            totals[r["source_kind"]] = int(r["kind_total"])
            out.append({**_now_row(r), "kind_total": int(r["kind_total"])})
        return out, sum(totals.values())

    async def arc_subjects(self, arc_ids: Iterable[str]) -> dict[str, str]:
        """arc_id -> subject_ref for the held arcs (#3783), so a row on a held arc's subject counts as held."""
        ids = sorted({str(i) for i in arc_ids if str(i).strip()})
        if not ids:
            return {}
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(NOW_ARC_SUBJECTS_SQL, {"ids": ids})).fetchall()
        return {str(r["arc_id"]): str(r["subject_ref"]) for r in rows if r["subject_ref"]}

    async def held_candidates(self, *, keys: Iterable[str], lo: datetime, as_of: datetime, own_kinds: Iterable[str],
                              window_lo: Optional[datetime] = None, limit: int = 50) -> list[dict]:
        """Other organs' rows in ``(lo, as_of]`` whose own ids equal one of ``keys`` (#3783), newest first; each
        says whether it landed before the read's own window (``before_window``: ingested at or before ``window_lo``)."""
        ks = sorted({str(k) for k in keys if str(k).strip()})
        if not ks:
            return []
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                rows = await (await conn.execute(NOW_HELD_SQL, {
                    "keys": ks, "lo": lo, "as_of": as_of, "own": sorted(own_kinds), "limit": int(limit),
                    "window_lo": window_lo or lo})).fetchall()
        return [{**_now_row(r), "before_window": bool(r["before_window"])} for r in rows]

    async def by_id(self, event_ids: Iterable[str], *, own_kinds: Iterable[str]) -> list[dict]:
        """Rows by event id (other organs' kinds only), in the candidates' shape: a reader's pinned items."""
        ids = sorted({str(i) for i in event_ids if str(i).strip()})
        if not ids:
            return []
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(NOW_BY_ID_SQL, {"ids": ids, "own": sorted(own_kinds)})).fetchall()
        return [_now_row(r) for r in rows]

    async def hunger_inputs(self, *, lo: datetime, hist_lo: datetime, as_of: datetime, own_kinds: Iterable[str],
                            cap: int, juniper_lo: datetime) -> tuple[list[dict], list[dict], Optional[str]]:
        """(surprise rows, Juniper's messages oldest first, the last message's chat_turn event id). Read only."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SET TRANSACTION READ ONLY")
                await conn.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                surprises = await (await conn.execute(HUNGER_SURPRISES_SQL, {
                    "lo": lo, "hist_lo": hist_lo, "as_of": as_of, "own": sorted(own_kinds), "cap": cap})).fetchall()
                msgs = await (await conn.execute(HUNGER_JUNIPER_SQL, {"lo": juniper_lo})).fetchall()
                event_id = None
                if msgs:
                    row = await (await conn.execute(HUNGER_CHAT_EVENT_SQL, {"ref": str(msgs[-1]["id"])})).fetchone()
                    event_id = row["event_id"] if row else None
        return [dict(r) for r in surprises], [dict(r) for r in msgs], event_id

    async def insert_shown(self, *, shown_id: str, reader_organ: str, reader_run_ref: Optional[str],
                           event_ids: list[str], since: datetime, shown_at: datetime,
                           advances_cursor: bool = True) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(INSERT_SHOWN_SQL, {
                "shown_id": shown_id, "reader_organ": reader_organ, "reader_run_ref": reader_run_ref,
                "event_ids": event_ids, "since": since, "shown_at": shown_at, "advances_cursor": advances_cursor})

    async def record_cites(self, shown_id: str, decide: Callable[[dict, list[str]], Any],
                           closed_at: Optional[datetime] = None) -> Any:
        """One transaction: lock the shown row, read the uses already stored for it, let ``decide(shown,
        already_used)`` (orion.temporal_self.now.record_use) choose, insert its uses, and close the read
        (``closed_at``, first close wins; #3498: this is what moves the organ's cursor). None: no such row."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(SHOWN_FOR_UPDATE_SQL, {"s": shown_id})
                    shown = await cur.fetchone()
                    if shown is None:
                        return None
                    await cur.execute(USED_FOR_SHOWN_SQL, {"s": shown_id})
                    already = [r["event_id"] for r in await cur.fetchall()]
                    result = decide(dict(shown), already)
                    rows = [u.model_dump() for u in result.uses]
                    if rows:
                        await cur.executemany(INSERT_USE_SQL, rows)
                    await cur.execute(CLOSE_SHOWN_SQL, {"s": shown_id, "at": closed_at or datetime.now(timezone.utc)})
        return result

    async def rate_inputs(self, lo: datetime, hi: datetime) -> tuple[dict[str, int], list[tuple[str, str, str]], dict[str, int]]:
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SET TRANSACTION READ ONLY")
                await conn.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                kinds = await (await conn.execute(RATE_KINDS_SQL, {"lo": lo, "hi": hi})).fetchall()
                uses = await (await conn.execute(RATE_USES_SQL, {"lo": lo, "hi": hi})).fetchall()
                made = await (await conn.execute(RATE_CITES_MADE_SQL, {"lo": lo, "hi": hi})).fetchall()
        return ({r["source_kind"]: int(r["n"]) for r in kinds},
                [(r["event_id"], r["source_kind"], r["citing_organ"]) for r in uses],
                {r["citing_organ"]: int(r["n"]) for r in made})
