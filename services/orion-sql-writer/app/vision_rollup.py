"""Walkway rollups reducer: hour, day, week and month rows in ``vision_rollup``.

The pure part (periods, counting rules, the template line) is
``orion/vision/walkway_rollup.py``. This module reads the raw tables, runs the
builds in the right order, and writes rows. It is split into a ``Reader`` (raw
rows, read-only) and a ``Store`` (rollup rows) so the same ``refresh`` runs in
three places:

- the clocked loop (app/vision_rollup_loop.py): DbReader + DbStore;
- the backfill script (scripts/backfill_vision_rollups.py): DbReader + DbStore
  with ``--apply``, DbReader + MemoryStore for the default dry run;
- the read-only replay eval (evals/run_vision_rollup_replay_eval.py):
  DbReader + MemoryStore, nothing written.

Order matters, because later rows read earlier ones: for each local date, the
date's closed hours first (an hour's "usual" reads the earlier day rows), then
the day (from its hour rows), then the week and month containing it (from
their day rows). So day totals are always the sum of their hours, and week and
month totals the sum of their days.

When an hour is ready: closed for ``settle_sec`` AND the individuals reducer's
cursor (landing time of crops it has turned into sightings) is past the hour's
end. A sighting can still grow after that (merge gap), so the newest
``recompute_hours`` closed hours are rebuilt every tick; older hours are built
once, when missing, back to ``catchup_hours``. An hour inside ``catchup_hours``
whose stored ``rules_version`` is older than ``RULES_VERSION`` is rebuilt too, once,
so a rules change (2026-10-20: people as a range) reaches the recent rows without a
backfill. Older rows keep their version; readers re-render them
(``walkway_rollup.display_line``), and ``scripts/backfill_vision_rollups.py``
rebuilds them for good.

Named people (rules_version 3): the hour build groups named individuals by the person
record (``person`` / ``person_link``, read once per tick by ``Reader.person_links``). An
hour inside ``catchup_hours`` whose stored ``person_record.digest`` differs from the
current links is rebuilt too, so a rename or a new link reaches the recent days without a
backfill. Person tables missing or unreadable: fail-open to label grouping, logged once.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple
from zoneinfo import ZoneInfo

from orion.vision.motion_story_gate import gate_passed, tally_from_counts
from orion.vision.walkway_rollup import (
    CaptionRow,
    HourInputs,
    Period,
    PersonRef,
    SightingRow,
    StoryRow,
    build_hour,
    build_parent,
    child_periods,
    day_period,
    hour_period,
    local_date,
    month_period,
    parked_vehicle_ids,
    render_line,
    week_period,
    DAY_BASELINE_DAYS,
    HOUR_BASELINE_DAYS,
    MIN_BASELINE_PERIODS,
    MONTH_BASELINE_MONTHS,
    RULES_VERSION,
    WEEK_BASELINE_WEEKS,
    person_record_digest,
)

logger = logging.getLogger("sql-writer.vision_rollup")

ROLLUP_LOCK_KEY = 0x0A1C_0004


class MigrationMissing(RuntimeError):
    """vision_rollup does not exist yet. The loop backs off instead of crash-looping."""


@dataclass(frozen=True)
class RollupConfig:
    streams: Tuple[str, ...] = ("walkway",)
    tz: str = "America/Denver"
    settle_sec: float = 120.0
    recompute_hours: int = 3
    catchup_hours: int = 48
    parked_min_hours: int = 4
    prop_individual_ids: Tuple[str, ...] = ()


# --- reader / store protocols ---------------------------------------------------------------------


class Reader(Protocol):
    def cursor(self, stream_id: str) -> Optional[datetime]: ...
    def sightings(self, stream_id: str, lo: datetime, hi: datetime) -> List[SightingRow]: ...
    def camera_minutes(self, stream_id: str, lo: datetime, hi: datetime) -> Dict[datetime, int]: ...
    def peaks(self, stream_id: str, lo: datetime, hi: datetime) -> Dict[datetime, Dict[str, int]]: ...
    def captions(self, stream_id: str, lo: datetime, hi: datetime) -> List[CaptionRow]: ...
    def story_gate(self) -> Dict[str, Any]: ...
    def stories(self, stream_id: str, lo: datetime, hi: datetime, rules_version: Optional[int]) -> List[StoryRow]: ...
    def person_links(self) -> Optional[Dict[str, PersonRef]]: ...


class Store(Protocol):
    def get(self, stream_id: str, grain: str, start: datetime) -> Optional[Dict[str, Any]]: ...
    def recent(self, stream_id: str, grain: str, before: datetime, limit: int) -> List[Dict[str, Any]]: ...
    def put(self, stream_id: str, period: Period, rollup: Dict[str, Any], line: str) -> None: ...


class MemoryStore:
    """Rollup rows in a dict: the dry-run backfill and the replay eval write nothing."""

    def __init__(self) -> None:
        self.rows: Dict[Tuple[str, str, datetime], Dict[str, Any]] = {}
        self.lines: Dict[Tuple[str, str, datetime], str] = {}

    def get(self, stream_id: str, grain: str, start: datetime) -> Optional[Dict[str, Any]]:
        return self.rows.get((stream_id, grain, start))

    def recent(self, stream_id: str, grain: str, before: datetime, limit: int) -> List[Dict[str, Any]]:
        keys = sorted((k for k in self.rows if k[0] == stream_id and k[1] == grain and k[2] < before),
                      key=lambda k: k[2], reverse=True)
        return [self.rows[k] for k in keys[:limit]]

    def put(self, stream_id: str, period: Period, rollup: Dict[str, Any], line: str) -> None:
        self.rows[(stream_id, period.grain, period.start)] = rollup
        self.lines[(stream_id, period.grain, period.start)] = line


# --- the build order ------------------------------------------------------------------------------


def _hour_baseline(store: Store, stream_id: str, hour: Period, tz: ZoneInfo) -> Optional[Dict[str, Any]]:
    """Mean passings for this local clock hour over earlier day rows where the camera saw it."""
    local_hour = hour.start.astimezone(tz).hour
    day_start = day_period(local_date(hour.start, tz), tz).start
    values: List[float] = []
    for d in store.recent(stream_id, "day", day_start, HOUR_BASELINE_DAYS):
        for e in d.get("series") or []:
            if e.get("status") == "ok" and int(e.get("hour", -1)) == local_hour:
                values.append(float(e.get("passings", 0)))
    if len(values) < MIN_BASELINE_PERIODS:
        return {"mean_passings": None, "periods": len(values)}
    return {"mean_passings": round(sum(values) / len(values), 2), "periods": len(values)}


@dataclass
class _Raw:
    sightings: List[SightingRow]
    minutes: Dict[datetime, int]
    peaks: Dict[datetime, Dict[str, int]]
    captions: List[CaptionRow]
    stories: List[StoryRow]
    persons: Optional[Dict[str, PersonRef]] = None


def build_hours(reader: Reader, store: Store, stream_id: str, hours: Sequence[Period], tz: ZoneInfo,
                cfg: RollupConfig, gate: Mapping[str, Any], raw: _Raw) -> int:
    written = 0
    prop = frozenset(cfg.prop_individual_ids)
    for h in hours:
        trailing = [s for s in raw.sightings if h.end - timedelta(hours=24) <= s.started_at < h.end]
        rollup = build_hour(HourInputs(
            stream_id=stream_id, period=h,
            sightings=[s for s in trailing if h.start <= s.started_at < h.end],
            camera_minutes=raw.minutes.get(h.start, 0),
            peak_at_once=raw.peaks.get(h.start, {}),
            captions=[c for c in raw.captions if h.start <= c.created_at < h.end],
            stories=[s for s in raw.stories if h.start <= s.started_at < h.end],
            story_gate=gate,
            parked_ids=parked_vehicle_ids(trailing, cfg.parked_min_hours),
            prop_ids=prop,
            baseline=_hour_baseline(store, stream_id, h, tz),
            persons=raw.persons,
        ), tz)
        store.put(stream_id, h, rollup, render_line(rollup))
        written += 1
    return written


def build_parent_row(store: Store, stream_id: str, period: Period, tz: ZoneInfo, covered_until: datetime,
                     gate: Mapping[str, Any]) -> Dict[str, Any]:
    grain = {"day": "hour", "week": "day", "month": "day"}[period.grain]
    children = [(c, store.get(stream_id, grain, c.start)) for c in child_periods(period, tz)]
    lookback = {"day": max(DAY_BASELINE_DAYS, HOUR_BASELINE_DAYS), "week": WEEK_BASELINE_WEEKS,
                "month": MONTH_BASELINE_MONTHS}[period.grain]
    history = store.recent(stream_id, period.grain, period.start, lookback)
    rollup = build_parent(stream_id, period, children, tz,
                          covered_until=None if covered_until >= period.end else covered_until,
                          history=history, story_gate=gate)
    store.put(stream_id, period, rollup, render_line(rollup))
    return rollup


def _stale(row: Optional[Mapping[str, Any]], digest: Optional[str] = None) -> bool:
    """Missing, written under older counting rules, or grouped by other person links than
    ``digest`` (the current ones). ``digest`` None: the record is unreadable now, so a row
    is not rebuilt just for that (it would only lose its person grouping)."""
    if row is None:
        return True
    try:
        if int(row.get("rules_version") or 1) < RULES_VERSION:
            return True
    except (TypeError, ValueError):
        return True
    if digest is not None and (row.get("person_record") or {}).get("digest") != digest:
        return True
    return False


def _person_links(reader: Any) -> Optional[Dict[str, PersonRef]]:
    fn = getattr(reader, "person_links", None)
    return fn() if callable(fn) else None


def ready_until(reader: Reader, stream_id: str, now: datetime, cfg: RollupConfig) -> Optional[datetime]:
    """End of the newest hour whose sightings are settled, or None if the individuals
    reducer has never run for this stream."""
    cur = reader.cursor(stream_id)
    if cur is None:
        return None
    edge = min(now - timedelta(seconds=cfg.settle_sec), cur - timedelta(seconds=60))
    return hour_period(edge).start  # hours ending at or before this are ready


def refresh_stream(reader: Reader, store: Store, stream_id: str, now: datetime, cfg: RollupConfig,
                   *, since: Optional[datetime] = None) -> Dict[str, Any]:
    """Build every ready hour that is recent or missing, then its day, week and month.

    ``since`` (backfill / replay) builds every hour from there, regardless of what exists."""
    tz = ZoneInfo(cfg.tz)
    summary: Dict[str, Any] = {"stream": stream_id, "hours": 0, "days": 0, "weeks": 0, "months": 0}
    ready = ready_until(reader, stream_id, now, cfg)
    if ready is None:
        summary["skipped"] = "individuals cursor missing"
        return summary
    summary["ready_until"] = ready.isoformat()
    persons = _person_links(reader)
    digest = person_record_digest(persons)
    summary["person_record"] = "ok" if persons is not None else "unavailable"
    if since is not None:
        # Never before the camera's first frame record: earlier hours would read "camera
        # down" (no inventory) while sightings exist, which is the record missing, not the camera.
        first_frame = getattr(reader, "first_frame", lambda _s: None)(stream_id)
        if first_frame is not None and first_frame > since:
            since = first_frame
            summary["clamped_to_first_frame"] = first_frame.isoformat()
        first = hour_period(since).start
        todo = []
        t = first
        while t < ready:
            todo.append(Period("hour", t, t + timedelta(hours=1)))
            t += timedelta(hours=1)
    else:
        todo = []
        t = ready - timedelta(hours=max(cfg.catchup_hours, cfg.recompute_hours))
        recompute_from = ready - timedelta(hours=cfg.recompute_hours)
        while t < ready:
            per = Period("hour", t, t + timedelta(hours=1))
            if t >= recompute_from or _stale(store.get(stream_id, "hour", t), digest):
                todo.append(per)
            t += timedelta(hours=1)
    gate = reader.story_gate()
    dates = sorted({local_date(h.start, tz) for h in todo} | {local_date(ready - timedelta(seconds=1), tz)})
    if todo:
        lo, hi = todo[0].start, todo[-1].end
        raw = _Raw(
            sightings=reader.sightings(stream_id, lo - timedelta(hours=24), hi),
            minutes=reader.camera_minutes(stream_id, lo, hi),
            peaks=reader.peaks(stream_id, lo, hi),
            captions=reader.captions(stream_id, lo, hi),
            stories=(reader.stories(stream_id, lo, hi, gate.get("rules_version")) if gate_passed(gate) else []),
            persons=persons,
        )
    else:
        raw = _Raw([], {}, {}, [], [], persons)
    by_date: Dict[date, List[Period]] = {}
    for h in todo:
        by_date.setdefault(local_date(h.start, tz), []).append(h)
    weeks_done: set = set()
    months_done: set = set()
    for d in dates:
        summary["hours"] += build_hours(reader, store, stream_id, by_date.get(d, []), tz, cfg, gate, raw)
        build_parent_row(store, stream_id, day_period(d, tz), tz, ready, gate)
        summary["days"] += 1
        # A week/month is rebuilt after each of its days, so its history and today's
        # partial day are both current; only the last rebuild per tick matters.
        wk, mo = week_period(d, tz), month_period(d, tz)
        build_parent_row(store, stream_id, wk, tz, ready, gate)
        build_parent_row(store, stream_id, mo, tz, ready, gate)
        weeks_done.add(wk.start)
        months_done.add(mo.start)
    summary["weeks"] = len(weeks_done)
    summary["months"] = len(months_done)
    summary["story_gate_passed"] = gate_passed(gate)
    return summary


# --- Postgres -------------------------------------------------------------------------------------


def _aware(ts: Any) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def is_missing_relation(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "undefinedtable" in type(getattr(exc, "orig", exc)).__name__.lower() or (
        "relation" in text and "does not exist" in text)


_PERSON_RECORD_WARNED = False


class DbReader:
    """Read-only access to the raw walkway tables (one SQLAlchemy connection)."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def _rows(self, sql: str, params: Mapping[str, Any]) -> List[Any]:
        from sqlalchemy import text

        return list(self.conn.execute(text(sql), dict(params)).mappings())

    def cursor(self, stream_id: str) -> Optional[datetime]:
        rows = self._rows("SELECT last_created_at FROM vision_individuals_cursor WHERE stream_id = :s",
                          {"s": stream_id})
        return _aware(rows[0]["last_created_at"]) if rows and rows[0]["last_created_at"] else None

    def sightings(self, stream_id: str, lo: datetime, hi: datetime) -> List[SightingRow]:
        rows = self._rows(
            "SELECT s.sighting_id, s.individual_id, s.started_at, s.zone_no_embed, s.stationary AS s_stat, "
            "i.kind, i.label, i.stationary AS i_stat "
            "FROM vision_individual_sighting s JOIN vision_individual i USING (individual_id) "
            "WHERE s.stream_id = :s AND s.started_at >= :lo AND s.started_at < :hi ORDER BY s.started_at",
            {"s": stream_id, "lo": lo, "hi": hi})
        return [SightingRow(sighting_id=r["sighting_id"], individual_id=r["individual_id"],
                            started_at=_aware(r["started_at"]), kind=r["kind"], label=r["label"],
                            zone_no_embed=bool(r["zone_no_embed"]), stationary=bool(r["s_stat"]),
                            individual_stationary=bool(r["i_stat"])) for r in rows]

    def first_frame(self, stream_id: str) -> Optional[datetime]:
        rows = self._rows("SELECT observed_at FROM vision_scene_inventory WHERE stream_id = :s "
                          "ORDER BY observed_at ASC LIMIT 1", {"s": stream_id})
        return _aware(rows[0]["observed_at"]) if rows else None

    def camera_minutes(self, stream_id: str, lo: datetime, hi: datetime) -> Dict[datetime, int]:
        rows = self._rows(
            "SELECT date_trunc('hour', observed_at AT TIME ZONE 'UTC') AS h, "
            "count(DISTINCT date_trunc('minute', observed_at AT TIME ZONE 'UTC')) AS n "
            "FROM vision_scene_inventory WHERE stream_id = :s AND observed_at >= :lo AND observed_at < :hi "
            "GROUP BY 1", {"s": stream_id, "lo": lo, "hi": hi})
        return {r["h"].replace(tzinfo=timezone.utc): int(r["n"]) for r in rows}

    def peaks(self, stream_id: str, lo: datetime, hi: datetime) -> Dict[datetime, Dict[str, int]]:
        """Most of each detector label in one vision window, per hour (a lower bound on how
        many were there; the sightings' passings are the upper bound)."""
        rows = self._rows(
            "SELECT date_trunc('hour', v.observed_at AT TIME ZONE 'UTC') AS h, c.key AS kind, "
            "max((c.value #>> '{}')::numeric) AS n "
            "FROM vision_scene_inventory v, jsonb_each(v.counts) c "
            "WHERE v.stream_id = :s AND v.observed_at >= :lo AND v.observed_at < :hi "
            "AND jsonb_typeof(c.value) = 'number' GROUP BY 1, 2", {"s": stream_id, "lo": lo, "hi": hi})
        out: Dict[datetime, Dict[str, int]] = {}
        for r in rows:
            out.setdefault(r["h"].replace(tzinfo=timezone.utc), {})[str(r["kind"])] = int(r["n"])
        return out

    def captions(self, stream_id: str, lo: datetime, hi: datetime) -> List[CaptionRow]:
        from orion.situational.street_material import CAMERA_CAPTION_EVENT_TYPE, detector_labels_from_tags

        rows = self._rows(
            "SELECT event_id, created_at, narrative, tags FROM vision_events "
            "WHERE stream_id = :s AND event_type = :t AND created_at >= :lo AND created_at < :hi "
            "AND narrative IS NOT NULL AND narrative <> '' ORDER BY created_at",
            {"s": stream_id, "t": CAMERA_CAPTION_EVENT_TYPE, "lo": lo, "hi": hi})
        return [CaptionRow(event_id=str(r["event_id"]), created_at=_aware(r["created_at"]), text=str(r["narrative"]),
                           detector=tuple(detector_labels_from_tags(r["tags"]))) for r in rows]

    def story_gate(self) -> Dict[str, Any]:
        try:
            with self.conn.begin_nested():
                version = self._rows(
                    "SELECT max(rules_version) AS v FROM vision_motion_story WHERE published AND clean", {})[0]["v"]
                if version is None:
                    return tally_from_counts({}, 0, None)
                rows = self._rows(
                    "SELECT CASE WHEN v.cant_tell THEN 'cant_tell' ELSE v.verdict END AS verdict, count(*) AS n, "
                    "count(*) FILTER (WHERE s.name_label IS NOT NULL) AS named "
                    "FROM vision_motion_story_verdict v JOIN vision_motion_story s ON s.story_id = v.story_id "
                    "WHERE s.published AND s.clean AND s.rules_version = :v GROUP BY 1", {"v": version})
        except Exception as exc:  # stories tables missing or unreadable: the gate is closed
            logger.info("vision_rollup_story_gate_unreadable error=%s", type(exc).__name__)
            return tally_from_counts({}, 0, None)
        counts = {r["verdict"]: int(r["n"]) for r in rows if r["verdict"]}
        named = sum(int(r["named"]) for r in rows if r["verdict"] in ("right", "wrong_name", "wrong_motion",
                                                                      "wrong_other"))
        return tally_from_counts(counts, named, int(version))

    def stories(self, stream_id: str, lo: datetime, hi: datetime, rules_version: Optional[int]) -> List[StoryRow]:
        rows = self._rows(
            "SELECT story_id, started_at, kind, sentence, name_label FROM vision_motion_story "
            "WHERE stream_id = :s AND published AND clean AND rules_version = :v "
            "AND started_at >= :lo AND started_at < :hi ORDER BY started_at",
            {"s": stream_id, "v": rules_version, "lo": lo, "hi": hi})
        return [StoryRow(story_id=r["story_id"], started_at=_aware(r["started_at"]), kind=r["kind"] or "other",
                         sentence=r["sentence"] or "", name_label=r["name_label"]) for r in rows]


    def person_links(self) -> Optional[Dict[str, PersonRef]]:
        """individual_id -> its active camera link (``vision_individual`` or
        ``vision_belonging``) with the person row. None when the person tables are missing
        or unreadable: the rollup then groups by label (logged once per process)."""
        global _PERSON_RECORD_WARNED
        try:
            with self.conn.begin_nested():
                rows = self._rows(
                    "SELECT l.source_ref, l.source_kind, p.person_id, p.display_name, p.status "
                    "FROM person_link l JOIN person p ON p.person_id = l.person_id "
                    "WHERE l.revoked_at IS NULL AND l.source_kind IN ('vision_individual', 'vision_belonging') "
                    "ORDER BY l.source_ref, l.source_kind DESC", {})
        except Exception as exc:  # noqa: BLE001 -- fail-open to label grouping
            if not _PERSON_RECORD_WARNED:
                _PERSON_RECORD_WARNED = True
                logger.warning("vision_rollup_person_record_unavailable error=%s missing=%s -- named grouped by label",
                               type(exc).__name__, is_missing_relation(exc))
            return None
        out: Dict[str, PersonRef] = {}
        for r in rows:
            # vision_individual sorts before vision_belonging (DESC): an individual link wins.
            out.setdefault(str(r["source_ref"]), PersonRef(
                person_id=str(r["person_id"]), display_name=" ".join(str(r["display_name"] or "").split()),
                status=str(r["status"] or "active"), source_kind=str(r["source_kind"])))
        return out


class DbStore:
    """``vision_rollup`` rows. ``put`` upserts on (stream_id, grain, period_start)."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def get(self, stream_id: str, grain: str, start: datetime) -> Optional[Dict[str, Any]]:
        from sqlalchemy import text

        row = self.conn.execute(text(
            "SELECT rollup FROM vision_rollup WHERE stream_id = :s AND grain = :g AND period_start = :p"),
            {"s": stream_id, "g": grain, "p": start}).first()
        return _json(row[0]) if row else None

    def recent(self, stream_id: str, grain: str, before: datetime, limit: int) -> List[Dict[str, Any]]:
        from sqlalchemy import text

        rows = self.conn.execute(text(
            "SELECT rollup FROM vision_rollup WHERE stream_id = :s AND grain = :g AND period_start < :b "
            "ORDER BY period_start DESC LIMIT :n"), {"s": stream_id, "g": grain, "b": before, "n": limit}).all()
        return [_json(r[0]) for r in rows]

    def put(self, stream_id: str, period: Period, rollup: Dict[str, Any], line: str) -> None:
        from sqlalchemy import text

        self.conn.execute(text(
            "INSERT INTO vision_rollup (stream_id, grain, period_start, period_end, label, complete, "
            "covered_until, rollup, line, rules_version, computed_at) VALUES (:s, :g, :p, :e, :l, :c, :cu, "
            "CAST(:j AS jsonb), :line, :rv, now()) "
            "ON CONFLICT (stream_id, grain, period_start) DO UPDATE SET period_end = EXCLUDED.period_end, "
            "label = EXCLUDED.label, complete = EXCLUDED.complete, covered_until = EXCLUDED.covered_until, "
            "rollup = EXCLUDED.rollup, line = EXCLUDED.line, rules_version = EXCLUDED.rules_version, "
            "computed_at = now()"),
            {"s": stream_id, "g": period.grain, "p": period.start, "e": period.end, "l": rollup.get("label"),
             "c": bool(rollup.get("complete")), "cu": rollup.get("covered_until") or period.end.isoformat(),
             "j": json.dumps(rollup, sort_keys=True, default=str), "line": line,
             "rv": int(rollup.get("rules_version") or 1)})


def _json(v: Any) -> Dict[str, Any]:
    if isinstance(v, str):
        return json.loads(v)
    return dict(v or {})


def check_table(conn: Any) -> None:
    from sqlalchemy import text

    if conn.execute(text("SELECT to_regclass('vision_rollup')")).scalar() is None:
        raise MigrationMissing("vision_rollup missing -- apply services/orion-sql-db/manual_migration_vision_rollup_v1.sql")


def run_one_rollup_cycle(*, postgres_uri: str, cfg: RollupConfig, now: Optional[datetime] = None,
                         since: Optional[datetime] = None, write: bool = True,
                         memory: Optional[MemoryStore] = None) -> Dict[str, Any]:
    """One pass over every configured stream. Blocking: run in a worker thread.

    ``write=False`` reads in a READ ONLY transaction and builds into ``memory`` (dry run / replay)."""
    from sqlalchemy import create_engine, text

    ts = now or datetime.now(timezone.utc)
    out: Dict[str, Any] = {"streams": []}
    engine = create_engine(postgres_uri, pool_pre_ping=True, pool_size=1, max_overflow=1)
    lock_conn = None
    try:
        if write:
            lock_conn = engine.connect()
            if not lock_conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": ROLLUP_LOCK_KEY}).scalar():
                out["skipped"] = "another instance holds the rollup lock"
                return out
            lock_conn.commit()
        for stream_id in cfg.streams:
            with engine.begin() as conn:
                if not write:
                    conn.execute(text("SET TRANSACTION READ ONLY"))
                else:
                    check_table(conn)
                reader = DbReader(conn)
                store: Any = DbStore(conn) if write else (memory if memory is not None else MemoryStore())
                try:
                    out["streams"].append(refresh_stream(reader, store, stream_id, ts, cfg, since=since))
                except Exception as exc:
                    if is_missing_relation(exc):
                        raise MigrationMissing(str(exc)) from exc
                    raise
    finally:
        if lock_conn is not None:
            try:
                lock_conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": ROLLUP_LOCK_KEY})
                lock_conn.close()
            except Exception:
                pass
        engine.dispose()
    return out
