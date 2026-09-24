"""Rhythm learner: when things usually happen on the street, predicted in
advance and graded afterwards.

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md ideas 2
and 8. Every expectation is a prediction written BEFORE its window opens and
scored by the same module after it closes: ``met``, ``missed``, or
``unscorable`` (the camera was not watching -- absence of evidence is not
evidence of absence).

Subjects:

- ``individual:<id>`` -- occurrences are sighting starts from the individuals
  reducer; "met" means a sighting overlapped the window.
- ``label:<label>`` -- from ``vision_scene_inventory`` counts. An arrival is
  the first window with count > 0 after at least ``arrival_gap_sec`` with no
  positive window (a debounced 0 -> >0, so detector flicker is not an
  arrival). "met" means the label was present in any census window inside.

Model: per (stream, subject, day_kind in weekday/weekend/any), a circular
(wrapped Gaussian) kernel density over minute-of-day in the local timezone,
so 23:50 and 00:10 are 20 minutes apart, not 1420. No expectation is emitted
under ``min_occurrences`` across ``min_days`` distinct days, and each window
must itself be hit on ``min_days`` distinct days. Window = peak +-
the half-width where density falls to half the peak (clamped). Confidence =
fraction of observed days of that day_kind with an occurrence inside the
window.

Rhythm surprise (met=0 / missed=1) lives in ``vision_percept_expectation``
only. It is NOT wired to the substrate graph: the metric gate's live-data
check cannot run until weeks of real data exist.

Pure functions first (explicit ``now``), then the blocking DB cycle.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger("sql-writer.vision_rhythm")

MINUTES_PER_DAY = 1440
DAY_KINDS = ("weekday", "weekend", "any")
EXPECT_KEY_PREFIX = "orion:vision:expect:"
_EXPECTATION_NS = uuid.UUID("6f1d3c52-8a0e-4a55-9a8e-7c1f0d2b9e41")


# ---------------------------------------------------------------------------
# Pure
# ---------------------------------------------------------------------------


def minute_of_day(ts: datetime, tz: ZoneInfo) -> int:
    local = ts.astimezone(tz)
    return local.hour * 60 + local.minute


def day_kind_of(d: date) -> str:
    return "weekend" if d.weekday() >= 5 else "weekday"


def matches_day_kind(d: date, day_kind: str) -> bool:
    return day_kind == "any" or day_kind_of(d) == day_kind


def circular_distance(a: float, b: float) -> float:
    d = abs(a - b) % MINUTES_PER_DAY
    return min(d, MINUTES_PER_DAY - d)


def circular_kde(minutes: Sequence[float], bandwidth: float) -> List[float]:
    """Wrapped-Gaussian density at every minute of the day (unnormalized)."""
    out = [0.0] * MINUTES_PER_DAY
    if not minutes or bandwidth <= 0:
        return out
    reach = int(min(MINUTES_PER_DAY // 2, math.ceil(4 * bandwidth)))
    for m in minutes:
        c = int(round(m)) % MINUTES_PER_DAY
        for d in range(-reach, reach + 1):
            out[(c + d) % MINUTES_PER_DAY] += math.exp(-0.5 * (d / bandwidth) ** 2)
    return out


def find_peaks(density: Sequence[float], *, rel_floor: float = 0.1) -> List[int]:
    """Circular local maxima at or above ``rel_floor`` * max; plateaus give one peak."""
    top = max(density) if density else 0.0
    if top <= 0:
        return []
    n = len(density)
    peaks = []
    for i in range(n):
        v = density[i]
        if v < rel_floor * top:
            continue
        left, right = density[(i - 1) % n], density[(i + 1) % n]
        if v > left and v >= right:
            peaks.append(i)
    return peaks


def half_width(density: Sequence[float], peak: int, *, min_w: int, max_w: int) -> int:
    """Minutes from the peak to where density first drops below half (max of both sides)."""
    n = len(density)
    half = density[peak] / 2.0
    widths = []
    for step in (-1, 1):
        w = 0
        while w < max_w and density[(peak + step * (w + 1)) % n] >= half:
            w += 1
        widths.append(w + 1)
    return int(min(max_w, max(min_w, max(widths))))


@dataclass(frozen=True)
class FittedWindow:
    day_kind: str
    peak_minute: int
    start_offset_min: int   # minutes from local midnight; may be < 0 or >= 1440 (wraps)
    end_offset_min: int
    confidence: float
    support_days: int
    support_sightings: int
    # Hit rate of this window on watched weekdays / weekend days separately
    # (None = no watched day of that kind). Lets an "any" window be refused
    # for a kind of day that plainly does not follow it.
    weekday_rate: Optional[float] = None
    weekend_rate: Optional[float] = None

    def rate_for(self, kind: str) -> Optional[float]:
        return self.weekday_rate if kind == "weekday" else self.weekend_rate


def _in_window(minute: float, peak: int, width: int) -> bool:
    return circular_distance(minute, peak) <= width


def fit_subject(
    occurrences: Sequence[datetime],
    *,
    tz: ZoneInfo,
    day_kind: str,
    observed_days: Optional[Set[date]] = None,
    min_occurrences: int = 5,
    min_days: int = 5,
    bandwidth: float = 20.0,
    min_confidence: float = 0.5,
    max_windows: int = 3,
    min_width: int = 10,
    max_width: int = 90,
) -> List[FittedWindow]:
    """Fit one (subject, day_kind). Empty list = not enough evidence (never a guess)."""
    occ = [o for o in occurrences if matches_day_kind(o.astimezone(tz).date(), day_kind)]
    days = {o.astimezone(tz).date() for o in occ}
    if len(occ) < min_occurrences or len(days) < min_days:
        return []
    minutes = [minute_of_day(o, tz) for o in occ]
    density = circular_kde(minutes, bandwidth)
    denom_days = {d for d in (observed_days or set()) if matches_day_kind(d, day_kind)} | days
    fits: List[FittedWindow] = []
    for p in find_peaks(density):
        w = half_width(density, p, min_w=min_width, max_w=max_width)
        hit_days = {o.astimezone(tz).date() for o, m in zip(occ, minutes) if _in_window(m, p, w)}
        # Minimum support applies to the WINDOW, not just the subject: three
        # coincidences near 18:00 out of six random visits is not a rhythm.
        if len(hit_days) < min_days:
            continue
        conf = len(hit_days) / len(denom_days) if denom_days else 0.0
        if conf < min_confidence:
            continue
        rates = {}
        for k in ("weekday", "weekend"):
            kd = {d for d in denom_days if day_kind_of(d) == k}
            rates[k] = round(len({d for d in hit_days if day_kind_of(d) == k}) / len(kd), 4) if kd else None
        fits.append(FittedWindow(
            day_kind=day_kind, peak_minute=p, start_offset_min=p - w, end_offset_min=p + w,
            confidence=round(min(1.0, conf), 4), support_days=len(days), support_sightings=len(occ),
            weekday_rate=rates["weekday"], weekend_rate=rates["weekend"],
        ))
    fits.sort(key=lambda f: -f.confidence)
    return fits[:max_windows]


@dataclass(frozen=True)
class PlannedExpectation:
    window_start: datetime
    window_end: datetime
    fit: FittedWindow


def _round_down_5(ts: datetime) -> datetime:
    return ts.replace(minute=ts.minute - ts.minute % 5, second=0, microsecond=0)


def plan_expectations(
    fits_by_day_kind: Dict[str, List[FittedWindow]], *, now: datetime, tz: ZoneInfo, horizon_h: float = 24.0,
    min_confidence: float = 0.5,
) -> List[PlannedExpectation]:
    """Upcoming windows starting in (now, now + horizon]. A specific day_kind model
    (weekday/weekend) is used for a date when it has fits, else the "any" model --
    but an "any" window is skipped for a kind of day on which it was hit less
    than ``min_confidence`` of the watched days (a weekday dog is not expected
    on Saturday just because weekends have too few days for their own model)."""
    local_today = now.astimezone(tz).date()
    horizon = now + timedelta(hours=horizon_h)
    out: List[PlannedExpectation] = []
    for delta in (-1, 0, 1, 2):  # -1: a window whose offset wraps past midnight
        d = local_today + timedelta(days=delta)
        kind = day_kind_of(d)
        fits = fits_by_day_kind.get(kind) or [
            f for f in (fits_by_day_kind.get("any") or [])
            if f.rate_for(kind) is None or f.rate_for(kind) >= min_confidence
        ]
        midnight = datetime(d.year, d.month, d.day, tzinfo=tz)
        for f in fits:
            ws = _round_down_5((midnight + timedelta(minutes=f.start_offset_min)).astimezone(timezone.utc))
            we = (midnight + timedelta(minutes=f.end_offset_min)).astimezone(timezone.utc)
            if now < ws <= horizon:
                out.append(PlannedExpectation(ws, we, f))
    out.sort(key=lambda p: p.window_start)
    return out


def overlaps(a_start: datetime, a_end: datetime, spans: Iterable[Tuple[datetime, datetime]]) -> bool:
    return any(a_start < e and s < a_end for s, e in spans)


# Census windows are ~5 s long and ~5 s apart (live p99 gap 5.1 s, measured
# 2026-09-24 on cam0). The camera counts as watching across a gap up to this.
CENSUS_GAP_TOLERANCE_SEC = 30.0


def coverage_fraction(
    spans: Sequence[Tuple[float, float]], start: datetime, end: datetime,
    *, gap_tolerance_sec: float = CENSUS_GAP_TOLERANCE_SEC,
) -> float:
    """Fraction of [start, end] the camera was watching: union of census spans,
    each extended by the gap tolerance, clipped to the window."""
    a, b = start.timestamp(), end.timestamp()
    if b <= a:
        return 0.0
    ivs = sorted((max(a, s0), min(b, s1 + gap_tolerance_sec)) for s0, s1 in spans)
    covered, cur_s, cur_e = 0.0, None, None
    for s0, s1 in ivs:
        if s1 <= s0:
            continue
        if cur_e is None or s0 > cur_e:
            if cur_e is not None:
                covered += cur_e - cur_s
            cur_s, cur_e = s0, s1
        else:
            cur_e = max(cur_e, s1)
    if cur_e is not None:
        covered += cur_e - cur_s
    return min(1.0, covered / (b - a))


def score_window(*, occurred: bool, coverage: float, min_coverage: float = 0.8) -> str:
    """met if it happened; missed only if the camera watched enough of the
    window to have seen it; otherwise unscorable."""
    if occurred:
        return "met"
    if coverage < min_coverage:
        return "unscorable"
    return "missed"


def arrivals_from_windows(windows: Sequence[Tuple[datetime, int]], *, gap_sec: float) -> List[datetime]:
    """Debounced 0 -> >0: a positive window with no positive window in the prior ``gap_sec``."""
    out: List[datetime] = []
    prev_pos: Optional[datetime] = None
    for ts, count in sorted(windows):
        if count and count > 0:
            if prev_pos is None or (ts - prev_pos).total_seconds() > gap_sec:
                out.append(ts)
            prev_pos = ts
    return out


def expectation_id(stream_id: str, subject_key: str, window_start: datetime) -> str:
    return str(uuid.uuid5(_EXPECTATION_NS, f"{stream_id}|{subject_key}|{window_start.isoformat()}"))


def expect_key_ttl(open_windows: Sequence[Tuple[datetime, datetime, bool]], now: datetime) -> Optional[int]:
    """TTL for orion:vision:expect:<stream>: seconds until the latest end among
    windows that are open now and not yet met. None = nothing to set."""
    ends = [we for ws, we, met in open_windows if ws <= now < we and not met]
    if not ends:
        return None
    return max(1, int(math.ceil((max(ends) - now).total_seconds())))


def _hhmm(ts: datetime, tz: ZoneInfo) -> str:
    return ts.astimezone(tz).strftime("%H:%M")


def outcome_narrative(
    *, status: str, subject_label: str, stream_id: str, window_start: datetime, window_end: datetime,
    confidence: float, support_days: int, tz: ZoneInfo, arrived_at: Optional[datetime] = None,
    coverage: float = 0.0,
) -> str:
    win = f"{_hhmm(window_start, tz)}-{_hhmm(window_end, tz)}"
    if status == "met":
        at = f" at {_hhmm(arrived_at, tz)}" if arrived_at else ""
        return (f"{subject_label} came{at} on the {stream_id} camera, inside the {win} window I expected "
                f"({confidence:.0%} sure, from {support_days} days of watching).")
    return (f"I expected {subject_label} on the {stream_id} camera between {win} "
            f"({confidence:.0%} sure, from {support_days} days of watching), and it did not come. "
            f"The camera was watching for {coverage:.0%} of that window.")


# ---------------------------------------------------------------------------
# Postgres + Redis. Blocking; run in a worker thread.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RhythmConfig:
    min_occurrences: int = 5
    min_days: int = 5
    history_days: float = 28.0
    bandwidth_min: float = 20.0
    min_confidence: float = 0.5
    max_per_subject: int = 3
    score_lag_sec: float = 600.0
    labels: Tuple[str, ...] = ("vehicle", "package", "mail truck")
    label_streams: Tuple[str, ...] = ("walkway",)
    arrival_gap_sec: float = 300.0
    local_tz: str = "America/Denver"
    min_coverage: float = 0.8


RHYTHM_LOCK_KEY = 0x0A1C_0002


@dataclass(frozen=True)
class Subject:
    stream_id: str
    key: str
    label: str
    occurrences: Tuple[datetime, ...]


def _check_tables(conn) -> None:
    from sqlalchemy import text

    from app.vision_individuals import MigrationMissing

    missing = conn.execute(text(
        "SELECT t FROM unnest(ARRAY['vision_percept_expectation','vision_rhythm_cursor',"
        "'vision_individual','vision_individual_sighting']) AS t WHERE to_regclass(t) IS NULL")).fetchall()
    if missing:
        raise MigrationMissing(
            "missing tables " + ",".join(r[0] for r in missing)
            + " -- apply services/orion-sql-db/manual_migration_walkway_camera_v1.sql")


def _label_arrivals(conn, stream_id: str, label: str, since: datetime, gap_sec: float) -> List[datetime]:
    from sqlalchemy import text

    rows = conn.execute(text("""
        SELECT observed_at FROM (
            SELECT observed_at, LAG(observed_at) OVER (ORDER BY observed_at) AS prev_pos
            FROM vision_scene_inventory
            WHERE stream_id = :s AND observed_at > :since
              AND COALESCE((counts->>:label)::numeric, 0) > 0
        ) q WHERE prev_pos IS NULL OR observed_at - prev_pos > make_interval(secs => :gap)
        ORDER BY observed_at
    """), {"s": stream_id, "since": since, "label": label, "gap": gap_sec}).fetchall()
    return [r[0] for r in rows]


def _load_subjects(conn, now: datetime, cfg: RhythmConfig) -> List[Subject]:
    from sqlalchemy import text

    from app.vision_individuals import individual_display_label

    since = now - timedelta(days=cfg.history_days)
    subjects: List[Subject] = []
    inds = conn.execute(text(
        "SELECT individual_id, stream_id, kind, label FROM vision_individual WHERE sighting_count >= :n"),
        {"n": cfg.min_occurrences}).fetchall()
    for iid, stream_id, kind, label in inds:
        occ = [r[0] for r in conn.execute(text(
            "SELECT started_at FROM vision_individual_sighting WHERE individual_id=:i AND started_at >= :since "
            "ORDER BY started_at"), {"i": iid, "since": since}).fetchall()]
        subjects.append(Subject(stream_id, f"individual:{iid}", individual_display_label(label, kind, iid), tuple(occ)))
    for stream_id in cfg.label_streams:
        for label in cfg.labels:
            occ = _label_arrivals(conn, stream_id, label, since, cfg.arrival_gap_sec)
            subjects.append(Subject(stream_id, f"label:{label}", label, tuple(occ)))
    return subjects


def _observed_days(conn, stream_id: str, since: datetime, tz: ZoneInfo) -> Set[date]:
    from sqlalchemy import text

    rows = conn.execute(text(
        "SELECT DISTINCT (observed_at AT TIME ZONE :tz)::date FROM vision_scene_inventory "
        "WHERE stream_id=:s AND observed_at > :since AND frame_count > 0"),
        {"tz": str(tz.key), "s": stream_id, "since": since}).fetchall()
    return {r[0] for r in rows}


def _occurred(conn, stream_id: str, subject_key: str, start: datetime, end: datetime) -> Optional[datetime]:
    """First occurrence time inside [start, end], or None."""
    from sqlalchemy import text

    kind, _, ref = subject_key.partition(":")
    if kind == "individual":
        row = conn.execute(text(
            "SELECT min(GREATEST(started_at, :a)) FROM vision_individual_sighting WHERE individual_id=:i "
            "AND started_at <= :b AND ended_at >= :a"), {"i": ref, "a": start, "b": end}).fetchone()
    else:
        row = conn.execute(text(
            "SELECT min(observed_at) FROM vision_scene_inventory WHERE stream_id=:s AND observed_at >= :a "
            "AND observed_at <= :b AND COALESCE((counts->>:l)::numeric, 0) > 0"),
            {"s": stream_id, "a": start, "b": end, "l": ref}).fetchone()
    return row[0] if row and row[0] else None


def _census_coverage(conn, stream_id: str, start: datetime, end: datetime) -> float:
    from sqlalchemy import text

    rows = conn.execute(text(
        "SELECT COALESCE(window_start_ts, extract(epoch FROM observed_at)), "
        "COALESCE(window_end_ts, extract(epoch FROM observed_at)) FROM vision_scene_inventory "
        "WHERE stream_id=:s AND frame_count > 0 AND observed_at >= :a AND observed_at <= :b"),
        {"s": stream_id, "a": start - timedelta(seconds=CENSUS_GAP_TOLERANCE_SEC),
         "b": end + timedelta(seconds=CENSUS_GAP_TOLERANCE_SEC)}).fetchall()
    return coverage_fraction([(float(r[0]), float(r[1])) for r in rows], start, end)


def _individuals_caught_up(conn, stream_id: str, through: datetime) -> bool:
    """An individual can only be graded once the individuals reducer has
    processed every crop through the window end -- a lagging or backed-off
    reducer must not turn into a false "missed"."""
    from sqlalchemy import text

    row = conn.execute(text("SELECT last_observed_at FROM vision_individuals_cursor WHERE stream_id=:s"),
                       {"s": stream_id}).fetchone()
    return bool(row and row[0] >= through)


def run_one_rhythm_cycle(
    *, postgres_uri: str, cfg: RhythmConfig, now: Optional[datetime] = None, redis_url: Optional[str] = None,
) -> dict:
    from sqlalchemy import create_engine, text

    from app.vision_individuals import MigrationMissing, is_missing_relation

    ts = now or datetime.now(timezone.utc)
    tz = ZoneInfo(cfg.local_tz)
    summary = {"subjects": 0, "fitted": 0, "emitted": 0, "met": 0, "missed": 0, "unscorable": 0,
               "expect_keys": 0}
    engine = create_engine(postgres_uri, pool_pre_ping=True)
    lock_conn = None
    per_stream: Dict[str, List[Tuple[datetime, datetime, bool]]] = {}
    per_stream_ids: Dict[str, List[str]] = {}
    try:
        lock_conn = engine.connect()
        if not lock_conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": RHYTHM_LOCK_KEY}).scalar():
            summary["skipped"] = "another instance holds the rhythm lock"
            return summary
        lock_conn.commit()
        # 1. Score closed windows first, so a fresh expectation never gets
        #    graded in the same pass that would also emit its successor.
        with engine.begin() as conn:
            _check_tables(conn)
            due = conn.execute(text(
                "SELECT expectation_id, stream_id, subject_key, subject_label, window_start, window_end, "
                "confidence, support_days FROM vision_percept_expectation WHERE status='open' AND window_end <= :t"),
                {"t": ts - timedelta(seconds=cfg.score_lag_sec)}).fetchall()
            for r in due:
                arrived = _occurred(conn, r.stream_id, r.subject_key, r.window_start, r.window_end)
                if (arrived is None and r.subject_key.startswith("individual:")
                        and not _individuals_caught_up(conn, r.stream_id, r.window_end)):
                    summary["deferred"] = summary.get("deferred", 0) + 1
                    continue
                coverage = _census_coverage(conn, r.stream_id, r.window_start, r.window_end)
                status = score_window(occurred=arrived is not None, coverage=coverage,
                                      min_coverage=cfg.min_coverage)
                summary[status] += 1
                event_id = None
                if status in ("met", "missed"):
                    event_id = f"rhythm-{r.expectation_id}"
                    narrative = outcome_narrative(
                        status=status, subject_label=r.subject_label, stream_id=r.stream_id,
                        window_start=r.window_start, window_end=r.window_end, confidence=r.confidence,
                        support_days=r.support_days, tz=tz, arrived_at=arrived, coverage=coverage)
                    conn.execute(text(
                        "INSERT INTO vision_events (event_id, event_type, narrative, entities, tags, confidence, "
                        "salience, evidence_refs, stream_id, created_at) VALUES (:id, :et, :n, CAST(:e AS jsonb), "
                        "CAST(:tg AS jsonb), :c, :sal, CAST(:ev AS jsonb), :sid, now()) "
                        "ON CONFLICT (event_id) DO NOTHING"),
                        {"id": event_id, "sid": r.stream_id, "et": "arrived_as_expected" if status == "met" else "expected_absent",
                         "n": narrative, "e": json.dumps([r.subject_label]),
                         "tg": json.dumps([r.stream_id, "rhythm", r.subject_key]), "c": r.confidence,
                         # a confident miss is more worth noticing than a confident hit
                         "sal": r.confidence if status == "missed" else 0.2 * r.confidence,
                         "ev": json.dumps([f"expectation:{r.expectation_id}"])})
                conn.execute(text(
                    "UPDATE vision_percept_expectation SET status=:st, scored_at=:t, outcome_event_id=:e "
                    "WHERE expectation_id=:id"), {"st": status, "t": ts, "e": event_id, "id": r.expectation_id})

        # 2. Fit and emit.
        since = ts - timedelta(days=cfg.history_days)
        with engine.begin() as conn:
            subjects = _load_subjects(conn, ts, cfg)
            obs_days: Dict[str, Set[date]] = {}
            for subj in subjects:
                summary["subjects"] += 1
                if subj.stream_id not in obs_days:
                    obs_days[subj.stream_id] = _observed_days(conn, subj.stream_id, since, tz)
                fits = {dk: fit_subject(
                    subj.occurrences, tz=tz, day_kind=dk, observed_days=obs_days[subj.stream_id],
                    min_occurrences=cfg.min_occurrences, min_days=cfg.min_days, bandwidth=cfg.bandwidth_min,
                    min_confidence=cfg.min_confidence, max_windows=cfg.max_per_subject) for dk in DAY_KINDS}
                if not any(fits.values()):
                    continue
                summary["fitted"] += 1
                existing = [(r[0], r[1]) for r in conn.execute(text(
                    "SELECT window_start, window_end FROM vision_percept_expectation WHERE stream_id=:s "
                    "AND subject_key=:k AND window_end > :t"), {"s": subj.stream_id, "k": subj.key, "t": ts}).fetchall()]
                planned = plan_expectations(fits, now=ts, tz=tz, min_confidence=cfg.min_confidence)
                planned = planned[: cfg.max_per_subject]
                for p in planned:
                    if overlaps(p.window_start, p.window_end, existing):
                        continue  # a refit moved the peak a minute; do not predict twice
                    f = p.fit
                    res = conn.execute(text("""
                        INSERT INTO vision_percept_expectation (expectation_id, stream_id, subject_key, subject_label,
                            day_kind, window_start, window_end, peak_minute, support_days, support_sightings,
                            confidence, status, emitted_at)
                        VALUES (:id, :s, :k, :l, :dk, :ws, :we, :pm, :sd, :ss, :c, 'open', :t)
                        ON CONFLICT (stream_id, subject_key, window_start) DO NOTHING
                    """), {"id": expectation_id(subj.stream_id, subj.key, p.window_start), "s": subj.stream_id,
                           "k": subj.key, "l": subj.label, "dk": f.day_kind, "ws": p.window_start,
                           "we": p.window_end, "pm": f.peak_minute % MINUTES_PER_DAY, "sd": f.support_days,
                           "ss": f.support_sightings, "c": f.confidence, "t": ts})
                    if res.rowcount:
                        summary["emitted"] += 1
                        existing.append((p.window_start, p.window_end))

            # 3. Idea 8: open, not-yet-met windows raise the camera's attention.
            open_now = conn.execute(text(
                "SELECT expectation_id, stream_id, subject_key, window_start, window_end FROM "
                "vision_percept_expectation WHERE status='open' AND window_start <= :t AND window_end > :t"),
                {"t": ts}).fetchall()
            for r in open_now:
                met = _occurred(conn, r.stream_id, r.subject_key, r.window_start, ts) is not None
                per_stream.setdefault(r.stream_id, []).append((r.window_start, r.window_end, met))
                if not met:
                    per_stream_ids.setdefault(r.stream_id, []).append(r.expectation_id)
            for stream_id in {s.stream_id for s in subjects} | set(per_stream):
                conn.execute(text(
                    "INSERT INTO vision_rhythm_cursor (stream_id, last_run_at, updated_at) VALUES (:s, :t, now()) "
                    "ON CONFLICT (stream_id) DO UPDATE SET last_run_at=EXCLUDED.last_run_at, updated_at=now()"),
                    {"s": stream_id, "t": ts})
        summary["expect_keys"] = _set_expect_keys(redis_url, per_stream, per_stream_ids, ts)
    except Exception as exc:
        if not isinstance(exc, MigrationMissing) and is_missing_relation(exc):
            raise MigrationMissing(str(exc)) from exc
        raise
    finally:
        if lock_conn is not None:
            try:
                lock_conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": RHYTHM_LOCK_KEY})
                lock_conn.close()
            except Exception:
                pass
        engine.dispose()
    return summary


def _set_expect_keys(redis_url: Optional[str], per_stream: Dict[str, List[Tuple[datetime, datetime, bool]]],
                     per_stream_ids: Dict[str, List[str]], now: datetime) -> int:
    if not redis_url:
        return 0
    n = 0
    try:
        import redis

        client = redis.Redis.from_url(redis_url, socket_timeout=3, socket_connect_timeout=3)
        try:
            for stream_id, windows in per_stream.items():
                ttl = expect_key_ttl(windows, now)
                if ttl is None:
                    # Everything open on this stream has already arrived:
                    # stop steering attention toward it now, not at the old TTL.
                    client.delete(f"{EXPECT_KEY_PREFIX}{stream_id}")
                    continue
                client.set(f"{EXPECT_KEY_PREFIX}{stream_id}",
                           json.dumps({"expectation_ids": per_stream_ids.get(stream_id, []),
                                       "set_at": now.isoformat()}), ex=ttl)
                n += 1
        finally:
            client.close()
    except Exception as exc:
        logger.warning("vision_expect_key_set_failed error=%s", exc)
    return n
