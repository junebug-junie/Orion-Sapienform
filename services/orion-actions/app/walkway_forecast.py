"""Tonight: grade what I expected on the walkway today, then forecast tomorrow.

Walkway spec idea 7 (docs/superpowers/specs/2026-09-22-walkway-camera-busy-
world-design.md). The rhythm reducer in orion-sql-writer writes
`vision_percept_expectation` rows ("the black dog, weekdays, 07:30-07:55") and
scores each one `met` / `missed` / `unscorable` once its window closes. This
module turns those rows into two journal triggers a night:

  walkway_grade     what was expected today and what happened, in the reverie
                    vocabulary (met -> confirmed, missed -> disconfirmed,
                    unscorable -> unscored), plus a 14-day tally per subject.
  walkway_forecast  what I expect tomorrow and which of those I am least sure
                    about.
  walkway_day       what the street held today, from the deterministic day
                    rollup (vision_rollup, orion/vision/walkway_rollup.py).
  walkway_week      the same for the week, on Sunday night.
  walkway_month     the same for the month, on its last night.

FACTS ONLY FROM ROWS (2026-10-18). On 10-17 the forecast entry said "the
walkway has been empty on at least five occasions" with zero expectation rows
behind it: the LLM turned "nothing has come back on at least 5 different days"
into a claim about the street. Now every number a walkway entry may use is in
its seed (the rollup's own counts for day/week/month), and ``guard_walkway_draft``
replaces any composed entry that states a number the seed does not hold with the
seed's plain template ``lines``. "Days watched" for the forecast counts days the
camera actually delivered frames (scene-inventory minutes), not days with sightings.

WHEN NOTHING IS WRITTEN, AND WHY. The journal has no "empty entry" shape, and a
composed entry about nothing is exactly the empty-shell cognition AGENTS.md
bans. So:

  * table missing / DSN unset / query failed  -> no entry, reason logged.
  * no expectations for today                 -> no grade entry.
  * no sightings at all in the last 14 days   -> no forecast entry. The camera
    is not producing; "I do not have enough days yet" would be a claim about a
    street nobody is watching.
  * sightings exist but no expectation has `min_support_days` of support
    -> a forecast entry that SAYS "I do not have enough days yet", with the
    real count of days watched. That sentence is backed by rows, so it is a
    finding, not filler -- and it is what the spec asks for instead of a guess.

Pure builders below take rows; the async `collect_walkway_triggers` does the
reads. Every read degrades to "nothing" on failure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Awaitable, Callable, Iterable, Sequence
from zoneinfo import ZoneInfo

from orion.journaler import JournalTriggerV1
from orion.vision.walkway_rollup import (
    RANGE_KINDS,
    bare_living_counts,
    display_line,
    person_count_range,
    split_note,
    unsupported_numbers,
)

from .vision_pg import fetch_rows

logger = logging.getLogger("orion-actions.walkway_forecast")

# The reducer's scoring words -> the reverie expectation-judge words
# (orion/schemas/reverie.py `expectation_verdict`), so both prediction loops
# speak one language. `open` is deliberately NOT mapped: an ungraded window is
# not a miss.
VERDICT_BY_STATUS: dict[str, str] = {
    "met": "confirmed",
    "missed": "disconfirmed",
    "unscorable": "unscored",
}

CALIBRATION_LOOKBACK_DAYS = 14
WATCH_LOOKBACK_DAYS = 14
# Expectations older than this are stale fits; the reducer re-emits nightly.
FORECAST_FRESH_DAYS = 14
MAX_LINES = 12

_EXPECTATION_COLUMNS = (
    "expectation_id, subject_key, subject_label, day_kind, window_start, "
    "window_end, peak_minute, support_days, support_sightings, confidence, "
    "status, emitted_at, scored_at"
)

# Latest expectation per subject that applies to tomorrow's kind of day.
FORECAST_SQL = f"""
SELECT DISTINCT ON (subject_key) {_EXPECTATION_COLUMNS}
FROM vision_percept_expectation
WHERE stream_id = %(stream_id)s
  AND day_kind IN (%(day_kind)s, 'any')
  AND emitted_at > now() - make_interval(days => %(fresh_days)s)
ORDER BY subject_key, emitted_at DESC
"""

GRADE_SQL = f"""
SELECT {_EXPECTATION_COLUMNS}
FROM vision_percept_expectation
WHERE stream_id = %(stream_id)s
  AND window_start >= %(day_start)s AND window_start < %(day_end)s
ORDER BY window_start
"""

CALIBRATION_SQL = """
SELECT subject_key, max(subject_label) AS subject_label,
       count(*) FILTER (WHERE status = 'met')        AS met,
       count(*) FILTER (WHERE status = 'missed')     AS missed,
       count(*) FILTER (WHERE status = 'unscorable') AS unscorable
FROM vision_percept_expectation
WHERE stream_id = %(stream_id)s
  AND status IN ('met', 'missed', 'unscorable')
  AND window_start > now() - make_interval(days => %(days)s)
GROUP BY subject_key
ORDER BY subject_key
"""

# Local days in the window on which the camera actually delivered frames for at
# least WATCHED_MIN_UP_MINUTES: minutes with a vision_scene_inventory row (the same
# frame heartbeat the rollups' camera minutes use, read directly so the count does
# not depend on how far the rollups have been backfilled). A day with the camera
# down is not a day "watched", whatever was or was not seen.
WATCHED_MIN_UP_MINUTES = 60
WATCH_DAYS_SQL = """
SELECT count(*) AS days FROM (
  SELECT (observed_at AT TIME ZONE %(tz)s)::date AS d,
         count(DISTINCT date_trunc('minute', observed_at)) AS up_minutes
  FROM vision_scene_inventory
  WHERE stream_id = %(stream_id)s
    AND observed_at > now() - make_interval(days => %(days)s)
  GROUP BY 1
) per_day
WHERE up_minutes >= %(min_up)s
"""

# One rollup row (vision_rollup, written by orion-sql-writer).
ROLLUP_SQL = """
SELECT grain, period_start, complete, covered_until, rollup, line
FROM vision_rollup
WHERE stream_id = %(stream_id)s AND grain = %(grain)s AND period_start = %(period_start)s
"""


def day_kind_for(d: date) -> str:
    return "weekend" if d.weekday() >= 5 else "weekday"


def _hhmm(minute_of_day: Any) -> str | None:
    try:
        m = int(minute_of_day)
    except (TypeError, ValueError):
        return None
    if not 0 <= m < 1440:
        return None
    return f"{m // 60:02d}:{m % 60:02d}"


def _local_hhmm(ts: Any, tz: ZoneInfo) -> str | None:
    if not isinstance(ts, datetime):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(tz).strftime("%H:%M")


def _float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


def build_forecast_seed(
    rows: Iterable[dict[str, Any]],
    *,
    tomorrow: date,
    tz: ZoneInfo,
    min_support_days: int,
    days_watched: int,
) -> dict[str, Any] | None:
    """Rows in, seed out -- or None when there is nothing honest to say."""
    wanted_kinds = {day_kind_for(tomorrow), "any"}
    latest: dict[str, dict[str, Any]] = {}
    below_support = 0
    malformed = 0
    for row in rows:
        if not isinstance(row, dict) or str(row.get("day_kind")) not in wanted_kinds:
            continue
        key = str(row.get("subject_key") or "").strip()
        label = str(row.get("subject_label") or "").strip()
        peak = _hhmm(row.get("peak_minute"))
        if not key or not label or peak is None:
            malformed += 1
            continue
        if _int(row.get("support_days")) < min_support_days:
            below_support += 1
            continue
        prev = latest.get(key)
        emitted = row.get("emitted_at")
        if prev is None or (
            isinstance(emitted, datetime)
            and isinstance(prev.get("emitted_at"), datetime)
            and emitted > prev["emitted_at"]
        ):
            latest[key] = row

    base = {
        "kind": "walkway_forecast",
        "for_date": tomorrow.isoformat(),
        "day_kind": day_kind_for(tomorrow),
        "min_support_days": min_support_days,
        "days_watched_last_14": days_watched,
    }

    if not latest:
        if days_watched <= 0:
            return None
        # "Nothing has come back on N days" is only true when the rows that
        # exist failed on support, or there were none. Rows dropped for being
        # malformed are a producer problem, not a fact about the street.
        if below_support == 0 and malformed > 0:
            return None
        return {
            **base,
            "enough_days": False,
            "expectations_below_support": below_support,
            "lines": [
                f"Tomorrow on the walkway I do not have enough days yet to expect "
                f"anything. My camera watched the street on {days_watched} of the last "
                f"{WATCH_LOOKBACK_DAYS} days, and a prediction needs the same thing "
                f"to come back on at least {min_support_days} different days."
            ],
        }

    forecasts: list[dict[str, Any]] = []
    for key, row in latest.items():
        forecasts.append(
            {
                "subject_key": key,
                "subject": str(row["subject_label"]).strip(),
                # Only the minute-of-day: the row's window is from some past
                # date, so its local clock time can be an hour off across a
                # DST change.
                "around": _hhmm(row.get("peak_minute")),
                "confidence": _float(row.get("confidence")),
                "support_days": _int(row.get("support_days")),
                "expectation_id": str(row.get("expectation_id") or ""),
            }
        )
    forecasts.sort(key=lambda f: f["around"] or "")
    forecasts = forecasts[:MAX_LINES]

    lines = [
        "Tomorrow on the walkway I expect: "
        + "; ".join(f"{f['subject']} around {f['around']}" for f in forecasts)
        + "."
    ]
    least_sure = None
    scored = [f for f in forecasts if f["confidence"] is not None]
    if len(scored) >= 2:
        least = min(scored, key=lambda f: f["confidence"])
        least_sure = least["subject"]
        lines.append(
            f"I am least sure about {least['subject']} "
            f"(confidence {least['confidence']:.2f}, seen on {least['support_days']} days)."
        )
    return {
        **base,
        "enough_days": True,
        "forecasts": forecasts,
        "least_sure": least_sure,
        "expectations_below_support": below_support,
        "lines": lines,
    }


# ---------------------------------------------------------------------------
# Grade
# ---------------------------------------------------------------------------


def build_grade_seed(
    rows: Iterable[dict[str, Any]],
    calibration_rows: Iterable[dict[str, Any]],
    *,
    day: date,
    tz: ZoneInfo,
) -> dict[str, Any] | None:
    """Today's expectations with their verdicts. None when there were none."""
    graded: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("subject_label") or "").strip()
        if not label:
            continue
        item = {
            "subject": label,
            "around": _hhmm(row.get("peak_minute")),
            "window_from": _local_hhmm(row.get("window_start"), tz),
            "window_to": _local_hhmm(row.get("window_end"), tz),
            "expectation_id": str(row.get("expectation_id") or ""),
        }
        verdict = VERDICT_BY_STATUS.get(str(row.get("status") or ""))
        if verdict is None:
            pending.append(item)
        else:
            graded.append({**item, "verdict": verdict})
    if not graded and not pending:
        return None

    lines: list[str] = []
    for word in ("confirmed", "disconfirmed", "unscored"):
        names = [g["subject"] + (f" (around {g['around']})" if g["around"] else "") for g in graded if g["verdict"] == word]
        if names:
            lines.append(f"{word}: " + "; ".join(names))
    if pending:
        lines.append("not scored yet: " + "; ".join(p["subject"] for p in pending))

    calibration: list[dict[str, Any]] = []
    for row in calibration_rows:
        if not isinstance(row, dict):
            continue
        confirmed, disconfirmed = _int(row.get("met")), _int(row.get("missed"))
        unscored = _int(row.get("unscorable"))
        if confirmed + disconfirmed + unscored == 0:
            continue
        decided = confirmed + disconfirmed
        calibration.append(
            {
                "subject": str(row.get("subject_label") or row.get("subject_key") or ""),
                "confirmed": confirmed,
                "disconfirmed": disconfirmed,
                "unscored": unscored,
                # None, not 0, when nothing was decidable: "no evidence" must
                # not read as "always wrong".
                "hit_rate": round(confirmed / decided, 2) if decided else None,
            }
        )

    return {
        "kind": "walkway_grade",
        "for_date": day.isoformat(),
        "graded": graded,
        "not_scored_yet": pending,
        "calibration_last_14_days": calibration,
        "lines": lines,
    }


# ---------------------------------------------------------------------------
# Day / week / month (from the deterministic rollup)
# ---------------------------------------------------------------------------

ROLLUP_KIND_BY_GRAIN = {"day": "walkway_day", "week": "walkway_week", "month": "walkway_month"}
# Kinds whose composed prose is checked against the seed (guard_walkway_draft).
WALKWAY_GUARDED_KINDS = frozenset({"walkway_forecast", "walkway_grade", *ROLLUP_KIND_BY_GRAIN.values()})


def _point_facts(point: Any) -> dict[str, Any] | None:
    """Busiest / quietest child: its passings as an upper bound, people as a range."""
    if not isinstance(point, dict):
        return None
    out: dict[str, Any] = {"at": point.get("at"), "passings_at_most": point.get("passings", 0)}
    if point.get("usual") is not None:
        out["usual_passings_at_most"] = point["usual"]
    people = point.get("people")
    if isinstance(people, dict) and "low" in people and "high" in people:
        out["people"] = {"at_least": people["low"], "at_most": people["high"]}
    peaks = {k: v for k, v in (point.get("peak_at_once") or {}).items() if k not in RANGE_KINDS}
    if peaks:
        out["most_at_once"] = peaks
    return out


def _baseline_facts(base: Any) -> dict[str, Any] | None:
    if not isinstance(base, dict):
        return None
    return {"mean_passings_at_most": base.get("mean_passings"), "periods": base.get("periods")}


def compact_rollup_facts(rollup: dict[str, Any]) -> dict[str, Any]:
    """The rollup without per-individual maps and ids: what the entry may say."""
    from orion.situational.street_material import label_camera_caption

    cam = dict(rollup.get("camera") or {})
    named = {
        name: {k: v for k, v in (entry or {}).items() if k in ("times", "days", "sightings")}
        for name, entry in (rollup.get("named") or {}).items()
    }
    # rules_version 3: a named thing linked to its owner, keyed by the owner's name.
    belongings = {
        name: {k: v for k, v in (entry or {}).items() if k in ("times", "days", "sightings", "kinds")}
        for name, entry in (rollup.get("belongings") or {}).items()
    }
    # People and dogs are a range (2026-10-20, dogs 2026-10-29): at least the most in one frame,
    # at most the clusters. Their bare passings/distinct (cluster counts) never reach the entry.
    kinds: dict[str, Any] = {}
    for k, v in (rollup.get("kinds") or {}).items():
        if k in RANGE_KINDS:
            r = person_count_range(rollup, k)
            if r:
                kinds[k] = {"at_least": r["low"], "at_most": r["high"], "note": split_note(k)}
        else:
            kinds[k] = {"passings": v.get("passings", 0), "distinct": v.get("distinct", 0)}
    reg = rollup.get("regulars") or {}
    facts: dict[str, Any] = {
        "period": rollup.get("label"),
        "local": rollup.get("local"),
        "complete": bool(rollup.get("complete")),
        "covered_until": rollup.get("covered_until"),
        # Passings include the people clusters, so the total is only an upper bound.
        "total": {"passings_at_most": int((rollup.get("total") or {}).get("passings", 0) or 0)},
        "kinds": kinds,
        "excluded_not_counted": rollup.get("excluded"),
        "named": named,
        "belongings": belongings,
        "unnamed_regulars": {"count": reg.get("count", 0), "by_kind": reg.get("by_kind", {})},
        "busiest": _point_facts(rollup.get("busiest")),
        "quietest": _point_facts(rollup.get("quietest")),
        "usual_total": _baseline_facts(rollup.get("baseline")),
        "camera": {k: cam.get(k) for k in ("up_minutes", "expected_minutes", "up_hours", "expected_hours",
                                            "status", "down_spans", "missing_spans")},
        "captions": [label_camera_caption(str(c.get("text") or ""), c.get("detector") or [], at=c.get("at"))
                     for c in (rollup.get("captions") or [])],
    }
    stories = rollup.get("stories") or {}
    if stories.get("gate_passed"):
        facts["stories"] = {k: stories.get(k) for k in ("count", "by_kind", "sentences") if k in stories}
    return facts


def build_rollup_seed(kind: str, row: dict[str, Any]) -> dict[str, Any] | None:
    rollup = row.get("rollup")
    if isinstance(rollup, str):
        rollup = json.loads(rollup)
    if not isinstance(rollup, dict):
        return None
    # A row from before rules_version 2 stored a bare people count: re-rendered as a range.
    line = display_line(row.get("line"), rollup)
    if not line:
        return None
    return {
        "kind": kind,
        "for_date": str((rollup.get("local") or {}).get("date") or ""),
        "period": rollup.get("label"),
        "facts": compact_rollup_facts(rollup),
        "lines": [line],
        "rule": ("Every number in the entry must come from facts; hours the camera was down are not "
                 "'nobody came'. People and dogs are a range: say 'at least N, at most M people' "
                 "(or dogs), never one number of people or dogs (the camera splits one into several)."),
    }


def guard_walkway_draft(draft: Any, trigger: JournalTriggerV1) -> tuple[Any, list[str]]:
    """Phrase, never invent. For walkway kinds: if the composed title/body states a number the
    seed does not hold, the body becomes the seed's template ``lines`` (and an offending title the
    trigger summary). Returns (draft, unsupported numbers found)."""
    if trigger.trigger_kind not in WALKWAY_GUARDED_KINDS or not trigger.prompt_seed:
        return draft, []
    try:
        seed = json.loads(trigger.prompt_seed)
    except ValueError:
        return draft, []
    lines = [str(x) for x in (seed.get("lines") or []) if str(x).strip()]
    # Rollup entries: people and dogs are a range, so "79 people" / "6 dogs" is refused even though
    # the number is a fact (it is the top of the range, not a head count).
    people = bare_living_counts if trigger.trigger_kind in ROLLUP_KIND_BY_GRAIN.values() else (lambda _t: [])
    body_bad = unsupported_numbers(str(draft.body or ""), seed) + people(str(draft.body or ""))
    title_bad = unsupported_numbers(str(draft.title or ""), seed) + people(str(draft.title or ""))
    if not body_bad and not title_bad:
        return draft, []
    update: dict[str, Any] = {}
    if body_bad and lines:
        update["body"] = "\n\n".join(lines)
    if title_bad:
        update["title"] = trigger.summary
    logger.warning("walkway_journal_guard kind=%s replaced=%s unsupported=%s", trigger.trigger_kind,
                   ",".join(sorted(update)) or "none", ",".join((body_bad + title_bad)[:10]))
    return (draft.model_copy(update=update) if update else draft), body_bad + title_bad


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkwayJournalJob:
    trigger: JournalTriggerV1
    audit_action: str
    dedupe_key: str


def build_walkway_trigger(kind: str, seed: dict[str, Any], *, stream_id: str) -> JournalTriggerV1:
    for_date = str(seed.get("for_date") or "")
    if kind == "walkway_forecast":
        summary = f"Walkway forecast for {for_date}: what I expect to see on the street tomorrow."
    elif kind == "walkway_grade":
        summary = f"Walkway grade for {for_date}: what I expected today and what happened."
    else:
        span = {"walkway_day": "today", "walkway_week": "this week", "walkway_month": "this month"}[kind]
        summary = f"Walkway {seed.get('period') or for_date}: what my outdoor camera counted on the street {span}."
    return JournalTriggerV1(
        trigger_kind=kind,  # type: ignore[arg-type]
        source_kind="scheduler",
        source_ref=f"{kind}:{stream_id}:{for_date}",
        summary=summary,
        prompt_seed=json.dumps(seed, sort_keys=True, default=str),
    )


def _dedupe_key(kind: str, stream_id: str, for_date: str, node: str) -> str:
    return f"actions:journal:{kind}:{stream_id}:{for_date}:{node}"


async def collect_walkway_jobs(
    *,
    dsn: str | None,
    stream_id: str,
    tz_name: str,
    now_utc: datetime,
    min_support_days: int,
    node: str,
) -> tuple[list[WalkwayJournalJob], list[str]]:
    """Read, build, and return (jobs to dispatch, skip reasons). Never raises."""
    skips: list[str] = []
    jobs: list[WalkwayJournalJob] = []
    try:
        tz = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        return [], ["bad_timezone"]
    today = now_utc.astimezone(tz).date()
    tomorrow = today + timedelta(days=1)
    day_start = datetime.combine(today, time.min, tzinfo=tz)
    day_end = day_start + timedelta(days=1)

    # Day always; week on Sunday night (ISO weeks end Sunday); month on its last night.
    grains = ["day"]
    if today.weekday() == 6:
        grains.append("week")
    if tomorrow.month != today.month:
        grains.append("month")
    for grain in grains:
        if grain == "day":
            start_local = today
        elif grain == "week":
            start_local = today - timedelta(days=today.weekday())
        else:
            start_local = today.replace(day=1)
        period_start = datetime.combine(start_local, time.min, tzinfo=tz)
        kind = ROLLUP_KIND_BY_GRAIN[grain]
        rows = await fetch_rows(
            dsn, ROLLUP_SQL,
            {"stream_id": stream_id, "grain": grain, "period_start": period_start},
            label=f"walkway_rollup_{grain}",
        )
        if rows is None:
            skips.append("rollup_unreadable")
            continue
        seed = build_rollup_seed(kind, rows[0]) if rows else None
        if seed is None:
            skips.append(f"{kind}_no_rollup")
            continue
        jobs.append(
            WalkwayJournalJob(
                trigger=build_walkway_trigger(kind, seed, stream_id=stream_id),
                audit_action=f"journal.{kind}",
                dedupe_key=_dedupe_key(kind, stream_id, f"{grain}:{period_start.date().isoformat()}", node),
            )
        )

    grade_rows = await fetch_rows(
        dsn, GRADE_SQL,
        {"stream_id": stream_id, "day_start": day_start, "day_end": day_end},
        label="walkway_grade",
    )
    if grade_rows is None:
        # Table missing or unreadable. The forecast reads the same table, so
        # there is nothing more to try tonight (the rollup jobs above stand).
        return jobs, skips + ["expectations_unreadable"]
    if grade_rows:
        calibration = await fetch_rows(
            dsn, CALIBRATION_SQL,
            {"stream_id": stream_id, "days": CALIBRATION_LOOKBACK_DAYS},
            label="walkway_calibration",
        ) or []
        grade = build_grade_seed(grade_rows, calibration, day=today, tz=tz)
        if grade is not None:
            jobs.append(
                WalkwayJournalJob(
                    trigger=build_walkway_trigger("walkway_grade", grade, stream_id=stream_id),
                    audit_action="journal.walkway_grade",
                    dedupe_key=_dedupe_key("walkway_grade", stream_id, today.isoformat(), node),
                )
            )
        else:
            skips.append("grade_no_usable_rows")
    else:
        skips.append("grade_no_expectations_today")

    forecast_rows = await fetch_rows(
        dsn, FORECAST_SQL,
        {"stream_id": stream_id, "day_kind": day_kind_for(tomorrow), "fresh_days": FORECAST_FRESH_DAYS},
        label="walkway_forecast",
    )
    watch = await fetch_rows(
        dsn, WATCH_DAYS_SQL,
        {"stream_id": stream_id, "tz": tz_name, "days": WATCH_LOOKBACK_DAYS, "min_up": WATCHED_MIN_UP_MINUTES},
        label="walkway_watch_days",
    )
    if forecast_rows is None:
        skips.append("forecast_unreadable")
    else:
        days_watched = _int(watch[0].get("days")) if watch else 0
        forecast = build_forecast_seed(
            forecast_rows,
            tomorrow=tomorrow,
            tz=tz,
            min_support_days=min_support_days,
            days_watched=days_watched,
        )
        if forecast is None:
            skips.append("forecast_no_sightings")
        else:
            jobs.append(
                WalkwayJournalJob(
                    trigger=build_walkway_trigger("walkway_forecast", forecast, stream_id=stream_id),
                    audit_action="journal.walkway_forecast",
                    dedupe_key=_dedupe_key("walkway_forecast", stream_id, tomorrow.isoformat(), node),
                )
            )
    return jobs, skips


def jobs_summary(jobs: Sequence[WalkwayJournalJob]) -> str:
    return ",".join(j.trigger.trigger_kind for j in jobs) or "none"


# A failed read (timeout, database blip, or a table that does not exist yet)
# is retried on the next scheduler ticks this many times, then the night is
# given up -- bounded, so a missing migration costs ~3 log lines, not a
# 45-second loop until midnight.
MAX_READ_ATTEMPTS = 3
READ_FAILURE_SKIPS = frozenset({"expectations_unreadable", "forecast_unreadable", "collect_failed",
                                "rollup_unreadable"})


async def run_walkway_tick(
    *,
    collect: Callable[[], Awaitable[tuple[list[WalkwayJournalJob], list[str]]]],
    dispatch: Callable[[WalkwayJournalJob], Awaitable[bool]],
    job_done_on: Callable[[str], str | None],
    mark_job_done: Callable[[str, str], None],
    local_date: str,
    read_attempts: dict[str, int],
) -> tuple[bool, list[str]]:
    """One scheduler tick. Returns (night complete, skip reasons).

    Each job's completion is recorded DURABLY under its own trigger kind
    (`mark_job_done`), so a restart after a partial night (grade written,
    forecast failed) retries only the forecast -- never a second grade.
    A failed dispatch holds the night open for the next tick. Read failures
    hold it open up to MAX_READ_ATTEMPTS, then give up.
    """
    try:
        jobs, skips = await collect()
    except Exception:  # noqa: BLE001
        logger.exception("walkway_journal_collect_failed")
        jobs, skips = [], ["collect_failed"]
    all_ok = True
    for job in jobs:
        kind = job.trigger.trigger_kind
        if job_done_on(kind) == local_date:
            continue
        if await dispatch(job):
            mark_job_done(kind, local_date)
        else:
            all_ok = False
    if READ_FAILURE_SKIPS.intersection(skips):
        read_attempts[local_date] = read_attempts.get(local_date, 0) + 1
        if read_attempts[local_date] < MAX_READ_ATTEMPTS:
            return False, skips
        logger.warning(
            "walkway_journal_giving_up date=%s attempts=%d skips=%s",
            local_date, read_attempts[local_date], ",".join(skips),
        )
    return all_ok, skips
