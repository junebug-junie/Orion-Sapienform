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
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

from orion.journaler import JournalTriggerV1

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

# Distinct local days on which the walkway saw anyone at all.
WATCH_DAYS_SQL = """
SELECT count(DISTINCT (started_at AT TIME ZONE %(tz)s)::date) AS days
FROM vision_individual_sighting
WHERE stream_id = %(stream_id)s
  AND started_at > now() - make_interval(days => %(days)s)
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
    for row in rows:
        if not isinstance(row, dict) or str(row.get("day_kind")) not in wanted_kinds:
            continue
        key = str(row.get("subject_key") or "").strip()
        label = str(row.get("subject_label") or "").strip()
        peak = _hhmm(row.get("peak_minute"))
        if not key or not label or peak is None:
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
        return {
            **base,
            "enough_days": False,
            "expectations_below_support": below_support,
            "lines": [
                f"Tomorrow on the walkway I do not have enough days yet to expect "
                f"anything. I have seen the street on {days_watched} of the last "
                f"{WATCH_LOOKBACK_DAYS} days, and nothing has come back on at "
                f"least {min_support_days} different days."
            ],
        }

    forecasts: list[dict[str, Any]] = []
    for key, row in latest.items():
        forecasts.append(
            {
                "subject_key": key,
                "subject": str(row["subject_label"]).strip(),
                "around": _hhmm(row.get("peak_minute")),
                "window_from": _local_hhmm(row.get("window_start"), tz),
                "window_to": _local_hhmm(row.get("window_end"), tz),
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
    else:
        summary = f"Walkway grade for {for_date}: what I expected today and what happened."
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

    grade_rows = await fetch_rows(
        dsn, GRADE_SQL,
        {"stream_id": stream_id, "day_start": day_start, "day_end": day_end},
        label="walkway_grade",
    )
    if grade_rows is None:
        # Table missing or unreadable. The forecast reads the same table, so
        # there is nothing more to try tonight.
        return [], ["expectations_unreadable"]
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
        {"stream_id": stream_id, "tz": tz_name, "days": WATCH_LOOKBACK_DAYS},
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
