"""Bounded, fail-open read of the most recent vision percept for the situation brief.

P4 of `docs/superpowers/specs/2026-08-12-perception-frontier-design.md`. Mirrors
`metacog_trend_reader.py`'s shape -- module-level cached engine, DSN resolution,
per-connection `statement_timeout` GUC, fail-open, never raises to the caller.

Read-only. This module never writes `vision_events`; `orion-vision-scribe` is
its only writer.

**Privacy.** Selects the narrative column and nothing else. `entities`, and any
future identity-bearing column, are deliberately not read -- see
`PerceptionContextV1`'s docstring for the exposed-field contract. A percept is
camera-derived content about a private home, so the cheapest way to keep that
promise is to never load the fields in the first place.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, NamedTuple, Sequence

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

_ENGINE = None
_ENGINE_URL: str | None = None

# Matches metacog_trend_reader's bound: this runs inside turn assembly, so a
# slow database must degrade to "no percept" rather than delay a reply.
_QUERY_STATEMENT_TIMEOUT_MS = 1500


def _dsn() -> str:
    return (
        os.getenv("SITUATION_PERCEPTION_DSN")
        or os.getenv("POSTGRES_URI")
        or os.getenv("DATABASE_URL")
        or ""
    ).strip()


def _get_engine():
    global _ENGINE, _ENGINE_URL
    url = _dsn()
    if not url:
        return None
    if _ENGINE is None or _ENGINE_URL != url:
        _ENGINE = create_engine(
            url,
            pool_pre_ping=True,
            connect_args={"options": f"-c statement_timeout={_QUERY_STATEMENT_TIMEOUT_MS}"},
        )
        _ENGINE_URL = url
    return _ENGINE


def _room_percept_stmt():
    """Newest narrated row from a ROOM camera.

    `vision_events` is shared by every camera: the council narrates the
    walkway (street and patio included) and the walkway reducers write
    `arrived_as_expected` / `expected_absent` / `attention_worthy` rows. The
    room percept must never be one of those, so the read is restricted to the
    configured room streams. Rows written before `stream_id` existed carry
    NULL and are still accepted -- they all came from the room cameras, the
    only ones that existed then.

    Expanding IN (not `= ANY`) so the same statement runs on SQLite in tests.
    """
    from sqlalchemy import DateTime, Text, bindparam

    return (
        text(
            "SELECT narrative, created_at FROM vision_events "
            "WHERE narrative IS NOT NULL AND narrative <> '' "
            "AND (stream_id IS NULL OR stream_id IN :stream_ids) "
            "ORDER BY created_at DESC LIMIT 1"
        )
        .bindparams(bindparam("stream_ids", expanding=True))
        .columns(narrative=Text, created_at=DateTime(timezone=True))
    )


def fetch_latest_percept(*, stream_ids: Sequence[str]) -> dict[str, Any] | None:
    """Return the newest vision percept, or None if there is none / on any error.

    Only rows from `stream_ids` (the room cameras) or legacy rows with no
    stream are considered -- see `_room_percept_stmt`. An empty list reads
    legacy rows only, never every camera.

    Returns ``{"scene_summary": str, "observed_at": datetime}``. The caller owns
    the staleness decision -- this returns the newest row regardless of age, so
    the age gate lives in one place (`situation.py`) rather than being split
    across the reader and the composer.
    """
    engine = _get_engine()
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            # A sentinel keeps the expanding IN valid when no stream is
            # configured: legacy rows only, never "every camera".
            ids = [str(s) for s in stream_ids if str(s).strip()] or ["\x00none"]
            row = conn.execute(_room_percept_stmt(), {"stream_ids": ids}).first()
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.warning("situation_perception_read_failed err=%s", exc)
        return None

    if row is None:
        return None

    observed_at = row[1]
    if observed_at is not None and observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    return {"scene_summary": str(row[0]).strip(), "observed_at": observed_at}


def fetch_presence(stream_id: str, *, engine: Any | None = None) -> dict[str, Any] | None:
    """Return the current embodied-presence snapshot for one stream, or None.

    Reads `substrate_embodied_presence` (`orion-vision-window`'s direct write,
    see `app/presence.py` in that service -- keyed one row per stream_id, JSONB
    blob: `{state, since_sec, last_seen_sec, subject}`).

    Deliberately the SAME fail-open contract as `fetch_latest_percept`: a
    presence read failure must degrade to "no presence enrichment", never to
    an exception that blocks turn assembly. Shares this module's cached
    engine/DSN rather than opening a second connection pool -- unless the
    caller already owns a shared pool of its own and passes it via
    ``engine=`` (review finding, 2026-08-25: orion-hub's endogenous_outreach
    tick already has one, `scripts.pg_engine.get_engine()`, built specifically
    to stop this exact class of duplicate-pool-per-tick; a bare call here
    would have opened a second pool against the identical database for no
    benefit). Passing a caller-owned engine skips this module's own
    statement_timeout GUC (baked into `_get_engine()`'s `connect_args` at
    creation time, not overridable per-call) -- accepted for `fetch_presence`
    specifically: that bound exists to protect live turn assembly from a slow
    query blocking a reply, and an outreach tick isn't blocking a live user
    response the same way. `fetch_latest_percept`, which genuinely IS on that
    live turn-assembly path, keeps its own bounded engine unconditionally.
    """
    engine = engine if engine is not None else _get_engine()
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT presence_json, updated_at FROM substrate_embodied_presence "
                    "WHERE presence_id = :stream_id"
                ),
                {"stream_id": stream_id},
            ).first()
        if row is None or not row[0]:
            return None
        # dict() INSIDE the try (review finding, 2026-08-29): a driver that
        # hands back a JSON string rather than a decoded mapping would raise
        # here, escaping a function whose stated contract is fail-open --
        # orion-hub's _fetch_embodied_presence relies on that contract.
        return dict(row[0])
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.warning("situation_presence_read_failed err=%s", exc)
        return None


def _presence_row_to_dict(presence_json: Any, updated_at: Any) -> dict[str, Any]:
    """Snapshot content plus the row's own write time under `row_updated_at`.

    The write time is NOT decoration: a camera that goes dark stops UPDATING
    this row rather than writing "absent" into it, so the JSONB content alone
    cannot distinguish "someone is present" from "someone was present when
    the webcam was last alive an hour ago". Every freshness judgement about
    presence has to come from this column. Named `row_updated_at` rather than
    `updated_at` so it can never be confused with, or shadowed by, a field
    inside the snapshot blob itself.

    Deliberately NOT applied to `fetch_presence` (review finding, 2026-08-29):
    that dict flows cross-service into orion-hub's
    `OutreachContext.embodied_presence`, and injecting a non-JSON-serializable
    datetime into a payload-shaped dict is a trap for the first caller that
    ever tries to serialize it. Only the resolved path, whose single consumer
    needs the age, carries this key.
    """
    out = dict(presence_json)
    if updated_at is not None and getattr(updated_at, "tzinfo", None) is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    out["row_updated_at"] = updated_at
    return out


def presence_row_age_seconds(presence: dict[str, Any] | None) -> float | None:
    """Seconds since this presence row was last written, or None if unknown."""
    if not presence:
        return None
    updated_at = presence.get("row_updated_at")
    if updated_at is None:
        return None
    try:
        return max(0.0, (datetime.now(timezone.utc) - updated_at).total_seconds())
    except Exception:  # noqa: BLE001 -- fail-open by module contract
        return None


class PresenceResolution(NamedTuple):
    """`read_ok` distinguishes "the database answered and there is nothing
    there" from "the read never happened" (review finding, 2026-08-29).

    Collapsing both into `(None, None)` was a real defect, not a style point:
    the caller treats "no presence" as evidence that Orion cannot see, so a
    Postgres blip -- or simply an unset `SITUATION_PERCEPTION_DSN` -- would
    have made Orion assert out loud that its camera was off. An infrastructure
    fault must never be laundered into a claim about the physical world.
    """

    stream_id: str | None
    presence: dict[str, Any] | None
    read_ok: bool


def fetch_presence_resolved(
    stream_ids: list[str],
    *,
    max_age_seconds: float,
    engine: Any | None = None,
) -> PresenceResolution:
    """Pick the one camera whose presence row should speak for "where is
    Juniper right now", across several streams, in a single query.

    A single hardcoded `perception_stream_id` was the wrong shape and was
    measurably wrong live (2026-08-29): cortex-exec read `cam0`, the interior
    room camera, which had been `absent` for 70 minutes, while `carbon`
    (the laptop webcam Juniper was actually sitting at) read `present` with
    `last_seen_sec=0.0`. The prompt was narrating an empty room at someone
    sitting at their desk.

    Preference order, first match wins:

    1. a FRESH row that says `present` -- someone is at this camera now
    2. a FRESH row that says `recent` -- someone just stepped out of frame
    3. the first configured stream that returned a row at all

    Tier 3 has NO age bound, deliberately -- it exists so a caller can still
    see what the last known state was. Callers must therefore check
    `presence_row_age_seconds` themselves before presenting a tier-3 row as
    current; `_build_perception_context` does exactly that before rendering
    any presence prose (review finding, 2026-08-29: it previously did not,
    and would narrate a frozen row's "in view for 27 minutes" as live).

    "Fresh" is judged from `row_updated_at`, never from the blob (see
    `_presence_row_to_dict`). Ties inside a tier break on the more recently
    written row, so two live cameras resolve deterministically rather than on
    dict ordering. Returns `(None, None)` when nothing is readable -- the same
    fail-open contract as `fetch_presence`, which this does not replace
    (single-stream callers such as endogenous_outreach still use that).
    """
    if not stream_ids:
        # Nothing was asked for, so nothing failed -- but there is also no
        # evidence, so read_ok stays False rather than asserting a clean miss.
        return PresenceResolution(None, None, False)
    engine = engine if engine is not None else _get_engine()
    if engine is None:
        # No DSN configured. Not an outage, but equally not an observation.
        logger.warning("situation_presence_multi_no_engine streams=%s", stream_ids)
        return PresenceResolution(None, None, False)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT presence_id, presence_json, updated_at FROM substrate_embodied_presence "
                    "WHERE presence_id = ANY(:stream_ids)"
                ),
                {"stream_ids": list(stream_ids)},
            ).all()
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.warning("situation_presence_multi_read_failed err=%s", exc)
        return PresenceResolution(None, None, False)

    try:
        found: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not row[1]:
                continue
            found[str(row[0])] = _presence_row_to_dict(row[1], row[2])
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.warning("situation_presence_multi_decode_failed err=%s", exc)
        return PresenceResolution(None, None, False)
    if not found:
        # A real answer: the table has no row for any configured stream.
        return PresenceResolution(None, None, True)

    def _sort_key(item: tuple[str, dict[str, Any]]) -> float:
        age = presence_row_age_seconds(item[1])
        # Unknown age sorts last within its tier rather than first: a row we
        # cannot date is not evidence of recency.
        return age if age is not None else float("inf")

    for wanted in ("present", "recent"):
        tier = [
            (sid, pres)
            for sid, pres in found.items()
            if pres.get("state") == wanted
            and (lambda a: a is not None and a <= max_age_seconds)(presence_row_age_seconds(pres))
        ]
        if tier:
            sid, pres = min(tier, key=_sort_key)
            return PresenceResolution(sid, pres, True)

    for sid in stream_ids:
        if sid in found:
            return PresenceResolution(sid, found[sid], True)
    return PresenceResolution(None, None, True)


def reset_perception_reader_engine_for_tests() -> None:
    global _ENGINE, _ENGINE_URL
    _ENGINE = None
    _ENGINE_URL = None


def presence_fragment(state: str | None, since_sec: float | None) -> str | None:
    """One clause, or None. Never mentions 'absent' -- an empty room is the
    default expectation for most rooms most of the time, and saying so every
    turn would be noise, not care. Only `present`/`recent` are worth a word.

    `since_sec` renders coarse on purpose: a felt-sense duration ("about 3
    hours") is the actual payload here, not a precise timer.

    Public (promoted from `orion.situational.context`'s own private copy,
    2026-08-25) so a second caller -- `endogenous_outreach.py`'s presence-
    aware outreach prompt block -- reads the exact same interpretation of a
    `fetch_presence()` row instead of a second, independently-drifting
    formatting of the same fields.
    """
    if state not in ("present", "recent") or since_sec is None or since_sec < 0:
        return None
    duration = coarse_duration(since_sec)
    if state == "present":
        return f"Someone has been in view for {duration}."
    return f"Someone stepped out of view {duration} ago."


def coarse_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{int(seconds)} seconds"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{int(round(minutes))} minutes"
    hours = minutes / 60.0
    if hours < 1.5:
        return "about an hour"
    return f"about {int(round(hours))} hours"


def percept_age_seconds(observed_at: datetime | None, now: datetime | None = None) -> int | None:
    if observed_at is None:
        return None
    reference = now or datetime.now(timezone.utc)
    return max(0, int((reference - observed_at).total_seconds()))


# ---------------------------------------------------------------------------
# The street (walkway camera spec ideas 5 and 9,
# docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md).
#
# Reads what orion-sql-writer's walkway reducers write -- individuals and their
# sightings, expectations and their grades, percepts nobody could name -- plus
# the patio presence row. Every read is its own statement with its own
# failure: the walkway tables ship as a manual migration and may not exist
# yet, and a missing table must cost that one line, never the rest.
#
# PRIVACY. Names appear only when Juniper gave them (`vision_individual.label`,
# set through the ask flow). The patio is family space: it is read ONLY from
# `substrate_embodied_presence` and yields a count, never a name, and patio
# sightings are excluded from the "who is around" line even if a reducer ever
# wrote one.
# ---------------------------------------------------------------------------

STREET_RECENT_MINUTES = 15
STREET_OUTCOME_LOOKBACK_MINUTES = 60
# Patio presence older than this is not "now".
PATIO_MAX_AGE_SECONDS = 900
_MAX_WHO_PHRASES = 4

_STREET_SIGHTINGS_SQL = text(
    "SELECT s.individual_id, i.kind, i.label, i.distinct_days, "
    "       min(s.started_at) AS first_at, max(s.ended_at) AS last_at "
    "FROM vision_individual_sighting s "
    "JOIN vision_individual i ON i.individual_id = s.individual_id "
    "WHERE s.stream_id = :stream_id "
    "  AND s.ended_at > now() - make_interval(mins => :lookback) "
    "  AND s.zone IS DISTINCT FROM 'patio' "
    "GROUP BY s.individual_id, i.kind, i.label, i.distinct_days "
    "ORDER BY max(s.ended_at) DESC LIMIT 50"
)

_STREET_EXPECTATIONS_SQL = text(
    "SELECT subject_key, subject_label, status, peak_minute, window_start, "
    "       window_end, scored_at "
    "FROM vision_percept_expectation "
    "WHERE stream_id = :stream_id AND ("
    "   (status IN ('met', 'missed') AND scored_at > now() - make_interval(mins => :lookback))"
    "   OR (status = 'open' AND window_start <= now() AND window_end > now())"
    ") ORDER BY window_start LIMIT 20"
)

_STREET_UNRESOLVED_SQL = text(
    "SELECT observed_at, description FROM vision_unresolved "
    "WHERE stream_id = :stream_id "
    "  AND observed_at > now() - make_interval(mins => :lookback) "
    "ORDER BY observed_at DESC LIMIT 20"
)

_PATIO_PRESENCE_SQL = text(
    "SELECT presence_json, updated_at FROM substrate_embodied_presence "
    "WHERE presence_id = :presence_id"
)


class StreetSummary(NamedTuple):
    """`read_ok=False` means nothing could be read at all (no DSN, database
    down) -- distinct from an empty `lines`, which means the street was read
    and was quiet. Same contract as `PresenceResolution`."""

    stream_id: str
    lines: list[str]
    read_ok: bool


def _article(noun: str) -> str:
    return "an" if noun[:1].lower() in "aeiou" else "a"


def _plural(kind: str) -> str:
    return {"person": "people"}.get(kind, kind + "s")


def _who_phrase(kind: str, label: str | None, distinct_days: int) -> str:
    if label:
        return label
    kind = (kind or "something").strip() or "something"
    if distinct_days >= 2:
        return f"{_article(kind)} {kind} I have seen on {distinct_days} days but have no name for"
    return f"{_article('unfamiliar')} unfamiliar {kind}"


def _hhmm(minute_of_day: Any) -> str | None:
    try:
        m = int(minute_of_day)
    except (TypeError, ValueError):
        return None
    return f"{m // 60:02d}:{m % 60:02d}" if 0 <= m < 1440 else None


def _utc(ts: Any) -> datetime | None:
    if not isinstance(ts, datetime):
        return None
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


def _past_peak(peak_minute: Any, window_start: datetime | None, *, now: datetime, tz: Any) -> bool:
    """Is `now` past the expected peak? Peak is minute-of-day local; anchor
    it on the window's own local date, rolled forward a day when it falls
    before the window start (a window that crosses midnight)."""
    try:
        minute = int(peak_minute)
    except (TypeError, ValueError):
        return False
    if window_start is None or not 0 <= minute < 1440:
        return False
    local_start = window_start.astimezone(tz)
    peak = local_start.replace(hour=minute // 60, minute=minute % 60, second=0, microsecond=0)
    if peak < local_start:
        peak += timedelta(days=1)
    return now > peak


def summarize_street(
    *,
    sightings: list[dict[str, Any]] | None,
    expectations: list[dict[str, Any]] | None,
    unresolved: list[dict[str, Any]] | None,
    patio: dict[str, Any] | None,
    now: datetime,
    tz: Any,
) -> list[str]:
    """Pure: rows in, at most four short plain-English lines out.

    `None` for any input means that read failed and contributes nothing --
    never a sentence claiming the street was empty.
    """
    lines: list[str] = []

    # 1. Who is around (last 15 minutes).
    recent_cut = now - timedelta(minutes=STREET_RECENT_MINUTES)
    who: list[str] = []
    unnamed_new: dict[str, int] = {}
    for row in sightings or []:
        last_at = _utc(row.get("last_at"))
        if last_at is None or last_at < recent_cut:
            continue
        label = str(row.get("label") or "").strip() or None
        kind = str(row.get("kind") or "").strip()
        days = int(row.get("distinct_days") or 0)
        if not label and days < 2:
            unnamed_new[kind or "something"] = unnamed_new.get(kind or "something", 0) + 1
            continue
        who.append(_who_phrase(kind, label, days))
    for kind, n in sorted(unnamed_new.items()):
        who.append(_who_phrase(kind, None, 0) if n == 1 else f"{n} unfamiliar {_plural(kind)}")
    if who:
        shown = who[:_MAX_WHO_PHRASES]
        extra = len(who) - len(shown)
        text_ = ", ".join(shown) + (f", and {extra} more" if extra > 0 else "")
        lines.append(f"On the walkway in the last {STREET_RECENT_MINUTES} minutes: {text_}.")

    # 2. What was expected, and did it happen.
    outcomes: list[str] = []
    for row in expectations or []:
        label = str(row.get("subject_label") or "").strip()
        if not label:
            continue
        status = str(row.get("status") or "")
        peak = _hhmm(row.get("peak_minute"))
        at = f" around {peak}" if peak else ""
        if status == "met":
            outcomes.append(f"{label} came as expected{at}")
        elif status == "missed":
            outcomes.append(f"{label} did not come (usually{at})")
        elif status == "open":
            key = str(row.get("subject_key") or "")
            kind, _, ref = key.partition(":")
            window_start = _utc(row.get("window_start"))
            here = False
            if kind == "individual":
                here = any(
                    str(r.get("individual_id")) == ref
                    and (_utc(r.get("last_at")) or now) >= (window_start or now)
                    for r in sightings or []
                )
            elif kind == "label":
                here = any(
                    str(r.get("kind")) == ref and (_utc(r.get("last_at")) or now) >= (window_start or now)
                    for r in sightings or []
                )
            past_peak = _past_peak(row.get("peak_minute"), window_start, now=now, tz=tz)
            # The sightings read only looks back STREET_OUTCOME_LOOKBACK_MINUTES;
            # a window that opened earlier could have been met before that, so
            # it cannot support an absence claim.
            if window_start is None or window_start < now - timedelta(minutes=STREET_OUTCOME_LOOKBACK_MINUTES):
                past_peak = False
            # "Usually here by now" is an absence claim, so it needs a
            # sightings read that answered AND an individual subject: label
            # subjects can be fed by scene counts this reader does not see.
            if here:
                outcomes.append(f"{label} is here, as expected")
            elif past_peak and sightings is not None and kind == "individual":
                outcomes.append(f"{label} is usually here by now")
            else:
                outcomes.append(f"{label} usually comes{at}")
    if outcomes:
        lines.append("Walkway rhythm: " + "; ".join(outcomes[:4]) + ".")

    # 3. Things I could not name (last hour).
    names = [r for r in unresolved or [] if str(r.get("description") or "").strip()]
    if names:
        latest = names[0]
        ts = _utc(latest.get("observed_at"))
        when = f" at {ts.astimezone(tz):%H:%M}" if ts else ""
        desc = " ".join(str(latest["description"]).split())[:100]
        if len(names) == 1:
            lines.append(f"In the last hour I saw one thing on the walkway I could not name{when}: {desc}.")
        else:
            lines.append(
                f"In the last hour I saw {len(names)} things on the walkway I could not name; "
                f"the latest{when}: {desc}."
            )

    # 4. The patio: counts only, never names, only when fresh.
    if patio:
        age = presence_row_age_seconds(patio)
        if age is not None and age <= PATIO_MAX_AGE_SECONDS and patio.get("state") == "present":
            count = patio.get("count")
            try:
                count = int(count) if count is not None else None
            except (TypeError, ValueError):
                count = None
            if count is not None and count >= 2:
                lines.append(f"{count} people are on the patio.")
            elif count == 1:
                lines.append("Someone is on the patio.")
            elif count is None:
                lines.append("People are on the patio.")
            # count == 0 with state present contradicts itself: say nothing.
    return lines


def _street_rows(conn: Any, stmt: Any, params: dict[str, Any], label: str) -> list[dict[str, Any]] | None:
    """One statement, one failure. Rolls back so the next read can proceed on
    the same connection after, say, an UndefinedTable."""
    try:
        return [dict(r._mapping) for r in conn.execute(stmt, params).all()]
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.info("situation_street_read_failed part=%s err=%s", label, type(exc).__name__)
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return None


def fetch_street_summary(
    stream_id: str,
    *,
    tz_name: str = "America/Denver",
    engine: Any | None = None,
    now: datetime | None = None,
) -> StreetSummary:
    """Plain-English lines about the street outside one walkway camera.

    Never raises. `read_ok=False` when no read could happen at all; an empty
    `lines` with `read_ok=True` means the tables answered and there was
    nothing worth saying (or the tables do not exist yet).
    """
    from zoneinfo import ZoneInfo

    stream_id = (stream_id or "").strip()
    if not stream_id:
        return StreetSummary("", [], False)
    engine = engine if engine is not None else _get_engine()
    if engine is None:
        return StreetSummary(stream_id, [], False)
    try:
        tz = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        tz = timezone.utc
    now = now or datetime.now(timezone.utc)
    try:
        with engine.connect() as conn:
            sightings = _street_rows(
                conn, _STREET_SIGHTINGS_SQL,
                # Wide enough to see an individual who arrived at the start of
                # an expectation window, not only the last 15 minutes.
                {"stream_id": stream_id, "lookback": STREET_OUTCOME_LOOKBACK_MINUTES},
                "sightings",
            )
            expectations = _street_rows(
                conn, _STREET_EXPECTATIONS_SQL,
                {"stream_id": stream_id, "lookback": STREET_OUTCOME_LOOKBACK_MINUTES},
                "expectations",
            )
            unresolved = _street_rows(
                conn, _STREET_UNRESOLVED_SQL,
                {"stream_id": stream_id, "lookback": STREET_OUTCOME_LOOKBACK_MINUTES},
                "unresolved",
            )
            patio_rows = _street_rows(
                conn, _PATIO_PRESENCE_SQL, {"presence_id": f"{stream_id}:patio"}, "patio"
            )
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.warning("situation_street_connect_failed err=%s", exc)
        return StreetSummary(stream_id, [], False)

    if all(r is None for r in (sightings, expectations, unresolved, patio_rows)):
        # Every read failed: that is "unread", not a quiet street.
        return StreetSummary(stream_id, [], False)

    patio = None
    if patio_rows:
        try:
            raw = patio_rows[0].get("presence_json")
            if isinstance(raw, str):
                raw = json.loads(raw)
            if isinstance(raw, dict):
                patio = _presence_row_to_dict(raw, patio_rows[0].get("updated_at"))
                # The sql-writer patio reducer writes the head count as
                # subject={"count": n}; lift it out before subject is dropped.
                subj = patio.get("subject")
                if patio.get("count") is None and isinstance(subj, dict):
                    patio["count"] = subj.get("count")
                # Belt and braces for the privacy rule: nothing identity-shaped
                # from the patio row survives past this point.
                for key in ("subject", "identity_confirmed", "identity_uncertain", "identity_confidence"):
                    patio.pop(key, None)
        except Exception as exc:  # noqa: BLE001
            logger.info("situation_street_patio_decode_failed err=%s", exc)
            patio = None

    try:
        lines = summarize_street(
            sightings=sightings,
            expectations=expectations,
            unresolved=unresolved,
            patio=patio,
            now=now,
            tz=tz,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("situation_street_summarize_failed err=%s", exc)
        return StreetSummary(stream_id, [], True)
    return StreetSummary(stream_id, lines, True)
