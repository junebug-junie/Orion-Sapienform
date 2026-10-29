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
from typing import Any, Mapping, NamedTuple, Sequence

from sqlalchemy import create_engine, text

from orion.situational.street_material import (
    REDUCER_EVENT_TYPES_SQL,
    dedupe_street_captions,
    split_caption_sql,
    street_caption_line,
)
from orion.vision.walkway_rollup import LIVING_KINDS, count_range, split_note, window_peak

logger = logging.getLogger(__name__)

_ENGINE = None
_ENGINE_URL: str | None = None

# Matches metacog_trend_reader's bound: this runs inside turn assembly, so a
# slow database must degrade to "no percept" rather than delay a reply.
_QUERY_STATEMENT_TIMEOUT_MS = 1500
_CONNECT_TIMEOUT_SEC = 3


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
            # connect_timeout (2026-10-07): statement_timeout only bounds a
            # query once connected. A blackholed host would otherwise hold
            # every uncached chat turn for the OS TCP timeout (minutes) now
            # that Hub's chat turns run this read too.
            connect_args={
                "options": f"-c statement_timeout={_QUERY_STATEMENT_TIMEOUT_MS}",
                "connect_timeout": _CONNECT_TIMEOUT_SEC,
            },
        )
        _ENGINE_URL = url
    return _ENGINE


# Rows with no stream_id are accepted as room rows only if written before
# this instant -- the day vision_events learned which camera a row is about.
# After it, a NULL stream_id means a producer (e.g. a not-yet-rebuilt
# orion-vision-scribe) dropped the camera, and it could be the walkway: fail
# closed. Override with ORION_VISION_EVENTS_LEGACY_CUTOFF (ISO-8601).
DEFAULT_VISION_EVENTS_LEGACY_CUTOFF = "2026-09-24T00:00:00+00:00"


def vision_events_legacy_cutoff() -> datetime:
    raw = (os.getenv("ORION_VISION_EVENTS_LEGACY_CUTOFF") or "").strip() or DEFAULT_VISION_EVENTS_LEGACY_CUTOFF
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("bad ORION_VISION_EVENTS_LEGACY_CUTOFF=%r; using default", raw)
        ts = datetime.fromisoformat(DEFAULT_VISION_EVENTS_LEGACY_CUTOFF)
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _room_percept_stmt():
    """Newest narrated row from a ROOM camera.

    `vision_events` is shared by every camera: the council narrates the
    walkway (street and patio included) and the walkway reducers write
    `arrived_as_expected` / `expected_absent` / `attention_worthy` rows. The
    room percept must never be one of those, so the read is restricted to the
    configured room streams. Rows written before `stream_id` existed carry
    NULL and are still accepted, but only if older than the legacy cutoff
    (`vision_events_legacy_cutoff`) -- they all came from the room cameras,
    the only ones that existed then. A NULL row after the cutoff is refused.

    Expanding IN (not `= ANY`) so the same statement runs on SQLite in tests.
    """
    from sqlalchemy import DateTime, Text, bindparam

    return (
        text(
            "SELECT narrative, created_at FROM vision_events "
            "WHERE narrative IS NOT NULL AND narrative <> '' "
            "AND (stream_id IN :stream_ids "
            "     OR (stream_id IS NULL AND created_at < :legacy_cutoff)) "
            "ORDER BY created_at DESC LIMIT 1"
        )
        .bindparams(bindparam("stream_ids", expanding=True))
        .columns(narrative=Text, created_at=DateTime(timezone=True))
    )


def fetch_latest_percept(
    *, stream_ids: Sequence[str], legacy_cutoff: datetime | None = None
) -> dict[str, Any] | None:
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
            cutoff = legacy_cutoff or vision_events_legacy_cutoff()
            row = conn.execute(
                _room_percept_stmt(), {"stream_ids": ids, "legacy_cutoff": cutoff}
            ).first()
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


_NOT_A_NAME = frozenset({"", "unknown", "none", "null", "n/a", "unknown_person"})


def presence_fragment(
    state: str | None, since_sec: float | None, subject: str | None = None
) -> str | None:
    """One clause, or None. Never mentions 'absent' -- an empty room is the
    default expectation for most rooms most of the time, and saying so every
    turn would be noise, not care. Only `present`/`recent` are worth a word.

    `since_sec` renders coarse on purpose: a felt-sense duration ("about 3
    hours") is the actual payload here, not a precise timer.

    `subject`, when it is a real name (not "unknown"/"none"), is the enrolled
    person identity_face matched to this camera's presence -- the clause then
    names them and says the name came from a face match, so the model has
    evidence to cite instead of filling "Someone" in from the chat. Callers
    pass it only for a FRESH row (2026-10-09: a turn that asked "can you see
    me?" was handed "Someone has been in view for 7 minutes" while the camera
    had matched Juniper at 0.70 a minute earlier).

    Public (promoted from `orion.situational.context`'s own private copy,
    2026-08-25) so a second caller -- `endogenous_outreach.py`'s presence-
    aware outreach prompt block -- reads the exact same interpretation of a
    `fetch_presence()` row instead of a second, independently-drifting
    formatting of the same fields.
    """
    if state not in ("present", "recent") or since_sec is None or since_sec < 0:
        return None
    duration = coarse_duration(since_sec)
    # Enrolled label from the DB: one line, bounded, and placeholders are not names.
    name = " ".join(str(subject or "").split())[:40]
    if name.lower() in _NOT_A_NAME:
        who, tag = "Someone", ""
    else:
        who, tag = name[:1].upper() + name[1:], " (matched by face)"
    if state == "present":
        return f"{who} has been in view for {duration}{tag}."
    return f"{who} stepped out of view {duration} ago{tag}."


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
# NAMES. Names appear only when Juniper gave them (`vision_individual.label`,
# set through the ask flow); everyone else is "an unfamiliar person". Since
# 2026-10-15 (manual_migration_walkway_privacy_v2.sql, Juniper lifted the
# walkway limits) the patio is fingerprinted like the rest of the walkway, so a
# person on the patio can be an individual and appear in the "who is around"
# line like anyone else. Zones are still found by the flags the writer sets
# from config/vision_zones.yaml, never by a zone's name:
#   * a head-count zone (`presence: true`, today the patio) keeps one
#     `substrate_embodied_presence` row flagged `head_count_zone` (older
#     writers: `no_embed_zone`) and yields a count line ("Someone is on the
#     patio"); identity fields on that row are dropped -- it is a count;
#   * a no-embed zone (`embed: false`, none configured today) never becomes an
#     individual: sightings flagged `zone_no_embed` are excluded here too.
# PEOPLE ARE A RANGE (spec PR #2763 decision 5, same rule as the rollups in
# PR #2781). The matcher splits one person into several individuals, so the
# number of unnamed person individuals is only the HIGH end. The LOW end is
# the most people in one vision window over the same 15 minutes
# (`vision_scene_inventory.counts`), less stationary person individuals
# (props) sighted then, less the named people already listed. The line says
# "at least L, at most H unfamiliar people", "at least N", "up to N" or "an
# unfamiliar person" -- never one bare number of people. Subtracting every
# named person seen in the 15 minutes is conservative: a named person who
# passed at another moment still comes off the busiest window, so the floor
# can read lower than it could ("up to 2" for two strangers together), never
# higher than what was seen. EVERY LIVING KIND (2026-10-29): dogs get the same
# range ("at least 1, at most 6 unfamiliar dogs (the camera splits one dog into
# several)"), with that kind's own window peak, stationary individuals and
# named ones; things (vehicles, bicycles, strollers) keep their count.
# STATIONARY FIXTURES. A box that has not moved for VISION_STATIONARY_MIN_SEC
# (the patio skeleton) is a prop, not a visitor: sightings flagged
# `stationary` are left out of "who is around". The column ships in the same
# migration; until it is applied the read falls back to the pre-flag statement.
# ---------------------------------------------------------------------------

STREET_RECENT_MINUTES = 15
STREET_OUTCOME_LOOKBACK_MINUTES = 60
# Patio presence older than this is not "now".
PATIO_MAX_AGE_SECONDS = 900
_MAX_WHO_PHRASES = 4

_STREET_SIGHTINGS_SELECT = (
    "SELECT s.individual_id, i.kind, i.label, i.distinct_days, "
    "       min(s.started_at) AS first_at, max(s.ended_at) AS last_at "
    "FROM vision_individual_sighting s "
    "JOIN vision_individual i ON i.individual_id = s.individual_id "
    "WHERE s.stream_id = :stream_id "
    "  AND s.ended_at > CAST(:now AS timestamptz) - make_interval(mins => :lookback) "
    "  AND s.started_at <= CAST(:until AS timestamptz) "
    "  AND NOT s.zone_no_embed "
    "{stationary}"
    "GROUP BY s.individual_id, i.kind, i.label, i.distinct_days "
    "ORDER BY max(s.ended_at) DESC LIMIT 50"
)
# Stationary fixtures excluded (manual_migration_walkway_privacy_v2.sql).
_STREET_SIGHTINGS_SQL = text(_STREET_SIGHTINGS_SELECT.format(stationary="  AND NOT s.stationary "))
# Fallback while that migration is unapplied (no `stationary` column yet).
_STREET_SIGHTINGS_PRE_STATIONARY_SQL = text(_STREET_SIGHTINGS_SELECT.format(stationary=""))

# Most people in one vision window (the floor of the people range). Rows only;
# the label mapping ("person", compound labels) is walkway_rollup.window_peak.
_STREET_PEOPLE_WINDOWS_SQL = text(
    "SELECT counts FROM vision_scene_inventory "
    "WHERE stream_id = :stream_id "
    "  AND observed_at > CAST(:now AS timestamptz) - make_interval(mins => :lookback) "
    "  AND observed_at <= CAST(:until AS timestamptz) "
    "  AND counts <> '{}'::jsonb "
    "ORDER BY observed_at DESC LIMIT 1000"
)

# Individuals sighted standing still, by kind (the patio skeleton is a
# "person"): each may be one of a window's detections, so that kind's floor
# drops by that many (as in walkway_rollup.count_range's `props`).
_STREET_STATIONARY_PEOPLE_SQL = text(
    "SELECT i.kind, count(DISTINCT s.individual_id) AS n FROM vision_individual_sighting s "
    "JOIN vision_individual i ON i.individual_id = s.individual_id "
    "WHERE s.stream_id = :stream_id AND s.stationary "
    "  AND s.ended_at > CAST(:now AS timestamptz) - make_interval(mins => :lookback) "
    "  AND s.started_at <= CAST(:until AS timestamptz) "
    "GROUP BY i.kind"
)

_STREET_EXPECTATIONS_SQL = text(
    "SELECT subject_key, subject_label, status, peak_minute, window_start, "
    "       window_end, scored_at "
    "FROM vision_percept_expectation "
    "WHERE stream_id = :stream_id AND ("
    "   (status IN ('met', 'missed') AND scored_at > CAST(:now AS timestamptz) - make_interval(mins => :lookback)"
    "    AND scored_at <= CAST(:until AS timestamptz))"
    "   OR (status = 'open' AND window_start <= CAST(:now AS timestamptz)"
    "       AND window_end > CAST(:now AS timestamptz))"
    ") ORDER BY window_start LIMIT 20"
)

_STREET_UNRESOLVED_SQL = text(
    "SELECT observed_at, description FROM vision_unresolved "
    "WHERE stream_id = :stream_id "
    "  AND observed_at > CAST(:now AS timestamptz) - make_interval(mins => :lookback) "
    "  AND observed_at <= CAST(:until AS timestamptz) "
    "ORDER BY observed_at DESC LIMIT 20"
)

# Latest walkway narratives (council captions), freshness-gated in SQL and
# deduped in Python (street_material.dedupe_street_captions). A different
# statement from `_room_percept_stmt` on purpose: this is the street, rendered
# as "On the walkway just now: ...", never as the room. Single stream and a
# bound `:now` window (`make_interval`), like the other street reads in this
# file, so it is its own statement rather than street_material's expanding one;
# the reducer-row exclusion is shared.
#
# CLOCK. Every street read windows on the bound `:now`, the same `now` the
# Python side uses for its 15-minute cut, never the server's `now()`: a caller
# (an eval, a replay) that passes `now=` gets rows from that moment's window,
# not from whatever the database clock says today. `:until` is `now` plus
# STREET_CLOCK_GRACE_SECONDS: events and sightings from after `now` are not
# part of that moment, and the grace keeps a live read from
# dropping the newest row when a writer's clock runs a little ahead. A
# sighting counts if it STARTED by then (someone mid-sighting is present).
# Only the event window is bounded: a replay still sees CURRENT state for the
# individual (`label`, `distinct_days`), so names given after `now` and
# today's day counts show up, and a sighting's `last_at` can run past `now`.
_STREET_CAPTIONS_SQL = text(
    split_caption_sql(
        "SELECT event_type, narrative, tags, created_at FROM vision_events "
        "WHERE stream_id = :stream_id "
        "  AND narrative IS NOT NULL AND narrative <> '' "
        f"  AND {REDUCER_EVENT_TYPES_SQL} "
        "  AND created_at > CAST(:now AS timestamptz) - make_interval(mins => :lookback) "
        "  AND created_at <= CAST(:until AS timestamptz)",
        order="ORDER BY created_at DESC",
        other_limit="20",
    )
)
STREET_CLOCK_GRACE_SECONDS = 60
STREET_CAPTION_MAX = 2
# Per-caption clip in the brief: two of these plus the prefix stay inside 420
# chars of chat stance's 520-char street clip (the rest is the camera intro),
# leaving room for the rhythm line. A camera caption adds a ~85-char label
# (detector labels capped at 3), and the brief shows at most one.
STREET_BRIEF_CAPTION_CHARS = 100

# One head-count row per counted zone, `<stream>:<zone>` (orion-sql-writer's
# vision_individuals.no_embed_presence_id), recognised by a flag in the
# snapshot rather than by a hardcoded zone name: `head_count_zone` (writers
# from 2026-10-15), else the older `no_embed_zone` (still written on every
# head-count row for compatibility).
_PATIO_PRESENCE_SQL = text(
    "SELECT presence_id, presence_json, updated_at FROM substrate_embodied_presence "
    "WHERE presence_id LIKE :prefix ESCAPE '\\' "
    "  AND (presence_json->>'head_count_zone' = 'true' "
    "       OR presence_json->>'no_embed_zone' = 'true') "
    "ORDER BY presence_id"
)


def _is_head_count_row(snapshot: Any) -> bool:
    return isinstance(snapshot, dict) and (
        snapshot.get("head_count_zone") is True or snapshot.get("no_embed_zone") is True
    )


def _like_prefix(stream_id: str) -> str:
    escaped = stream_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}:%"


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


# Kinds that are someone rather than something: unnamed ones of these come
# before vehicles and other things in the "who is around" line, and each is
# said as a range (the matcher splits a dog into several clusters the way it
# splits a person). Anything not listed (vehicle, bicycle, stroller, ...) is a
# thing and keeps its count. One list, shared with the rollups.
_LIVING_KINDS = LIVING_KINDS


def _recurring_phrase(kind: str, days: list[int]) -> str:
    """Unnamed individuals of one kind each seen on 2+ days. One phrase per
    kind: two parked cars are "2 vehicles", not the same sentence twice. The
    day count is exact when they agree, else the smallest ("at least")."""
    n = len(days)
    if n == 1:
        return f"{_article(kind)} {kind} I have seen on {days[0]} days but have no name for"
    low = min(days)
    span = f"{low} days" if low == max(days) else f"at least {low} days"
    return f"{n} {_plural(kind)} I have each seen on {span} but have no name for"


def _new_phrase(kind: str, n: int) -> str:
    return f"{_article('unfamiliar')} unfamiliar {kind}" if n == 1 else f"{n} unfamiliar {_plural(kind)}"


def unnamed_people_range(
    *, unnamed: int, named: int, peak: int | None, props: int = 0
) -> tuple[int | None, int]:
    """(low, high) for the unnamed people in the who line.

    High: the unnamed person individuals, raised to the low when more stood in
    one window than were matched. Low: `count_range`'s floor over everyone
    (most in one window less props, at least 1 when a person was sighted),
    less the `named` people listed by name; None when the window count could
    not be read (then only "up to" can be said).
    """
    unnamed, named = max(0, int(unnamed)), max(0, int(named))
    if peak is None:
        return None, unnamed
    r = count_range(peak, unnamed + named, props=props)
    low = max(0, int(r["low"]) - named)
    return low, max(unnamed, low)


def _living_phrase(kind: str, low: int | None, high: int, *, recurring: list[int]) -> str:
    """Unnamed individuals of one living kind as a range (decision 5, every
    living kind since 2026-10-29). `recurring`: distinct_days of the
    individuals seen on 2+ days (said as "some", never as a count; one that
    is surely one keeps the day count)."""
    if recurring and high == 1 and low != 0:
        return _recurring_phrase(kind, recurring[:1])
    if recurring:
        one, many = f"{_article(kind)} {kind} I have no name for", f"{_plural(kind)} I have no name for"
    else:
        one, many = f"an unfamiliar {kind}", f"unfamiliar {_plural(kind)}"
    notes: list[str] = []
    if recurring:
        notes.append("seen here on other days" if len(recurring) >= high else "some seen here on other days")
    if low is None or low < 1:
        text = (f"possibly {one}" if low == 0 else one) if high == 1 else f"up to {high} {many}"
        if high > 1:
            notes.append(split_note(kind))
    elif low == high:
        text = one if low == 1 else f"at least {low} {many}"
    else:
        text = f"at least {low}, at most {high} {many}"
        notes.append(split_note(kind))
    return text + (f" ({'; '.join(notes)})" if notes else "")


def _people_phrase(low: int | None, high: int, *, recurring: list[int]) -> str:
    return _living_phrase("person", low, high, recurring=recurring)


def _who_line(
    sightings: list[dict[str, Any]],
    *,
    recent_cut: datetime,
    people_peak: int | None = None,
    people_props: int = 0,
    peaks: Mapping[str, int] | None = None,
    props: Mapping[str, int] | None = None,
) -> str:
    """The "who is around" list, most important first.

    Order: names Juniper gave (most recent first), then unnamed people and
    animals, then vehicles and other things; within each unnamed group a
    kind stays together, the ones seen on several days before the new ones,
    and kinds with recurring members come first. Identical unnamed
    phrases are grouped by kind. At most `_MAX_WHO_PHRASES` phrases are
    shown, so a name is only cut when there are more names than that; "and
    N more" counts the individuals left out, not the phrases ("and others"
    when the people range is among them: it has no single count).

    Unnamed people, and every other living kind (dogs), are one range phrase
    each (`_living_phrase`), never a count of individuals: `peaks` is the most
    of each kind in one window over the same minutes (None: unread), `props`
    the stationary individuals of each kind. `people_peak` / `people_props`
    are the person-only form (used when `peaks` / `props` are not given).
    """
    if peaks is None and people_peak is not None:
        peaks = {"person": people_peak}
    if props is None:
        props = {"person": people_props}
    named: list[str] = []
    seen_names: set[str] = set()
    named_by_kind: dict[str, set[str]] = {}
    # kind -> distinct_days per individual, in first-seen (most recent) order.
    recurring: dict[str, list[int]] = {}
    new: dict[str, int] = {}
    for row in sightings:
        last_at = _utc(row.get("last_at"))
        if last_at is None or last_at < recent_cut:
            continue
        label = str(row.get("label") or "").strip()
        if label:
            # Two individuals Juniper gave the same name are one name here.
            if label.casefold() not in seen_names:
                seen_names.add(label.casefold())
                named.append(label)
            named_by_kind.setdefault(str(row.get("kind") or "").strip(), set()).add(label.casefold())
            continue
        kind = str(row.get("kind") or "").strip() or "something"
        try:
            days = int(row.get("distinct_days") or 0)
        except (TypeError, ValueError):
            days = 0
        if days >= 2:
            recurring.setdefault(kind, []).append(days)
        else:
            new[kind] = new.get(kind, 0) + 1

    # (phrase, how many individuals it stands for; None = a range, no count)
    phrases: list[tuple[str, int | None]] = [(name, 1) for name in named]
    # One kind stays together: its recurring phrase, then its new one.
    kinds = list(recurring) + sorted(k for k in new if k not in recurring)
    for living in (True, False):
        for kind in kinds:
            if (kind in _LIVING_KINDS) is not living:
                continue
            if living:
                rec = recurring.get(kind, [])
                low, high = unnamed_people_range(
                    unnamed=len(rec) + new.get(kind, 0), named=len(named_by_kind.get(kind, ())),
                    peak=None if peaks is None else int(peaks.get(kind, 0) or 0),
                    props=int(props.get(kind, 0) or 0),
                )
                phrases.append((_living_phrase(kind, low, high, recurring=rec), None))
                continue
            if kind in recurring:
                phrases.append((_recurring_phrase(kind, recurring[kind]), len(recurring[kind])))
            if kind in new:
                phrases.append((_new_phrase(kind, new[kind]), new[kind]))
    if not phrases:
        return ""
    shown = [p for p, _ in phrases[:_MAX_WHO_PHRASES]]
    cut = phrases[_MAX_WHO_PHRASES:]
    if any(n is None for _, n in cut):
        return ", ".join(shown) + ", and others"
    extra = sum(n or 0 for _, n in cut)
    if extra > 0:
        return ", ".join(shown) + f", and {extra} more"
    if len(shown) == 1:
        return shown[0]
    return ", ".join(shown[:-1]) + ", and " + shown[-1]


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
    patio: dict[str, Any] | list[dict[str, Any]] | None,
    now: datetime,
    tz: Any,
    camera: str = "walkway",
    captions: list[dict[str, Any]] | None = None,
    people_windows: list[dict[str, Any]] | None = None,
    stationary_people: int | Mapping[str, int] | None = None,
) -> list[str]:
    """Pure: rows in, a few short plain-English lines out.

    `None` for any input means that read failed and contributes nothing --
    never a sentence claiming the street was empty. `patio` is one presence
    snapshot per no-embed zone (a single dict is accepted too); each names its
    own zone under `zone`. `camera` is the stream the rows came from.
    `people_windows` are `vision_scene_inventory` rows (`counts`) from the
    last 15 minutes and `stationary_people` the stationary individuals then,
    by kind (a bare int: persons only); together they are the floor of the
    range of every living kind, people and dogs (None: unread, the line can
    only say "up to").
    """
    lines: list[str] = []
    camera = (camera or "").strip() or "camera"

    # 0. What the street camera's captions said (last 15 minutes). First, so
    # the chat-stance clip keeps it: it is the only line about the scene
    # itself rather than about who was recognised.
    # At most one camera caption (dedupe's default): its label is ~85 chars.
    caption = street_caption_line(
        dedupe_street_captions(
            captions or [], now=now, max_age_sec=STREET_RECENT_MINUTES * 60, limit=STREET_CAPTION_MAX,
            max_chars=STREET_BRIEF_CAPTION_CHARS,
        ),
        camera=camera,
        tz=tz,
    )
    if caption:
        lines.append(caption)

    # 1. Who is around (last 15 minutes).
    living = sorted({str(r.get("kind") or "").strip() for r in sightings or []} & _LIVING_KINDS)
    peaks: dict[str, int] | None = None
    if people_windows is not None:
        counts = [_json_obj(r.get("counts")) for r in people_windows]
        peaks = {k: max((window_peak(c, k) for c in counts), default=0) for k in living}
    if isinstance(stationary_people, Mapping):
        props = {str(k): int(v or 0) for k, v in stationary_people.items()}
    else:
        props = {"person": int(stationary_people or 0)}
    who = _who_line(
        sightings or [],
        recent_cut=now - timedelta(minutes=STREET_RECENT_MINUTES),
        peaks=peaks,
        props=props,
    )
    if who:
        lines.append(f"On the {camera} in the last {STREET_RECENT_MINUTES} minutes: {who}.")

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
        lines.append(f"{camera[:1].upper()}{camera[1:]} rhythm: " + "; ".join(outcomes[:4]) + ".")

    # 3. Things I could not name (last hour).
    names = [r for r in unresolved or [] if str(r.get("description") or "").strip()]
    if names:
        latest = names[0]
        ts = _utc(latest.get("observed_at"))
        when = f" at {ts.astimezone(tz):%H:%M}" if ts else ""
        desc = " ".join(str(latest["description"]).split())[:100]
        if len(names) == 1:
            lines.append(f"In the last hour I saw one thing on the {camera} I could not name{when}: {desc}.")
        else:
            lines.append(
                f"In the last hour I saw {len(names)} things on the {camera} I could not name; "
                f"the latest{when}: {desc}."
            )

    # 4. Head-count zones (the patio): a count line, only when fresh. People
    # on the patio may also appear by name/kind in line 1 via their sightings.
    zones = [patio] if isinstance(patio, dict) else list(patio or [])
    for snap in zones:
        if not isinstance(snap, dict):
            continue
        zone = str(snap.get("zone") or "").strip()
        if not zone:
            continue  # cannot say where; say nothing rather than guess
        age = presence_row_age_seconds(snap)
        if age is None or age > PATIO_MAX_AGE_SECONDS or snap.get("state") != "present":
            continue
        count = snap.get("count")
        try:
            count = int(count) if count is not None else None
        except (TypeError, ValueError):
            count = None
        if count is not None and count >= 2:
            # One frame's head count: a floor (the detector can miss people).
            lines.append(f"At least {count} people are on the {zone}.")
        elif count == 1:
            lines.append(f"Someone is on the {zone}.")
        elif count is None:
            lines.append(f"People are on the {zone}.")
        # count == 0 with state present contradicts itself: say nothing.
    return lines


def _json_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


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
    until = now + timedelta(seconds=STREET_CLOCK_GRACE_SECONDS)
    window = {"stream_id": stream_id, "now": now, "until": until}
    try:
        with engine.connect() as conn:
            # Wide enough to see an individual who arrived at the start of
            # an expectation window, not only the last 15 minutes.
            sightings_params = {**window, "lookback": STREET_OUTCOME_LOOKBACK_MINUTES}
            sightings = _street_rows(conn, _STREET_SIGHTINGS_SQL, sightings_params, "sightings")
            if sightings is None:
                # Most likely the `stationary` column is not there yet (the
                # walkway privacy v2 migration is unapplied): read as before.
                sightings = _street_rows(
                    conn, _STREET_SIGHTINGS_PRE_STATIONARY_SQL, sightings_params, "sightings_pre_stationary"
                )
            expectations = _street_rows(
                conn, _STREET_EXPECTATIONS_SQL,
                {**window, "lookback": STREET_OUTCOME_LOOKBACK_MINUTES},
                "expectations",
            )
            unresolved = _street_rows(
                conn, _STREET_UNRESOLVED_SQL,
                {**window, "lookback": STREET_OUTCOME_LOOKBACK_MINUTES},
                "unresolved",
            )
            patio_rows = _street_rows(
                conn, _PATIO_PRESENCE_SQL, {"prefix": _like_prefix(stream_id)}, "no_embed_zones"
            )
            captions = _street_rows(
                conn, _STREET_CAPTIONS_SQL,
                {**window, "lookback": STREET_RECENT_MINUTES},
                "captions",
            )
            people_windows = _street_rows(
                conn, _STREET_PEOPLE_WINDOWS_SQL,
                {**window, "lookback": STREET_RECENT_MINUTES},
                "people_windows",
            )
            stationary_rows = _street_rows(
                conn, _STREET_STATIONARY_PEOPLE_SQL,
                {**window, "lookback": STREET_RECENT_MINUTES},
                "stationary_people",
            )
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.warning("situation_street_connect_failed err=%s", exc)
        return StreetSummary(stream_id, [], False)

    if all(r is None for r in (sightings, expectations, unresolved, patio_rows, captions)):
        # Every read failed: that is "unread", not a quiet street.
        return StreetSummary(stream_id, [], False)

    patio: list[dict[str, Any]] = []
    prefix = f"{stream_id}:"
    for prow in patio_rows or []:
        try:
            raw = prow.get("presence_json")
            if isinstance(raw, str):
                raw = json.loads(raw)
            if not _is_head_count_row(raw):
                continue
            snap = _presence_row_to_dict(raw, prow.get("updated_at"))
            # The zone's own name from the snapshot, else from the row id.
            pid = str(prow.get("presence_id") or "")
            if not snap.get("zone") and pid.startswith(prefix):
                snap["zone"] = pid[len(prefix):]
            # The sql-writer reducer writes the head count as
            # subject={"count": n}; lift it out before subject is dropped.
            subj = snap.get("subject")
            if snap.get("count") is None and isinstance(subj, dict):
                snap["count"] = subj.get("count")
            # A head-count row is a count: nothing identity-shaped from it
            # survives past this point (names come from sightings, above).
            for key in ("subject", "identity_confirmed", "identity_uncertain", "identity_confidence"):
                snap.pop(key, None)
            patio.append(snap)
        except Exception as exc:  # noqa: BLE001
            logger.info("situation_street_patio_decode_failed err=%s", exc)

    try:
        lines = summarize_street(
            sightings=sightings,
            expectations=expectations,
            unresolved=unresolved,
            patio=patio,
            now=now,
            tz=tz,
            camera=stream_id,
            captions=captions,
            # Without the props read the floor could still hold the patio
            # skeleton, so an unread props count makes the floor unread too.
            people_windows=people_windows if stationary_rows is not None else None,
            stationary_people={str(r.get("kind") or ""): int(r.get("n") or 0) for r in stationary_rows or []},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("situation_street_summarize_failed err=%s", exc)
        return StreetSummary(stream_id, [], True)
    return StreetSummary(stream_id, lines, True)
