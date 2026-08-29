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

import logging
import os
from datetime import datetime, timezone
from typing import Any

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


def fetch_latest_percept() -> dict[str, Any] | None:
    """Return the newest vision percept, or None if there is none / on any error.

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
            row = conn.execute(
                text(
                    "SELECT narrative, created_at FROM vision_events "
                    "WHERE narrative IS NOT NULL AND narrative <> '' "
                    "ORDER BY created_at DESC LIMIT 1"
                )
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
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.warning("situation_presence_read_failed err=%s", exc)
        return None

    if row is None or not row[0]:
        return None
    return _presence_row_to_dict(row[0], row[1])


def _presence_row_to_dict(presence_json: Any, updated_at: Any) -> dict[str, Any]:
    """Snapshot content plus the row's own write time under `row_updated_at`.

    The write time is NOT decoration: a camera that goes dark stops UPDATING
    this row rather than writing "absent" into it, so the JSONB content alone
    cannot distinguish "someone is present" from "someone was present when
    the webcam was last alive an hour ago". Every freshness judgement about
    presence has to come from this column. Named `row_updated_at` rather than
    `updated_at` so it can never be confused with, or shadowed by, a field
    inside the snapshot blob itself.
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


def fetch_presence_resolved(
    stream_ids: list[str],
    *,
    max_age_seconds: float,
    engine: Any | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
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

    "Fresh" is judged from `row_updated_at`, never from the blob (see
    `_presence_row_to_dict`). Ties inside a tier break on the more recently
    written row, so two live cameras resolve deterministically rather than on
    dict ordering. Returns `(None, None)` when nothing is readable -- the same
    fail-open contract as `fetch_presence`, which this does not replace
    (single-stream callers such as endogenous_outreach still use that).
    """
    if not stream_ids:
        return (None, None)
    engine = engine if engine is not None else _get_engine()
    if engine is None:
        return (None, None)
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
        return (None, None)

    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row[1]:
            continue
        found[str(row[0])] = _presence_row_to_dict(row[1], row[2])
    if not found:
        return (None, None)

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
            return (sid, pres)

    for sid in stream_ids:
        if sid in found:
            return (sid, found[sid])
    return (None, None)


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
