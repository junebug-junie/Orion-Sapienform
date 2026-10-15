"""Direct read of recent vision events from Postgres, for reverie perception context.

orion-thought is a thin bus service (see `test_reverie_thin_import_boundary.py`) --
this deliberately does NOT import `orion.substrate` (drags the graph engine /
`requests`) and does NOT import the vision service's own models. It queries the
`vision_events` table (`services/orion-sql-writer/app/models/vision_event.py`)
directly with sqlalchemy + a psycopg driver.

This mirrors `orion/situational/perception_reader.py` (P4 of
`docs/superpowers/specs/2026-08-12-perception-frontier-design.md`, the situation-
brief's percept read -- relocated 2026-08-22 from cortex-exec's private `app/`
into that shared `orion/` package so orion-hub could reuse it too) as closely
as the service boundary allows -- same table, same statement_timeout GUC, same
narrative-only privacy contract. It remains a separate module here rather than
importing that shared one directly: this file predates the relocation, and
orion-thought's own thin-import-boundary test (`test_reverie_thin_import_boundary.py`)
is deliberately narrow about what this service pulls in. Worth revisiting as a
follow-up, not assumed safe to merge in this comment alone.
Deliberately NOT merged with this module's own `broadcast_reader.py` sibling's
connection pool either: `perception_reader.py`'s statement_timeout only takes
effect on first-use engine construction for a given DSN, so sharing one lazily-
built engine across readers with different timeout requirements would silently
let whichever reader runs first decide the GUC for both -- a subtler bug than
the two-small-pools cost it would save.

**Privacy.** Selects the narrative column only -- same contract as
`PerceptionContextV1` (`orion/schemas/situation.py`): "the exposed-field list is
the privacy contract, and it is short on purpose." `entities`, `tags`, and any
other column are deliberately not read here, not because this consumer trusts
them less than the situation brief does, but because there is no case yet for
the LLM prompt to need more than the narrative sentence.

Fail-open: any failure (missing table, unavailable DB, no rows, slow query past
the statement timeout) returns an empty list so a reverie tick degrades to "no
percepts" -- never raises, never blocks narration.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

logger = logging.getLogger("orion-thought.vision_reader")

_engine = None
_engine_url: str | None = None

# Matches perception_reader.py's bound: a slow database must degrade to "no
# percepts" rather than stall the reverie tick's shared event loop.
_QUERY_STATEMENT_TIMEOUT_MS = 1500


def _database_url() -> str:
    return (
        os.getenv("POSTGRES_URI", "").strip()
        or "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
    )


def _get_engine():
    global _engine, _engine_url
    url = _database_url()
    if _engine is None or _engine_url != url:
        from sqlalchemy import create_engine

        _engine = create_engine(
            url,
            pool_pre_ping=True,
            connect_args={"options": f"-c statement_timeout={_QUERY_STATEMENT_TIMEOUT_MS}"},
        )
        _engine_url = url
    return _engine


DEFAULT_VISION_EVENTS_LEGACY_CUTOFF = "2026-09-24T00:00:00+00:00"


def _legacy_cutoff() -> datetime:
    """Same rule and same env key as orion/situational/perception_reader.py."""
    raw = (os.getenv("ORION_VISION_EVENTS_LEGACY_CUTOFF") or "").strip() or DEFAULT_VISION_EVENTS_LEGACY_CUTOFF
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("bad ORION_VISION_EVENTS_LEGACY_CUTOFF=%r; using default", raw)
        ts = datetime.fromisoformat(DEFAULT_VISION_EVENTS_LEGACY_CUTOFF)
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def read_recent_vision_events(
    *,
    max_age_sec: float,
    limit: int,
    stream_ids: Sequence[str],
    legacy_cutoff: datetime | None = None,
) -> list[dict[str, Any]]:
    """Most recent fresh, narrated vision events, newest first. Never raises.

    Each entry: {narrative: str, age_sec: float | None}. The empty/whitespace-
    narrative and staleness filters both apply before LIMIT -- a caption-less
    or aged-out row must not crowd out an older, real, still-fresh one.

    Only rows from ``stream_ids`` (the ROOM cameras) or legacy rows with a
    NULL ``stream_id`` written before ``legacy_cutoff`` (default: setting
    ``ORION_VISION_EVENTS_LEGACY_CUTOFF``, 2026-09-24T00:00:00Z) are read -- a
    NULL row after that means a producer dropped the camera, so it is refused. ``vision_events`` also carries the walkway
    camera's narratives (street and patio) and its reducers' rows; reverie is
    about the room. An empty ``stream_ids`` reads legacy rows only, never
    every camera. Same rule as ``orion/situational/perception_reader.py``.
    """
    if limit <= 0:
        return []
    ids = [str(s).strip() for s in stream_ids if str(s).strip()] or ["\x00none"]
    try:
        from sqlalchemy import bindparam, text

        engine = _get_engine()
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=float(max_age_sec))
        stmt = text(
            "SELECT narrative, created_at FROM vision_events "
            "WHERE narrative IS NOT NULL AND narrative <> '' "
            "AND created_at >= :cutoff "
            "AND (stream_id IN :stream_ids "
            "     OR (stream_id IS NULL AND created_at < :legacy_cutoff)) "
            "ORDER BY created_at DESC "
            "LIMIT :limit"
        ).bindparams(bindparam("stream_ids", expanding=True))
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    stmt,
                    {"limit": limit, "cutoff": cutoff, "stream_ids": ids,
                     "legacy_cutoff": legacy_cutoff or _legacy_cutoff()},
                )
                .mappings()
                .all()
            )
    except Exception as exc:
        logger.warning("vision events read failed: %s", exc)
        return []

    now = datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    for row in rows:
        created_at = row.get("created_at")
        age_sec: float | None = None
        if isinstance(created_at, datetime):
            ts = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
            age_sec = (now - ts).total_seconds()
        out.append({"narrative": str(row["narrative"]).strip(), "age_sec": age_sec})
    return out


# --- The street (walkway camera), 2026-10-15 ---------------------------------
#
# A SEPARATE read from `read_recent_vision_events` above, never merged into it:
# that one is the room; these are "outside, on the walkway". Reverie renders
# them under their own heading, and the room-only expectation judge never sees
# them. Same narrative-only privacy contract. Shared rules (what is a caption,
# dedupe, how a graded prediction is worded) live in
# orion/situational/street_material.py, which is stdlib-only.

# Rows fetched before Python-side dedupe: the council narrates the walkway
# every ~30-90 s and often repeats itself, so read a few more than shown.
_STREET_FETCH_ROWS = 20


def read_recent_street_events(
    *,
    max_age_sec: float,
    limit: int,
    stream_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Newest fresh, distinct walkway captions. Never raises; [] on any failure
    or when ``stream_ids`` is empty (switched off -- never "every camera").

    Each entry: {narrative, age_sec}.
    """
    ids = [str(s).strip() for s in stream_ids if str(s).strip()]
    if limit <= 0 or not ids:
        return []
    from orion.situational.street_material import STREET_CAPTIONS_SQL, dedupe_street_captions

    now = datetime.now(timezone.utc)
    try:
        from sqlalchemy import bindparam, text

        stmt = text(STREET_CAPTIONS_SQL).bindparams(bindparam("stream_ids", expanding=True))
        with _get_engine().connect() as conn:
            rows = conn.execute(
                stmt,
                {"stream_ids": ids, "since": now - timedelta(seconds=float(max_age_sec)),
                 "limit": _STREET_FETCH_ROWS},
            ).mappings().all()
    except Exception as exc:
        logger.warning("street events read failed: %s", exc)
        return []
    kept = dedupe_street_captions([dict(r) for r in rows], now=now, max_age_sec=max_age_sec, limit=limit)
    return [{"narrative": c["narrative"], "age_sec": c["age_sec"]} for c in kept]


def read_recent_street_graded(
    *,
    stream_ids: Sequence[str],
    lookback_hours: float,
    limit: int,
    tz_name: str,
) -> list[str]:
    """Recent met/missed walkway expectations as plain lines, each with the
    subject's 14-day hit rate when there is one. Never raises; [] on failure,
    when nothing was graded, or when ``stream_ids`` is empty.

    The rhythm reducer only emits expectations with >= 5 days of support, so
    this is empty for the walkway's first week -- by design, not a fault.
    """
    ids = [str(s).strip() for s in stream_ids if str(s).strip()]
    if limit <= 0 or not ids:
        return []
    from zoneinfo import ZoneInfo

    from orion.situational.street_material import (
        CALIBRATION_LOOKBACK_DAYS,
        CALIBRATION_SQL,
        GRADED_EXPECTATIONS_SQL,
        graded_expectation_lines,
    )

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    now = datetime.now(timezone.utc)
    try:
        from sqlalchemy import bindparam, text

        graded_stmt = text(GRADED_EXPECTATIONS_SQL).bindparams(bindparam("stream_ids", expanding=True))
        cal_stmt = text(CALIBRATION_SQL).bindparams(bindparam("stream_ids", expanding=True))
        with _get_engine().connect() as conn:
            rows = conn.execute(
                graded_stmt,
                {"stream_ids": ids, "since": now - timedelta(hours=float(lookback_hours)),
                 "until": now, "limit": limit},
            ).mappings().all()
            if not rows:
                return []
            cal = conn.execute(
                cal_stmt,
                {"stream_ids": ids, "cal_since": now - timedelta(days=CALIBRATION_LOOKBACK_DAYS)},
            ).mappings().all()
    except Exception as exc:
        logger.warning("street graded read failed: %s", exc)
        return []
    return graded_expectation_lines([dict(r) for r in rows], [dict(r) for r in cal], tz=tz, limit=limit)


def reset_vision_reader_engine_for_tests() -> None:
    global _engine, _engine_url
    _engine = None
    _engine_url = None
