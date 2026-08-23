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


def fetch_presence(stream_id: str) -> dict[str, Any] | None:
    """Return the current embodied-presence snapshot for one stream, or None.

    Reads `substrate_embodied_presence` (`orion-vision-window`'s direct write,
    see `app/presence.py` in that service -- keyed one row per stream_id, JSONB
    blob: `{state, since_sec, last_seen_sec, subject}`).

    Deliberately the SAME fail-open contract as `fetch_latest_percept`: a
    presence read failure must degrade to "no presence enrichment", never to
    an exception that blocks turn assembly. Shares this module's cached
    engine/DSN rather than opening a second connection pool.
    """
    engine = _get_engine()
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
    return dict(row[0])


def reset_perception_reader_engine_for_tests() -> None:
    global _ENGINE, _ENGINE_URL
    _ENGINE = None
    _ENGINE_URL = None


def percept_age_seconds(observed_at: datetime | None, now: datetime | None = None) -> int | None:
    if observed_at is None:
        return None
    reference = now or datetime.now(timezone.utc)
    return max(0, int((reference - observed_at).total_seconds()))
