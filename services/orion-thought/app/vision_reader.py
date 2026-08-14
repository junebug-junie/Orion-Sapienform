"""Direct read of recent vision events from Postgres, for reverie perception context.

orion-thought is a thin bus service (see `test_reverie_thin_import_boundary.py`) --
this deliberately does NOT import `orion.substrate` (drags the graph engine /
`requests`) and does NOT import the vision service's own models. It queries the
`vision_events` table (`services/orion-sql-writer/app/models/vision_event.py`)
directly with sqlalchemy + a psycopg driver.

This mirrors `services/orion-cortex-exec/app/perception_reader.py` (P4 of
`docs/superpowers/specs/2026-08-12-perception-frontier-design.md`, the situation-
brief's percept read) as closely as the service boundary allows -- same table,
same statement_timeout GUC, same narrative-only privacy contract. It is a
separate module rather than a shared import because orion-thought and
orion-cortex-exec are separate services/containers with no shared `app/`
package (CLAUDE.md §5: cross-service seams are `orion/`-package or documented
API/bus contracts, not reaching into another service's `app/` internals).
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
from datetime import datetime, timezone
from typing import Any

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


def read_recent_vision_events(
    *,
    max_age_sec: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Most recent fresh, narrated vision events, newest first. Never raises.

    Each entry: {narrative: str, age_sec: float | None}. The empty/whitespace-
    narrative and staleness filters both apply before LIMIT -- a caption-less
    or aged-out row must not crowd out an older, real, still-fresh one.
    """
    if limit <= 0:
        return []
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT narrative, created_at FROM vision_events "
                        "WHERE narrative IS NOT NULL AND narrative <> '' "
                        "AND created_at >= now() - make_interval(secs => :max_age_sec) "
                        "ORDER BY created_at DESC "
                        "LIMIT :limit"
                    ),
                    {"limit": limit, "max_age_sec": max_age_sec},
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


def reset_vision_reader_engine_for_tests() -> None:
    global _engine, _engine_url
    _engine = None
    _engine_url = None
