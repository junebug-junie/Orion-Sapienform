"""Bounded, fail-open read of Orion's most recent reverie/dream
interpretations for the situation brief.

Mirrors `perception_reader.py`'s shape exactly: module-level cached engine,
DSN resolution (own override -> `POSTGRES_URI` -> `DATABASE_URL`), a
per-connection `statement_timeout` GUC, fail-open, never raises to the
caller.

Reads the same `substrate_reverie_thought` table
`services/orion-hub/scripts/reverie_routes.py`'s `/api/reverie/text/recent`
route already renders for Juniper's reverie cockpit -- this is a second,
narrower reader of that same table, not an import of the route handler's own
`_fetch_text_recent`. `orion/` is shared code services import FROM (see e.g.
`reverie_routes.py`'s own `from orion.reverie.visual_storage import ...`);
importing a private, underscore-prefixed function out of a service's route
script the other way round would invert that dependency direction and tie
this module to Hub's FastAPI app/settings module being importable at all.

Read-only. This module never writes `substrate_reverie_thought`; the
substrate reverie pipeline is its only writer.

**Privacy.** Selects `interpretation`, `created_at`, `salience` and nothing
else -- no raw diffusion-prompt internals, no chain linkage, no visual
artifact references. See `ReverieContextV1`'s docstring for the exposed-field
contract.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import NamedTuple

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

_ENGINE = None
_ENGINE_URL: str | None = None

# Matches perception_reader's bound: this runs inside turn assembly, so a
# slow database must degrade to "no reverie" rather than delay a reply.
_QUERY_STATEMENT_TIMEOUT_MS = 1500


def _dsn() -> str:
    return (
        os.getenv("SITUATION_REVERIE_DSN")
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


class ReverieRow(NamedTuple):
    text: str
    observed_at: datetime | None
    salience: float | None


def fetch_recent_reverie_snippets(limit: int) -> list[ReverieRow]:
    """Newest `limit` non-empty reverie interpretations, newest first.

    Returns `[]` on no DSN configured, no rows yet, or any read error --
    the caller cannot distinguish those cases from this return value alone
    by design, same collapse `perception_reader.fetch_latest_percept`
    already makes for its own three equivalent cases (an unpatched/
    unconfigured/erroring reader must degrade to "nothing to show", never
    raise into turn assembly).
    """
    engine = _get_engine()
    if engine is None:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT interpretation, created_at, salience "
                    "FROM substrate_reverie_thought "
                    "WHERE interpretation IS NOT NULL AND interpretation <> '' "
                    "ORDER BY created_at DESC LIMIT :limit"
                ),
                {"limit": max(0, int(limit))},
            ).all()
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.warning("situation_reverie_read_failed err=%s", exc)
        return []

    out: list[ReverieRow] = []
    for row in rows:
        observed_at = row[1]
        if observed_at is not None and observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        out.append(ReverieRow(text=str(row[0]).strip(), observed_at=observed_at, salience=row[2]))
    return out
