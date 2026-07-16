"""Read-only lookup of human Resolve/Dismiss verdicts for the rung-3 workspace
competition (``attention_loop_outcome``, written by a human action in the Hub
via ``services/orion-hub/scripts/attention_loops_store.py``).

Same table, same connection/query style as the two existing readers of this
table (``attention_loops_store.py`` itself and
``services/orion-thought/app/store.py::load_recent_loop_outcomes``) -- direct
SQLAlchemy engine, ``POSTGRES_URI`` env with the same conjourney default,
``DISTINCT ON (loop_id) ... ORDER BY created_at DESC`` for "most recent verdict
per loop". This module only needs the verdict itself (not note/features), and
is bounded to the loop_ids actually competing in a given tick -- never scans
the whole table.

Read-only, fail-open: any error (missing table, connection failure, bad env)
returns an empty set so a DB hiccup never blocks a broadcast tick. Callers
must treat the result as "loops known to be closed" and proceed as if no
verdicts exist otherwise.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

logger = logging.getLogger("orion.substrate.attention.verdicts")

# Verdicts that mean a human explicitly closed the loop; a third valid verdict,
# "decayed_unattended" (see attention_loops_store.py's _VALID_VERDICTS), is an
# implicit non-engagement signal, not an explicit closure -- left eligible to
# compete.
TERMINAL_VERDICTS = {"resolved", "dismissed"}


def _database_url() -> str:
    return (
        os.getenv("POSTGRES_URI", "").strip()
        or "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
    )


_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from sqlalchemy import create_engine

        _ENGINE = create_engine(_database_url(), pool_pre_ping=True)
    return _ENGINE


def load_terminal_verdict_loop_ids(loop_ids: Iterable[str]) -> set[str]:
    """Return the subset of ``loop_ids`` whose most recent verdict is terminal.

    Bounded to the given loop_ids (the loops actually competing this tick).
    Best-effort: returns an empty set on any failure, including an empty/None
    input, so a lookup failure never blocks frame-building.
    """
    ids = sorted({str(i) for i in (loop_ids or []) if i})
    if not ids:
        return set()
    try:
        from sqlalchemy import bindparam, text

        stmt = text(
            """
            SELECT DISTINCT ON (loop_id) loop_id, verdict
            FROM attention_loop_outcome
            WHERE loop_id IN :ids
            ORDER BY loop_id, created_at DESC
            """
        ).bindparams(bindparam("ids", expanding=True))
        with _engine().connect() as conn:
            rows = conn.execute(stmt, {"ids": ids}).mappings().all()
        return {
            str(row["loop_id"])
            for row in rows
            if str(row.get("verdict") or "") in TERMINAL_VERDICTS
        }
    except Exception as exc:
        logger.warning("attention_loop_outcome_verdict_lookup_failed ids=%s err=%s", ids, exc)
        return set()
