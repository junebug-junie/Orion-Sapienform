"""Bounded, fail-open read of the questions Orion's quiet reveries left, for the situation brief.

Reveries that ask (2026-11-24, docs/superpowers/specs/2026-11-18-reverie-questions-design.md): a reverie
no longer narrates; it leaves one question (one ``reverie_question`` row) or nothing. This reader shows the
newest ACCEPTED questions (not refused, not none; ``orion/reverie/questions.py``) from the last
``_MAX_AGE_DAYS`` days. The retired narration table (substrate_reverie_thought) is no longer read.

Mirrors `perception_reader.py`'s shape: module-level cached engine, DSN resolution (own override ->
`POSTGRES_URI` -> `DATABASE_URL`), a per-connection `statement_timeout` GUC, fail-open, never raises to
the caller. Read-only; orion-thought is the table's only writer.

**Privacy.** Selects the question text and its time, nothing else: no seed material, no refs.
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


def reset_reverie_reader_engine_for_tests() -> None:
    """Mirrors `perception_reader.reset_perception_reader_engine_for_tests`
    -- not currently called by any test in this repo (the situation-layer
    tests monkeypatch `fetch_recent_reverie_snippets` at the `context`
    module level instead), but kept for parity so a future test exercising
    `_get_engine()`/DSN-switching directly has the same reset seam
    `perception_reader.py` already offers."""
    global _ENGINE, _ENGINE_URL
    _ENGINE = None
    _ENGINE_URL = None


class ReverieRow(NamedTuple):
    text: str
    observed_at: datetime | None
    salience: float | None


# Rows read per requested snippet before de-duplication (code already refuses near-duplicate reverie
# questions, but an exact repeat across threads costs nothing to drop here).
_OVERFETCH_FACTOR = 5
_OVERFETCH_CAP = 100
# Questions older than this are not "recent" for the brief.
_MAX_AGE_DAYS = 7


def _normalize_reverie_text(value: str) -> str:
    return " ".join(str(value or "").split()).casefold().rstrip(" .!?;:,")


def fetch_recent_reverie_snippets(limit: int) -> list[ReverieRow]:
    """Newest `limit` distinct accepted reverie questions of the last `_MAX_AGE_DAYS` days, newest first.

    Distinct by normalized text (case, whitespace and trailing punctuation
    ignored); the newest occurrence of a repeated line is the one kept.

    Returns `[]` on no DSN configured, no rows yet, or any read error --
    the caller cannot distinguish those cases from this return value alone
    by design, same collapse `perception_reader.fetch_latest_percept`
    already makes for its own three equivalent cases (an unpatched/
    unconfigured/erroring reader must degrade to "nothing to show", never
    raise into turn assembly).
    """
    want = max(0, int(limit))
    if want == 0:
        return []
    engine = _get_engine()
    if engine is None:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT question, created_at, NULL::double precision AS salience "
                    "FROM reverie_question "
                    "WHERE question IS NOT NULL AND question <> '' "
                    "AND refused_reason IS NULL AND none_reason IS NULL "
                    "AND created_at > now() - make_interval(days => :days) "
                    "ORDER BY created_at DESC LIMIT :limit"
                ),
                {"limit": min(want * _OVERFETCH_FACTOR, max(want, _OVERFETCH_CAP)), "days": _MAX_AGE_DAYS},
            ).all()
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.warning("situation_reverie_read_failed err=%s", exc)
        return []

    out: list[ReverieRow] = []
    seen: set[str] = set()
    for row in rows:
        text_value = str(row[0]).strip()
        key = _normalize_reverie_text(text_value)
        if not key or key in seen:
            continue
        seen.add(key)
        observed_at = row[1]
        if observed_at is not None and observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        out.append(ReverieRow(text=text_value, observed_at=observed_at, salience=row[2]))
        if len(out) >= want:
            break
    return out
