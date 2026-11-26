"""Reading reverie questions back: one SQL and one line renderer every consumer shares.

Reveries that ask, Patch 1 (docs/superpowers/specs/2026-11-18-reverie-questions-design.md). orion-thought
writes one ``reverie_question`` row per reverie (services/orion-sql-db/manual_migration_reverie_question.sql;
contract orion/schemas/reverie.py ReverieQuestionV1). The surfaces that used to show reverie chains (Prime's
frame, the day letter, the situation brief, outreach, Hub) read this table through these helpers, so "what a
reverie asked" is worded the same way everywhere.

Only questions code let through are shown as questions: a row with ``refused_reason`` is a record of a
refusal, and a ``none_reason`` row says the material led nowhere. Neither is ever shown as a question.
Stdlib only: imported by thin services.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

TABLE = "reverie_question"
MIGRATION = "services/orion-sql-db/manual_migration_reverie_question.sql"

COLUMNS = (
    "question_id, thread_id, seed_kind, seed_refs, question, why, none_reason, refused_reason, "
    "routed_to, offer_status, outcome, created_at"
)

# Accepted questions (not refused, not none) created in [lo, hi), newest first. Positional params: lo, hi,
# limit. psycopg / SQLAlchemy callers use the ``%(lo)s`` form below; asyncpg callers use ``$1..$3``.
ACCEPTED_SQL_PG = (
    f"SELECT {COLUMNS} FROM {TABLE} "
    "WHERE created_at >= %(lo)s AND created_at < %(hi)s AND question IS NOT NULL "
    "AND refused_reason IS NULL AND none_reason IS NULL "
    "ORDER BY created_at DESC LIMIT %(limit)s"
)
ACCEPTED_SQL_ASYNCPG = (
    f"SELECT {COLUMNS} FROM {TABLE} "
    "WHERE created_at >= $1 AND created_at < $2 AND question IS NOT NULL "
    "AND refused_reason IS NULL AND none_reason IS NULL "
    "ORDER BY created_at DESC LIMIT $3"
)
ACCEPTED_SQL_SA = (
    f"SELECT {COLUMNS} FROM {TABLE} "
    "WHERE created_at >= :lo AND created_at < :hi AND question IS NOT NULL "
    "AND refused_reason IS NULL AND none_reason IS NULL "
    "ORDER BY created_at DESC LIMIT :limit"
)

SEED_WORDS = {
    "absence": "a routine that did not happen",
    "lost_bet": "a bet I keep losing",
    "far_memory": "an old memory beside a new one",
}


def _aware(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def is_accepted(row: Mapping[str, Any]) -> bool:
    return bool(str(row.get("question") or "").strip()) and not row.get("refused_reason") and not row.get(
        "none_reason"
    )


# reverie_question.offer_status (Patch 3, services/orion-thought/app/reverie_threads.py), in words.
OFFER_WORDS = {
    "waiting": "waiting for the focus slot",
    "lapsed": "waited for the focus slot, lapsed",
    "dropped": "not sent: repeated an open question",
}


def routed_words(row: Mapping[str, Any]) -> str:
    routed = str(row.get("routed_to") or "none")
    if routed.startswith("focus:"):
        return "sent to curiosity"
    words = OFFER_WORDS.get(str(row.get("offer_status") or ""))
    return words or "kept, not sent"


def outcome_words(row: Mapping[str, Any]) -> str | None:
    """What the dig found, when Patch 3 has written it (``outcome`` jsonb); None before that."""
    outcome = row.get("outcome")
    if not isinstance(outcome, Mapping):
        return None
    parts = [str(outcome.get(k)) for k in ("disposition", "finding") if outcome.get(k)]
    note = str(outcome.get("continue_note") or "").strip()
    if note:
        parts.append(note[:160])
    return "; ".join(parts) or None


def render_question_line(row: Mapping[str, Any], *, now: datetime | None = None, max_chars: int = 400) -> str:
    """One plain line: when, what it started from, the question, and where it went.

    Example: ``- 2h ago (a bet I keep losing): Why do I call long turns unlikely when they happen 37% of the
    time? (sent to curiosity) [reverie_question:abc]``"""
    q = " ".join(str(row.get("question") or "").split())
    at = _aware(row.get("created_at"))
    when = ""
    if at is not None:
        age = max(0.0, ((now or datetime.now(timezone.utc)) - at).total_seconds())
        when = f"{int(age // 60)}m ago" if age < 3600 else f"{age / 3600:.0f}h ago"
    seed = SEED_WORDS.get(str(row.get("seed_kind") or ""), str(row.get("seed_kind") or "material"))
    tail = routed_words(row)
    found = outcome_words(row)
    if found:
        tail += f"; the dig: {found}"
    line = f"- {when + ' ' if when else ''}({seed}): {q} ({tail}) [reverie_question:{row.get('question_id')}]"
    return line if len(line) <= max_chars else line[: max_chars - 1] + "…"


__all__ = [
    "ACCEPTED_SQL_ASYNCPG",
    "ACCEPTED_SQL_PG",
    "ACCEPTED_SQL_SA",
    "COLUMNS",
    "MIGRATION",
    "OFFER_WORDS",
    "SEED_WORDS",
    "TABLE",
    "is_accepted",
    "outcome_words",
    "render_question_line",
    "routed_words",
]
