"""Read side of ``orion_day_letter`` for Hub (asyncpg). The durable run is the only writer."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from orion.schemas.orion_day import OrionDayLetterV1

LETTER_COLUMNS = (
    "letter_date, run_id, window_start, window_end, note_md, carry_forward_md, material, sources, "
    "journal_entry_id, created_at, emailed_at, email_notification_id, carry_forward_expires_at, "
    "carry_forward_offered_at, carry_forward_offered_run_id"
)
SELECT_LETTER_SQL = f"SELECT {LETTER_COLUMNS} FROM orion_day_letter WHERE letter_date = $1"


def letter_from_row(row: Any) -> OrionDayLetterV1:
    data = dict(row)
    for key in ("material", "sources"):
        if isinstance(data.get(key), (str, bytes)):
            data[key] = json.loads(data[key])
    return OrionDayLetterV1.model_validate(data)


async def fetch_letter(conn: Any, letter_date: date) -> OrionDayLetterV1 | None:
    row = await conn.fetchrow(SELECT_LETTER_SQL, letter_date)
    return letter_from_row(row) if row is not None else None


SELECT_LATEST_LETTER_SQL = f"SELECT {LETTER_COLUMNS} FROM orion_day_letter ORDER BY letter_date DESC LIMIT 1"
# The text a search index needs, without the ~700 KB material column.
SELECT_LETTER_TEXTS_SQL = (
    "SELECT letter_date, note_md, carry_forward_md, created_at FROM orion_day_letter ORDER BY letter_date"
)
LETTER_CREATED_SINCE_SQL = "SELECT EXISTS (SELECT 1 FROM orion_day_letter WHERE created_at >= $1)"


async def fetch_latest_letter(conn: Any) -> OrionDayLetterV1 | None:
    """The most recent letter by letter_date; None when no letter was ever written."""
    row = await conn.fetchrow(SELECT_LATEST_LETTER_SQL)
    return letter_from_row(row) if row is not None else None


async def fetch_letter_texts(conn: Any) -> list[dict[str, Any]]:
    """letter_date, note_md, carry_forward_md, created_at for every letter, oldest first."""
    return [dict(r) for r in await conn.fetch(SELECT_LETTER_TEXTS_SQL)]
