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
