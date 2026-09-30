"""The one writer of ``orion_day_letter`` (services/orion-sql-db/manual_migration_orion_day_letter_v1.sql).

Runs on AdmissionRuntime's psycopg pool (autocommit, dict rows). One letter per day: the first
run to insert wins, a replay or a later run is a no-op, and the caller learns whose row it is.
"""
from __future__ import annotations

from typing import Any

INSERT_LETTER_SQL = """
INSERT INTO orion_day_letter (
    letter_date, run_id, window_start, window_end, note_md, carry_forward_md,
    material, sources, journal_entry_id, created_at, carry_forward_expires_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (letter_date) DO NOTHING
RETURNING run_id, note_md, created_at
"""
SELECT_STORED_SQL = "SELECT run_id, note_md, created_at FROM orion_day_letter WHERE letter_date = %s"


async def persist_letter(pool: Any, row: dict[str, Any]) -> dict[str, Any]:
    """Insert the day's letter unless one exists. Returns the STORED row's
    ``{"run_id", "note_md", "created_at"}`` -- this run's, or the earlier run's that won."""
    from psycopg.types.json import Jsonb

    params = (
        row["letter_date"], row["run_id"], row["window_start"], row["window_end"], row["note_md"],
        row["carry_forward_md"], Jsonb(row["material"]), Jsonb(row["sources"]), row["journal_entry_id"],
        row["created_at"], row["carry_forward_expires_at"],
    )
    async with pool.connection() as conn:
        inserted = await (await conn.execute(INSERT_LETTER_SQL, params)).fetchone()
        if inserted is not None:
            return dict(inserted)
        existing = await (await conn.execute(SELECT_STORED_SQL, (row["letter_date"],))).fetchone()
    if existing is None:
        raise RuntimeError("orion_day_letter insert conflicted but no row is readable")
    return dict(existing)
