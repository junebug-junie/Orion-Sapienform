from __future__ import annotations

from datetime import datetime

import psycopg2

from .models import JournalEntry


class JournalRepository:
    def __init__(self, dsn: str, table: str) -> None:
        self._dsn = dsn
        self._table = table

    def fetch_entries(self) -> list[JournalEntry]:
        sql = f"""
            SELECT
                entry_id,
                created_at,
                mode,
                source_kind,
                source_ref,
                title,
                body
            FROM {self._table}
            ORDER BY created_at DESC, entry_id DESC
        """
        with psycopg2.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
        return [
            JournalEntry(
                entry_id=row[0],
                created_at=row[1],
                mode=row[2],
                source_kind=row[3],
                source_ref=row[4],
                title=row[5],
                body=row[6] or "",
            )
            for row in rows
        ]

    def stats(self) -> dict[str, object]:
        sql = f"SELECT COUNT(*), MAX(created_at) FROM {self._table}"
        with psycopg2.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                count, latest = cur.fetchone()
        return {"count": int(count or 0), "latest_created_at": latest if isinstance(latest, datetime) else None}
