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
                trigger_kind,
                trigger_summary,
                conversation_frame,
                task_mode,
                identity_salience,
                answer_strategy,
                stance_summary,
                active_identity_facets,
                active_growth_axes,
                active_relationship_facets,
                social_posture,
                reflective_themes,
                active_tensions,
                dream_motifs,
                response_hazards,
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
                trigger_kind=row[5],
                trigger_summary=row[6],
                conversation_frame=row[7],
                task_mode=row[8],
                identity_salience=row[9],
                answer_strategy=row[10],
                stance_summary=row[11],
                active_identity_facets=row[12],
                active_growth_axes=row[13],
                active_relationship_facets=row[14],
                social_posture=row[15],
                reflective_themes=row[16],
                active_tensions=row[17],
                dream_motifs=row[18],
                response_hazards=row[19],
                title=row[20],
                body=row[21] or "",
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
