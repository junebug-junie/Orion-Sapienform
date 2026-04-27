from __future__ import annotations

from datetime import datetime

import psycopg2
import re

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

    def search_entries(self, query: str, limit: int = 8) -> list[JournalEntry]:
        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "have",
            "been",
            "what",
            "your",
            "from",
            "about",
            "lately",
        }
        terms = [
            term
            for term in re.findall(r"[a-zA-Z0-9_]+", query.lower())
            if len(term) > 2 and term not in stop_words
        ]
        if not terms:
            return []
        predicates = " OR ".join(
            ["(COALESCE(title, '') ILIKE %s OR COALESCE(body, '') ILIKE %s)" for _ in terms]
        )
        score_expr = " + ".join(
            [
                "(CASE WHEN COALESCE(title, '') ILIKE %s OR COALESCE(body, '') ILIKE %s THEN 1 ELSE 0 END)"
                for _ in terms
            ]
        )
        params: list[object] = []
        for term in terms:
            like = f"%{term}%"
            params.extend([like, like])
        score_params: list[object] = []
        for term in terms:
            like = f"%{term}%"
            score_params.extend([like, like])
        params = score_params + params
        params.append(limit)

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
                body,
                ({score_expr}) as match_score
            FROM {self._table}
            WHERE {predicates}
            ORDER BY match_score DESC, created_at DESC, entry_id DESC
            LIMIT %s
        """
        with psycopg2.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
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
