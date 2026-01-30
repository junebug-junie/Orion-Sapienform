from __future__ import annotations

from datetime import timedelta, datetime
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor


class TopicRailReader:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def fetch_training_rows(
        self,
        *,
        limit: int,
        time_window_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query = """
        SELECT id, correlation_id, session_id, prompt, response, created_at
        FROM chat_history_log
        {where_clause}
        ORDER BY created_at DESC
        LIMIT %s
        """
        params: list[Any] = []
        where_clause = ""
        if time_window_days:
            where_clause = "WHERE created_at >= %s"
            since = datetime.utcnow() - timedelta(days=int(time_window_days))
            params.append(since)
        query = query.format(where_clause=where_clause)
        params.append(int(limit))
        return self._fetch(query, params)

    def fetch_unassigned_rows(
        self,
        *,
        model_version: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        query = """
        SELECT c.id, c.correlation_id, c.session_id, c.prompt, c.response, c.created_at
        FROM chat_history_log c
        LEFT JOIN chat_topic t
          ON t.chat_id = c.id AND t.model_version = %s
        WHERE t.chat_id IS NULL
        ORDER BY c.created_at DESC
        LIMIT %s
        """
        return self._fetch(query, [model_version, int(limit)])

    def fetch_summary_counts(
        self,
        *,
        model_version: str,
        window_start: datetime,
        window_end: datetime,
    ) -> List[Dict[str, Any]]:
        query = """
        SELECT t.topic_id, COUNT(*) AS doc_count
        FROM chat_topic t
        JOIN chat_history_log c ON c.id = t.chat_id
        WHERE t.model_version = %s
          AND c.created_at >= %s
          AND c.created_at < %s
        GROUP BY t.topic_id
        ORDER BY doc_count DESC
        """
        return self._fetch(query, [model_version, window_start, window_end])

    def fetch_drift_rows(
        self,
        *,
        model_version: str,
        window_start: datetime,
        window_end: datetime,
    ) -> List[Dict[str, Any]]:
        query = """
        SELECT c.session_id, t.topic_id, c.created_at
        FROM chat_topic t
        JOIN chat_history_log c ON c.id = t.chat_id
        WHERE t.model_version = %s
          AND c.session_id IS NOT NULL
          AND c.created_at >= %s
          AND c.created_at < %s
        ORDER BY c.session_id, c.created_at
        """
        return self._fetch(query, [model_version, window_start, window_end])

    def _fetch(self, query: str, params: list[Any]) -> List[Dict[str, Any]]:
        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def count_rows_since(self, created_at_iso: str) -> int:
        query = """
        SELECT COUNT(*) AS doc_count
        FROM chat_history_log
        WHERE created_at >= %s
        """
        rows = self._fetch(query, [created_at_iso])
        if not rows:
            return 0
        return int(rows[0].get("doc_count") or 0)
