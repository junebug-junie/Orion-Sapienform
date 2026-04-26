from __future__ import annotations

from datetime import datetime
from typing import Any

from psycopg2.extras import RealDictCursor

from app.storage.pg import pg_conn


def fetch_chat_turn_rows(*, start_at: datetime, end_at: datetime, limit: int) -> list[dict[str, Any]]:
    query = """
        SELECT
            id,
            correlation_id,
            created_at,
            prompt,
            response,
            thought_process,
            source,
            memory_status,
            memory_tier,
            memory_reason,
            spark_meta,
            client_meta
        FROM chat_history_log
        WHERE created_at >= %s AND created_at < %s
        ORDER BY created_at ASC
        LIMIT %s
    """
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (start_at, end_at, limit))
            return cur.fetchall() or []
