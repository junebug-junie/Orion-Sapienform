from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from psycopg2.extras import RealDictCursor

from app.storage.pg import pg_conn

from .exclusions import cached_excluded_turn_ids

logger = logging.getLogger(__name__)


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
          AND NOT (id = ANY(%s))
        ORDER BY created_at ASC
        LIMIT %s
    """
    # Excluded in SQL rather than filtered in Python so the LIMIT applies to
    # the rows that will actually be used. Filtering after the fetch would let
    # excluded turns consume LIMIT slots and silently shrink the real window.
    excluded = sorted(cached_excluded_turn_ids())
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (start_at, end_at, excluded, limit))
            rows = cur.fetchall() or []
    if excluded:
        logger.info(
            "topic_foundry_corpus_excluded_turns count=%d rows_returned=%d",
            len(excluded), len(rows),
        )
    return rows
