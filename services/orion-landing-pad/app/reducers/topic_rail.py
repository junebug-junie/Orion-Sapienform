from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import asyncpg


async def _resolve_model_version(conn: asyncpg.Connection, table_name: str) -> Optional[str]:
    query = f"""
        SELECT model_version
        FROM {table_name}
        ORDER BY window_end DESC, created_at DESC
        LIMIT 1
    """
    row = await conn.fetchrow(query)
    return row["model_version"] if row else None


def _build_topic_summary_query(
    window_minutes: int,
    model_version: str,
    max_topics: int,
) -> Tuple[str, Sequence[Any]]:
    query = """
        SELECT
            model_version,
            window_start,
            window_end,
            topic_id,
            topic_label,
            topic_keywords,
            doc_count,
            pct_of_window,
            outlier_count,
            outlier_pct
        FROM chat_topic_summary
        WHERE model_version = $1
          AND window_end >= (NOW() - ($2 * interval '1 minute'))
        ORDER BY doc_count DESC
        LIMIT $3
    """
    return query, [model_version, window_minutes, max_topics]


def _build_topic_drift_query(
    window_minutes: int,
    model_version: str,
    min_turns: int,
    max_sessions: int,
) -> Tuple[str, Sequence[Any]]:
    query = """
        SELECT
            model_version,
            window_start,
            window_end,
            session_id,
            turns,
            unique_topics,
            entropy,
            switch_rate,
            dominant_topic_id,
            dominant_pct
        FROM chat_topic_session_drift
        WHERE model_version = $1
          AND window_end >= (NOW() - ($2 * interval '1 minute'))
          AND turns >= $3
        ORDER BY switch_rate DESC, entropy DESC
        LIMIT $4
    """
    return query, [model_version, window_minutes, min_turns, max_sessions]


async def get_topic_summary(
    window_minutes: int,
    model_version: Optional[str] = None,
    max_topics: int = 20,
    *,
    dsn: Optional[str] = None,
) -> Dict[str, Any]:
    if dsn is None:
        from ..settings import settings

        dsn = settings.postgres_uri
    conn = await asyncpg.connect(dsn=dsn)
    try:
        resolved_model = model_version or await _resolve_model_version(conn, "chat_topic_summary")
        if not resolved_model:
            return {
                "model_version": None,
                "window_minutes": window_minutes,
                "topics": [],
            }
        query, params = _build_topic_summary_query(window_minutes, resolved_model, max_topics)
        rows = await conn.fetch(query, *params)
        return {
            "model_version": resolved_model,
            "window_minutes": window_minutes,
            "topics": [dict(row) for row in rows],
        }
    finally:
        await conn.close()


async def get_topic_drift(
    window_minutes: int,
    model_version: Optional[str] = None,
    min_turns: int = 10,
    max_sessions: int = 50,
    *,
    dsn: Optional[str] = None,
) -> Dict[str, Any]:
    if dsn is None:
        from ..settings import settings

        dsn = settings.postgres_uri
    conn = await asyncpg.connect(dsn=dsn)
    try:
        resolved_model = model_version or await _resolve_model_version(conn, "chat_topic_session_drift")
        if not resolved_model:
            return {
                "model_version": None,
                "window_minutes": window_minutes,
                "min_turns": min_turns,
                "sessions": [],
            }
        query, params = _build_topic_drift_query(window_minutes, resolved_model, min_turns, max_sessions)
        rows = await conn.fetch(query, *params)
        return {
            "model_version": resolved_model,
            "window_minutes": window_minutes,
            "min_turns": min_turns,
            "sessions": [dict(row) for row in rows],
        }
    finally:
        await conn.close()
