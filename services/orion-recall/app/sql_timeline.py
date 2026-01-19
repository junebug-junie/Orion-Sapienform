from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    psycopg2 = None

try:
    from .settings import settings
except ImportError:  # pragma: no cover - test harness path
    from settings import settings  # type: ignore


@dataclass
class TimelineItem:
    id: Any
    ts: float
    source: str
    title: Optional[str]
    text: str
    tags: List[str]
    session_id: Optional[str]
    node_id: Optional[str]
    source_ref: Optional[str] = None


def _connect():
    if psycopg2 is None:
        raise RuntimeError("psycopg2 not available")
    return psycopg2.connect(settings.RECALL_PG_DSN)


def _parse_row(row: Dict[str, Any]) -> TimelineItem:
    tags_raw = row.get("tags")
    tags: List[str] = []
    if isinstance(tags_raw, list):
        tags = [str(t) for t in tags_raw]
    elif tags_raw:
        try:
            import json
            parsed = json.loads(tags_raw)
            if isinstance(parsed, list):
                tags = [str(t) for t in parsed]
        except Exception:
            tags = [str(tags_raw)]

    ts_val = row.get("ts")
    try:
        ts = float(ts_val)
    except Exception:
        ts = 0.0

    return TimelineItem(
        id=row.get("id"),
        ts=ts,
        source="sql_timeline",
        title=row.get("title"),
        text=row.get("text") or "",
        tags=tags,
        session_id=row.get("session_id"),
        node_id=row.get("node_id"),
        source_ref=row.get("source_ref"),
    )


async def fetch_recent_fragments(
    session_id: Optional[str],
    node_id: Optional[str],
    since_minutes: int,
    limit: int,
) -> List[TimelineItem]:
    """
    Fetch recent fragments from the configured collapse mirror timeline.
    """
    def _query() -> List[TimelineItem]:
        conn = _connect()
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT id,
                       {settings.RECALL_SQL_TIMELINE_TS_COL} AS ts,
                       {settings.RECALL_SQL_TIMELINE_TEXT_COL} AS text,
                       {settings.RECALL_SQL_TIMELINE_SESSION_COL} AS session_id,
                       {settings.RECALL_SQL_TIMELINE_NODE_COL} AS node_id,
                       {settings.RECALL_SQL_TIMELINE_TAGS_COL} AS tags
                FROM {settings.RECALL_SQL_TIMELINE_TABLE}
                WHERE {settings.RECALL_SQL_TIMELINE_TS_COL} >= NOW() - INTERVAL '%s minutes'
                  AND (%s IS NULL OR {settings.RECALL_SQL_TIMELINE_SESSION_COL} = %s)
                ORDER BY {settings.RECALL_SQL_TIMELINE_TS_COL} DESC
                LIMIT %s
                """,
                (since_minutes, session_id, session_id, limit),
            )
            cols = [desc[0] for desc in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            for r in rows:
                r["source_ref"] = settings.RECALL_SQL_TIMELINE_TABLE
            return [_parse_row(r) for r in rows]
        finally:
            conn.close()

    return await asyncio.to_thread(_query)


async def fetch_related_by_entities(
    entities: List[str],
    since_hours: int,
    limit: int,
    *,
    session_id: Optional[str] = None,
) -> List[TimelineItem]:
    """
    Fetch fragments mentioning any of the provided entities.
    """
    if not entities:
        return []

    def _query() -> List[TimelineItem]:
        conn = _connect()
        try:
            cur = conn.cursor()
            placeholders = ", ".join(["%s"] * len(entities))
            cur.execute(
                f"""
                SELECT id,
                       {settings.RECALL_SQL_TIMELINE_TS_COL} AS ts,
                       {settings.RECALL_SQL_TIMELINE_TEXT_COL} AS text,
                       {settings.RECALL_SQL_TIMELINE_SESSION_COL} AS session_id,
                       {settings.RECALL_SQL_TIMELINE_NODE_COL} AS node_id,
                       {settings.RECALL_SQL_TIMELINE_TAGS_COL} AS tags
                FROM {settings.RECALL_SQL_TIMELINE_TABLE}
                WHERE {settings.RECALL_SQL_TIMELINE_TS_COL} >= NOW() - INTERVAL '%s hours'
                  AND ({settings.RECALL_SQL_TIMELINE_TEXT_COL} ILIKE ANY (ARRAY[{placeholders}]))
                  AND (%s IS NULL OR {settings.RECALL_SQL_TIMELINE_SESSION_COL} = %s)
                ORDER BY {settings.RECALL_SQL_TIMELINE_TS_COL} DESC
                LIMIT %s
                """,
                (since_hours, *[f"%{e}%" for e in entities], session_id, session_id, limit),
            )
            cols = [desc[0] for desc in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            for r in rows:
                r["source_ref"] = settings.RECALL_SQL_TIMELINE_TABLE
            return [_parse_row(r) for r in rows]
        finally:
            conn.close()

    return await asyncio.to_thread(_query)
