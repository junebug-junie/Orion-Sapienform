from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    psycopg2 = None

try:
    from .settings import settings
except ImportError:  # pragma: no cover - test harness path
    from settings import settings  # type: ignore

logger = logging.getLogger(__name__)


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


def _epoch(ts_val: Any, default: Optional[float] = None) -> float:
    if ts_val is None:
        return default or datetime.utcnow().timestamp()
    if isinstance(ts_val, datetime):
        return ts_val.timestamp()
    try:
        return datetime.fromisoformat(str(ts_val)).timestamp()
    except Exception:
        return default or datetime.utcnow().timestamp()


def _split_table_name(table: str) -> tuple[str, str]:
    if "." in table:
        schema, name = table.split(".", 1)
        return schema.strip('"'), name.strip('"')
    return "public", table.strip('"')


def _table_has_column(cur, table: str, column: str) -> bool:
    schema, name = _split_table_name(table)
    cur.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
          AND column_name = %s
        LIMIT 1
        """,
        (schema, name, column),
    )
    return cur.fetchone() is not None


def _should_filter_juniper() -> bool:
    if settings.RECALL_SQL_TIMELINE_REQUIRE_JUNIPER_OBSERVER is None:
        return settings.RECALL_SQL_TIMELINE_TABLE == "collapse_mirror"
    return bool(settings.RECALL_SQL_TIMELINE_REQUIRE_JUNIPER_OBSERVER)


def _juniper_filter_clause() -> str:
    observer_col = settings.RECALL_SQL_TIMELINE_SESSION_COL
    return (
        f"(lower({observer_col}) LIKE 'junip%' "
        f"OR lower({observer_col}) IN ('juniper','june','juniperk','juniperr','juniperfeld'))"
    )


async def fetch_recent_fragments(
    session_id: Optional[str],
    node_id: Optional[str],
    since_minutes: int,
    limit: int,
) -> List[TimelineItem]:
    """
    Fetch recent fragments from the configured SQL timeline source.
    By default this reads chat_history_log (prompt + response). If RECALL_SQL_TIMELINE_TABLE
    is overridden (e.g. collapse_mirror), the legacy timeline columns are used instead.
    """
    def _query() -> List[TimelineItem]:
        conn = _connect()
        try:
            cur = conn.cursor()
            timeline_table = settings.RECALL_SQL_TIMELINE_TABLE
            if timeline_table == settings.RECALL_SQL_CHAT_TABLE:
                logger.debug("sql_timeline: using chat-based timeline source")
                prompt_col = settings.RECALL_SQL_CHAT_TEXT_COL
                response_col = settings.RECALL_SQL_CHAT_RESPONSE_COL
                ts_col = settings.RECALL_SQL_CHAT_CREATED_AT_COL
                has_trace_id = _table_has_column(cur, timeline_table, "trace_id")
                session_clause = ""
                params: List[Any] = [since_minutes]
                if has_trace_id:
                    session_clause = "AND (%s IS NULL OR trace_id = %s)"
                    params.extend([session_id, session_id])
                params.append(limit)
                cur.execute(
                    f"""
                    SELECT
                        {"trace_id," if has_trace_id else ""}
                        {prompt_col} AS prompt,
                        {response_col} AS response,
                        {ts_col} AS created_at
                    FROM {timeline_table}
                    WHERE {ts_col} >= NOW() - INTERVAL '%s minutes'
                      {session_clause}
                    ORDER BY {ts_col} DESC
                    LIMIT %s
                    """,
                    params,
                )
                cols = [desc[0] for desc in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                formatted: List[TimelineItem] = []
                for row in rows:
                    prompt = (row.get("prompt") or "").strip()
                    response = (row.get("response") or "").strip()
                    text = f"User: {prompt}\nOrion: {response}".strip()
                    created_at = row.get("created_at")
                    trace_id = row.get("trace_id")
                    row_data = {
                        "id": str(trace_id or f"chat_{int(_epoch(created_at))}"),
                        "ts": _epoch(created_at),
                        "text": text,
                        "session_id": str(trace_id) if trace_id is not None else None,
                        "node_id": None,
                        "tags": ["chat_timeline"],
                        "source_ref": timeline_table,
                    }
                    formatted.append(_parse_row(row_data))
                return formatted

            logger.debug("sql_timeline: using generic timeline source=%s", timeline_table)
            juniper_clause = ""
            if _should_filter_juniper() and timeline_table == "collapse_mirror":
                juniper_clause = f" AND {_juniper_filter_clause()}"
            cur.execute(
                f"""
                SELECT id,
                       {settings.RECALL_SQL_TIMELINE_TS_COL} AS ts,
                       {settings.RECALL_SQL_TIMELINE_TEXT_COL} AS text,
                       {settings.RECALL_SQL_TIMELINE_SESSION_COL} AS session_id,
                       {settings.RECALL_SQL_TIMELINE_NODE_COL} AS node_id,
                       {settings.RECALL_SQL_TIMELINE_TAGS_COL} AS tags
                FROM {timeline_table}
                WHERE {settings.RECALL_SQL_TIMELINE_TS_COL} >= NOW() - INTERVAL '%s minutes'
                  AND (%s IS NULL OR {settings.RECALL_SQL_TIMELINE_SESSION_COL} = %s)
                  {juniper_clause}
                ORDER BY {settings.RECALL_SQL_TIMELINE_TS_COL} DESC
                LIMIT %s
                """,
                (since_minutes, session_id, session_id, limit),
            )
            cols = [desc[0] for desc in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            for r in rows:
                r["source_ref"] = timeline_table
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
    Defaults to searching recent chat prompt/response history unless the timeline
    table is explicitly set to a legacy source like collapse_mirror.
    """
    if not entities:
        return []

    def _query() -> List[TimelineItem]:
        conn = _connect()
        try:
            cur = conn.cursor()
            timeline_table = settings.RECALL_SQL_TIMELINE_TABLE
            if timeline_table == settings.RECALL_SQL_CHAT_TABLE:
                logger.debug("sql_timeline: using chat-based entity search")
                prompt_col = settings.RECALL_SQL_CHAT_TEXT_COL
                response_col = settings.RECALL_SQL_CHAT_RESPONSE_COL
                ts_col = settings.RECALL_SQL_CHAT_CREATED_AT_COL
                patterns = [f"%{e}%" for e in entities]
                has_trace_id = _table_has_column(cur, timeline_table, "trace_id")
                session_clause = ""
                params: List[Any] = [since_hours, patterns, patterns]
                if has_trace_id:
                    session_clause = "AND (%s IS NULL OR trace_id = %s)"
                    params.extend([session_id, session_id])
                params.append(limit)
                cur.execute(
                    f"""
                    SELECT
                        {"trace_id," if has_trace_id else ""}
                        {prompt_col} AS prompt,
                        {response_col} AS response,
                        {ts_col} AS created_at
                    FROM {timeline_table}
                    WHERE {ts_col} >= NOW() - INTERVAL '%s hours'
                      AND (
                        {prompt_col} ILIKE ANY (%s)
                        OR {response_col} ILIKE ANY (%s)
                      )
                      {session_clause}
                    ORDER BY {ts_col} DESC
                    LIMIT %s
                    """,
                    params,
                )
                cols = [desc[0] for desc in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                formatted: List[TimelineItem] = []
                for row in rows:
                    prompt = (row.get("prompt") or "").strip()
                    response = (row.get("response") or "").strip()
                    text = f"User: {prompt}\nOrion: {response}".strip()
                    created_at = row.get("created_at")
                    trace_id = row.get("trace_id")
                    row_data = {
                        "id": str(trace_id or f"chat_{int(_epoch(created_at))}"),
                        "ts": _epoch(created_at),
                        "text": text,
                        "session_id": str(trace_id) if trace_id is not None else None,
                        "node_id": None,
                        "tags": ["chat_timeline"],
                        "source_ref": timeline_table,
                    }
                    formatted.append(_parse_row(row_data))
                return formatted

            logger.debug("sql_timeline: using generic entity search source=%s", timeline_table)
            placeholders = ", ".join(["%s"] * len(entities))
            juniper_clause = ""
            if _should_filter_juniper() and timeline_table == "collapse_mirror":
                juniper_clause = f" AND {_juniper_filter_clause()}"
            cur.execute(
                f"""
                SELECT id,
                       {settings.RECALL_SQL_TIMELINE_TS_COL} AS ts,
                       {settings.RECALL_SQL_TIMELINE_TEXT_COL} AS text,
                       {settings.RECALL_SQL_TIMELINE_SESSION_COL} AS session_id,
                       {settings.RECALL_SQL_TIMELINE_NODE_COL} AS node_id,
                       {settings.RECALL_SQL_TIMELINE_TAGS_COL} AS tags
                FROM {timeline_table}
                WHERE {settings.RECALL_SQL_TIMELINE_TS_COL} >= NOW() - INTERVAL '%s hours'
                  AND ({settings.RECALL_SQL_TIMELINE_TEXT_COL} ILIKE ANY (ARRAY[{placeholders}]))
                  AND (%s IS NULL OR {settings.RECALL_SQL_TIMELINE_SESSION_COL} = %s)
                  {juniper_clause}
                ORDER BY {settings.RECALL_SQL_TIMELINE_TS_COL} DESC
                LIMIT %s
                """,
                (since_hours, *[f"%{e}%" for e in entities], session_id, session_id, limit),
            )
            cols = [desc[0] for desc in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            for r in rows:
                r["source_ref"] = timeline_table
            return [_parse_row(r) for r in rows]
        finally:
            conn.close()

    return await asyncio.to_thread(_query)
