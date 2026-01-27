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


def _pick_session_col(cur, table: str) -> Optional[str]:
    for column in ("session_id", "correlation_id", "trace_id"):
        if _table_has_column(cur, table, column):
            return column
    return None


def _pick_id_col(cur, table: str) -> Optional[str]:
    for column in ("id", "correlation_id"):
        if _table_has_column(cur, table, column):
            return column
    return None


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
                id_col = _pick_id_col(cur, timeline_table)
                session_col = _pick_session_col(cur, timeline_table)
                select_row_id = f"{id_col} AS row_id," if id_col else ""
                select_sid = f"{session_col} AS sid," if session_col else ""
                session_clause = ""
                session_presence_clause = ""
                params: List[Any] = [since_minutes]
                if session_col:
                    session_presence_clause = f"AND {session_col} IS NOT NULL AND {session_col} <> ''"
                    session_clause = f"AND (%s IS NULL OR {session_col} = %s)"
                    params.extend([session_id, session_id])
                params.append(limit)
                cur.execute(
                    f"""
                    SELECT
                        {select_row_id}
                        {select_sid}
                        {prompt_col} AS prompt,
                        {response_col} AS response,
                        {ts_col} AS created_at
                    FROM {timeline_table}
                    WHERE {ts_col} >= NOW() - INTERVAL '%s minutes'
                      {session_presence_clause}
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
                    sid = row.get("sid")
                    row_id = row.get("row_id")
                    tags = ["chat_timeline"]
                    if sid is not None and str(sid) != "":
                        tags.append(f"session_id:{sid}")
                        if session_col:
                            tags.append(f"session_col:{session_col}")
                    row_data = {
                        "id": str(row_id or f"chat_{int(_epoch(created_at))}"),
                        "ts": _epoch(created_at),
                        "text": text,
                        "session_id": str(sid) if sid is not None else None,
                        "node_id": None,
                        "tags": tags,
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
                id_col = _pick_id_col(cur, timeline_table)
                session_col = _pick_session_col(cur, timeline_table)
                select_row_id = f"{id_col} AS row_id," if id_col else ""
                select_sid = f"{session_col} AS sid," if session_col else ""
                session_clause = ""
                session_presence_clause = ""
                params: List[Any] = [since_hours, patterns, patterns]
                if session_col:
                    session_presence_clause = f"AND {session_col} IS NOT NULL AND {session_col} <> ''"
                    session_clause = f"AND (%s IS NULL OR {session_col} = %s)"
                    params.extend([session_id, session_id])
                params.append(limit)
                cur.execute(
                    f"""
                    SELECT
                        {select_row_id}
                        {select_sid}
                        {prompt_col} AS prompt,
                        {response_col} AS response,
                        {ts_col} AS created_at
                    FROM {timeline_table}
                    WHERE {ts_col} >= NOW() - INTERVAL '%s hours'
                      AND (
                        {prompt_col} ILIKE ANY (%s)
                        OR {response_col} ILIKE ANY (%s)
                      )
                      {session_presence_clause}
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
                    sid = row.get("sid")
                    row_id = row.get("row_id")
                    tags = ["chat_timeline"]
                    if sid is not None and str(sid) != "":
                        tags.append(f"session_id:{sid}")
                        if session_col:
                            tags.append(f"session_col:{session_col}")
                    row_data = {
                        "id": str(row_id or f"chat_{int(_epoch(created_at))}"),
                        "ts": _epoch(created_at),
                        "text": text,
                        "session_id": str(sid) if sid is not None else None,
                        "node_id": None,
                        "tags": tags,
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


async def fetch_exact_fragments(
    tokens: List[str],
    session_id: Optional[str],
    node_id: Optional[str],
    limit: int,
) -> List[TimelineItem]:
    if not tokens:
        return []

    def _query() -> List[TimelineItem]:
        timeline_table = settings.RECALL_SQL_TIMELINE_TABLE
        if not timeline_table:
            return []
        conn = _connect()
        try:
            with conn.cursor() as cur:
                if timeline_table == settings.RECALL_SQL_CHAT_TABLE:
                    id_col = _pick_id_col(cur, timeline_table)
                    session_col = _pick_session_col(cur, timeline_table)
                    prompt_col = settings.RECALL_SQL_CHAT_TEXT_COL
                    response_col = settings.RECALL_SQL_CHAT_RESPONSE_COL
                    ts_col = settings.RECALL_SQL_CHAT_CREATED_AT_COL
                    select_row_id = f"{id_col} AS row_id," if id_col else ""
                    select_sid = f"{session_col} AS sid," if session_col else ""
                    session_clause = ""
                    params = []
                    if session_col:
                        session_clause = f"AND (%s IS NULL OR {session_col} = %s)"
                        params.extend([session_id, session_id])
                    term_filters = []
                    for token in tokens:
                        term_filters.append(f"{prompt_col} ILIKE %s")
                        params.append(f"%{token}%")
                        term_filters.append(f"{response_col} ILIKE %s")
                        params.append(f"%{token}%")
                    filter_clause = " OR ".join(term_filters) if term_filters else "TRUE"
                    params.append(limit)
                    cur.execute(
                        f"""
                        SELECT
                            {select_row_id}
                            {select_sid}
                            {prompt_col} AS prompt,
                            {response_col} AS response,
                            {ts_col} AS created_at
                        FROM {timeline_table}
                        WHERE ({filter_clause})
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
                        sid = row.get("sid")
                        row_id = row.get("row_id")
                        tags = ["chat_timeline"]
                        if sid is not None and str(sid) != "":
                            tags.append(f"session_id:{sid}")
                            if session_col:
                                tags.append(f"session_col:{session_col}")
                        row_data = {
                            "id": str(row_id or f"chat_{int(_epoch(created_at))}"),
                            "ts": _epoch(created_at),
                            "text": text,
                            "session_id": str(sid) if sid is not None else None,
                            "node_id": None,
                            "tags": tags,
                            "source_ref": timeline_table,
                        }
                        formatted.append(_parse_row(row_data))
                    return formatted

                logger.debug("sql_timeline: using generic exact-match source=%s", timeline_table)
                term_filters = []
                params = []
                for token in tokens:
                    term_filters.append(f"{settings.RECALL_SQL_TIMELINE_TEXT_COL} ILIKE %s")
                    params.append(f"%{token}%")
                filter_clause = " OR ".join(term_filters) if term_filters else "TRUE"
                session_clause = ""
                node_clause = ""
                if settings.RECALL_SQL_TIMELINE_SESSION_COL:
                    session_clause = f"AND (%s IS NULL OR {settings.RECALL_SQL_TIMELINE_SESSION_COL} = %s)"
                    params.extend([session_id, session_id])
                if node_id and settings.RECALL_SQL_TIMELINE_NODE_COL:
                    node_clause = f"AND {settings.RECALL_SQL_TIMELINE_NODE_COL} = %s"
                    params.append(node_id)
                params.append(limit)
                cur.execute(
                    f"""
                    SELECT id,
                           {settings.RECALL_SQL_TIMELINE_TS_COL} AS ts,
                           {settings.RECALL_SQL_TIMELINE_TEXT_COL} AS text,
                           {settings.RECALL_SQL_TIMELINE_SESSION_COL} AS session_id,
                           {settings.RECALL_SQL_TIMELINE_NODE_COL} AS node_id,
                           {settings.RECALL_SQL_TIMELINE_TAGS_COL} AS tags
                    FROM {timeline_table}
                    WHERE ({filter_clause})
                      {session_clause}
                      {node_clause}
                    ORDER BY {settings.RECALL_SQL_TIMELINE_TS_COL} DESC
                    LIMIT %s
                    """,
                    params,
                )
                cols = [desc[0] for desc in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                for r in rows:
                    r["source_ref"] = timeline_table
                return [_parse_row(r) for r in rows]
        finally:
            conn.close()

    return await asyncio.to_thread(_query)
