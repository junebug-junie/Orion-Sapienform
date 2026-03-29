from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import asyncpg  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    asyncpg = None

try:
    from .settings import settings
except ImportError:  # pragma: no cover - test harness path
    from settings import settings  # type: ignore


@dataclass
class ChatItem:
    id: str
    ts: float
    text: str
    source_ref: str


def _to_epoch(value: Any) -> float:
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return " ".join(text.split())


def _contains_active_prompt(record_text: str, active_text: str) -> bool:
    normalized_active = _normalize_text(active_text)
    if not normalized_active:
        return False
    normalized_record = _normalize_text(record_text)
    if not normalized_record:
        return False
    return normalized_active in normalized_record


async def fetch_chat_history_pairs(
    limit: int,
    since_minutes: int,
    *,
    exclude_text: Optional[str] = None,
    exclude_ids: Optional[List[str]] = None,
) -> List[ChatItem]:
    if asyncpg is None:
        return []
    query = f"""
        SELECT {settings.RECALL_SQL_CHAT_TEXT_COL} AS prompt,
               {settings.RECALL_SQL_CHAT_RESPONSE_COL} AS response,
               {settings.RECALL_SQL_CHAT_CREATED_AT_COL} AS created_at
        FROM {settings.RECALL_SQL_CHAT_TABLE}
        WHERE {settings.RECALL_SQL_CHAT_CREATED_AT_COL} >= NOW() - INTERVAL '{since_minutes} minutes'
        ORDER BY {settings.RECALL_SQL_CHAT_CREATED_AT_COL} DESC
        LIMIT {limit}
    """
    try:
        conn = await asyncpg.connect(settings.RECALL_PG_DSN)
        try:
            rows = await conn.fetch(query)
        finally:
            await conn.close()
    except Exception:
        return []

    items: List[ChatItem] = []
    excluded_id_set = {str(v).strip() for v in (exclude_ids or []) if str(v).strip()}
    suppressed = 0
    for idx, row in enumerate(rows):
        prompt = row.get("prompt") or ""
        response = row.get("response") or ""
        created_at = row.get("created_at")
        row_id = str(row.get("correlation_id") or row.get("id") or f"{settings.RECALL_SQL_CHAT_TABLE}:{idx}")
        if row_id in excluded_id_set:
            suppressed += 1
            continue
        if _contains_active_prompt(prompt, exclude_text or ""):
            suppressed += 1
            continue
        text = f'ExactUserText: "{prompt}"\nOrionResponse: "{response}"'
        items.append(
            ChatItem(
                id=row_id,
                ts=_to_epoch(created_at),
                text=text,
                source_ref=settings.RECALL_SQL_CHAT_TABLE,
            )
        )
    if suppressed:
        logger.info("sql_chat self-hit suppression backend=chat_pairs suppressed=%s", suppressed)
    return items


async def fetch_chat_messages(
    limit: int,
    since_minutes: int,
    *,
    exclude_text: Optional[str] = None,
    exclude_ids: Optional[List[str]] = None,
) -> List[ChatItem]:
    table = settings.RECALL_SQL_MESSAGE_TABLE
    if not table:
        return []
    if asyncpg is None:
        return []
    query = f"""
        SELECT {settings.RECALL_SQL_MESSAGE_ROLE_COL} AS role,
               {settings.RECALL_SQL_MESSAGE_TEXT_COL} AS text,
               {settings.RECALL_SQL_MESSAGE_CREATED_AT_COL} AS created_at
        FROM {table}
        WHERE {settings.RECALL_SQL_MESSAGE_CREATED_AT_COL} >= NOW() - INTERVAL '{since_minutes} minutes'
        ORDER BY {settings.RECALL_SQL_MESSAGE_CREATED_AT_COL} DESC
        LIMIT {limit}
    """
    try:
        conn = await asyncpg.connect(settings.RECALL_PG_DSN)
        try:
            rows = await conn.fetch(query)
        finally:
            await conn.close()
    except Exception:
        return []

    items: List[ChatItem] = []
    excluded_id_set = {str(v).strip() for v in (exclude_ids or []) if str(v).strip()}
    suppressed = 0
    for idx, row in enumerate(rows):
        role = row.get("role") or "unknown"
        text = row.get("text") or ""
        created_at = row.get("created_at")
        row_id = str(row.get("correlation_id") or row.get("id") or f"{table}:{idx}")
        if row_id in excluded_id_set:
            suppressed += 1
            continue
        if role.lower() == "user" and _contains_active_prompt(text, exclude_text or ""):
            suppressed += 1
            continue
        items.append(
            ChatItem(
                id=row_id,
                ts=_to_epoch(created_at),
                text=f"{role}: {text}",
                source_ref=table,
            )
        )
    if suppressed:
        logger.info("sql_chat self-hit suppression backend=chat_messages suppressed=%s", suppressed)
    return items
