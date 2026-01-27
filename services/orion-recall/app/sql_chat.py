from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

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


async def fetch_chat_history_pairs(limit: int, since_minutes: int) -> List[ChatItem]:
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
    for idx, row in enumerate(rows):
        prompt = row.get("prompt") or ""
        response = row.get("response") or ""
        created_at = row.get("created_at")
        text = f'ExactUserText: "{prompt}"\nOrionResponse: "{response}"'
        items.append(
            ChatItem(
                id=f"{settings.RECALL_SQL_CHAT_TABLE}:{idx}",
                ts=_to_epoch(created_at),
                text=text,
                source_ref=settings.RECALL_SQL_CHAT_TABLE,
            )
        )
    return items


async def fetch_chat_messages(limit: int, since_minutes: int) -> List[ChatItem]:
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
    for idx, row in enumerate(rows):
        role = row.get("role") or "unknown"
        text = row.get("text") or ""
        created_at = row.get("created_at")
        items.append(
            ChatItem(
                id=f"{table}:{idx}",
                ts=_to_epoch(created_at),
                text=f"{role}: {text}",
                source_ref=table,
            )
        )
    return items
