"""Hub attention minutes: record that Juniper is actually looking at the Hub.

The browser (static/js/hub-attention-tracker.js) sends one heartbeat per minute,
but only while the page is visible, the window has focus, and there was real
input in the last 120 s. This route folds each heartbeat into one row per
(minute, client) in ``hub_attention_minute``. The minute is
``date_trunc('minute', now())`` on the Postgres server -- client clocks are
never trusted.

Best-effort on every path: a missing table (migration
``services/orion-sql-db/manual_migration_hub_attention_minute_v1.sql`` not
applied), an unset ``POSTGRES_URI``, or any DB error is logged (rate-limited)
and answered with ``{"ok": false, ...}`` at HTTP 200. It never 500s.

No reader and no bus event yet, on purpose: the prediction service's reachable
forecast reads this table later, once ~2 weeks of live rows pass the metric gate.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from .settings import settings

logger = logging.getLogger("orion-hub.hub_attention")

router = APIRouter(prefix="/api/hub-attention", tags=["hub-attention"])

TAB_MAX_LEN = 64
_TAB_ALLOWED = re.compile(r"[^A-Za-z0-9_\-/.:]")
_LOG_EVERY_SEC = 600.0

UPSERT_SQL = """
INSERT INTO hub_attention_minute (minute, client, tab, beats, first_at, last_at)
VALUES (date_trunc('minute', now()), :client, :tab, 1, now(), now())
ON CONFLICT (minute, client) DO UPDATE SET
    beats = hub_attention_minute.beats + 1,
    tab = COALESCE(EXCLUDED.tab, hub_attention_minute.tab),
    last_at = EXCLUDED.last_at
RETURNING minute
"""

_engine_instance: Any = None
_engine_lock = threading.Lock()
_last_error_log_at: float = 0.0


class HubAttentionHeartbeat(BaseModel):
    model_config = ConfigDict(extra="ignore")

    client: Literal["desktop", "mobile"]
    tab: str | None = Field(default=None, max_length=512)


def normalize_tab(tab: str | None) -> str | None:
    """Hash route -> short safe label ('#reading' -> 'reading'); empty -> None."""
    if not tab:
        return None
    cleaned = _TAB_ALLOWED.sub("", tab.strip().lstrip("#"))[:TAB_MAX_LEN]
    return cleaned or None


def _engine():
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            uri = os.getenv("POSTGRES_URI", "").strip()
            if not uri:
                return None
            from sqlalchemy import create_engine

            _engine_instance = create_engine(uri, pool_pre_ping=True, pool_size=2, max_overflow=2)
        return _engine_instance


def _log_failure(reason: str, exc: Exception | None = None) -> None:
    """At most one warning per _LOG_EVERY_SEC so a missing table can't flood logs."""
    global _last_error_log_at
    now = time.monotonic()
    if _last_error_log_at and now - _last_error_log_at < _LOG_EVERY_SEC:
        return
    _last_error_log_at = now
    logger.warning("hub_attention_write_failed reason=%s error=%s", reason, exc)


def write_heartbeat(client: str, tab: str | None) -> str | None:
    """Upsert one beat; returns the server minute (ISO) or raises."""
    from sqlalchemy import text

    engine = _engine()
    if engine is None:
        raise RuntimeError("postgres_uri_not_configured")
    with engine.begin() as conn:
        row = conn.execute(text(UPSERT_SQL), {"client": client, "tab": tab}).first()
    minute = row[0] if row else None
    return minute.isoformat() if hasattr(minute, "isoformat") else (str(minute) if minute else None)


@router.post("/heartbeat")
def hub_attention_heartbeat(beat: HubAttentionHeartbeat) -> dict[str, Any]:
    if not bool(getattr(settings, "HUB_ATTENTION_TRACKER_ENABLED", True)):
        return {"ok": False, "enabled": False}
    tab = normalize_tab(beat.tab)
    try:
        minute = write_heartbeat(beat.client, tab)
    except Exception as exc:  # best-effort: never 500 the page over a heartbeat
        _log_failure("db_error", exc)
        return {"ok": False, "enabled": True, "reason": "db_error"}
    return {"ok": True, "enabled": True, "minute": minute, "client": beat.client, "tab": tab}
