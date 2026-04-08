from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db import Base, engine
from app.settings import settings
from app.worker import build_hunter
from app.api_notify import router as notify_router

logging.basicConfig(
    level=logging.INFO,
    format="[SQL_WRITER] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("sql-writer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure schema exists
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("🛠️  Ensured DB schema is present")
    except Exception as e:
        logger.warning("Schema init warning: %s", e)

    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS correlation_id TEXT;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS trace_id TEXT;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS memory_status TEXT;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS memory_tier TEXT;"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_chat_message_corr_id ON chat_message (correlation_id);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_chat_message_corr_role_ts ON chat_message (correlation_id, role, timestamp);"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS memory_status TEXT;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS memory_tier TEXT;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS memory_reason TEXT;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS thought_process TEXT;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS client_meta JSONB;"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_chat_history_log_mem_status ON chat_history_log (memory_status);"
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS orion_metacognitive_trace (
                    trace_id TEXT PRIMARY KEY,
                    correlation_id TEXT NOT NULL,
                    session_id TEXT NULL,
                    message_id TEXT NULL,
                    trace_role TEXT NOT NULL,
                    trace_stage TEXT NOT NULL,
                    content TEXT NOT NULL,
                    model TEXT NOT NULL,
                    token_count INT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_orion_metacog_trace_corr ON orion_metacognitive_trace (correlation_id);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_orion_metacog_trace_created_at ON orion_metacognitive_trace (created_at);"
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS social_room_turns (
                    turn_id TEXT PRIMARY KEY,
                    correlation_id TEXT NULL,
                    session_id TEXT NULL,
                    user_id TEXT NULL,
                    source TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    text TEXT NOT NULL,
                    recall_profile TEXT NULL,
                    trace_verb TEXT NULL,
                    tags JSONB NULL,
                    concept_evidence JSONB NULL,
                    grounding_state JSONB NULL,
                    redaction JSONB NULL,
                    client_meta JSONB NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_social_room_turns_session_id ON social_room_turns (session_id);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_social_room_turns_corr_id ON social_room_turns (correlation_id);"
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS external_room_messages (
                    event_id TEXT PRIMARY KEY,
                    correlation_id TEXT NULL,
                    platform TEXT NOT NULL,
                    room_id TEXT NOT NULL,
                    thread_id TEXT NULL,
                    direction TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    transport_message_id TEXT NOT NULL,
                    reply_to_message_id TEXT NULL,
                    sender_id TEXT NOT NULL,
                    sender_name TEXT NULL,
                    sender_kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    transport_ts TEXT NULL,
                    raw_payload JSONB NULL,
                    metadata JSONB NULL,
                    delivery_ok BOOLEAN NULL,
                    delivery_error TEXT NULL,
                    skip_reason TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_external_room_messages_room_id ON external_room_messages (room_id);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_external_room_messages_transport_msg_id ON external_room_messages (transport_message_id);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_external_room_messages_corr_id ON external_room_messages (correlation_id);"
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS external_room_participants (
                    participant_ref TEXT PRIMARY KEY,
                    platform TEXT NOT NULL,
                    room_id TEXT NOT NULL,
                    participant_id TEXT NOT NULL,
                    participant_name TEXT NULL,
                    participant_kind TEXT NOT NULL,
                    last_message_id TEXT NULL,
                    last_seen_at TEXT NOT NULL,
                    metadata JSONB NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_external_room_participants_room_id ON external_room_participants (room_id);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_external_room_participants_platform ON external_room_participants (platform);"
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS journal_entries (
                    entry_id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    author TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    title TEXT NULL,
                    body TEXT NOT NULL,
                    source_kind TEXT NULL,
                    source_ref TEXT NULL,
                    correlation_id TEXT NULL
                );
                """
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_journal_entries_created_at ON journal_entries (created_at);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_journal_entries_correlation_id ON journal_entries (correlation_id);"
            )
        logger.info("🧬 chat_message correlation/trace columns ensured")
        retention_days = int(getattr(settings, "metacog_trace_retention_days", 0) or 0)
        if retention_days > 0:
            with engine.begin() as conn:
                conn.exec_driver_sql(
                    "DELETE FROM orion_metacognitive_trace WHERE created_at < (NOW() - (%s || ' days')::INTERVAL);",
                    (str(retention_days),),
                )
            logger.info("🧹 Applied metacog trace retention window=%s days", retention_days)
    except Exception as e:
        logger.warning("chat_message migration warning: %s", e)

    task: asyncio.Task | None = None
    if settings.orion_bus_enabled:
        svc = build_hunter()
        logger.info("🚀 starting Hunter")
        logger.info("🧲 sql-writer subscribing to channels: %s", settings.effective_subscribe_channels)
        task = asyncio.create_task(svc.start())
    else:
        logger.warning("Bus disabled; writer will be idle.")

    try:
        yield
    finally:
        if task:
            task.cancel()
            with contextlib.suppress(Exception):
                await task


app = FastAPI(
    title=settings.service_name,
    version=settings.service_version,
    lifespan=lifespan,
)

app.include_router(notify_router, prefix="/api/notify-read", tags=["notify-read"])

@app.get("/health")
def health():
    return {"ok": True, "service": settings.service_name, "version": settings.service_version}
