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
                CREATE TABLE IF NOT EXISTS chat_response_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    target_turn_id TEXT NULL,
                    target_message_id TEXT NULL,
                    target_correlation_id TEXT NULL,
                    target_key TEXT NULL,
                    session_id TEXT NULL,
                    user_id TEXT NULL,
                    feedback_value TEXT NOT NULL,
                    categories TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                    free_text TEXT NULL,
                    source TEXT NULL,
                    ui_context JSONB NULL,
                    submission_fingerprint TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_chat_response_feedback_turn_id ON chat_response_feedback (target_turn_id);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_chat_response_feedback_message_id ON chat_response_feedback (target_message_id);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_chat_response_feedback_corr_id ON chat_response_feedback (target_correlation_id);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_chat_response_feedback_target_key ON chat_response_feedback (target_key);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_chat_response_feedback_session_id ON chat_response_feedback (session_id);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_chat_response_feedback_user_id ON chat_response_feedback (user_id);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_chat_response_feedback_value ON chat_response_feedback (feedback_value);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_chat_response_feedback_created_at ON chat_response_feedback (created_at);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_chat_response_feedback_fingerprint ON chat_response_feedback (submission_fingerprint);"
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
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS journal_entry_index (
                    entry_id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL,
                    author TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    title TEXT NULL,
                    body TEXT NOT NULL,
                    source_kind TEXT NULL,
                    source_ref TEXT NULL,
                    correlation_id TEXT NULL,
                    trigger_kind TEXT NULL,
                    trigger_summary TEXT NULL,
                    conversation_frame TEXT NULL,
                    task_mode TEXT NULL,
                    identity_salience TEXT NULL,
                    answer_strategy TEXT NULL,
                    stance_summary TEXT NULL,
                    active_identity_facets JSONB NULL,
                    active_growth_axes JSONB NULL,
                    active_relationship_facets JSONB NULL,
                    social_posture JSONB NULL,
                    reflective_themes JSONB NULL,
                    active_tensions JSONB NULL,
                    dream_motifs JSONB NULL,
                    response_hazards JSONB NULL
                );
                """
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_journal_entry_index_created_at ON journal_entry_index (created_at);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_journal_entry_index_mode ON journal_entry_index (mode);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_journal_entry_index_source_kind ON journal_entry_index (source_kind);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_journal_entry_index_trigger_kind ON journal_entry_index (trigger_kind);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_journal_entry_index_author ON journal_entry_index (author);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_journal_entry_index_source_ref ON journal_entry_index (source_ref);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_journal_entry_index_correlation_id ON journal_entry_index (correlation_id);"
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS evidence_units (
                    unit_id TEXT PRIMARY KEY,
                    unit_kind TEXT NOT NULL,
                    source_family TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    correlation_id TEXT NULL,
                    parent_unit_id TEXT NULL,
                    sibling_prev_id TEXT NULL,
                    sibling_next_id TEXT NULL,
                    title TEXT NULL,
                    summary TEXT NULL,
                    body TEXT NULL,
                    facets JSONB NOT NULL DEFAULT '[]'::jsonb,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NULL
                );
                """
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_evidence_units_created_at ON evidence_units (created_at);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_evidence_units_unit_kind ON evidence_units (unit_kind);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_evidence_units_source_family ON evidence_units (source_family);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_evidence_units_source_kind ON evidence_units (source_kind);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_evidence_units_source_ref ON evidence_units (source_ref);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_evidence_units_correlation_id ON evidence_units (correlation_id);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_evidence_units_parent_unit_id ON evidence_units (parent_unit_id);"
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS world_pulse_run (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    date TEXT NOT NULL,
                    requested_by TEXT NULL,
                    dry_run TEXT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    schema_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NULL
                );
                """
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS world_pulse_digest (
                    run_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    date TEXT NOT NULL,
                    executive_summary TEXT NOT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    schema_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS world_pulse_digest_item (
                    item_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    confidence TEXT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    schema_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_world_pulse_digest_item_run_id ON world_pulse_digest_item (run_id);")
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS world_pulse_article (
                    article_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    schema_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_world_pulse_article_run_id ON world_pulse_article (run_id);")
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_world_pulse_article_source_id ON world_pulse_article (source_id);")
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS world_pulse_article_cluster (
                    cluster_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    article_count TEXT NOT NULL DEFAULT '0',
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    schema_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_world_pulse_cluster_run_id ON world_pulse_article_cluster (run_id);")
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_world_pulse_cluster_category ON world_pulse_article_cluster (category);")
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS world_pulse_claim (
                    claim_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    topic_id TEXT NULL,
                    promotion_status TEXT NOT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    schema_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS world_pulse_event (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    schema_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS world_pulse_entity (
                    entity_id TEXT PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    schema_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS world_pulse_situation_brief (
                    topic_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    tracking_status TEXT NOT NULL,
                    current_assessment TEXT NOT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    schema_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS world_pulse_situation_change (
                    change_id TEXT PRIMARY KEY,
                    topic_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    schema_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS world_pulse_learning_delta (
                    learning_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    schema_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS world_pulse_worth_reading (
                    reading_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    schema_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS world_pulse_worth_watching (
                    watch_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    schema_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS world_pulse_context_capsule (
                    capsule_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    schema_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS world_pulse_publish_status (
                    status_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    state TEXT NOT NULL,
                    detail TEXT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    schema_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS world_pulse_hub_message (
                    message_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    date TEXT NOT NULL,
                    executive_summary TEXT NOT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    schema_version TEXT NOT NULL DEFAULT 'v1',
                    created_at TIMESTAMPTZ NOT NULL
                );
                """
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
