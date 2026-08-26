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
            # ROADMAP B1. This MUST be applied before the writer serves traffic: _write_row
            # filters payload keys against the mapper's columns, so the moment
            # BiometricsSummarySQL declares `measurements` the key enters the INSERT. Against
            # a database without the column that is psycopg2 UndefinedColumn -> ProgrammingError,
            # and the only handlers here catch IntegrityError -- so ALL biometrics-summary
            # persistence would stop, not just the new field, while the bus publish and the Hub
            # panel keep looking healthy. Boot-time DDL is the existing convention for exactly
            # this hazard (the chat_* statements below), so it lives here rather than in a
            # hand-applied file an operator has to remember.
            conn.exec_driver_sql(
                "ALTER TABLE IF EXISTS orion_biometrics_summary "
                "ADD COLUMN IF NOT EXISTS measurements JSONB;"
            )
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
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS llm_uncertainty_source TEXT;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS llm_mean_logprob DOUBLE PRECISION;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS llm_min_logprob DOUBLE PRECISION;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS llm_mean_top1_margin DOUBLE PRECISION;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS llm_low_margin_token_count INTEGER;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS llm_low_logprob_token_count INTEGER;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS llm_unstable_span_count INTEGER;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS llm_uncertainty_available BOOLEAN;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS response_identity TEXT;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE IF EXISTS aitown_chat_history_log ADD COLUMN IF NOT EXISTS response_identity TEXT;"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_chat_history_log_mem_status ON chat_history_log (memory_status);"
            )
            # 2026-08-26: JuniperMultimodalAffectSQL gained chat_correlation_id
            # (Hub's per-chat-turn affect bracket). Same hazard the ROADMAP B1
            # comment at the top of this block spells out: the moment the model
            # declares the column, _write_row's column-filter lets the key into
            # the INSERT, and against a database lacking it that is an
            # UndefinedColumn ProgrammingError -- which the handlers here do not
            # catch, so ALL juniper_multimodal_affect_log persistence would stop
            # while the bus publish and the Hub panel kept looking healthy.
            conn.exec_driver_sql(
                "ALTER TABLE IF EXISTS juniper_multimodal_affect_log "
                "ADD COLUMN IF NOT EXISTS chat_correlation_id TEXT;"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_juniper_multimodal_affect_chat_corr "
                "ON juniper_multimodal_affect_log (chat_correlation_id);"
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
                    target_artifact_ref TEXT NULL,
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
                "ALTER TABLE journal_entry_index ADD COLUMN IF NOT EXISTS llm_uncertainty JSONB;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE journal_entry_index ADD COLUMN IF NOT EXISTS llm_mean_logprob DOUBLE PRECISION;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE journal_entry_index ADD COLUMN IF NOT EXISTS llm_mean_top1_margin DOUBLE PRECISION;"
            )
            conn.exec_driver_sql(
                "ALTER TABLE journal_entry_index ADD COLUMN IF NOT EXISTS llm_unstable_span_count INTEGER;"
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
                CREATE TABLE IF NOT EXISTS mind_runs (
                    mind_run_id TEXT PRIMARY KEY,
                    correlation_id TEXT NOT NULL,
                    session_id TEXT NULL,
                    trigger TEXT NOT NULL,
                    ok BOOLEAN NOT NULL,
                    error_code TEXT NULL,
                    snapshot_hash TEXT NOT NULL DEFAULT '',
                    router_profile_id TEXT NOT NULL DEFAULT '',
                    result_jsonb JSONB NOT NULL,
                    request_summary_jsonb JSONB NOT NULL,
                    redaction_profile_id TEXT NULL,
                    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_mind_runs_correlation_id ON mind_runs (correlation_id);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_mind_runs_created_at ON mind_runs (created_at_utc DESC);"
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
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS action_outcomes (
                    action_id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    success BOOLEAN NULL,
                    surprise DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    observed_at TIMESTAMPTZ NULL,
                    correlation_id TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            # Composite covers the read query (WHERE subject=? ORDER BY observed_at DESC);
            # a standalone subject index would be redundant with this prefix.
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_action_outcomes_subject_observed_at ON action_outcomes (subject, observed_at DESC);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_action_outcomes_correlation_id ON action_outcomes (correlation_id);"
            )
            # goal_provenance_streak_ticks: debug-tier telemetry (2026-08-11) -- see
            # services/orion-sql-writer/app/models/dominance_streak_tick.py's docstring and
            # docs/superpowers/specs/2026-07-30-goal-system-remaining-gaps-design.md Part H.
            # High-volume (~1 row per real field tick); bounded by
            # goal_provenance_streak_ticks_retention_days (default 14, applied at boot below;
            # previously matched the now-removed drive_audits_retention_days' pattern).
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS goal_provenance_streak_ticks (
                    tick_telemetry_id TEXT PRIMARY KEY,
                    target_id TEXT NULL,
                    streak_count INTEGER NOT NULL,
                    min_streak_at_tick INTEGER NOT NULL,
                    qualified BOOLEAN NOT NULL,
                    source_field_tick_id TEXT NOT NULL,
                    source_attention_frame_id TEXT NOT NULL,
                    observed_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_goal_provenance_streak_ticks_observed_at ON goal_provenance_streak_ticks (observed_at DESC);"
            )
            # drive_audits table (CREATE/ALTER/INDEX boot DDL) removed
            # 2026-08-13, same patch as the Hub Drives Analytics tab removal
            # (docs/superpowers/pr-reports/2026-08-13-remove-hub-drives-
            # analytics-tab-pr.md). Review finding: the table had already
            # been dropped out-of-band (snapshotted first,
            # /tmp/drive_audits_drop_2026-08-13/), but this boot DDL was
            # left in place -- `CREATE TABLE IF NOT EXISTS` unconditionally
            # on every startup would have silently resurrected an empty
            # table on the next orion-sql-writer restart, undoing the drop.
            # The rest of the write-path wiring (worker.py's DriveAuditSQL
            # model/MODEL_MAP/INSERT_ONLY_MODELS entries, settings.py's
            # route-map + subscribe-channel-default entries, .env_example,
            # and orion/bus/channels.yaml's orion:memory:drives:audit entry)
            # was deliberately left in place at that time and fully removed
            # in a same-day follow-up patch (docs/superpowers/pr-reports/
            # 2026-08-13-untangle-drive-audit-sql-writer-pr.md).
            conn.exec_driver_sql(
                "ALTER TABLE bus_fallback_log ADD COLUMN IF NOT EXISTS created_at_ts TIMESTAMPTZ;"
            )
            conn.exec_driver_sql(
                """
                UPDATE bus_fallback_log
                SET created_at_ts = CASE
                  WHEN created_at IS NULL OR btrim(created_at) = '' THEN NULL
                  WHEN created_at ~ '^\\d{4}-' THEN created_at::timestamptz
                  WHEN created_at ~ '^\\d+(\\.\\d+)?$' THEN to_timestamp(created_at::double precision)
                  ELSE NULL
                END
                WHERE created_at_ts IS NULL;
                """
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_bus_fallback_log_created_at_ts ON bus_fallback_log (created_at_ts);"
            )
            conn.exec_driver_sql(
                """
                CREATE INDEX IF NOT EXISTS idx_bus_fallback_log_kind_created_at_ts
                ON bus_fallback_log (kind, created_at_ts);
                """
            )
            conn.exec_driver_sql(
                """
                DELETE FROM spark_telemetry st
                WHERE telemetry_id NOT IN (
                  SELECT DISTINCT ON (correlation_id) telemetry_id
                  FROM spark_telemetry
                  WHERE correlation_id IS NOT NULL AND btrim(correlation_id) <> ''
                  ORDER BY correlation_id, timestamp DESC NULLS LAST, telemetry_id ASC
                );
                """
            )
            conn.exec_driver_sql(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_spark_telemetry_correlation_id
                ON spark_telemetry (correlation_id);
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
        logger.info(
            "ℹ️ grammar index idx_grammar_events_source_created: apply via "
            "services/orion-sql-db/manual_migration_grammar_atlas.sql (CONCURRENTLY if table is large)"
        )
    except Exception as e:
        logger.warning("chat_message migration warning: %s", e)

    # Deliberately NOT inside the long bootstrap transaction above (review
    # finding). That block runs ~700 statements under one engine.begin() with a
    # single swallowing handler, so any earlier statement failing rolls the
    # whole thing back -- including this ALTER -- while
    # app/models/chat_response_feedback.py has ALREADY declared the column. The
    # ORM would then put target_artifact_ref in every INSERT, Postgres would
    # raise UndefinedColumn, only IntegrityError is handled downstream, and ALL
    # chat feedback persistence would stop with a green health check and one
    # warning line among hundreds. Its own transaction, its own handler.
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "ALTER TABLE chat_response_feedback "
                "ADD COLUMN IF NOT EXISTS target_artifact_ref TEXT NULL;"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_chat_response_feedback_artifact_ref "
                "ON chat_response_feedback (target_artifact_ref);"
            )
    except Exception as e:
        logger.warning("chat_response_feedback artifact_ref migration warning: %s", e)

    # drive_audits retention startup job removed 2026-08-13 (same patch that
    # fully untangled DriveAuditSQL's write path) -- the table and its boot
    # DDL are both gone, so a DELETE against it was dead weight even guarded
    # by the try/except. See settings.py for the matching field removal.

    goal_provenance_streak_ticks_retention_days = int(
        getattr(settings, "goal_provenance_streak_ticks_retention_days", 0) or 0
    )
    if goal_provenance_streak_ticks_retention_days > 0:
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql(
                    "DELETE FROM goal_provenance_streak_ticks WHERE observed_at < (NOW() - (%s || ' days')::INTERVAL);",
                    (str(goal_provenance_streak_ticks_retention_days),),
                )
            logger.info(
                "🧹 Applied goal_provenance_streak_ticks retention window=%s days",
                goal_provenance_streak_ticks_retention_days,
            )
        except Exception as exc:
            logger.warning("goal_provenance_streak_ticks retention startup failed (continuing boot): %s", exc)

    # NO STARTUP RETENTION PASS. Deliberate, 2026-08-20 -- this used to be four synchronous
    # blocking blocks here (grammar_events, grammar_edges, grammar_atoms,
    # substrate_organ_emissions), each with 100-batch/120s caps.
    #
    # Three problems, all measured live:
    #   1. It ran on the event loop, before `svc.start()` below, so it delayed not just
    #      readiness but the BUS SUBSCRIPTION, which is a far worse failure than retention
    #      starting a minute late. The ~260s figure is carried forward from
    #      docs/superpowers/pr-reports/2026-08-20-grammar-retention-periodic-pr.md ("observed
    #      ~260s across four tables"), not re-measured here -- the old container is gone. What
    #      IS measured on this branch: boot is now 4.6s (container start 17:34:15.209 ->
    #      "Application startup complete" 17:34:19.844).
    #   2. It could not converge anyway. Retention that runs only at process start deletes a
    #      fixed amount per restart against continuous arrival; that is the entire reason
    #      app/grammar_retention_loop.py exists.
    #   3. It was drifting out of step with the periodic path. It hand-listed four tables
    #      while GRAMMAR_RETENTION_TABLES had six, so grammar_traces and
    #      substrate_proposal_frames were startup-exempt by omission and the divergence had to
    #      be pinned by a test rather than being impossible.
    #
    # The periodic loop reaches the same steady state within one interval of boot and, with
    # the cycle budget below, is strictly more capable than the startup pass ever was. One
    # retention path, not two.
    task: asyncio.Task | None = None
    if settings.orion_bus_enabled:
        svc = build_hunter()
        logger.info("🚀 starting Hunter")
        logger.info("🧲 sql-writer subscribing to channels: %s", settings.effective_subscribe_channels)
        task = asyncio.create_task(svc.start())
    else:
        logger.warning("Bus disabled; writer will be idle.")

    # Backlog watcher for bus_fallback_log. Independent of orion_bus_enabled on
    # purpose: the rows it reads are already in Postgres, and a writer with the
    # bus disabled still has a backlog worth reporting.
    watch_task: asyncio.Task | None = None
    if settings.sql_writer_fallback_watch_enabled:
        from app.fallback_watch import fallback_watch_loop

        watch_task = asyncio.create_task(fallback_watch_loop(settings))
    else:
        logger.warning(
            "bus_fallback_log backlog watcher DISABLED "
            "(SQL_WRITER_FALLBACK_WATCH_ENABLED=false); unrouted events will accumulate silently"
        )

    # Periodic retention. Retention itself already existed and worked per-run; its only
    # trigger was process start, so it could never converge against continuous arrival.
    # See app/grammar_retention_loop.py for the measured numbers.
    retention_task: asyncio.Task | None = None
    if float(getattr(settings, "grammar_retention_interval_sec", 0.0) or 0.0) > 0:
        from app.grammar_retention_loop import grammar_retention_loop

        retention_task = asyncio.create_task(grammar_retention_loop(settings))
    else:
        logger.warning(
            "periodic grammar retention DISABLED (GRAMMAR_RETENTION_INTERVAL_SEC=0); "
            "retention runs only at startup, which cannot keep up with arrival"
        )

    # Object-permanence sweep -- see app/vision_object_permanence.py. Timer-
    # driven because a departure is a non-event: nothing frame-triggered can
    # ever detect one.
    vision_permanence_task: asyncio.Task | None = None
    if float(getattr(settings, "vision_permanence_sweep_interval_sec", 0.0) or 0.0) > 0:
        from app.vision_object_permanence_loop import vision_object_permanence_loop

        vision_permanence_task = asyncio.create_task(vision_object_permanence_loop(settings))
    else:
        logger.info(
            "vision object-permanence sweep DISABLED (VISION_PERMANENCE_SWEEP_INTERVAL_SEC=0)"
        )

    try:
        yield
    finally:
        pending = [
            t for t in (task, watch_task, retention_task, vision_permanence_task)
            if t is not None
        ]
        for background in pending:
            background.cancel()
        if pending:
            # gather(return_exceptions=True) rather than a loop of
            # `suppress(Exception)`: CancelledError derives from BaseException,
            # not Exception, so suppress(Exception) does NOT catch it. The bus
            # chassis's start() has no CancelledError handler, so awaiting it
            # re-raised out of the loop and the SECOND task was never cancelled
            # at all -- left pending at loop close, possibly mid-DB-query.
            await asyncio.gather(*pending, return_exceptions=True)


app = FastAPI(
    title=settings.service_name,
    version=settings.service_version,
    lifespan=lifespan,
)

app.include_router(notify_router, prefix="/api/notify-read", tags=["notify-read"])

@app.get("/health")
def health():
    from app.grammar_truth import build_grammar_truth_snapshot

    snap = build_grammar_truth_snapshot()
    return {
        "ok": snap["ok"],
        "degraded": snap["degraded"],
        "service": settings.service_name,
        "version": settings.service_version,
    }


@app.get("/grammar/truth")
def grammar_truth():
    from app.grammar_truth import build_grammar_truth_snapshot

    return build_grammar_truth_snapshot()
