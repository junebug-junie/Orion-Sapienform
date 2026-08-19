from sqlalchemy import Boolean, Column, Float, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base


class AitownChatHistoryLogSQL(Base):
    """Mirror of ``ChatHistoryLogSQL`` (chat_history_log), for AI Town rows only.

    Track B of docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md
    -- physical table split. Phase 1 (PR #1734, 2026-08-19) shipped this
    table with an additive dual-write bridge (every AI-Town row landed in
    both tables). Retired the same day, with AI Town's own backend
    confirmed dead and zero concurrent-write risk: ``worker.py`` now
    *routes* each row to exactly one table
    (``_resolve_chat_history_model_cls``) instead of duplicating it. An
    AI-Town row (``client_meta -> 'external_room' ->> 'platform' == 'aitown'``,
    the same signal ``services/orion-recall/app/chat_source_tagging.py::chat_source_platform()``
    uses -- reimplemented locally rather than cross-imported, matching this
    repo's service-boundary convention (CLAUDE.md section 5), same choice
    ``services/orion-hub/scripts/concept_atlas_routes.py``'s ``_AITOWN_PLATFORM_TAG``
    already made for the same signal) now lands here ONLY, never in
    ``chat_history_log`` too.

    Column-for-column identical to ``ChatHistoryLogSQL`` so the same
    ``upsert_chat_history_row()``/conflict-merge logic can target either
    table via a ``model_cls`` parameter -- one source of truth for the
    concurrency-hardened merge semantics, not a second hand-copied
    implementation of it.
    """

    __tablename__ = "aitown_chat_history_log"

    id = Column(String, primary_key=True)
    correlation_id = Column(String, index=True, nullable=True)
    source = Column(String)
    prompt = Column(Text)
    response = Column(Text)
    user_id = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    spark_meta = Column(JSONB, nullable=True)
    memory_status = Column(String, index=True, nullable=True)
    memory_tier = Column(String, index=True, nullable=True)
    memory_reason = Column(String, nullable=True)
    thought_process = Column(Text, nullable=True)
    client_meta = Column(JSONB, nullable=True)
    llm_uncertainty_source = Column(String, nullable=True)
    llm_mean_logprob = Column(Float, nullable=True)
    llm_min_logprob = Column(Float, nullable=True)
    llm_mean_top1_margin = Column(Float, nullable=True)
    llm_low_margin_token_count = Column(Integer, nullable=True)
    llm_low_logprob_token_count = Column(Integer, nullable=True)
    llm_unstable_span_count = Column(Integer, nullable=True)
    llm_uncertainty_available = Column(Boolean, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
