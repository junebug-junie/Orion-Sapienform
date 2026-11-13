"""Audit of Claude's tool calls in Hub chat (Patch 1b of
docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md).

Writer: sql-writer off ``orion:claude:tool:audit`` (``ClaudeToolCallV1``; producer: the broker in
orion-harness-governor). One row per phase of a call: ``ask`` (before Orion is asked), ``answer``
(Orion's answer, or the timeout) and ``result`` (what Claude got). PK ``event_id``
(``<call_id>:<phase>``), so a redelivered event is one row. Read by Hub's
``GET /api/claude-tools/audit``.

Table and indexes: ``services/orion-sql-db/manual_migration_claude_tool_call_log_v1.sql`` (jsonb
there; generic JSON here so the sqlite write-path test runs). Index names match, so create_all at boot
and the migration build the same thing.
"""

from sqlalchemy import JSON, Boolean, Column, DateTime, Index, Integer, String, Text
from sqlalchemy.sql import func

from app.db import Base


class ClaudeToolCallLogSQL(Base):
    __tablename__ = "claude_tool_call_log"

    event_id = Column(String, primary_key=True)
    phase = Column(String, nullable=False)
    call_id = Column(String, nullable=False)
    turn_request_id = Column(String, nullable=False)
    room_id = Column(String, nullable=False)
    claude_session_id = Column(String, nullable=False)
    hub_session_id = Column(String, nullable=True)
    trigger = Column(String, nullable=True)
    tool = Column(String, nullable=False)
    category = Column(String, nullable=True)
    arguments = Column(JSON, nullable=True)
    reason = Column(Text, nullable=True)
    setting = Column(String, nullable=True)
    decision_path = Column(String, nullable=True)
    shared = Column(Boolean, nullable=True)
    orion_answer = Column(String, nullable=True)
    orion_scope = Column(String, nullable=True)
    orion_reason = Column(Text, nullable=True)
    consent_path = Column(String, nullable=True)
    consent_latency_ms = Column(Integer, nullable=True)
    act_id = Column(String, nullable=True)
    result_chars = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    written_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_claude_tool_call_log_created", "created_at"),
        Index("idx_claude_tool_call_log_call", "call_id"),
        Index("idx_claude_tool_call_log_turn", "turn_request_id"),
        Index("idx_claude_tool_call_log_category_created", "category", "created_at"),
    )
