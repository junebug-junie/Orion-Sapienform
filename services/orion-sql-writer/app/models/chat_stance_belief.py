from sqlalchemy import Column, DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db import Base


class ChatStanceBeliefLogSQL(Base):
    __tablename__ = "chat_stance_belief_log"

    entry_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    correlation_id = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    shift_kind = Column(String, nullable=True)
    anchor_summary = Column(Text, nullable=True)
    degraded_producers = Column(JSONB, nullable=True)
    lineage_summary = Column(Text, nullable=True)
    # Added 2026-09-10 -- see orion/schemas/chat_stance_belief.py's docstring
    # for why these are written from a separate call site/row than the
    # fields above. Boot-time ALTER TABLE lives in app/main.py's lifespan
    # (same ADD COLUMN IF NOT EXISTS convention as every other table here) --
    # this table already exists live, so create_all() alone will not add
    # these columns to it.
    interaction_regime = Column(String, nullable=True)
    task_mode = Column(String, nullable=True)

    __table_args__ = (
        Index("idx_chat_stance_belief_log_created_at", "created_at"),
        Index("idx_chat_stance_belief_log_session_id", "session_id"),
        Index("idx_chat_stance_belief_log_correlation_id", "correlation_id"),
    )
