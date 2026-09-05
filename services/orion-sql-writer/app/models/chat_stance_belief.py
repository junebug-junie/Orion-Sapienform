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

    __table_args__ = (
        Index("idx_chat_stance_belief_log_created_at", "created_at"),
        Index("idx_chat_stance_belief_log_session_id", "session_id"),
        Index("idx_chat_stance_belief_log_correlation_id", "correlation_id"),
    )
