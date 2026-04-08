from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db import Base


class ChatResponseFeedbackSQL(Base):
    __tablename__ = "chat_response_feedback"

    feedback_id = Column(String, primary_key=True)
    correlation_id = Column(String, index=True, nullable=True)
    session_id = Column(String, index=True, nullable=True)
    user_id = Column(String, index=True, nullable=True)
    response_id = Column(String, index=True, nullable=True)
    rating = Column(String, nullable=False)
    feedback_text = Column(Text, nullable=True)
    tags = Column(JSONB, nullable=False)
    metadata = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    ingested_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
