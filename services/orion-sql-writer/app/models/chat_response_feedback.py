from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.sql import func

from app.db import Base


class ChatResponseFeedbackSQL(Base):
    __tablename__ = "chat_response_feedback"

    feedback_id = Column(String, primary_key=True)
    target_turn_id = Column(String, index=True, nullable=True)
    target_message_id = Column(String, index=True, nullable=True)
    target_correlation_id = Column(String, index=True, nullable=True)
    target_key = Column(String, index=True, nullable=True)
    session_id = Column(String, index=True, nullable=True)
    user_id = Column(String, index=True, nullable=True)
    feedback_value = Column(String, index=True, nullable=False)
    categories = Column(ARRAY(String), nullable=False, default=list)
    free_text = Column(Text, nullable=True)
    source = Column(String, nullable=True)
    ui_context = Column(JSONB, nullable=True)
    submission_fingerprint = Column(String, index=True, nullable=True)
    created_at = Column(DateTime, index=True, nullable=False, server_default=func.now())
