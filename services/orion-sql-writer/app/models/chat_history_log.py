from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base

class ChatHistoryLogSQL(Base):
    __tablename__ = "chat_history_log"

    id = Column(String, primary_key=True)
    trace_id = Column(String, index=True)
    correlation_id = Column(String, index=True, nullable=True)
    source = Column(String)
    prompt = Column(Text)
    response = Column(Text)
    user_id = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    spark_meta = Column(JSONB, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
