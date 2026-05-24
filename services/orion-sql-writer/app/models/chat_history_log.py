from sqlalchemy import Boolean, Column, Float, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base

class ChatHistoryLogSQL(Base):
    __tablename__ = "chat_history_log"

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
