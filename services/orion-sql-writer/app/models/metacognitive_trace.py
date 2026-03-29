from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

from app.db import Base


class MetacognitiveTraceSQL(Base):
    __tablename__ = "orion_metacognitive_trace"

    trace_id = Column(String, primary_key=True)
    correlation_id = Column(String, index=True, nullable=False)
    session_id = Column(String, index=True, nullable=True)
    message_id = Column(String, index=True, nullable=True)
    trace_role = Column(String, nullable=False)
    trace_stage = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    model = Column(String, nullable=False)
    token_count = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
