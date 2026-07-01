from sqlalchemy import Column, DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db import Base


class VisionEventSQL(Base):
    __tablename__ = "vision_events"

    event_id = Column(String, primary_key=True, index=True)
    event_type = Column(String, nullable=True)
    narrative = Column(Text, nullable=True)
    entities = Column(JSONB, default=list)
    tags = Column(JSONB, default=list)
    confidence = Column(Float, nullable=True)
    salience = Column(Float, nullable=True)
    evidence_refs = Column(JSONB, default=list)
    correlation_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
