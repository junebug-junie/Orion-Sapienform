from sqlalchemy import Column, DateTime, Float, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import deferred
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
    # Which camera (2026-09-24, walkway camera). NULL on rows written before
    # the column existed -- all room-camera rows. Added at boot by main.py
    # (ALTER ... ADD COLUMN IF NOT EXISTS) and by the walkway migration.
    # Deferred + server_default + no eager defaults: never in a SELECT, and
    # only in an INSERT when actually set, so a DB without the column still
    # accepts writes that omit it (worker.py strips it until boot confirms the
    # column exists). Verified against postgres:16 with and without the column.
    stream_id = deferred(Column(String, nullable=True, server_default=text("NULL")))

    __mapper_args__ = {"eager_defaults": False}
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
