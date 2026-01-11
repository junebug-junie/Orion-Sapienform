from sqlalchemy import Column, DateTime, Float, Integer, JSON, String

from app.db import Base


class MetacognitionEnrichedSQL(Base):
    __tablename__ = "metacognition_enriched"

    # Same tick_id, now with enrichment
    tick_id = Column(String, primary_key=True, index=True)

    correlation_id = Column(String, index=True, nullable=True)

    source_service = Column(String, nullable=True)
    source_node = Column(String, nullable=True)

    distress_score = Column(Float, nullable=True)
    zen_score = Column(Float, nullable=True)
    services_tracked = Column(Integer, nullable=False, default=0)

    generated_at = Column(DateTime(timezone=True), nullable=True)

    tags = Column(JSON, nullable=True)
    entities = Column(JSON, nullable=True)

    # Full original tick for audit/debug
    raw_tick = Column(JSON, nullable=True)
