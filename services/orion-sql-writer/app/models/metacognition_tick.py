from sqlalchemy import Column, DateTime, Float, Integer, JSON, String

from app.db import Base


class MetacognitionTickSQL(Base):
    __tablename__ = "metacognition_ticks"

    # Tick id is the natural primary key for telemetry rows
    tick_id = Column(String, primary_key=True, index=True)

    # For joins / trace linkage (optional but helpful)
    correlation_id = Column(String, index=True, nullable=True)

    source_service = Column(String, nullable=True)
    source_node = Column(String, nullable=True)

    distress_score = Column(Float, nullable=True)
    zen_score = Column(Float, nullable=True)
    services_tracked = Column(Integer, nullable=False, default=0)

    # When created
    generated_at = Column(DateTime(timezone=True), nullable=True)

    # Raw structured tick snapshot
    snapshot = Column(JSON, nullable=True)
