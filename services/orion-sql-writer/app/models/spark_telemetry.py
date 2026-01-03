from sqlalchemy import Column, String, Float, JSON, DateTime
from sqlalchemy.sql import func
from app.db import Base

class SparkTelemetrySQL(Base):
    __tablename__ = "spark_telemetry"

    # Use correlation_id as primary key (one telemetry point per trace)
    correlation_id = Column(String, primary_key=True)
    
    # Core Metrics
    phi = Column(Float, nullable=True)
    novelty = Column(Float, nullable=True)
    
    # Context
    trace_mode = Column(String, nullable=True)
    trace_verb = Column(String, nullable=True)
    stimulus_summary = Column(String, nullable=True)
    
    # Full stats (valence, energy, etc)
    metadata_ = Column("metadata", JSON, nullable=True)
    
    timestamp = Column(DateTime, server_default=func.now())
