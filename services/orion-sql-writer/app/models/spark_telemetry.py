import uuid

from sqlalchemy import Column, DateTime, Float, JSON, String

from app.db import Base


class SparkTelemetrySQL(Base):
    __tablename__ = "spark_telemetry"

    # NOTE: We want each telemetry row to be uniquely addressable. We keep
    # correlation_id as an indexed foreign key / join key back to the trace.
    telemetry_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    correlation_id = Column(String, index=True, nullable=False)

    phi = Column(Float)
    novelty = Column(Float)
    trace_mode = Column(String)
    trace_verb = Column(String)
    stimulus_summary = Column(String)

    # "metadata" is a reserved attribute name on SQLAlchemy Declarative Base.
    # We map the DB column name -> python attribute name via "metadata_".
    metadata_ = Column("metadata", JSON)

    timestamp = Column(DateTime(timezone=True))
