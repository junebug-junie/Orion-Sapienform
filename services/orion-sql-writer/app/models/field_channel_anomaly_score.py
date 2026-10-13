from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Index, Integer, String
from sqlalchemy.sql import func

from app.db import Base


class FieldChannelAnomalyScoreSQL(Base):
    """Every mood-arc anomaly score orion-field-digester publishes (flagged or
    not), one row per 60s scoring window (FieldChannelAnomalyScoreV1 on
    ``orion:field_channel:anomaly_score``). Single writer: sql-writer.

    Exists so the telemetry_anomaly trigger threshold (recon_error_p95 x
    EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_THRESHOLD_MULTIPLIER, an uncalibrated
    3.0) can be fit from the full recon_loss distribution: before this table
    only flagged windows were stored (as metacog_trigger rows), so the calm
    side of the distribution was never seen. ~1,440 rows/day; insert-only;
    created by ``Base.metadata.create_all`` at boot."""

    __tablename__ = "field_channel_anomaly_score"

    correlation_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    encoder_id = Column(String, nullable=False)
    encoder_version = Column(String, nullable=False)
    recon_loss = Column(Float, nullable=False)
    recon_error_p95 = Column(Float, nullable=False)
    threshold_multiplier = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    anomalous = Column(Boolean, nullable=False)
    window_start = Column(DateTime(timezone=True), nullable=False)
    window_end = Column(DateTime(timezone=True), nullable=False)
    window_size = Column(Integer, nullable=False)
    top_channels = Column(JSON, nullable=True)
    mean_signed_deviation = Column(Float, nullable=True)
    deviation_direction = Column(String, nullable=True)
    attribution_ok = Column(Boolean, nullable=True)

    __table_args__ = (
        Index("idx_field_channel_anomaly_score_window_end", "window_end"),
        Index("idx_field_channel_anomaly_score_encoder_window", "encoder_id", "window_end"),
    )
