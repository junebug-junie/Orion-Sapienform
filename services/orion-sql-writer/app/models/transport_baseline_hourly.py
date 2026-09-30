from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Index, Integer, String
from sqlalchemy.sql import func

from app.db import Base


class TransportBaselineHourlySQL(Base):
    """One row per (service, instance, hop, UTC hour) of orion-equilibrium-service's
    per-hop transport baseline gate (TransportBaselineHourlyV1,
    orion/schemas/telemetry/transport_baseline_hourly.py). Single writer: sql-writer off
    ``orion:equilibrium:transport_baseline:hourly``. Read by
    scripts/analysis/grade_transport_baseline.py (spec 2026-09-24 acceptance check 1).
    A restart mid-hour gives two rows for that hour (flush_reason shutdown + hour_end).
    ~1,500 rows/day at ~64 hops; no retention policy yet (small, and the grading window
    is weeks)."""

    __tablename__ = "transport_baseline_hourly"

    summary_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    service = Column(String, nullable=False)
    instance = Column(String, nullable=True)
    key = Column(String, nullable=False)
    hour_start = Column(DateTime(timezone=True), nullable=False)
    flush_reason = Column(String, nullable=False)
    flushed_at = Column(DateTime(timezone=True), nullable=False)
    windows_seen = Column(Integer, nullable=False)
    windows_evaluated = Column(Integer, nullable=False)
    success_count = Column(Integer, nullable=False, default=0)
    timeout_count = Column(Integer, nullable=False, default=0)
    z_p50 = Column(Float, nullable=True)
    z_p90 = Column(Float, nullable=True)
    saturation_ratio_p50 = Column(Float, nullable=True)
    baseline_ms = Column(Float, nullable=True)
    floor_ms_start = Column(Float, nullable=True)
    floor_ms = Column(Float, nullable=True)
    calls_per_min_mean = Column(Float, nullable=False, default=0.0)
    conditions_opened = Column(JSON, nullable=True)
    open_at_hour_end = Column(JSON, nullable=True)
    would_emit_by_condition = Column(JSON, nullable=True)
    excluded = Column(Boolean, nullable=False, default=False)
    warm = Column(Boolean, nullable=False, default=False)
    warm_at_start = Column(Boolean, nullable=False, default=False)
    emit_effective = Column(Boolean, nullable=False, default=False)
    config_fingerprint = Column(String, nullable=False)

    __table_args__ = (
        Index("idx_transport_baseline_hourly_hour", "hour_start"),
        Index("idx_transport_baseline_hourly_key_hour", "service", "instance", "key", "hour_start"),
    )
