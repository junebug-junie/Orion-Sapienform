from sqlalchemy import JSON, Column, DateTime, Float, Index, Integer, String
from sqlalchemy.sql import func

from app.db import Base


class GpuPoolEventSQL(Base):
    """One row per GPU pool lease/pool fact (GpuPoolEventV1, orion/schemas/gpu_pool.py).
    Single writer: orion-sql-writer off `orion:gpu_pool:event`. The pool's own projection
    (gpu_pool_leases) is live state; this is the history the Hub panel's historical views
    and the lease walker's "what happened around it" read."""

    __tablename__ = "gpu_pool_events"

    event_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    generated_at = Column(DateTime(timezone=True), nullable=False)
    event = Column(String, nullable=False)
    lease_id = Column(String, nullable=True)
    holder = Column(String, nullable=True)
    work_class = Column(String, nullable=True)
    priority = Column(String, nullable=True)
    role = Column(String, nullable=True)
    cards = Column(JSON, nullable=True)
    turn_correlation_id = Column(String, nullable=True)
    attempt = Column(Integer, nullable=True)
    waited_ms = Column(Float, nullable=True)
    held_ms = Column(Float, nullable=True)
    reason = Column(String, nullable=True)
    detail = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_gpu_pool_events_generated", "generated_at"),
        Index("idx_gpu_pool_events_lease", "lease_id", "generated_at"),
        Index("idx_gpu_pool_events_class_generated", "work_class", "generated_at"),
    )
