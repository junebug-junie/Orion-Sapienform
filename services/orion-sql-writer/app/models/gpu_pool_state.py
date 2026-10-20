"""Minimal history of the existing state broadcast; no lease bodies or prompts."""
from datetime import timezone
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import JSON, Column, DateTime, Index, String
from sqlalchemy.sql import func

from app.db import Base
from orion.gpu_pool.shed import shed_held_queue


class GpuPoolStateSQL(Base):
    __tablename__ = "gpu_pool_state_history"

    snapshot_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    generated_at = Column(DateTime(timezone=True), nullable=False)
    host = Column(String, nullable=False)
    mode = Column(String, nullable=False)
    config_digest = Column(String, nullable=False)
    backlog_depth = Column(JSON, nullable=False)
    queue_depth = Column(JSON, nullable=False)
    # Of queue_depth, what an active shed (heat) is holding back: {work_class: n}, and the reasons
    # (``+``-joined). NULL on rows from before 2026-10-20 or from a pool that sends no shed view:
    # unknown, not zero. manual_migration_regulation_gpu_shed_held_v1.sql.
    shed_held_depth = Column(JSON, nullable=True)
    shed_reason = Column(String, nullable=True)
    __table_args__ = (
        Index("idx_gpu_pool_state_generated", "host", "generated_at"),
        Index("idx_gpu_pool_state_created", "created_at"),
    )


def state_history_row(state):
    """Missing fields cannot turn into fresh, empty backlog via schema defaults."""
    required = {"generated_at", "host", "backlog_depth", "queue_depth"}
    if not required <= state.model_fields_set or not state.host or state.generated_at.tzinfo is None:
        raise ValueError("GPU history needs explicit source, aware timestamp and depths")
    for depths in (state.backlog_depth, state.queue_depth):
        if any(type(n) is not int or n < 0 for n in depths.values()):
            raise ValueError("GPU history depths must be nonnegative integers")
    at = state.generated_at.astimezone(timezone.utc)
    # The held count needs both the shed view and the lease list; without either it is unknown.
    held, reason = (shed_held_queue(state.leases, state.shed.get("blocked"))
                    if {"shed", "leases"} <= state.model_fields_set and state.shed else (None, None))
    # Redelivery is the same observation even if the envelope id changes.
    identity = f"gpu_pool.state.v1:{state.host}:{at.isoformat()}:{state.config_digest}"
    return dict(snapshot_id=str(uuid5(NAMESPACE_URL, identity)), generated_at=at,
                host=state.host, mode=state.mode, config_digest=state.config_digest,
                backlog_depth=state.backlog_depth, queue_depth=state.queue_depth,
                shed_held_depth=held, shed_reason=reason)
