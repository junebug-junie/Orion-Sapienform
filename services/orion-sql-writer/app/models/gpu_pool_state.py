"""Minimal history of the existing state broadcast; no lease bodies or prompts."""
from datetime import timezone
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import JSON, Column, DateTime, Index, String
from sqlalchemy.sql import func

from app.db import Base


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
    # Redelivery is the same observation even if the envelope id changes.
    identity = f"gpu_pool.state.v1:{state.host}:{at.isoformat()}:{state.config_digest}"
    return dict(snapshot_id=str(uuid5(NAMESPACE_URL, identity)), generated_at=at,
                host=state.host, mode=state.mode, config_digest=state.config_digest,
                backlog_depth=state.backlog_depth, queue_depth=state.queue_depth)
