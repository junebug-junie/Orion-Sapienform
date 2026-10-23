"""Minimal history of the existing state broadcast; no lease bodies or prompts."""
from datetime import timezone
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import JSON, Column, DateTime, Index, String
from sqlalchemy.sql import func

from app.db import Base
from orion.gpu_pool.shed import REFLEX_REASONS, deferrable_queued, shed_held_queue


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
    # Of queue_depth, what an active heat-reflex shed (REFLEX_REASONS: cabinet_hot, cabinet_unknown,
    # cooling_incident; not orion_self_shed) is holding back: {work_class: n}, and the reasons
    # (``+``-joined). NULL on rows from before 2026-10-20 or from a pool that sends no shed view:
    # unknown, not zero. manual_migration_regulation_gpu_shed_held_v1.sql.
    shed_held_depth = Column(JSON, nullable=True)
    shed_reason = Column(String, nullable=True)
    # Of queue_depth, Orion's own deferrable self-work: queued background HOLDS with no parent hold
    # (orion.gpu_pool.shed.deferrable_queued), {work_class: n}, never counting a lease already in
    # shed_held_depth. Arousal's S2 reads it as hunger, not load (body-economy S2 decision,
    # 2026-10-23). NULL on rows from before 2026-10-23, or a snapshot with a queue but no lease
    # rows: unknown. manual_migration_regulation_gpu_deferrable_depth_v1.sql.
    deferrable_depth = Column(JSON, nullable=True)
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
    # A pool with no shed lever sends shed={} (the schema default): unknown, stored as NULL.
    held, reason = (shed_held_queue(state.leases, state.shed.get("blocked"), reasons=REFLEX_REASONS)
                    if state.shed else (None, None))
    queued_rows = any(getattr(r, "status", None) == "queued" for r in state.leases)
    deferrable = (deferrable_queued(state.leases, state.shed.get("blocked") if state.shed else None,
                                    reasons=REFLEX_REASONS)
                  if queued_rows or not any(state.queue_depth.values()) else None)
    # Redelivery is the same observation even if the envelope id changes.
    identity = f"gpu_pool.state.v1:{state.host}:{at.isoformat()}:{state.config_digest}"
    return dict(snapshot_id=str(uuid5(NAMESPACE_URL, identity)), generated_at=at,
                host=state.host, mode=state.mode, config_digest=state.config_digest,
                backlog_depth=state.backlog_depth, queue_depth=state.queue_depth,
                shed_held_depth=held, shed_reason=reason,
                deferrable_depth=deferrable)
