from sqlalchemy import Boolean, Column, DateTime, Float, String
from sqlalchemy.sql import func

from app.db import Base


class ActionOutcomeSQL(Base):
    """Durable record of an autonomous action outcome (e.g. readonly web fetch).

    Produced by orion-spark-concept-induction via `action.outcome.emit.v1` and
    read back per-subject by orion-cortex-exec's chat stance pipeline. `action_id`
    is the primary key so re-delivered events upsert idempotently.

    Secondary indexes are intentionally NOT declared here via `index=True`. They are
    owned by the boot DDL in `app/main.py` (and the standalone migration
    `services/orion-sql-db/manual_migration_action_outcomes_v1.sql`) so there is a
    single source of truth: a composite `(subject, observed_at DESC)` matching the
    read query plus a `correlation_id` lookup index. Declaring them here too would
    make `Base.metadata.create_all` emit duplicate `ix_*` indexes alongside the DDL's
    `idx_*` indexes on this write-hot table.
    """

    __tablename__ = "action_outcomes"

    action_id = Column(String, primary_key=True)
    subject = Column(String, nullable=False)
    kind = Column(String, nullable=False)
    summary = Column(String, nullable=False)
    success = Column(Boolean, nullable=True)
    surprise = Column(Float, nullable=False, default=0.0)
    observed_at = Column(DateTime(timezone=True), nullable=True)
    correlation_id = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
