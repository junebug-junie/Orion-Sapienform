from sqlalchemy import Boolean, Column, DateTime, Float, String
from sqlalchemy.sql import func

from app.db import Base


class ActionOutcomeSQL(Base):
    """Durable record of an autonomous action outcome (e.g. readonly web fetch).

    Produced by orion-spark-concept-induction via `action.outcome.emit.v1` and
    read back per-subject by orion-cortex-exec's chat stance pipeline. `action_id`
    is the primary key so re-delivered events upsert idempotently.
    """

    __tablename__ = "action_outcomes"

    action_id = Column(String, primary_key=True)
    subject = Column(String, index=True, nullable=False)
    kind = Column(String, nullable=False)
    summary = Column(String, nullable=False)
    success = Column(Boolean, nullable=True)
    surprise = Column(Float, nullable=False, default=0.0)
    observed_at = Column(DateTime(timezone=True), index=True, nullable=True)
    correlation_id = Column(String, index=True, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
