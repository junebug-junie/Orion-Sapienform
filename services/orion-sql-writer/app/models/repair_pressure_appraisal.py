from sqlalchemy import JSON, Column, DateTime, Float, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.db import Base

# Generic JSON on SQLite (tests), JSONB on Postgres (prod) -- same pattern as
# app/models/metacog_entry.py.
_JSONB = JSON().with_variant(JSONB(), "postgresql")


class RepairPressureAppraisalLog(Base):
    """Durable log of every repair_pressure_v2 appraisal, gated or not.

    Insert-only, standalone table -- not a patch onto chat_history_log (which
    would race the row-creation timing, since repair_pressure computes
    pre-turn, before that row necessarily exists). See
    orion/schemas/repair_pressure_appraisal.py and
    docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md.
    """

    __tablename__ = "repair_pressure_appraisal_log"

    id = Column(String, primary_key=True, index=True)
    correlation_id = Column(String, index=True, nullable=True)
    created_at = Column(String, nullable=True)
    inserted_at = Column(DateTime(timezone=True), server_default=func.now())

    level = Column(Float, nullable=True)
    level_label = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    evidence = Column(_JSONB, default=list)
    behavior_applied = Column(String, nullable=True)
