from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, JSON, String
from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base

# Generic JSON on SQLite (tests), JSONB on Postgres (prod) -- same pattern as
# app/models/harness_turn_trace.py.
_JSONB = JSON().with_variant(JSONB(), "postgresql")


class CockpitTurnSightingSQL(Base):
    """Append-only hop row for one cockpit.hop.v1 publish.

    PK (correlation_id, seq) is the rewind cursor: Soft HUD reads
    ORDER BY seq and treats a missing canonical stage as a gap.
    Republish of the same (correlation_id, seq) is a no-op.
    """

    __tablename__ = "cockpit_turn_sighting"

    correlation_id = Column(String, primary_key=True)
    seq = Column(Integer, primary_key=True)
    ts = Column(DateTime, nullable=False)
    stage = Column(String, nullable=False)
    visor_line = Column(String, nullable=False)
    status = Column(String, nullable=False)
    summary = Column(_JSONB, nullable=False)
    raw = Column(_JSONB, nullable=False)
    producer = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
