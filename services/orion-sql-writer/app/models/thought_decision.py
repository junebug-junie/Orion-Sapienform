from uuid import uuid4
from sqlalchemy import Boolean, Column, DateTime, Float, JSON, String
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base
from datetime import datetime

# Generic JSON on SQLite (tests), JSONB on Postgres (prod) -- same pattern as
# app/models/metacog_entry.py.
_JSONB = JSON().with_variant(JSONB(), "postgresql")


class ThoughtDecisionSQL(Base):
    """Durable record of a ThoughtEventV1 stance decision (proceed/defer/refuse).

    Deliberately narrow: consumes the full ThoughtEventV1 wire payload on
    orion:thought:artifact, but this model only declares the redacted subset
    of columns ThoughtDecisionRecordV1 defines (orion/schemas/thought.py) --
    _write_row() filters the validated payload down to whatever columns
    actually exist here, so imperative/tone/strain_refs/stance_harness_slice/
    grounding_capsule/autonomy_slice never reach this table.
    """

    __tablename__ = "thought_decision"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid4()))
    event_id = Column(String, index=True, nullable=True)
    correlation_id = Column(String, index=True, nullable=True)
    session_id = Column(String, index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    disposition = Column(String, index=True, nullable=False)
    disposition_reasons = Column(_JSONB, default=list)
    boundary_register = Column(Boolean, default=False)
    repair_pressure_level = Column(Float, nullable=True)
    trust_rupture_score = Column(Float, nullable=True)

    llm_profile = Column(String, nullable=True)
    producer = Column(String, nullable=True)
    model_id = Column(String, nullable=True)
