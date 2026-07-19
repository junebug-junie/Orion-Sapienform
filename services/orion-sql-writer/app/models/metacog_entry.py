from sqlalchemy import Column, String, Text, Boolean, JSON
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base

# Generic JSON on SQLite (tests), JSONB on Postgres (prod) -- same pattern as
# app/models/drive_audit.py and app/models/causal_geometry_snapshot.py, declared
# once so Base.metadata.create_all and sqlite-backed tests agree with the type.
_JSONB = JSON().with_variant(JSONB(), "postgresql")


class MetacogEntry(Base):
    """Orion's machine-generated self-observation entries (real-artifact model).

    Genuinely separate from collapse_mirror (Juniper's manually-authored,
    strict-lane journal entries) -- not a v2/successor table, a different
    table for a different lane. See
    docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md.
    """

    __tablename__ = "orion_metacog"

    id = Column(String, primary_key=True, index=True)  # Maps to Pydantic event_id
    event_id = Column(String, index=True, nullable=True)
    correlation_id = Column(String, index=True, nullable=True)

    timestamp = Column(String, nullable=True)
    environment = Column(String, nullable=True)

    trigger_kind = Column(String, index=True, nullable=True)
    trigger_reason = Column(Text, nullable=True)

    summary = Column(Text)
    mantra = Column(Text)

    # Added in the correction pass to orion/schemas/metacog_entry.py --
    # without matching columns here, _row_dict's generic column-name filter
    # silently drops these two on every real insert. Caught while adding the
    # repair_pressure_appraisal_log table below, not by the original tests.
    severity = Column(String, nullable=True)
    touches = Column(_JSONB, default=list)

    snapshot_kind = Column(String, default="baseline")
    is_causally_dense = Column(Boolean, default=False)
    epistemic_status = Column(String, default="observed")
    visibility = Column(String, default="internal")
    redaction_level = Column(String, default="low")
    source_service = Column(String, nullable=True)
    source_node = Column(String, nullable=True)

    # --- Complex structures (JSONB) ---
    what_changed = Column(_JSONB, nullable=True)
    state = Column(_JSONB, nullable=True)
    causal_density = Column(_JSONB, nullable=True)
    provenance = Column(_JSONB, nullable=True)
    tags = Column(_JSONB, default=list)
