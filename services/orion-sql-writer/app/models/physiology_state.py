"""One row per Body-organ tick (body economy Patch A3, Organ 1 Body).

Writer: sql-writer off ``orion:physiology:state`` (``PhysiologyStateV1``, producer: orion-durable-runs'
``physiology`` node in the regulation thread, ~120 s). This is the persisted history of Orion's
constraints (heat, energy, lane:<role>, attention:juniper, claude) that the design notes is
missing today. PK ``tick_id`` (deterministic per tick), so a redelivered event is one row.

Table and indexes come from ``services/orion-sql-db/manual_migration_physiology_state_v1.sql``
(jsonb there; generic JSON here so the sqlite write-path test runs). ``constraints`` keeps every
reading verbatim, including ``no_reading`` ones: a missing reading is stored as missing, never 0.
"""

from sqlalchemy import JSON, Column, DateTime, Index, String
from sqlalchemy.sql import func

from app.db import Base


class PhysiologyStateSQL(Base):
    __tablename__ = "physiology_state"

    tick_id = Column(String, primary_key=True)
    generated_at = Column(DateTime(timezone=True), nullable=False)
    constraints = Column(JSON, nullable=False)
    warnings = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_physiology_state_generated", "generated_at"),
        Index("idx_physiology_state_created", "created_at"),
    )


def physiology_state_row(state) -> dict:
    """``PhysiologyStateV1`` -> row. Nested readings go in as JSON-safe dicts (datetimes as ISO)."""
    as_json = state.model_dump(mode="json")
    return dict(tick_id=state.tick_id, generated_at=state.generated_at,
                constraints=as_json["constraints"], warnings=as_json["warnings"])
