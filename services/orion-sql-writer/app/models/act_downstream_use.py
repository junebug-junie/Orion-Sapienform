from sqlalchemy import Boolean, Column, DateTime, Index, Integer, String, Text

from app.db import Base


class ActDownstreamUseSQL(Base):
    """One row per ``grammar_acts`` act: was its output used by a later decision site within 72 h.

    Single writer: app/act_downstream_use.py (read-only over its sources). Weight source (a) of the
    body economy's Organ 3b. ``used`` is NULL for ``link_status='no_link_path'`` (cannot be measured),
    never false. Migration: services/orion-sql-db/manual_migration_act_downstream_use_v1.sql.
    """

    __tablename__ = "act_downstream_use"

    act_id = Column(String, primary_key=True)
    actor = Column(String, nullable=False)
    verb = Column(String, nullable=False)
    started_at = Column(DateTime(timezone=True))   # opened_at, else created_at (close-before-open)
    link_status = Column(String, nullable=False)  # linked | no_output | no_link_path
    link_path = Column(String)                    # direct | painting | chat_row | retro_leisure
    output_ref = Column(Text)                     # <table>:<id> of the act's output (the used one if used)
    used = Column(Boolean)
    first_use_at = Column(DateTime(timezone=True))
    site = Column(String)
    evidence_ref = Column(Text)
    arc_id = Column(String)
    arc_evidence_n = Column(Integer)               # refs in the citing arc (arc-granular credit)
    decision_id = Column(String)
    window_end = Column(DateTime(timezone=True))
    window_closed = Column(Boolean, nullable=False)
    refreshed_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("idx_act_downstream_use_actor_started", "actor", "started_at"),)
