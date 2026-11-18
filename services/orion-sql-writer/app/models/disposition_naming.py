from sqlalchemy import JSON, Column, DateTime, Index, String, Text
from sqlalchemy.sql import func

from app.db import Base


class DispositionNamingSQL(Base):
    """One row per self-inquiry naming run (DispositionNamingV1, orion/schemas/personality.py):
    Orion's verdict on one personality disposition (name / reject / merge / no_answer), the packet
    Orion was shown, and Hub's structural check. Append-only, PK naming_id (uuid5 of run+packet), so
    a redelivered publish is a no-op. Writer: sql-writer off ``orion:personality:naming`` (producer:
    Hub's self-inquiry naming branch). Reader: orion-durable-runs, the single writer of
    ``personality_*``. Table and indexes come from
    ``services/orion-sql-db/manual_migration_personality_naming_v1.sql`` (jsonb there; generic JSON
    here so the sqlite write-path test runs)."""

    __tablename__ = "orion_disposition_naming"

    naming_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False)
    packet_id = Column(String, nullable=False)
    disposition_id = Column(String, nullable=False)
    verdict = Column(String, nullable=False)
    name = Column(Text, nullable=True)
    sentence = Column(Text, nullable=True)
    merge_with = Column(String, nullable=True)
    evidence_refs = Column(JSON, nullable=False, default=list)
    shown = Column(JSON, nullable=False)
    hub_check = Column(String, nullable=False)
    refused_reasons = Column(JSON, nullable=False, default=list)
    written_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("orion_disposition_naming_disp_idx", "disposition_id", "written_at"),
        Index("orion_disposition_naming_created_idx", "created_at", "naming_id"),
    )
