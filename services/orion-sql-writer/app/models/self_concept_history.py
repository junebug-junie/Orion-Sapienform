from sqlalchemy import JSON, Column, DateTime, Index, Integer, String, Text
from sqlalchemy.sql import func

from app.db import Base


class SelfConceptHistorySQL(Base):
    __tablename__ = "self_concept_history"

    entry_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    concept_id = Column(String, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    content = Column(Text, nullable=False)
    # Generic JSON (not the Postgres-only JSONB dialect type) -- portable
    # across sqlite, which the real end-to-end write-path test uses; nothing
    # here needs JSONB-specific indexing/querying.
    evidence_refs = Column(JSON, nullable=True)
    produced_by = Column(String, nullable=False)

    __table_args__ = (
        Index("idx_self_concept_history_created_at", "created_at"),
        Index("idx_self_concept_history_concept_id", "concept_id"),
    )
