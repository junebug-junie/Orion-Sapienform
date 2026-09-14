from sqlalchemy import JSON, Column, DateTime, Index, String, Text
from sqlalchemy.sql import func

from app.db import Base


class CuriosityPeerBriefSQL(Base):
    """One row per contractor-peer brief (PeerBriefV1).

    Single writer: orion-sql-writer, off `orion:curiosity:peer:brief`.
    Columns mirror the schema one-to-one; `created_at` is write-time only
    (same pattern as AttentionSchemaSQL).
    """

    __tablename__ = "curiosity_peer_brief"

    brief_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    help_id = Column(String, nullable=False)
    run_id = Column(String, nullable=False)
    prior_id = Column(String, nullable=True)
    peer = Column(String, nullable=False)
    status = Column(String, nullable=False)
    summary = Column(Text, nullable=False, default="")
    evidence_pointers = Column(JSON, nullable=True)
    open_questions = Column(JSON, nullable=True)
    suggested_next_looks = Column(JSON, nullable=True)
    refusal_reason = Column(String, nullable=True)
    written_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_curiosity_peer_brief_run_id", "run_id"),
        Index("idx_curiosity_peer_brief_help_id", "help_id"),
        Index("idx_curiosity_peer_brief_created_at", "created_at"),
    )
