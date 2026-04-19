from sqlalchemy import Column, DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base


class JournalEntryIndexSQL(Base):
    __tablename__ = "journal_entry_index"

    entry_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    author = Column(String, nullable=False)
    mode = Column(String, nullable=False)
    title = Column(Text, nullable=True)
    body = Column(Text, nullable=False)

    source_kind = Column(String, nullable=True)
    source_ref = Column(String, nullable=True)
    correlation_id = Column(String, nullable=True)
    trigger_kind = Column(String, nullable=True)
    trigger_summary = Column(Text, nullable=True)

    conversation_frame = Column(String, nullable=True)
    task_mode = Column(String, nullable=True)
    identity_salience = Column(String, nullable=True)
    answer_strategy = Column(String, nullable=True)
    stance_summary = Column(Text, nullable=True)

    active_identity_facets = Column(JSONB, nullable=True)
    active_growth_axes = Column(JSONB, nullable=True)
    active_relationship_facets = Column(JSONB, nullable=True)
    social_posture = Column(JSONB, nullable=True)
    reflective_themes = Column(JSONB, nullable=True)
    active_tensions = Column(JSONB, nullable=True)
    dream_motifs = Column(JSONB, nullable=True)
    response_hazards = Column(JSONB, nullable=True)

    __table_args__ = (
        Index("idx_journal_entry_index_created_at", "created_at"),
        Index("idx_journal_entry_index_mode", "mode"),
        Index("idx_journal_entry_index_source_kind", "source_kind"),
        Index("idx_journal_entry_index_trigger_kind", "trigger_kind"),
        Index("idx_journal_entry_index_author", "author"),
        Index("idx_journal_entry_index_source_ref", "source_ref"),
        Index("idx_journal_entry_index_correlation_id", "correlation_id"),
    )
