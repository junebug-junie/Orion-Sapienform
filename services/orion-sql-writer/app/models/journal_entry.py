from sqlalchemy import Column, DateTime, Index, String, Text
from sqlalchemy.sql import func

from app.db import Base


class JournalEntrySQL(Base):
    __tablename__ = "journal_entries"

    entry_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    author = Column(String, nullable=False)
    mode = Column(String, nullable=False)
    title = Column(String, nullable=True)
    body = Column(Text, nullable=False)
    source_kind = Column(String, nullable=True)
    source_ref = Column(String, nullable=True)
    correlation_id = Column(String, nullable=True)

    __table_args__ = (
        Index("idx_journal_entries_created_at", "created_at"),
        Index("idx_journal_entries_correlation_id", "correlation_id"),
        Index("idx_journal_entries_source_ref", "source_ref"),
    )
