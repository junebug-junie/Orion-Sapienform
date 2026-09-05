from sqlalchemy import Column, DateTime, Index, String, Text
from sqlalchemy.sql import func

from app.db import Base


class SelfKnowledgeItemLogSQL(Base):
    __tablename__ = "self_knowledge_items"

    entry_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    item_id = Column(String, nullable=False)
    run_id = Column(String, nullable=False)
    category = Column(String, nullable=False)
    name = Column(String, nullable=False)
    trust_tier = Column(String, nullable=False)
    observed_at = Column(String, nullable=False)
    source_path = Column(String, nullable=False)
    symbol_name = Column(String, nullable=True)
    metadata_text = Column(Text, nullable=True)

    # Review finding: run_id/item_id/category indexes added speculatively,
    # with no query pattern in this diff that uses them -- Patch 3's
    # topic-foundry DatasetSpec (the only planned consumer) reads by
    # time_column (created_at) for its training window, not by these.
    # created_at is the one real, currently-justified index; the others are
    # cheap to add later once an actual query pattern exists.
    __table_args__ = (
        Index("idx_self_knowledge_items_created_at", "created_at"),
    )
