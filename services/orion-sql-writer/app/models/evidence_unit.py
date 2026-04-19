from sqlalchemy import Column, DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base


class EvidenceUnitSQL(Base):
    __tablename__ = "evidence_units"

    unit_id = Column(String, primary_key=True)
    unit_kind = Column(String, nullable=False)
    source_family = Column(String, nullable=False)
    source_kind = Column(String, nullable=False)
    source_ref = Column(String, nullable=False)
    correlation_id = Column(String, nullable=True)
    parent_unit_id = Column(String, nullable=True)
    sibling_prev_id = Column(String, nullable=True)
    sibling_next_id = Column(String, nullable=True)
    title = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    body = Column(Text, nullable=True)
    facets = Column(JSONB, nullable=False, default=list)
    metadata_json = Column("metadata", JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_evidence_units_created_at", "created_at"),
        Index("idx_evidence_units_unit_kind", "unit_kind"),
        Index("idx_evidence_units_source_family", "source_family"),
        Index("idx_evidence_units_source_kind", "source_kind"),
        Index("idx_evidence_units_source_ref", "source_ref"),
        Index("idx_evidence_units_correlation_id", "correlation_id"),
        Index("idx_evidence_units_parent_unit_id", "parent_unit_id"),
    )
