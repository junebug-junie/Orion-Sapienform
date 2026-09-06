from sqlalchemy import Column, DateTime, Float, Index, String, Text
from sqlalchemy.sql import func

from app.db import Base


class AttentionSchemaSQL(Base):
    """One row per attention event from any of Orion's attending processes
    (AttentionSchemaV1, orion/schemas/attention_schema.py). Single writer:
    orion-sql-writer, off the `orion:attention:schema` bus channel.

    Columns mirror the schema one-to-one on purpose -- the acceptance checks
    in docs/superpowers/specs/2026-09-04-attention-schema-surface-design.md
    are `GROUP BY process` / `count(DISTINCT reason_narrative)` queries, and
    a jsonb blob would make every one of them a `->>` scan. The composite
    (process, created_at) index exists for the stratified sample those checks
    require (equal N per process), not for a bare timestamp scan.
    """

    __tablename__ = "substrate_attention_schema"

    entry_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    generated_at = Column(DateTime(timezone=True), nullable=False)
    process = Column(String, nullable=False)
    correlation_id = Column(String, nullable=True)
    attended_id = Column(String, nullable=True)
    attended_label = Column(String, nullable=False, default="")
    attention_reason = Column(String, nullable=False)
    reason_narrative = Column(Text, nullable=False, default="")
    narrative_kind = Column(String, nullable=False, default="computed")
    confidence = Column(Float, nullable=True)
    confidence_basis = Column(String, nullable=True)
    predicted_next = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_substrate_attention_schema_process_created", "process", "created_at"),
    )
