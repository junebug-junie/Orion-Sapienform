from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.sql import func

from app.db import Base


class CuriosityHopReadingSQL(Base):
    """One row per hop reading (HopReadingV1,
    orion/schemas/curiosity_supervisor.py). A READING of what a hop was
    doing, not a write to Orion's own graph -- see that module's docstring.

    Single writer: orion-sql-writer, off `orion:curiosity:supervisor:reading`.
    `reading_id` is the row key, not `(hop_run_id, hop_n)`: that pair is
    exactly what this arc's own hop-identity patch found colliding on real
    data (run 58b638778228 held six real hops under three hop_n values).
    """

    __tablename__ = "curiosity_hop_reading"

    reading_id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    generated_at = Column(DateTime(timezone=True), nullable=True)
    hop_run_id = Column(String, nullable=False)
    hop_n = Column(Integer, nullable=False)
    # Epoch ms, same type as HopRecord.written_at/HopReadingV1.hop_written_at
    # -- NOT a DateTime column. It comes straight from FalkorDB's timestamp()
    # via the supervisor's own reading, not from this service's clock, and
    # storing the raw int avoids a silent timezone/precision reinterpretation
    # on the one hop-reading path where "what the graph's clock said" matters
    # (see orion/curiosity/worldview.py's hop_order_key).
    hop_written_at = Column(BigInteger, nullable=True)
    about_prior_id = Column(String, nullable=True)
    kind = Column(String, nullable=False)
    moved_the_claim = Column(Boolean, nullable=True)
    reading_confidence = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=False, default="")

    __table_args__ = (
        Index("idx_curiosity_hop_reading_run_id", "hop_run_id"),
        Index("idx_curiosity_hop_reading_prior_id", "about_prior_id"),
        Index("idx_curiosity_hop_reading_created_at", "created_at"),
    )
