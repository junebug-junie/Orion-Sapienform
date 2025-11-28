from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.sql import func

from app.db import Base


class SparkIntrospectionLogSQL(Base):
    """
    Stores Orion's internal reflections on Spark introspection candidates.

    Each row represents one introspection note about a specific turn / trace_id.
    """
    __tablename__ = "spark_introspection_log"

    # Primary key: can be trace_id, or trace_id plus some suffix if you ever
    # want multiple introspections per turn. For MVP we can just use trace_id.
    id = Column(String, primary_key=True)

    # Link back to the original chat turn or event
    trace_id = Column(String, index=True)

    # Where the candidate came from (e.g. "brain", "hub", "spark-introspector")
    source = Column(String)

    # Optional type/kind if you ever differentiate
    kind = Column(String, nullable=True)

    # Original human prompt and Orion chat response (for local context)
    prompt = Column(Text)
    response = Column(Text)

    # The introspection text that Cortex/Brain produced
    introspection = Column(Text)

    # Spark meta (phi_before/after, SelfField, tags, etc.) as JSON string
    spark_meta = Column(Text)

    created_at = Column(DateTime, server_default=func.now())
