from sqlalchemy import Column, String, Boolean, Text, JSON, DateTime
from app.db import Base

class CognitionTraceSQL(Base):
    __tablename__ = "cognition_traces"

    correlation_id = Column(String, primary_key=True)
    mode = Column(String, nullable=False)
    verb = Column(String, nullable=False)
    final_text = Column(Text, nullable=True)

    # Stored as TIMESTAMPTZ in Postgres.
    # The SQL writer worker will coerce float epoch seconds and ISO8601
    # strings into timezone-aware datetimes before insertion.
    timestamp = Column(DateTime(timezone=True), nullable=False)
    source_service = Column(String, nullable=True)
    source_node = Column(String, nullable=True)

    recall_used = Column(Boolean, default=False)

    # Store complex structures as JSON
    packs = Column(JSON, nullable=True)
    options = Column(JSON, nullable=True)
    steps = Column(JSON, nullable=True)
    recall_debug = Column(JSON, nullable=True)
    metadata_ = Column("metadata", JSON, nullable=True)
