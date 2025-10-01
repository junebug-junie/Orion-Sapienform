from pydantic import BaseModel
from typing import List, Dict, Optional
from sqlalchemy import Column, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, timezone

Base = declarative_base()

class CollapseMirrorEntrySQL(Base):
    __tablename__ = "collapse_mirror"

    id = Column(String, primary_key=True, index=True)
    observer = Column(String)
    trigger = Column(Text)
    observer_state = Column(String)
    field_resonance = Column(Text)
    type = Column(String)
    emergent_entity = Column(Text)
    summary = Column(Text)
    mantra = Column(Text)
    causal_echo = Column(Text, nullable=True)
    timestamp = Column(String, nullable=True)
    environment = Column(String, nullable=True)

    # 🚩 Link to enrichments
    enrichments = relationship("CollapseEnrichmentSQL", back_populates="collapse")


class CollapseEnrichmentSQL(Base):
    __tablename__ = "collapse_enrichment"

    id = Column(String, primary_key=True, index=True)
    collapse_id = Column(String, ForeignKey("collapse_mirror.id"), nullable=False)

    service_name = Column(String, nullable=False)
    service_version = Column(String, nullable=False)
    enrichment_type = Column(String, nullable=False)

    tags = Column(Text, nullable=True)      # JSON string of list[str]
    entities = Column(Text, nullable=True)  # JSON string of list[dict]
    salience = Column(Float, nullable=True)

    ts = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # 🚩 Link back
    collapse = relationship("CollapseMirrorEntrySQL", back_populates="enrichments")

