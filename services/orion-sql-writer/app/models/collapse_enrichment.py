from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base

class CollapseEnrichment(Base):
    __tablename__ = "collapse_enrichment"

    id = Column(String, primary_key=True, nullable=False)        # enrichment record id
    collapse_id = Column(String, nullable=False)                 # FK (app-level) → collapse_mirror.id
    service_name = Column(String)
    service_version = Column(String)
    enrichment_type = Column(String)
    tags = Column(JSONB)                                         # array[str]
    entities = Column(JSONB)                                     # array[dict]
    salience = Column(Float)
    ts = Column(DateTime, default=datetime.utcnow)
