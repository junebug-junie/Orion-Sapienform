from datetime import datetime
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base


# ---------- Pydantic input models ----------

class EnrichmentInput(BaseModel):
    """
    Enriched tag/meta events.
    If collapse_id is missing, derive it from id (your earlier constraint).
    """
    id: Optional[str] = None                 # enrichment record id (may be generated upstream)
    collapse_id: Optional[str] = None        # link back to collapse_mirror.id
    service_name: str
    service_version: str
    enrichment_type: str
    tags: Optional[List[str]] = []
    entities: Optional[List[Dict[str, Any]]] = []
    salience: Optional[float] = None
    ts: Optional[datetime] = None

    def normalize(self):
        # Ensure timestamps & collapse_id
        if not self.ts:
            self.ts = datetime.utcnow()
        if not self.collapse_id:
            # fall back to incoming id if upstream didn't pass collapse_id
            self.collapse_id = self.id
        return self


class MirrorInput(BaseModel):
    """
    Raw collapse mirror intake (schema kept intentionally loose).
    You can extend this as your mirror contract firms up.
    """
    id: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    ts: Optional[datetime] = None

    def normalize(self):
        if not self.ts:
            self.ts = datetime.utcnow()
        return self


# ---------- SQLAlchemy persistence models ----------

class CollapseEnrichment(Base):
    __tablename__ = "collapse_enrichment"

    id = Column(String, primary_key=True, nullable=False)        # enrichment record id
    collapse_id = Column(String, nullable=False)                 # FK → collapse_mirror.id (application-level)
    service_name = Column(String)
    service_version = Column(String)
    enrichment_type = Column(String)
    tags = Column(JSONB)                                         # array of strings
    entities = Column(JSONB)                                     # array of dicts
    salience = Column(Float)
    ts = Column(DateTime, default=datetime.utcnow)


class CollapseMirror(Base):
    __tablename__ = "collapse_mirror"

    id = Column(String, primary_key=True, nullable=False)        # mirror record id
    payload = Column(JSONB)                                      # raw document/event
    ts = Column(DateTime, default=datetime.utcnow)
