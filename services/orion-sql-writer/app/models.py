from datetime import datetime
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy import Column, String, Text, Float, DateTime
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
    id: str
    observer: Optional[str] = None
    trigger: Optional[str] = None
    observer_state: Optional[List[str]] = None
    field_resonance: Optional[str] = None
    type: Optional[str] = None
    emergent_entity: Optional[str] = None
    summary: Optional[str] = None
    mantra: Optional[str] = None
    causal_echo: Optional[str] = None
    timestamp: Optional[str] = None
    environment: Optional[str] = None

    def normalize(self):
        """Flatten list fields and ensure strings are clean."""
        if isinstance(self.observer_state, list):
            self.observer_state = ", ".join(self.observer_state)
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


# === ORM (Postgres table) ===
class CollapseMirror(Base):
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
