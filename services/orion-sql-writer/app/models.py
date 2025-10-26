from datetime import datetime, date
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy import Column, String, Text, Float, DateTime, Date, Integer
from sqlalchemy.sql import func
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


class ChatHistoryInput(BaseModel):
    """Pydantic model for validating incoming chat history log messages."""
    id: Optional[str] = None # Will be populated from trace_id
    trace_id: str
    source: str
    prompt: Optional[str] = None
    response: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: Optional[datetime] = None

    def normalize(self) -> "ChatHistoryInput":
        """
        Populates the primary key and timestamp before database insertion.
        This follows the pattern in your main.py.
        """
        self.id = self.trace_id
        self.created_at = datetime.utcnow()
        return self

class DreamFragmentMeta(BaseModel):
    """Represents the metadata for a fragment used in a dream."""
    id: str
    kind: str
    salience: float
    tags: List[str] = []

class DreamMetrics(BaseModel):
    """Represents the metrics associated with a dream."""
    gpu_w: Optional[float] = None
    util: Optional[float] = None
    mem_mb: Optional[float] = None
    cpu_c: Optional[float] = None

class DreamInput(BaseModel):
    """Pydantic model for validating the dream output JSON."""
    # Core dream fields
    tldr: Optional[str] = None
    themes: Optional[List[str]] = []
    symbols: Optional[Dict[str, str]] = {}
    narrative: Optional[str] = None

    # Provenance and debugging fields
    fragments: Optional[List[DreamFragmentMeta]] = []
    metrics: Optional[DreamMetrics] = Field(default=None)
    dream_date: Optional[date] = None
    created_at: Optional[datetime] = None

    def normalize(self) -> "DreamInput":
        """Sets timestamps and default date if missing."""
        if not self.dream_date:
            self.dream_date = date.today()
        if not self.created_at:
            self.created_at = datetime.utcnow()
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

def generate_chat_log_model():
    """Dynamically creates the SQLAlchemy model with the configured table name."""
    return type(
        "ChatHistoryLogSQL",
        (Base,),
        {
            "__tablename__": "chat_history_log",
            "id": Column(String, primary_key=True),
            "trace_id": Column(String, index=True),
            "source": Column(String),
            "prompt": Column(Text),
            "response": Column(Text),
            "user_id": Column(String, nullable=True),
            "session_id": Column(String, nullable=True),
            "created_at": Column(DateTime, server_default=func.now()),
        },
    )

class Dream(Base):
    """SQLAlchemy model for storing dream results."""
    __tablename__ = "dreams"

    # Use an auto-incrementing integer as the primary key for simplicity
    id = Column(Integer, primary_key=True, autoincrement=True)
    dream_date = Column(Date, nullable=False, index=True, unique=True) # Ensure only one dream per date

    # Core dream fields
    tldr = Column(Text, nullable=True)
    themes = Column(JSONB, nullable=True) # Stores List[str]
    symbols = Column(JSONB, nullable=True) # Stores Dict[str, str]
    narrative = Column(Text, nullable=True)

    # Provenance and debugging fields stored as JSON blobs
    fragments = Column(JSONB, nullable=True) # Stores List[DreamFragmentMeta]
    metrics = Column(JSONB, nullable=True) # Stores DreamMetrics

    # Standard timestamp
    created_at = Column(DateTime, server_default=func.now())

ChatHistoryLogSQL = generate_chat_log_model()
