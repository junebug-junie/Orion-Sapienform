from typing import List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

class MetaTagsPayload(BaseModel):
    """
    Standard schema for enrichment tags emitted by analysis services.
    Conforms to the Titanium Contract for Enrichment data.
    """
    # Service Identifiers
    service_name: str
    service_version: str
    node: Optional[str] = None

    # Lineage / Traceability
    timestamp: datetime = Field(default_factory=datetime.now)
    source_message_id: Optional[str] = None
    correlation_id: Optional[str] = None

    # Domain Data (The actual enrichment)
    tags: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)

    # Meta
    processing_meta: Optional[Any] = None

    # Required for DB Mapping (CollapseEnrichment)
    id: Optional[str] = None
    collapse_id: Optional[str] = None
    enrichment_type: Optional[str] = None
    salience: Optional[float] = None
    ts: Optional[datetime] = None # Duplicate of timestamp, used by some writers
