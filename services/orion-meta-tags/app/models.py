from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class EventIn(BaseModel):
    id: str
    summary: Optional[str] = None
    trigger: Optional[str] = None
    observer: Optional[str] = None

    @property
    def text(self) -> str:
        # Fallback order: summary > trigger > observer > empty string
        return self.summary or self.trigger or self.observer or ""

class Enrichment(BaseModel):
    id: str
    service_name: str
    service_version: str
    enrichment_type: str
    tags: Optional[List[str]] = None
    entities: Optional[List[Dict[str, str]]] = None
    salience: Optional[float] = None
    ts: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat())

