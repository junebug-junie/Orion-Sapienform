from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class EnrichmentInput(BaseModel):
    id: Optional[str] = None
    collapse_id: Optional[str] = None
    service_name: str
    service_version: str
    enrichment_type: str
    tags: Optional[List[str]] = []
    entities: Optional[List[Dict[str, Any]]] = []
    salience: Optional[float] = None
    ts: Optional[datetime] = None

    def normalize(self):
        if not self.ts:
            self.ts = datetime.utcnow()
        if not self.collapse_id:
            self.collapse_id = self.id
        return self
