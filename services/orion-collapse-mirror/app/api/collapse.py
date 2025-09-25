from pydantic import BaseModel
from typing import List, Optional

class CollapseMirrorEntry(BaseModel):
    observer: str
    trigger: str
    observer_state: List[str]
    field_resonance: str
    intent: str
    type: str
    emergent_entity: str
    summary: str
    mantra: str
    causal_echo: Optional[str] = None
    timestamp: str
    environment: str

    def with_timestamp(self):
        """Ensure the entry has a UTC ISO timestamp."""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        return self
