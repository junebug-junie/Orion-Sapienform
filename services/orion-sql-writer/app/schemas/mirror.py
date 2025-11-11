from typing import List, Optional
from pydantic import BaseModel

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
        if isinstance(self.observer_state, list):
            self.observer_state = ", ".join(self.observer_state)
        return self
