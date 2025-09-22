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
