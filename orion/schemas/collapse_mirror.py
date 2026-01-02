from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Union
from datetime import datetime, timezone
import os


class CollapseMirrorEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    observer: str
    trigger: str
    observer_state: List[str]
    field_resonance: str
    type: str
    emergent_entity: str
    summary: str
    mantra: str
    causal_echo: Optional[str] = None
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp")
    environment: Optional[str] = Field(default=None, description="Environment (dev, prod, etc.)")

    def with_defaults(self):
        """Fill in defaults for timestamp + environment if missing."""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.environment:
            self.environment = os.getenv("CHRONICLE_ENVIRONMENT", "dev")
        return self
