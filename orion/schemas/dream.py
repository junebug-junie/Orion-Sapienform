from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

class DreamRequest(BaseModel):
    """
    Request to synthesize a dream/imagination sequence.
    """
    model_config = ConfigDict(extra="ignore")

    context_text: str
    mood: Optional[str] = None
    duration_seconds: int = 60
    integration_mode: str = "visual" # "text", "visual", "audio"
    metadata: Dict[str, Any] = Field(default_factory=dict)
