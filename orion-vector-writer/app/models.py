from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class BaseEvent(BaseModel):
    """A base model for incoming events, ensuring an ID is present."""
    id: str

class CollapseTriageEvent(BaseEvent):
    """
    Schema for raw collapse events from the `orion:collapse:triage` channel.
    """
    summary: str
    trigger: str
    observer: str
    # Use extra='allow' to capture all other fields from the raw event
    class Config:
        extra = "allow"

class TagsEnrichedEvent(BaseEvent):
    """
    Schema for enriched events from the `orion:tags:enriched` channel.
    """
    collapse_id: str
    tags: List[str] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Optional[str] = None # The summary might be in the enriched event
    # Use extra='allow' to capture all other fields
    class Config:
        extra = "allow"
