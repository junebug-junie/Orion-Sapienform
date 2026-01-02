from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class CortexChatRequest(BaseModel):
    prompt: str = Field(..., description="The user's prompt text")
    mode: str = Field(default="brain", description="Execution mode: brain, agent, council")

    # Optional overrides
    verb: Optional[str] = Field(default=None, description="Cognition verb override")
    packs: Optional[List[str]] = Field(default=None, description="Cognition packs override")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Execution options")
    recall: Optional[Dict[str, Any]] = Field(default=None, description="Recall configuration overrides")

    # Context overrides
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    trace_id: Optional[str] = Field(default=None, description="Trace identifier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional context metadata")
