from typing import Optional
from pydantic import BaseModel, Field
from orion.schemas.cortex.contracts import CortexChatRequest, CortexClientResult

class CortexChatResult(BaseModel):
    cortex_result: CortexClientResult = Field(..., description="The raw result from Cortex Orchestrator")
    final_text: Optional[str] = Field(default=None, description="Convenience field for the final text response")
