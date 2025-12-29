# services/orion-cortex-orch/app/models.py
from __future__ import annotations
from typing import Any, Dict
from pydantic import BaseModel, ConfigDict, Field

# Hub -> Orch Input (This is specific to Orch's public face)
class CortexOrchInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    verb_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    origin_node: str = "unknown"
    context: Dict[str, Any] = Field(default_factory=dict)
