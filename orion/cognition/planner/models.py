# orion/cognition/planner/models.py
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VerbConfig(BaseModel):
    name: str
    description: Optional[str] = None
    group: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    services: List[str] = Field(default_factory=list)
    prompt_template: Optional[str] = None
    timeout_ms: Optional[int] = None
    requires_gpu: Optional[bool] = None
    requires_memory: Optional[bool] = None
