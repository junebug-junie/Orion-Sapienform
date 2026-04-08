# services/orion-agent-chain/app/models.py
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel


class ToolDef(BaseModel):
    """
    Generic “tool definition” for Agent Chain.

    This is what we pass to Planner as the list of tools it’s allowed to use.
    """
    tool_id: str                     # must match verb name in VerbRegistry
    description: str                 # human-readable description
    input_schema: Dict[str, Any]     # JSON schema style
    output_schema: Dict[str, Any]    # JSON schema style
    execution_mode: Optional[str] = None
    requires_capability_selector: bool = False
    preferred_skill_families: Dict[str, Any] | list[str] | None = None
    side_effect_level: Optional[str] = None
