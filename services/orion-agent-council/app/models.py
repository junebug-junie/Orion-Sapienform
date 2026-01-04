# services/orion-agent-council/app/models.py
from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field

# Import shared schemas (DRY)
from orion.schemas.agents.schemas import (
    PhiSnapshot,
    SelfField,
    AgentOpinion,
    BlinkScores,
    BlinkJudgement,
    AuditVerdict,
    RoundResult,
    CouncilResult,
    DeliberationRequest
)

# --- Personas / Agents (Internal Config) ---

class AgentConfig(BaseModel):
    """
    Configuration for an internal Council agent.
    This remains local because consumers don't need to know about internal weights/configs.
    """
    name: str
    role_description: str
    temperature: float = 0.7
    weight: float = 1.0
    universe: str = "core"
    tags: List[str] = Field(default_factory=list)
    use_phi: bool = True
