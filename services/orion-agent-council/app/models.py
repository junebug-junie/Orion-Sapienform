# services/orion-agent-council/app/models.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# --- φ + SelfField snapshots (optional) ---


class PhiSnapshot(BaseModel):
    valence: float = 0.0
    energy: float = 0.0
    coherence: float = 1.0
    novelty: float = 0.0


class SelfField(BaseModel):
    calm: Optional[float] = None
    stress_load: Optional[float] = None
    uncertainty: Optional[float] = None
    focus: Optional[float] = None
    attunement_to_juniper: Optional[float] = None
    curiosity: Optional[float] = None


# --- Personas / agents ---


class AgentConfig(BaseModel):
    name: str
    role_description: str
    backend: str
    model: Optional[str] = None
    temperature: float = 0.7
    weight: float = 1.0
    universe: str = "core"
    tags: List[str] = Field(default_factory=list)
    use_phi: bool = True  # whether this agent conditions on φ/self_field


class AgentOpinion(BaseModel):
    agent_name: str
    model: Optional[str]
    backend: str
    text: str


# --- Blink / arbiter ---


class BlinkScores(BaseModel):
    coherence: float = 0.7
    faithfulness: float = 0.7
    usefulness: float = 0.7
    risk: float = 0.3
    effort_cost: float = 0.5
    novelty: float = 0.5
    overall: float = 0.7


class BlinkJudgement(BaseModel):
    proposed_answer: str
    scores: BlinkScores
    disagreement: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


# --- Auditor verdict ---


class AuditVerdict(BaseModel):
    action: str  # "accept" | "revise_same_round" | "new_round"
    reason: str
    constraints: Dict[str, Any] = Field(default_factory=dict)
    override_answer: Optional[str] = None


# --- Round / final council result ---


class RoundResult(BaseModel):
    round_index: int
    opinions: List[AgentOpinion]


class CouncilResult(BaseModel):
    trace_id: str
    prompt: str
    final_text: str
    opinions: List[AgentOpinion]
    blink: BlinkJudgement
    verdict: AuditVerdict
    meta: Dict[str, Any] = Field(default_factory=dict)


# --- Incoming request from hub / cortex / whoever ---


class DeliberationRequest(BaseModel):
    event: str = "council_deliberation"
    trace_id: Optional[str] = None
    source: Optional[str] = None

    prompt: str
    history: Optional[List[Dict[str, Any]]] = None
    tags: Optional[List[str]] = None
    universe: Optional[str] = None

    response_channel: Optional[str] = None

    # φ + SelfField (optional, from Spark)
    phi: Optional[PhiSnapshot] = None
    self_field: Optional[SelfField] = None

    # For future per-agent state (“n-dimensional stubs”)
    persona_state: Optional[Dict[str, Any]] = None
