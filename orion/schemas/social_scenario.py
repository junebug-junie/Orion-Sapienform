from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SocialScenarioTurnFixtureKind = Literal["social_turn", "bridge_message"]
SocialScenarioPromptFixtureKind = Literal["captured_bridge_payload", "custom_payload"]


class SocialScenarioTurnFixtureV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    fixture_id: str = Field(default_factory=lambda: f"social-scenario-turn-{uuid4()}")
    fixture_kind: SocialScenarioTurnFixtureKind
    thread_id: Optional[str] = None
    participant_id: str = "peer-1"
    participant_name: str = "CallSyne Peer"
    participant_kind: str = "peer_ai"
    prompt: str = ""
    response: str = ""
    text: str = ""
    mentions_orion: bool = False
    target_participant_id: Optional[str] = None
    target_participant_name: Optional[str] = None
    reply_to_message_id: Optional[str] = None
    reply_to_sender_id: Optional[str] = None
    reply_to_sender_name: Optional[str] = None
    mentioned_participant_ids: List[str] = Field(default_factory=list)
    mentioned_participant_names: List[str] = Field(default_factory=list)
    turn_id: Optional[str] = None
    correlation_id: Optional[str] = None
    message_id: Optional[str] = None
    created_at: Optional[str] = None
    stored_at: Optional[str] = None
    client_meta: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_fixture_kind(self) -> "SocialScenarioTurnFixtureV1":
        if self.fixture_kind == "social_turn":
            if not self.prompt.strip() or not self.response.strip():
                raise ValueError("social_turn fixtures require both prompt and response")
        if self.fixture_kind == "bridge_message" and not self.text.strip():
            raise ValueError("bridge_message fixtures require text")
        return self


class SocialScenarioSeedStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    participant: Optional[Dict[str, Any]] = None
    room: Optional[Dict[str, Any]] = None
    stance: Optional[Dict[str, Any]] = None
    peer_style: Optional[Dict[str, Any]] = None
    room_ritual: Optional[Dict[str, Any]] = None


class SocialScenarioExpectationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    turn_policy_decision: Optional[str] = None
    routing_decision: Optional[str] = None
    audience_scope: Optional[str] = None
    repair_decision: Optional[str] = None
    repair_type: Optional[str] = None
    epistemic_claim_kind: Optional[str] = None
    epistemic_decision: Optional[str] = None
    epistemic_lead_in_contains: Optional[str] = None
    deliberation_decision_kind: Optional[str] = None
    floor_decision_kind: Optional[str] = None
    bridge_summary_expected: Optional[bool] = None
    clarifying_question_expected: Optional[bool] = None
    handoff_decision_kind: Optional[str] = None
    closure_expected: Optional[bool] = None
    gif_decision_kind: Optional[str] = None
    gif_allowed_expected: Optional[bool] = None
    gif_intent_kind: Optional[str] = None
    selected_context_kinds: List[str] = Field(default_factory=list)
    softened_context_kinds: List[str] = Field(default_factory=list)
    excluded_context_kinds: List[str] = Field(default_factory=list)
    inspection_section_kinds: List[str] = Field(default_factory=list)
    prompt_must_contain: List[str] = Field(default_factory=list)
    prompt_must_not_contain: List[str] = Field(default_factory=list)
    blocked_strings: List[str] = Field(default_factory=list)
    require_pending_artifact_non_active: bool = False
    require_private_material_blocked: bool = False
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialScenarioFixtureV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    scenario_id: str
    platform: str
    room_id: str
    participant_id: str = "peer-1"
    description: str = ""
    active_participants: List[str] = Field(default_factory=list)
    seeded_state: SocialScenarioSeedStateV1 = Field(default_factory=SocialScenarioSeedStateV1)
    transcript_turns: List[SocialScenarioTurnFixtureV1] = Field(default_factory=list)
    prompt_fixture_kind: SocialScenarioPromptFixtureKind = "captured_bridge_payload"
    prompt_payload_overrides: Dict[str, Any] = Field(default_factory=dict)
    expectation: SocialScenarioExpectationV1 = Field(default_factory=SocialScenarioExpectationV1)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SocialScenarioEvaluationResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    evaluation_id: str = Field(default_factory=lambda: f"social-scenario-eval-{uuid4()}")
    platform: str
    room_id: str
    scenario_id: str
    passed: bool
    mismatch_reasons: List[str] = Field(default_factory=list)
    seams_exercised: List[str] = Field(default_factory=list)
    transcript_turn_count: int = Field(default=0, ge=0)
    observed_outcomes: Dict[str, Any] = Field(default_factory=dict)
    safety_observations: List[str] = Field(default_factory=list)
    evaluated_at: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, str] = Field(default_factory=dict)
