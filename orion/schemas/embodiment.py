from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

EMBODIMENT_INTENT_KIND = "embodiment.intent.v1"
EMBODIMENT_OUTCOME_KIND = "embodiment.outcome.v1"
EMBODIMENT_PERCEPTION_KIND = "embodiment.perception.v1"
EMBODIMENT_PERSONA_KIND = "embodiment.persona.v1"

IntentKind = Literal[
    "approach_player",
    "start_conversation",
    "wander",
    "go_to_location",
    "idle",
]
IntentSource = Literal["deliberate", "involuntary"]
OutcomeStatus = Literal["actuated", "resolved_noop", "preempted", "denied", "error"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EmbodimentIntentV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: IntentKind
    source: IntentSource
    ref: Optional[str] = None
    urgency: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str
    correlation_id: str
    world_id: Optional[str] = None
    player_id: Optional[str] = None
    issued_at: datetime = Field(default_factory=_utcnow)

    @field_validator("reason")
    @classmethod
    def _reason_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("reason must be non-empty (anti empty-shell)")
        return v


class EmbodimentOutcomeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_correlation_id: str
    source: IntentSource
    status: OutcomeStatus
    resolved_destination: Optional[dict[str, float]] = None
    player_id: Optional[str] = None
    send_input_ok: bool = False
    reason: str
    actuated_at: datetime = Field(default_factory=_utcnow)


class WorldPerceptionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: str
    position: dict[str, float]
    # Orion's own orientation ({dx,dy}) and whether the engine is currently
    # moving Orion along a path. The engine only orients stopped participants
    # toward each other (Conversation.tick), so surfacing pathfinding lets the
    # worker guarantee the `!pathfinding` precondition and makes facing inspectable.
    facing: Optional[dict[str, float]] = None
    pathfinding: bool = False
    nearby_players: list[dict[str, Any]] = Field(default_factory=list)
    active_conversation: Optional[dict[str, Any]] = None
    world_generation: Optional[int] = None
    perceived_at: datetime = Field(default_factory=_utcnow)


class OrionTownPersonaV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "Orion"
    identity_blurb: str
    plan: str
    spritesheet: str
    persona_source: Literal["projection", "fallback", "card"] = "projection"
    provenance: dict[str, Any] = Field(default_factory=dict)
