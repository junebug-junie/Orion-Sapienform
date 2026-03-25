from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatStanceBrief(BaseModel):
    """Bounded internal stance brief used by chat_general speech pass."""

    conversation_frame: Literal[
        "technical",
        "planning",
        "reflective",
        "playful_relational",
        "identity_emergence",
        "mixed",
    ] = Field(...)

    user_intent: str = Field(...)
    self_relevance: str = Field(...)
    juniper_relevance: str = Field(...)

    active_identity_facets: list[str] = Field(default_factory=list)
    active_growth_axes: list[str] = Field(default_factory=list)
    active_relationship_facets: list[str] = Field(default_factory=list)
    social_posture: list[str] = Field(default_factory=list)

    reflective_themes: list[str] = Field(default_factory=list)
    active_tensions: list[str] = Field(default_factory=list)
    dream_motifs: list[str] = Field(default_factory=list)

    response_priorities: list[str] = Field(default_factory=list)
    response_hazards: list[str] = Field(default_factory=list)

    answer_strategy: str = Field(...)
    stance_summary: str = Field(...)
