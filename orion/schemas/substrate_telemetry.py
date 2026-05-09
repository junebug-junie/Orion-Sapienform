"""Bus payloads for substrate observability (tier merge outcomes)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SubstrateTierOutcomesPayloadV1(BaseModel):
    """Emitted on ``orion:substrate:tier_outcomes`` when cold-path materialization runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Literal["substrate.tier_outcomes"] = "substrate.tier_outcomes"
    generated_at: str
    cold_anchors: list[str] = Field(default_factory=list)
    tier_outcomes: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per-anchor tier outcome strings, e.g. operator_static_protected:2",
    )
    degraded_producers: list[str] = Field(default_factory=list)
