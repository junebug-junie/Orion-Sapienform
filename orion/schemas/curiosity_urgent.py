"""Urgent curiosity runs: a seeded investigation Juniper (or the hardware watcher) asks for.

Spec: docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md (Parts 2, 2b, 3).
Plan: docs/superpowers/plans/2026-09-28-urgent-curiosity-plan-3-seeded-urgent-runs.md.

The seed rides the curiosity durable run end to end -- ``CuriosityRunBriefV1.urgent`` at
kickoff, ``CuriosityTurnRequestV1.urgent`` on the turn RPC -- so the turn runs the focused
investigation prompt instead of Orion's self-directed one. Both fields are optional and
omitted on the wire when ``None``, so producers and consumers deploy in any order.

``CuriosityUrgentRequestV1`` is the same shape published on ``orion:curiosity:urgent:request``
so the hardware watcher can start an urgent run without reaching into Hub.

``question`` is passed through verbatim as the assignment; nothing branches on its words.
``evidence`` is hardware telemetry and pool state only -- never chat, memory, or journal content.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from pydantic_core import to_json

URGENT_REQUEST_CHANNEL = "orion:curiosity:urgent:request"
URGENT_REQUEST_KIND = "curiosity.urgent.request.v1"
URGENT_EVIDENCE_MAX_BYTES = 32_000

Trigger = Literal["manual", "heat", "cooling"]


class CuriosityUrgentSeedV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: str = Field(pattern=r"^[0-9a-f]{12,32}$")
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    trigger: Trigger
    # What the incident is about, e.g. "athena", "circe/gpu2", "cabinet_ac".
    subject: str = Field(default="", max_length=120)
    evidence: dict[str, Any] = Field(default_factory=dict)
    requested_at: datetime
    requested_by: str = Field(default="hub", max_length=64)

    @field_validator("evidence")
    @classmethod
    def _evidence_is_bounded_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            size = len(to_json(value))
        except Exception as exc:  # PydanticSerializationError on non-JSON values
            raise ValueError(f"evidence must be JSON-serialisable: {exc}") from exc
        if size > URGENT_EVIDENCE_MAX_BYTES:
            raise ValueError(f"evidence is {size} bytes serialized; limit {URGENT_EVIDENCE_MAX_BYTES}")
        return value


class CuriosityUrgentRequestV1(CuriosityUrgentSeedV1):
    """Bus request on ``orion:curiosity:urgent:request``: the seed, nothing more."""
