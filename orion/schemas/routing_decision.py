"""What the chat router decided, and whether the confidence gate changed it.

Orion can adjust one knob about its own behaviour: ``chat_reflective_lane_threshold``,
the confidence it must have before acting rather than replying
(``services/orion-cortex-orch/app/decision_router.py``). Until this schema
existed, the gate fired and left no trace -- it wrote its inputs into an
in-memory options dict that nothing read and nothing persisted.

That absence is why the mutation loop ended up wired to the wrong evidence. With
no record of routing behaviour, the only signal available to justify a routing
change came from the graph-review runtime, which the threshold does not affect;
the two were joined by the string "routing" and nothing else. This record is the
missing observation: the evidence a routing change should actually be judged on.

Deliberately carries no message content. The question it answers is "how did
Orion decide", not "what was said".
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

RoutingDecisionSourceV1 = Literal["heuristic", "llm", "fallback"]


class RoutingDecisionRecordV1(BaseModel):
    """One chat routing decision, with the gate's effect on it made explicit."""

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(default_factory=lambda: f"routing-decision-{uuid4()}")
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    #: Where the depth decision came from. "heuristic" is a keyword table with
    #: hardcoded confidences; "llm" is the model-scored router. Recorded because
    #: a confidence threshold means something different against each.
    source: RoutingDecisionSourceV1 = "heuristic"
    reason: str = ""

    #: Depth the router chose before the gate, and after it. They differ exactly
    #: when the gate demoted the turn.
    execution_depth_before_gate: int = Field(ge=0, le=8)
    execution_depth: int = Field(ge=0, le=8)
    primary_verb: Optional[str] = None

    #: The comparison the gate actually made. Both sides are recorded so a later
    #: reader can tell a threshold change from a confidence change -- moving the
    #: threshold is the only half Orion controls.
    decision_confidence: float = Field(ge=0.0, le=1.0)
    routing_threshold: float = Field(ge=0.0, le=1.0)

    #: True when the gate forced depth to 0. This is the outcome a routing
    #: mutation is trying to move, and the honest thing to measure it against.
    gate_demoted: bool = False
