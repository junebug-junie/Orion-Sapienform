from __future__ import annotations

import json
from typing import Any, Literal, cast

from orion.schemas.thought import StanceReactRequestV1, ThoughtEventV1
from orion.thought.coalition import coalition_ids_from_association
from orion.thought.policy_refusal import evaluate_thought_disposition
from orion.thought.stance_quality import enforce_thought_stance_quality


def parse_stance_react_payload(raw: dict[str, Any] | str) -> ThoughtEventV1:
    """Validate JSON from cortex stance_react step into ThoughtEventV1."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    return ThoughtEventV1.model_validate(raw)


def apply_stance_react_pipeline(
    thought: ThoughtEventV1,
    request: StanceReactRequestV1,
) -> ThoughtEventV1:
    enriched, _ = enforce_thought_stance_quality(thought, request.stance_inputs)
    coalition_ids = coalition_ids_from_association(request.association)
    decision = evaluate_thought_disposition(
        enriched,
        association_stale=request.association.broadcast_stale,
        coalition_ids=coalition_ids,
    )
    enriched.disposition = cast(Literal["proceed", "defer", "refuse"], decision.disposition)
    enriched.disposition_reasons = list(decision.reasons)
    enriched.boundary_register = decision.boundary_register
    return enriched
