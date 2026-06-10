from __future__ import annotations

from orion.memory.crystallization.schemas import (
    MemoryCrystallizationProposeRequestV1,
    MemoryCrystallizationV1,
)
from orion.memory.crystallization.salience import apply_salience
from orion.memory.crystallization.validator import apply_validation_to_governance, validate_proposal


def propose(request: MemoryCrystallizationProposeRequestV1) -> MemoryCrystallizationV1:
    """Local model / operator may propose; proposal is never canonical."""
    crystallization = apply_salience(request.to_crystallization())
    result = validate_proposal(crystallization)
    return apply_validation_to_governance(crystallization, result)
