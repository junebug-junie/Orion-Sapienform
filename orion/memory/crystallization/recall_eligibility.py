from __future__ import annotations

import os

from orion.memory.crystallization.schemas import MemoryCrystallizationV1

ACTIVATION_RECALL_FLOOR = float(os.getenv("ACTIVATION_RECALL_FLOOR", "0.08"))
_INELIGIBLE_STATUSES = frozenset({"deprecated", "archived", "rejected", "quarantined"})


def eligible_for_recall(
    crystallization: MemoryCrystallizationV1,
    *,
    floor: float | None = None,
) -> bool:
    if crystallization.status != "active":
        return False
    if crystallization.status in _INELIGIBLE_STATUSES:
        return False
    threshold = ACTIVATION_RECALL_FLOOR if floor is None else floor
    return float(crystallization.dynamics.activation) >= threshold
