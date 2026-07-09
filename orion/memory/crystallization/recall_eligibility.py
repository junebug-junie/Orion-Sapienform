from __future__ import annotations

import os

from orion.memory.crystallization.schemas import MemoryCrystallizationV1

ACTIVATION_RECALL_FLOOR = float(os.getenv("ACTIVATION_RECALL_FLOOR", "0.08"))


def eligible_for_recall(
    crystallization: MemoryCrystallizationV1,
    *,
    floor: float | None = None,
) -> bool:
    """Return True when an active crystallization is strong enough for recall injection.

    Legacy governor-approved rows with activation=0.0 remain ineligible until re-approved
    (post-M1 governor.approve seeds dynamics) or manually reinforced.
    """
    if crystallization.status != "active":
        return False
    threshold = ACTIVATION_RECALL_FLOOR if floor is None else floor
    return float(crystallization.dynamics.activation) >= threshold
