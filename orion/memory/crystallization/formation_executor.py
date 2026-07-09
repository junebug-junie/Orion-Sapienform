from __future__ import annotations

from datetime import datetime, timezone

from orion.memory.crystallization.dynamics import seed_weak_dynamics
from orion.memory.crystallization.formation_policy import FormationPolicy, resolve_formation_policy
from orion.memory.crystallization.schemas import MemoryCrystallizationV1
from orion.memory.crystallization.validator import validate_proposal


class GovernorPathRequired(ValueError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def auto_activate(
    crystallization: MemoryCrystallizationV1,
    *,
    actor: str = "system:formation_policy",
    encode_ratio: float = 0.4,
) -> tuple[MemoryCrystallizationV1, dict]:
    policy, reasons = resolve_formation_policy(crystallization)
    if policy != FormationPolicy.AUTO_ACTIVATE:
        raise GovernorPathRequired("; ".join(reasons) or policy.value)
    validation = validate_proposal(crystallization)
    if not validation.valid:
        raise GovernorPathRequired("; ".join(validation.errors))
    now = _utc_now()
    updated = crystallization.model_copy(deep=True)
    updated.status = "active"
    updated.governance.approval_mode = "auto_policy"
    updated.governance.requires_manual_review = False
    updated.governance.approved_by = actor
    updated.governance.validation_status = "valid"
    updated.governance.last_reviewed_at = now
    updated = seed_weak_dynamics(updated, now=now, ratio=encode_ratio)
    history = {
        "op": "auto_activate",
        "actor": actor,
        "reasons": reasons,
        "before": {"status": crystallization.status},
        "after": {"status": "active", "activation": updated.dynamics.activation},
    }
    return updated, history
