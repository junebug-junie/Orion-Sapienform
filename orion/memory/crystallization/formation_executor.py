from __future__ import annotations

from datetime import datetime, timezone

from orion.memory.crystallization.dynamics import seed_weak_dynamics
from orion.memory.crystallization.formation_policy import (
    DEFAULT_AUTO_ACTIVATE_PLATFORMS,
    FormationPolicy,
    resolve_formation_policy,
)
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
    auto_activate_platforms: frozenset[str] = DEFAULT_AUTO_ACTIVATE_PLATFORMS,
) -> tuple[MemoryCrystallizationV1, dict]:
    # Forwarded rather than left to the default: this function re-resolves the
    # policy independently of the caller's own resolve_formation_policy() call, so
    # a caller-supplied platform set that is not passed through here would be
    # silently ignored on exactly the path that decides activation.
    policy, reasons = resolve_formation_policy(
        crystallization, auto_activate_platforms=auto_activate_platforms
    )
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
