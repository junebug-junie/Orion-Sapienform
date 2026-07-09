from __future__ import annotations

from enum import Enum

from orion.memory.crystallization.schemas import MemoryCrystallizationV1

AUTO_ACTIVE_KINDS = frozenset({"semantic", "episode", "open_loop", "procedure"})
GATED_KINDS = frozenset({"stance", "decision", "contradiction", "attractor", "failure_mode"})
IDENTITY_SCOPE_PREFIX = "identity:"


class FormationPolicy(str, Enum):
    AUTO_ACTIVATE = "auto_activate"
    GOVERNOR_QUEUE = "governor_queue"
    REINFORCE_EXISTING = "reinforce_existing"


def _has_identity_scope(crystallization: MemoryCrystallizationV1, *, prefix: str = IDENTITY_SCOPE_PREFIX) -> bool:
    return any(str(s).startswith(prefix) for s in crystallization.scope)


def resolve_formation_policy(
    crystallization: MemoryCrystallizationV1,
    *,
    duplicate_id: str | None = None,
    identity_scope_prefix: str = IDENTITY_SCOPE_PREFIX,
) -> tuple[FormationPolicy, list[str]]:
    reasons: list[str] = []
    if duplicate_id:
        return FormationPolicy.REINFORCE_EXISTING, [f"duplicate:{duplicate_id}"]
    if crystallization.governance.sensitivity == "intimate":
        return FormationPolicy.GOVERNOR_QUEUE, ["intimate_sensitivity"]
    if _has_identity_scope(crystallization, prefix=identity_scope_prefix):
        return FormationPolicy.GOVERNOR_QUEUE, ["identity_scope"]
    if crystallization.kind in GATED_KINDS:
        return FormationPolicy.GOVERNOR_QUEUE, [f"gated_kind:{crystallization.kind}"]
    if crystallization.kind in AUTO_ACTIVE_KINDS:
        return FormationPolicy.AUTO_ACTIVATE, reasons
    return FormationPolicy.GOVERNOR_QUEUE, [f"unknown_kind:{crystallization.kind}"]
