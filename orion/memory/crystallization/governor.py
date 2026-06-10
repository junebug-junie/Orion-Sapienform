from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from orion.memory.crystallization.schemas import MemoryCrystallizationV1
from orion.memory.crystallization.validator import ValidationResult, validate_proposal


GovernorAction = Literal["approve", "reject", "quarantine", "supersede"]


class GovernorError(ValueError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def can_activate(crystallization: MemoryCrystallizationV1) -> ValidationResult:
    """A crystallization can only become active via governor path."""
    if crystallization.status != "proposed":
        return ValidationResult(valid=False, errors=[f"cannot activate from status={crystallization.status}"])

    result = validate_proposal(crystallization)
    if not result.valid:
        return result

    if crystallization.governance.validation_status not in ("valid", "unvalidated"):
        return ValidationResult(
            valid=False,
            errors=[f"validation_status={crystallization.governance.validation_status}"],
        )

    if crystallization.governance.requires_manual_review and not crystallization.governance.approved_by:
        return ValidationResult(valid=False, errors=["manual_review_required"])

    return ValidationResult(valid=True)


def approve(
    crystallization: MemoryCrystallizationV1,
    *,
    actor: str,
    reason: str | None = None,
) -> tuple[MemoryCrystallizationV1, dict]:
    """Governor approves a proposed crystallization → active."""
    check = can_activate(crystallization)
    if not check.valid:
        raise GovernorError("; ".join(check.errors))

    now = _utc_now()
    updated = crystallization.model_copy(deep=True)
    updated.status = "active"
    updated.governance.approved_by = actor
    updated.governance.validation_status = "valid"
    updated.governance.last_reviewed_at = now
    updated.updated_at = now

    history = {
        "op": "approve",
        "actor": actor,
        "reason": reason,
        "before": {"status": crystallization.status},
        "after": {"status": "active"},
    }
    return updated, history


def reject(
    crystallization: MemoryCrystallizationV1,
    *,
    actor: str,
    reason: str | None = None,
) -> tuple[MemoryCrystallizationV1, dict]:
    if crystallization.status not in ("proposed", "quarantined"):
        raise GovernorError(f"cannot reject from status={crystallization.status}")

    now = _utc_now()
    updated = crystallization.model_copy(deep=True)
    updated.status = "rejected"
    updated.governance.approved_by = None
    updated.governance.last_reviewed_at = now
    updated.updated_at = now

    history = {
        "op": "reject",
        "actor": actor,
        "reason": reason,
        "before": {"status": crystallization.status},
        "after": {"status": "rejected"},
    }
    return updated, history


def quarantine(
    crystallization: MemoryCrystallizationV1,
    *,
    actor: str,
    errors: list[str],
    reason: str | None = None,
) -> tuple[MemoryCrystallizationV1, dict]:
    now = _utc_now()
    updated = crystallization.model_copy(deep=True)
    updated.status = "quarantined"
    updated.governance.validation_status = "quarantined"
    updated.governance.validation_errors = list(errors)
    updated.governance.last_reviewed_at = now
    updated.updated_at = now

    history = {
        "op": "quarantine",
        "actor": actor,
        "reason": reason,
        "before": {"status": crystallization.status},
        "after": {"status": "quarantined", "errors": errors},
    }
    return updated, history


def supersede(
    crystallization: MemoryCrystallizationV1,
    *,
    actor: str,
    superseded_by_id: str,
    reason: str | None = None,
) -> tuple[MemoryCrystallizationV1, dict]:
    """Mark crystallization superseded; preserves artifact."""
    if crystallization.status not in ("active", "proposed"):
        raise GovernorError(f"cannot supersede from status={crystallization.status}")

    now = _utc_now()
    updated = crystallization.model_copy(deep=True)
    updated.status = "superseded"
    updated.governance.last_reviewed_at = now
    updated.updated_at = now

    history = {
        "op": "supersede",
        "actor": actor,
        "reason": reason,
        "before": {"status": crystallization.status},
        "after": {"status": "superseded", "superseded_by": superseded_by_id},
    }
    return updated, history
