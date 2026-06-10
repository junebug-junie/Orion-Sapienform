"""Governor transitions for MemoryCrystallizationV1.

The governor is the only path to canonical status. All transitions are pure:
they take a crystallization, return a new crystallization plus a
GovernanceHistoryEntry, and never mutate inputs. Persistence and bus emission
live in the repository/service layer.

Invariants enforced here:
- only validated proposals can become active
- rejection/quarantine preserve the artifact (status change, not deletion)
- supersession is explicit and preserves the superseded artifact
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from orion.memory.crystallization.salience import score_salience
from orion.memory.crystallization.validator import mark_validation, validate_proposal
from orion.schemas.memory_crystallization import (
    CrystallizationLinkV1,
    MemoryCrystallizationV1,
)


class GovernanceError(Exception):
    """Raised when a governance transition is not allowed."""


@dataclass(frozen=True)
class GovernanceHistoryEntry:
    op: str
    actor: str
    crystallization_id: str
    reason: str | None = None
    before_status: str | None = None
    after_status: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def validate(
    proposal: MemoryCrystallizationV1, actor: str
) -> tuple[MemoryCrystallizationV1, GovernanceHistoryEntry]:
    """Run validation and record the outcome on the proposal's governance block."""
    errors = validate_proposal(proposal)
    validated = mark_validation(proposal, errors).model_copy(update={"updated_at": _now()})
    entry = GovernanceHistoryEntry(
        op="validate",
        actor=actor,
        crystallization_id=proposal.crystallization_id,
        before_status=proposal.status,
        after_status=validated.status,
        detail={"validation_errors": errors},
    )
    return validated, entry


def approve(
    proposal: MemoryCrystallizationV1,
    actor: str,
    *,
    approval_mode: str = "operator",
    salience_override: float | None = None,
) -> tuple[MemoryCrystallizationV1, GovernanceHistoryEntry]:
    """Canonize a proposal. Requires a prior successful validation pass."""
    if proposal.status != "proposed":
        raise GovernanceError(f"cannot approve from status {proposal.status!r}")

    errors = validate_proposal(proposal)
    if errors:
        raise GovernanceError(f"cannot approve invalid proposal: {errors}")

    salience = salience_override if salience_override is not None else score_salience(proposal)
    governance = proposal.governance.model_copy(
        update={
            "approved_by": actor,
            "approval_mode": approval_mode,
            "validation_status": "valid",
            "validation_errors": [],
            "last_reviewed_at": _now(),
        }
    )
    approved = proposal.model_copy(
        update={
            "status": "active",
            "salience": salience,
            "governance": governance,
            "updated_at": _now(),
        }
    )
    entry = GovernanceHistoryEntry(
        op="approve",
        actor=actor,
        crystallization_id=proposal.crystallization_id,
        before_status=proposal.status,
        after_status="active",
        detail={"salience": salience, "approval_mode": approval_mode},
    )
    return approved, entry


def reject(
    proposal: MemoryCrystallizationV1, actor: str, *, reason: str | None = None
) -> tuple[MemoryCrystallizationV1, GovernanceHistoryEntry]:
    if proposal.status != "proposed":
        raise GovernanceError(f"cannot reject from status {proposal.status!r}")
    governance = proposal.governance.model_copy(update={"last_reviewed_at": _now()})
    rejected = proposal.model_copy(
        update={"status": "rejected", "governance": governance, "updated_at": _now()}
    )
    entry = GovernanceHistoryEntry(
        op="reject",
        actor=actor,
        crystallization_id=proposal.crystallization_id,
        reason=reason,
        before_status=proposal.status,
        after_status="rejected",
    )
    return rejected, entry


def quarantine(
    proposal: MemoryCrystallizationV1, actor: str, *, reason: str | None = None
) -> tuple[MemoryCrystallizationV1, GovernanceHistoryEntry]:
    """Quarantined artifacts are preserved but excluded from all projections."""
    if proposal.status in {"superseded", "archived"}:
        raise GovernanceError(f"cannot quarantine from status {proposal.status!r}")
    governance = proposal.governance.model_copy(
        update={"validation_status": "quarantined", "last_reviewed_at": _now()}
    )
    quarantined = proposal.model_copy(
        update={"status": "quarantined", "governance": governance, "updated_at": _now()}
    )
    entry = GovernanceHistoryEntry(
        op="quarantine",
        actor=actor,
        crystallization_id=proposal.crystallization_id,
        reason=reason,
        before_status=proposal.status,
        after_status="quarantined",
    )
    return quarantined, entry


def supersede(
    old: MemoryCrystallizationV1,
    new: MemoryCrystallizationV1,
    actor: str,
    *,
    reason: str | None = None,
) -> tuple[MemoryCrystallizationV1, MemoryCrystallizationV1, list[GovernanceHistoryEntry]]:
    """Mark `old` superseded by `new`. Both artifacts are preserved.

    `new` must already be active (governed); supersession never silently edits
    the old crystallization's content, only its status and links.
    """
    if new.status != "active":
        raise GovernanceError("superseding crystallization must be active")
    if old.status not in {"active", "deprecated"}:
        raise GovernanceError(f"cannot supersede from status {old.status!r}")
    if old.crystallization_id == new.crystallization_id:
        raise GovernanceError("a crystallization cannot supersede itself")

    superseded_old = old.model_copy(update={"status": "superseded", "updated_at": _now()})
    link = CrystallizationLinkV1(
        target_crystallization_id=old.crystallization_id,
        relation="supersedes",
        confidence=1.0,
        note=reason,
    )
    new_links = list(new.links)
    if not any(
        l.target_crystallization_id == old.crystallization_id and l.relation == "supersedes"
        for l in new_links
    ):
        new_links.append(link)
    updated_new = new.model_copy(update={"links": new_links, "updated_at": _now()})

    entries = [
        GovernanceHistoryEntry(
            op="supersede",
            actor=actor,
            crystallization_id=old.crystallization_id,
            reason=reason,
            before_status=old.status,
            after_status="superseded",
            detail={"superseded_by": new.crystallization_id},
        ),
        GovernanceHistoryEntry(
            op="supersede_link",
            actor=actor,
            crystallization_id=new.crystallization_id,
            reason=reason,
            detail={"supersedes": old.crystallization_id},
        ),
    ]
    return superseded_old, updated_new, entries


def set_status(
    crystallization: MemoryCrystallizationV1,
    new_status: str,
    actor: str,
    *,
    reason: str | None = None,
) -> tuple[MemoryCrystallizationV1, GovernanceHistoryEntry]:
    """Lifecycle transitions on canonical artifacts (deprecate/archive).

    `active` is only reachable via approve(); `superseded` only via
    supersede(); `proposed` is never re-entered.
    """
    allowed_targets = {"deprecated", "archived"}
    if new_status not in allowed_targets:
        raise GovernanceError(f"set_status target must be one of {sorted(allowed_targets)}")
    if crystallization.status in {"proposed", "rejected", "quarantined"}:
        raise GovernanceError(
            f"cannot transition non-canonical status {crystallization.status!r} via set_status"
        )
    updated = crystallization.model_copy(update={"status": new_status, "updated_at": _now()})
    entry = GovernanceHistoryEntry(
        op="status_change",
        actor=actor,
        crystallization_id=crystallization.crystallization_id,
        reason=reason,
        before_status=crystallization.status,
        after_status=new_status,
    )
    return updated, entry
