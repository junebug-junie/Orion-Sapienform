from __future__ import annotations

from dataclasses import dataclass, field

from orion.memory.crystallization.schemas import (
    CONTRADICTION_KIND,
    STANCE_PROCEDURE_DECISION_KINDS,
    MemoryCrystallizationV1,
)


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    quarantine: bool = False


def validate_proposal(crystallization: MemoryCrystallizationV1) -> ValidationResult:
    """Validate a crystallization proposal before governor review."""
    errors: list[str] = []

    if crystallization.schema_version != "memory_crystallization.v1":
        errors.append("schema_version must be memory_crystallization.v1")

    if crystallization.status != "proposed":
        errors.append("proposals must start with status=proposed")

    if not (crystallization.summary or "").strip():
        errors.append("summary is required")

    if not crystallization.evidence:
        errors.append("evidence is required")

    if not crystallization.scope:
        errors.append("scope is required")

    if crystallization.governance is None:
        errors.append("governance block is required")

    if crystallization.kind in STANCE_PROCEDURE_DECISION_KINDS:
        if not crystallization.planning_effects:
            errors.append(f"{crystallization.kind} requires planning_effects")
        if crystallization.kind == "stance" and not crystallization.retrieval_affordances:
            errors.append("stance requires retrieval_affordances")

    if crystallization.kind == CONTRADICTION_KIND:
        contradiction_targets = [
            link
            for link in crystallization.links
            if link.relation in ("contradicts", "evidence_against")
        ]
        if len(contradiction_targets) < 2:
            errors.append("contradiction requires at least two linked targets")

    refs = crystallization.projection_refs
    if refs and refs.synced_at is not None:
        errors.append("projection_refs must be empty or pre-canonical on proposals")

    if any(refs.memory_card_ids) or any(refs.chroma_doc_ids) or any(refs.graphiti_episode_ids):
        errors.append("projection_refs must be empty on proposals")

    quarantine = bool(errors)
    return ValidationResult(valid=not errors, errors=errors, quarantine=quarantine)


def apply_validation_to_governance(crystallization: MemoryCrystallizationV1, result: ValidationResult) -> MemoryCrystallizationV1:
    updated = crystallization.model_copy(deep=True)
    if result.valid:
        updated.governance.validation_status = "valid"
        updated.governance.validation_errors = []
    elif result.quarantine:
        updated.governance.validation_status = "quarantined"
        updated.governance.validation_errors = list(result.errors)
        updated.status = "quarantined"
    else:
        updated.governance.validation_status = "invalid"
        updated.governance.validation_errors = list(result.errors)
    return updated
