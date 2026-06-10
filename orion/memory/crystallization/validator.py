"""Proposal validation for MemoryCrystallizationV1 (spec section 22).

Validation is pure: it returns a list of error strings and never mutates the
proposal. The governor decides what to do with invalid proposals (quarantine).
"""

from __future__ import annotations

from orion.schemas.memory_crystallization import MemoryCrystallizationV1

# Kinds whose acceptance implies behavioral commitments.
PLANNING_EFFECT_KINDS = frozenset({"stance", "procedure", "decision"})

# Minimum link/evidence targets for an explicit contradiction artifact.
CONTRADICTION_MIN_TARGETS = 2


def validate_proposal(proposal: MemoryCrystallizationV1) -> list[str]:
    """Return validation errors for a crystallization proposal.

    An empty list means the proposal is structurally eligible for governance.
    Pydantic already enforces schema_version, kind, confidence literals and
    salience bounds; this layer enforces the cross-field governance rules.
    """
    errors: list[str] = []

    if proposal.status != "proposed":
        errors.append(f"status must be 'proposed' at validation time, got {proposal.status!r}")

    if not proposal.subject.strip():
        errors.append("subject must be non-empty")
    if not proposal.summary.strip():
        errors.append("summary must be non-empty")

    if not proposal.evidence:
        errors.append("evidence must be non-empty")
    for idx, ev in enumerate(proposal.evidence):
        if not ev.source_id.strip():
            errors.append(f"evidence[{idx}].source_id must be non-empty")

    if not proposal.scope:
        errors.append("scope must be non-empty")

    if not proposal.governance.proposed_by.strip():
        errors.append("governance.proposed_by must be non-empty")

    claim_ids = {claim.claim_id for claim in proposal.claims}
    if len(claim_ids) != len(proposal.claims):
        errors.append("claims must have unique claim_id values")

    if proposal.kind in PLANNING_EFFECT_KINDS and not proposal.planning_effects:
        errors.append(f"kind={proposal.kind!r} requires non-empty planning_effects")

    if proposal.kind == "stance" and not proposal.governance.requires_manual_review:
        errors.append("stance crystallizations require governance.requires_manual_review=true")

    if proposal.kind == "contradiction":
        targets = {link.target_crystallization_id for link in proposal.links}
        # The contradicted artifacts themselves also count as targets when
        # referenced as evidence.
        targets.update(
            ev.source_id for ev in proposal.evidence if ev.source_kind in {"memory_card", "graphiti_episode"}
        )
        if len(targets) < CONTRADICTION_MIN_TARGETS:
            errors.append(
                "contradiction crystallizations require at least "
                f"{CONTRADICTION_MIN_TARGETS} distinct targets (links/evidence)"
            )

    # Proposals must not claim canonical projections before governance.
    refs = proposal.projection_refs
    if refs.memory_card_ids or refs.chroma_doc_ids or refs.graphiti_episode_ids or refs.rdf_named_graphs:
        errors.append("projection_refs must be empty on a pre-canonical proposal")

    return errors


def mark_validation(
    proposal: MemoryCrystallizationV1, errors: list[str]
) -> MemoryCrystallizationV1:
    """Return a copy of the proposal with governance validation fields set."""
    governance = proposal.governance.model_copy(
        update={
            "validation_status": "valid" if not errors else "invalid",
            "validation_errors": list(errors),
        }
    )
    return proposal.model_copy(update={"governance": governance})
