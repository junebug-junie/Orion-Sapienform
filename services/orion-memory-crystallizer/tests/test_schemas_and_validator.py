"""MemoryCrystallizationV1 schema and proposal validation rules (spec 22/23)."""

from __future__ import annotations

import pytest
from conftest import make_proposal
from pydantic import ValidationError

from orion.memory.crystallization.validator import validate_proposal
from orion.schemas.memory_crystallization import (
    CrystallizationEvidenceRefV1,
    CrystallizationLinkV1,
    MemoryCrystallizationV1,
)


def test_schema_validates_and_roundtrips(proposal) -> None:
    dumped = proposal.model_dump(mode="json")
    assert dumped["schema_version"] == "memory_crystallization.v1"
    restored = MemoryCrystallizationV1.model_validate(dumped)
    assert restored == proposal


def test_schema_forbids_extra_fields(proposal) -> None:
    dumped = proposal.model_dump(mode="json")
    dumped["unexpected"] = "nope"
    with pytest.raises(ValidationError):
        MemoryCrystallizationV1.model_validate(dumped)


def test_valid_stance_proposal_passes(proposal) -> None:
    assert validate_proposal(proposal) == []


def test_evidence_required() -> None:
    proposal = make_proposal(evidence=[])
    errors = validate_proposal(proposal)
    assert any("evidence" in e for e in errors)


def test_scope_required() -> None:
    proposal = make_proposal(scope=[])
    errors = validate_proposal(proposal)
    assert any("scope" in e for e in errors)


def test_stance_requires_planning_effects() -> None:
    proposal = make_proposal(planning_effects=[])
    errors = validate_proposal(proposal)
    assert any("planning_effects" in e for e in errors)


def test_contradiction_requires_two_targets() -> None:
    proposal = make_proposal(
        kind="contradiction",
        planning_effects=[],
        links=[
            CrystallizationLinkV1(target_crystallization_id="crys_a", relation="contradicts"),
        ],
        evidence=[
            CrystallizationEvidenceRefV1(source_kind="grammar_event", source_id="gev_1"),
        ],
    )
    errors = validate_proposal(proposal)
    assert any("contradiction" in e for e in errors)

    ok = make_proposal(
        kind="contradiction",
        planning_effects=[],
        links=[
            CrystallizationLinkV1(target_crystallization_id="crys_a", relation="contradicts"),
            CrystallizationLinkV1(target_crystallization_id="crys_b", relation="contradicts"),
        ],
    )
    assert validate_proposal(ok) == []


def test_grammar_event_sources_supported() -> None:
    proposal = make_proposal(
        kind="semantic",
        planning_effects=[],
        evidence=[CrystallizationEvidenceRefV1(source_kind="grammar_event", source_id="gev_001")],
        source_grammar_event_ids=["gev_001"],
    )
    assert validate_proposal(proposal) == []


def test_pre_canonical_projection_refs_rejected(proposal) -> None:
    bad = proposal.model_copy(
        update={
            "projection_refs": proposal.projection_refs.model_copy(
                update={"memory_card_ids": ["card-x"]}
            )
        }
    )
    errors = validate_proposal(bad)
    assert any("projection_refs" in e for e in errors)


def test_non_proposed_status_rejected(proposal) -> None:
    active = proposal.model_copy(update={"status": "active"})
    errors = validate_proposal(active)
    assert any("status" in e for e in errors)


def test_salience_bounds_enforced(proposal) -> None:
    dumped = proposal.model_dump(mode="json")
    dumped["salience"] = 1.5
    with pytest.raises(ValidationError):
        MemoryCrystallizationV1.model_validate(dumped)
