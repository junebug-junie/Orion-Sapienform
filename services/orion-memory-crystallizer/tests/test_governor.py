"""Governor transition rules: only the governor path canonizes."""

from __future__ import annotations

import pytest
from conftest import make_proposal

from orion.memory.crystallization import governor


def test_approve_canonizes_valid_proposal(proposal) -> None:
    approved, entry = governor.approve(proposal, "operator:juniper")
    assert approved.status == "active"
    assert approved.governance.approved_by == "operator:juniper"
    assert 0.0 <= approved.salience <= 1.0
    assert entry.op == "approve"
    # input not mutated
    assert proposal.status == "proposed"


def test_approve_rejects_invalid_proposal() -> None:
    invalid = make_proposal(evidence=[])
    with pytest.raises(governor.GovernanceError):
        governor.approve(invalid, "operator:juniper")


def test_approve_requires_proposed_status(proposal) -> None:
    rejected, _ = governor.reject(proposal, "operator:juniper")
    with pytest.raises(governor.GovernanceError):
        governor.approve(rejected, "operator:juniper")


def test_reject_preserves_artifact(proposal) -> None:
    rejected, entry = governor.reject(proposal, "operator:juniper", reason="not supported")
    assert rejected.status == "rejected"
    assert rejected.summary == proposal.summary
    assert entry.reason == "not supported"


def test_quarantine_marks_governance(proposal) -> None:
    quarantined, entry = governor.quarantine(proposal, "governor", reason="suspicious")
    assert quarantined.status == "quarantined"
    assert quarantined.governance.validation_status == "quarantined"
    assert entry.op == "quarantine"


def test_supersession_preserves_old_and_links_new(proposal) -> None:
    old, _ = governor.approve(proposal, "operator:juniper")
    new_proposal = make_proposal(subject="Updated stance")
    new, _ = governor.approve(new_proposal, "operator:juniper")

    superseded_old, updated_new, entries = governor.supersede(old, new, "operator:juniper")
    assert superseded_old.status == "superseded"
    # old content untouched
    assert superseded_old.summary == old.summary
    # explicit supersedes link on the new artifact
    assert any(
        l.relation == "supersedes" and l.target_crystallization_id == old.crystallization_id
        for l in updated_new.links
    )
    assert len(entries) == 2


def test_supersede_requires_active_new(proposal) -> None:
    old, _ = governor.approve(proposal, "operator:juniper")
    not_active = make_proposal()
    with pytest.raises(governor.GovernanceError):
        governor.supersede(old, not_active, "operator:juniper")


def test_set_status_only_for_canonical(proposal) -> None:
    with pytest.raises(governor.GovernanceError):
        governor.set_status(proposal, "deprecated", "operator:juniper")

    active, _ = governor.approve(proposal, "operator:juniper")
    deprecated, entry = governor.set_status(active, "deprecated", "operator:juniper")
    assert deprecated.status == "deprecated"
    assert entry.before_status == "active"


def test_set_status_cannot_force_active(proposal) -> None:
    active, _ = governor.approve(proposal, "operator:juniper")
    with pytest.raises(governor.GovernanceError):
        governor.set_status(active, "active", "operator:juniper")


def test_validate_marks_governance_block(proposal) -> None:
    validated, entry = governor.validate(proposal, "governor")
    assert validated.governance.validation_status == "valid"
    assert entry.op == "validate"

    invalid = make_proposal(scope=[])
    marked, _ = governor.validate(invalid, "governor")
    assert marked.governance.validation_status == "invalid"
    assert marked.governance.validation_errors
