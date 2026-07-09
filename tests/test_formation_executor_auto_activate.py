import pytest

from orion.memory.crystallization.formation_executor import GovernorPathRequired, auto_activate
from orion.memory.crystallization.proposer import propose
from orion.memory.crystallization.schemas import (
    CrystallizationEvidenceRefV1,
    MemoryCrystallizationProposeRequestV1,
)


def _proposed_semantic():
    req = MemoryCrystallizationProposeRequestV1(
        kind="semantic",
        subject="Topic",
        summary="We agreed on Postgres for memory store",
        scope=["project:orion"],
        evidence=[CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="corr-1")],
        proposed_by="memory_consolidation_intake",
    )
    return propose(req)


def test_auto_activate_sets_active_and_weak_dynamics():
    crys, hist = auto_activate(_proposed_semantic(), encode_ratio=0.4)
    assert crys.status == "active"
    assert crys.governance.approval_mode == "auto_policy"
    assert crys.governance.requires_manual_review is False
    assert crys.governance.approved_by == "system:formation_policy"
    assert 0.05 <= crys.dynamics.activation <= 0.35
    assert hist["op"] == "auto_activate"


def test_auto_activate_rejects_contradiction():
    req = MemoryCrystallizationProposeRequestV1(
        kind="contradiction",
        subject="x",
        summary="y",
        scope=["p"],
        evidence=[CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="corr-2")],
        proposed_by="t",
    )
    with pytest.raises(GovernorPathRequired):
        auto_activate(propose(req))
