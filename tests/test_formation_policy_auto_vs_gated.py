from orion.memory.crystallization.formation_policy import FormationPolicy, resolve_formation_policy
from orion.memory.crystallization.proposer import propose
from orion.memory.crystallization.schemas import (
    CrystallizationEvidenceRefV1,
    MemoryCrystallizationProposeRequestV1,
)


def _crys(*, kind="semantic", sensitivity="private", scope=None):
    req = MemoryCrystallizationProposeRequestV1(
        kind=kind,
        subject="Deploy plan",
        summary="We chose k3s for staging",
        scope=scope or ["project:orion"],
        evidence=[CrystallizationEvidenceRefV1(source_kind="memory_card", source_id="card_1", excerpt="fact")],
        proposed_by="test",
        sensitivity=sensitivity,
    )
    return propose(req)


def test_semantic_auto_activate():
    policy, reasons = resolve_formation_policy(_crys(kind="semantic"))
    assert policy == FormationPolicy.AUTO_ACTIVATE
    assert not reasons


def test_contradiction_governor_queue():
    policy, reasons = resolve_formation_policy(_crys(kind="contradiction"))
    assert policy == FormationPolicy.GOVERNOR_QUEUE


def test_intimate_governor_queue():
    policy, _ = resolve_formation_policy(_crys(sensitivity="intimate"))
    assert policy == FormationPolicy.GOVERNOR_QUEUE


def test_identity_scope_governor_queue():
    policy, _ = resolve_formation_policy(_crys(scope=["identity:orion"]))
    assert policy == FormationPolicy.GOVERNOR_QUEUE
