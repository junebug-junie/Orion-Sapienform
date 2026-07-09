from __future__ import annotations

from datetime import datetime, timezone

from orion.memory.crystallization.recall_eligibility import ACTIVATION_RECALL_FLOOR, eligible_for_recall
from orion.memory.crystallization.schemas import (
    CrystallizationDynamicsV1,
    CrystallizationGovernanceV1,
    MemoryCrystallizationV1,
)


def _active(activation: float) -> MemoryCrystallizationV1:
    now = datetime.now(timezone.utc)
    return MemoryCrystallizationV1(
        crystallization_id="crys_test",
        kind="semantic",
        subject="s",
        summary="s",
        status="active",
        dynamics=CrystallizationDynamicsV1(activation=activation, formed_at=now),
        governance=CrystallizationGovernanceV1(proposed_by="t"),
        created_at=now,
        updated_at=now,
    )


def test_eligible_when_active_and_above_floor():
    assert eligible_for_recall(_active(0.2)) is True


def test_ineligible_when_below_floor():
    assert eligible_for_recall(_active(0.01)) is False
    assert eligible_for_recall(_active(ACTIVATION_RECALL_FLOOR - 0.01)) is False


def test_ineligible_when_deprecated():
    crys = _active(0.2)
    crys.status = "deprecated"
    assert eligible_for_recall(crys) is False
