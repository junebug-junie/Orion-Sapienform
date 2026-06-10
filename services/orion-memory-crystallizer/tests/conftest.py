import os
import sys
from datetime import datetime, timezone

import pytest

# Make app/ importable and repo root importable (service dir is hyphenated).
SERVICE_ROOT = os.path.join(os.path.dirname(__file__), "..")
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, SERVICE_ROOT)
sys.path.insert(0, REPO_ROOT)

from orion.schemas.memory_crystallization import (  # noqa: E402
    CrystallizationEvidenceRefV1,
    CrystallizationGovernanceV1,
    MemoryCrystallizationV1,
    new_crystallization_id,
)


def make_proposal(**overrides) -> MemoryCrystallizationV1:
    now = datetime.now(timezone.utc)
    base = dict(
        crystallization_id=new_crystallization_id(),
        kind="stance",
        subject="Local-first Orion memory governance",
        summary="Prefer local-first memory governance; avoid frontier models inside canonical memory loops.",
        status="proposed",
        confidence="likely",
        scope=["project:orion", "layer:memory"],
        evidence=[
            CrystallizationEvidenceRefV1(
                source_kind="memory_card",
                source_id="card-123",
                excerpt="operator note about local-first governance",
                strength=0.8,
            )
        ],
        planning_effects=["prefer local extraction", "require governor before canonical write"],
        retrieval_affordances=["retrieve_when:memory_architecture"],
        governance=CrystallizationGovernanceV1(proposed_by="test-proposer"),
        created_at=now,
        updated_at=now,
    )
    base.update(overrides)
    return MemoryCrystallizationV1(**base)


@pytest.fixture
def proposal() -> MemoryCrystallizationV1:
    return make_proposal()
