from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.schemas.drives import DriveStateV1
from orion.spark.concept_induction.goals import GoalProposalEngine


def test_goal_provenance_carries_spawned_correlation_id() -> None:
    now = datetime(2026, 7, 6, tzinfo=timezone.utc)
    env = BaseEnvelope(
        id=uuid4(),
        kind="world.pulse.run.result.v1",
        correlation_id=uuid4(),
        created_at=now,
        source=ServiceRef(name="t", version="0", node="n"),
        payload={},
    )
    drive_state = DriveStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind="memory.drives.state.v1",
        pressures={"predictive": 0.7},
        activations={"predictive": True},
        updated_at=now,
        provenance={"intake_channel": "orion:world_pulse:run:result"},
    )
    engine = GoalProposalEngine(cooldown_minutes=0)
    store = MagicMock()
    store.load_goal_cooldown.return_value = None
    store.load_goal_slot.return_value = {}
    decision = engine.propose(
        env=env,
        intake_channel="orion:world_pulse:run:result",
        drive_state=drive_state,
        tensions=[],
        store=store,
        dominant_drive="predictive",
        window_summary="hardware_compute_gpu gap",
        spawned_correlation_id="wp-run-gap-gpu",
    )
    assert decision.proposal is not None
    assert decision.proposal.provenance.spawned_correlation_id == "wp-run-gap-gpu"
