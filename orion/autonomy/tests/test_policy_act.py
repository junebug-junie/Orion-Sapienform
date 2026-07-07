from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orion.autonomy.policy_act import (
    build_readonly_fetch_query,
    maybe_execute_readonly_fetch_after_goal,
)
from orion.core.schemas.drives import DriveStateV1, GoalProposalV1
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1


def _gap_signal() -> FrontierInvocationSignalV1:
    return FrontierInvocationSignalV1(
        signal_type="world_coverage_gap",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="concept_graph",
        task_type_candidate="concept_expand",
        focal_node_refs=["section:hardware_compute_gpu"],
        signal_strength=0.65,
        evidence_summary="world coverage gap: hardware_compute_gpu had zero digest items",
        confidence=0.65,
    )


def _goal() -> GoalProposalV1:
    return GoalProposalV1.model_validate(
        {
            "artifact_id": "goal-gap-gpu",
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "kind": "memory.goals.proposed.v1",
            "goal_statement": "Reduce predictive uncertainty for hardware_compute_gpu.",
            "proposal_signature": "sig",
            "drive_origin": "predictive",
            "proposal_status": "proposed",
            "provenance": {"intake_channel": "orion:world_pulse:run:result"},
        }
    )


def _drive_state(predictive: float = 0.7) -> DriveStateV1:
    return DriveStateV1.model_validate(
        {
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "kind": "memory.drives.state.v1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "provenance": {"intake_channel": "orion:world_pulse:run:result"},
            "pressures": {
                "coherence": 0.5,
                "continuity": 0.5,
                "capability": 0.5,
                "relational": 0.5,
                "predictive": predictive,
                "autonomy": 0.5,
            },
            "activations": {
                "coherence": False,
                "continuity": False,
                "capability": False,
                "relational": False,
                "predictive": True,
                "autonomy": False,
            },
        }
    )


def test_build_readonly_fetch_query_from_gap_section() -> None:
    query = build_readonly_fetch_query([_gap_signal()])
    assert "hardware compute gpu" in query


@pytest.mark.asyncio
async def test_policy_act_executes_fetch_when_allowed(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))

    backend = AsyncMock(return_value={"success": True, "urls": ["https://example.com/a"]})
    decision, outcome = await maybe_execute_readonly_fetch_after_goal(
        goal=_goal(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        spawned_correlation_id="wp-run-gap-gpu",
        fetch_backend=backend,
    )
    assert decision.outcome == "allowed"
    assert decision.auto_execute is True
    assert outcome is not None
    assert outcome.success is True
    backend.assert_awaited_once()


@pytest.mark.asyncio
async def test_policy_act_denied_when_pressure_low(monkeypatch) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    decision, outcome = await maybe_execute_readonly_fetch_after_goal(
        goal=_goal(),
        drive_state=_drive_state(predictive=0.2),
        curiosity_signals=[_gap_signal()],
        spawned_correlation_id="wp-run-gap-gpu",
        fetch_backend=AsyncMock(),
    )
    assert decision.outcome == "denied"
    assert decision.reason_code == "predictive_pressure_insufficient"
    assert outcome is None
