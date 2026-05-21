"""Bus catalog and schema registry coverage for autonomy goal artifacts."""

from __future__ import annotations

from pathlib import Path

import yaml

from orion.core.schemas.drives import AutonomyGoalPlannedV1, GoalProposalV1
from orion.schemas.registry import _REGISTRY, resolve

ROOT = Path(__file__).resolve().parents[1]
CHANNELS_YAML = ROOT / "orion" / "bus" / "channels.yaml"


def _channel_entry(name: str) -> dict:
    doc = yaml.safe_load(CHANNELS_YAML.read_text(encoding="utf-8")) or {}
    for entry in doc.get("channels") or []:
        if isinstance(entry, dict) and entry.get("name") == name:
            return entry
    raise AssertionError(f"Missing channel catalog entry: {name}")


def test_goal_proposal_schema_in_registry() -> None:
    assert "GoalProposalV1" in _REGISTRY
    assert resolve("GoalProposalV1") is GoalProposalV1
    goal = GoalProposalV1.model_validate(
        {
            "artifact_id": "goal-1",
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "kind": "memory.goals.proposed.v1",
            "goal_statement": "Clarify autonomy boundaries without executing any new action.",
            "proposal_signature": "sig-1",
            "drive_origin": "autonomy",
            "priority": 0.5,
            "provenance": {
                "intake_channel": "orion:metacognition:tick",
                "source_event_refs": [],
                "evidence_items": [],
                "tension_refs": [],
            },
        }
    )
    assert goal.proposal_status == "proposed"
    assert goal.semantic_source == "template"


def test_memory_goals_proposed_channel_cataloged() -> None:
    entry = _channel_entry("orion:memory:goals:proposed")
    assert entry["schema_id"] == "GoalProposalV1"
    assert entry["message_kind"] == "memory.goals.proposed.v1"
    assert "orion-spark-concept-induction" in (entry.get("producer_services") or [])


def test_autonomy_goal_planned_schema_and_channel_cataloged() -> None:
    assert "AutonomyGoalPlannedV1" in _REGISTRY
    assert resolve("AutonomyGoalPlannedV1") is AutonomyGoalPlannedV1

    entry = _channel_entry("orion:autonomy:goal:planned")
    assert entry["schema_id"] == "AutonomyGoalPlannedV1"
    assert entry["message_kind"] == "autonomy.goal.planned.v1"
    assert "orion-planner-react" in (entry.get("producer_services") or [])

    planned = AutonomyGoalPlannedV1.model_validate(
        {
            "goal_artifact_id": "goal-abc",
            "goal_statement": "Archive stale goals",
            "drive_origin": "autonomy",
            "task_id": "goal-task-1",
        }
    )
    assert planned.kind == "autonomy.goal.planned.v1"
